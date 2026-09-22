"""``ScriptedLLM``: ordered rules, the reply/memory split, and the JSON rules file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rt_agent.contracts import ChatLLM, ChatMessage
from rt_agent.llm import DEFAULT_REPLY, RULES_SCHEMA_VERSION, ScriptedLLM, ScriptedRule
from rt_agent.llm.prompts import memory_summary_messages, reply_messages
from tests.conftest import make_context, make_utterance

RULES = [
    {"match": r"\btime\b", "reply": "It is just after four."},
    {"match": r"\bdrink\b", "reply": "There is tea in the pot."},
]

MEMORY_RULES = [{"match": r"\bpeanut", "reply": "S2 is allergic to peanuts."}]

DOCUMENT: dict[str, Any] = {
    "schema_version": RULES_SCHEMA_VERSION,
    "default": "I am not sure.",
    "rules": RULES,
    "memory_default": "",
    "memory_rules": MEMORY_RULES,
}


def user(text: str) -> list[ChatMessage]:
    """A bare one-message prompt."""
    return [ChatMessage(role="user", content=text)]


class TestRules:
    def test_it_satisfies_the_chat_llm_protocol(self) -> None:
        assert isinstance(ScriptedLLM(), ChatLLM)

    async def test_first_matching_rule_wins(self) -> None:
        llm = ScriptedLLM([{"match": "a", "reply": "first"}, {"match": "a", "reply": "second"}])
        assert await llm.complete(user("banana")) == "first"

    async def test_matching_is_case_insensitive(self) -> None:
        llm = ScriptedLLM(RULES)
        assert await llm.complete(user("What TIME is it?")) == "It is just after four."

    async def test_default_when_nothing_matches(self) -> None:
        assert await ScriptedLLM(RULES).complete(user("hello")) == DEFAULT_REPLY

    async def test_only_the_last_user_message_is_matched(self) -> None:
        llm = ScriptedLLM(RULES)
        messages = [
            ChatMessage(role="system", content="what time is it"),
            ChatMessage(role="user", content="what should I drink"),
        ]
        assert await llm.complete(messages) == "There is tea in the pot."

    async def test_calls_are_recorded(self) -> None:
        llm = ScriptedLLM(RULES)
        await llm.complete(user("time"))
        assert llm.calls == [tuple(user("time"))]

    def test_a_bad_regex_fails_at_load_time(self) -> None:
        with pytest.raises(Exception, match=r"(?i)unterminated|error"):
            ScriptedRule(match="(unclosed", reply="x")

    def test_a_malformed_rule_is_refused(self) -> None:
        with pytest.raises(ValueError, match="match"):
            ScriptedLLM([{"pattern": "x", "reply": "y"}])


class TestReplyAndMemoryModes:
    async def test_auto_mode_routes_memory_prompts_to_memory_rules(self) -> None:
        llm = ScriptedLLM.from_mapping(DOCUMENT)
        turns = [make_utterance("I am allergic to peanuts.", speaker_label="S2")]
        assert await llm.complete(memory_summary_messages(turns, "S2")) == (
            "S2 is allergic to peanuts."
        )

    async def test_auto_mode_routes_reply_prompts_to_reply_rules(self) -> None:
        llm = ScriptedLLM.from_mapping(DOCUMENT)
        ctx = make_context(current=make_utterance("Alice, what time is it?"))
        assert await llm.complete(reply_messages(ctx)) == "It is just after four."

    async def test_memory_mode_pins_the_memory_rules(self) -> None:
        llm = ScriptedLLM.from_mapping(DOCUMENT, mode="memory")
        assert await llm.complete(user("peanuts")) == "S2 is allergic to peanuts."
        assert await llm.complete(user("what time is it")) == ""

    async def test_reply_mode_pins_the_reply_rules(self) -> None:
        llm = ScriptedLLM.from_mapping(DOCUMENT, mode="reply")
        turns = [make_utterance("I am allergic to peanuts.", speaker_label="S2")]
        assert await llm.complete(memory_summary_messages(turns, "S2")) == "I am not sure."

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            ScriptedLLM(mode="sideways")  # type: ignore[arg-type]


class TestRulesFile:
    async def test_round_trip_through_a_json_file(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.json"
        path.write_text(json.dumps(DOCUMENT), encoding="utf-8")
        llm = ScriptedLLM.from_file(path)
        assert await llm.complete(user("what time is it")) == "It is just after four."
        assert await llm.complete(user("nothing matches")) == "I am not sure."

    def test_a_bare_list_is_a_reply_rule_set(self) -> None:
        llm = ScriptedLLM.from_mapping(RULES)
        assert llm.rules[0].reply == "It is just after four."
        assert llm.default == DEFAULT_REPLY

    def test_unknown_keys_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown keys"):
            ScriptedLLM.from_mapping({**DOCUMENT, "replies": []})

    def test_an_unknown_schema_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schema"):
            ScriptedLLM.from_mapping({**DOCUMENT, "schema_version": "scripted-llm/v99"})


class TestFailureInjection:
    async def test_armed_failure_raises(self) -> None:
        llm = ScriptedLLM(RULES)
        llm.arm_failure(RuntimeError("model is down"))
        with pytest.raises(RuntimeError, match="model is down"):
            await llm.complete(user("time"))

    async def test_a_counted_failure_stops(self) -> None:
        llm = ScriptedLLM(RULES, fail_with=RuntimeError("flaky"), fail_count=1)
        with pytest.raises(RuntimeError):
            await llm.complete(user("time"))
        assert await llm.complete(user("time")) == "It is just after four."
