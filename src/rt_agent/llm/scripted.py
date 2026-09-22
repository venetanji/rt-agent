"""``ScriptedLLM`` — a deterministic, network-free stand-in for the remote model.

It answers by walking an ordered list of rules: the first whose regex matches the last
*user* message wins, otherwise a default string is returned. The same object serves
both LLM paths, because a memory-summary request is recognised by its system prompt
(:data:`~rt_agent.llm.prompts.MEMORY_SUMMARY_SYSTEM`) and answered from a second rule
set — so one scripted model can drive a whole replay: replies *and* the sentences the
:class:`~rt_agent.memory.writer.MemoryWriter` stores.

Rule sets are literals in tests and JSON files in demos; the file format is documented
in ``docs/memory.md`` and parsed by :meth:`ScriptedLLM.from_file`.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rt_agent.contracts.protocols import ChatMessage
from rt_agent.llm.prompts import MEMORY_SUMMARY_SYSTEM

__all__ = [
    "DEFAULT_REPLY",
    "RULES_SCHEMA_VERSION",
    "Mode",
    "ScriptedLLM",
    "ScriptedRule",
]

#: Version tag of the JSON rules file format.
RULES_SCHEMA_VERSION = "scripted-llm/v1"

#: What a reply-mode :class:`ScriptedLLM` says when no rule matches.
DEFAULT_REPLY = "I heard you, but I do not know what to say about that."

Mode = Literal["auto", "reply", "memory"]


@dataclass(frozen=True)
class ScriptedRule:
    """One ordered rule: ``match`` is a regex searched in the last user message."""

    match: str
    reply: str

    def __post_init__(self) -> None:
        """Compile the pattern once so a bad regex fails at load time, not mid-replay."""
        re.compile(self.match, re.IGNORECASE)

    @property
    def pattern(self) -> re.Pattern[str]:
        """The compiled, case-insensitive pattern (``re`` caches the compilation)."""
        return re.compile(self.match, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        """True when this rule claims ``text``."""
        return self.pattern.search(text) is not None

    @classmethod
    def parse(cls, raw: ScriptedRule | Mapping[str, Any]) -> ScriptedRule:
        """Accept a rule object or a ``{"match": ..., "reply": ...}`` mapping."""
        if isinstance(raw, ScriptedRule):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError(f"a scripted rule must be an object, got {type(raw).__name__}")
        try:
            match = raw["match"]
            reply = raw["reply"]
        except KeyError as error:
            raise ValueError(f"a scripted rule needs both 'match' and 'reply': {raw!r}") from error
        if not isinstance(match, str) or not isinstance(reply, str):
            raise ValueError(f"'match' and 'reply' must be strings: {raw!r}")
        return cls(match=match, reply=reply)


def _parse_rules(raw: Iterable[ScriptedRule | Mapping[str, Any]]) -> tuple[ScriptedRule, ...]:
    return tuple(ScriptedRule.parse(rule) for rule in raw)


def _last_user_message(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content if messages else ""


def _is_memory_request(messages: Sequence[ChatMessage]) -> bool:
    return any(
        message.role == "system" and message.content == MEMORY_SUMMARY_SYSTEM
        for message in messages
    )


class ScriptedLLM:
    """A :class:`~rt_agent.contracts.protocols.ChatLLM` that never leaves the process.

    ``mode`` decides which rule set answers a request: ``"auto"`` (the default) routes
    memory-summary prompts to ``memory_rules`` and everything else to ``rules``;
    ``"reply"`` and ``"memory"`` pin one rule set regardless of the prompt.
    """

    def __init__(
        self,
        rules: Iterable[ScriptedRule | Mapping[str, Any]] = (),
        *,
        default: str = DEFAULT_REPLY,
        memory_rules: Iterable[ScriptedRule | Mapping[str, Any]] = (),
        memory_default: str = "",
        mode: Mode = "auto",
        latency_ms: float = 0.0,
        fail_with: Exception | None = None,
        fail_count: int | None = None,
    ) -> None:
        if mode not in ("auto", "reply", "memory"):
            raise ValueError(f"unknown ScriptedLLM mode {mode!r}")
        self.rules = _parse_rules(rules)
        self.default = default
        self.memory_rules = _parse_rules(memory_rules)
        self.memory_default = memory_default
        self.mode: Mode = mode
        self.latency_ms = latency_ms
        self._fail_with = fail_with
        self._fail_remaining = fail_count
        #: Every message list this model was asked to complete, in order.
        self.calls: list[tuple[ChatMessage, ...]] = []

    # -- construction -------------------------------------------------------------

    @classmethod
    def from_mapping(cls, document: Any, **kwargs: Any) -> ScriptedLLM:
        """Build from an already-parsed rules document (see ``docs/memory.md``)."""
        if isinstance(document, list):
            return cls(document, **kwargs)
        if not isinstance(document, Mapping):
            raise ValueError("a scripted rules document must be an object or a list of rules")
        version = document.get("schema_version", RULES_SCHEMA_VERSION)
        if version != RULES_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scripted rules schema {version!r}, expected {RULES_SCHEMA_VERSION!r}"
            )
        unknown = set(document) - {
            "schema_version",
            "default",
            "rules",
            "memory_default",
            "memory_rules",
        }
        if unknown:
            raise ValueError(f"unknown keys in scripted rules document: {sorted(unknown)}")
        options: dict[str, Any] = {
            "rules": document.get("rules", ()),
            "memory_rules": document.get("memory_rules", ()),
        }
        if "default" in document:
            options["default"] = document["default"]
        if "memory_default" in document:
            options["memory_default"] = document["memory_default"]
        options.update(kwargs)
        rules = options.pop("rules")
        return cls(rules, **options)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> ScriptedLLM:
        """Load a JSON rules file. Keyword arguments override the file's settings."""
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(document, **kwargs)

    # -- failure injection --------------------------------------------------------

    def arm_failure(self, error: Exception, count: int | None = None) -> None:
        """Make the next ``count`` completions (or all of them) raise ``error``."""
        self._fail_with = error
        self._fail_remaining = count

    def clear_failure(self) -> None:
        """Stop injecting failures."""
        self._fail_with = None
        self._fail_remaining = None

    def _maybe_fail(self) -> None:
        if self._fail_with is None:
            return
        if self._fail_remaining is None:
            raise self._fail_with
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            if self._fail_remaining == 0:
                error, self._fail_with = self._fail_with, None
                raise error
            raise self._fail_with

    # -- lookup -------------------------------------------------------------------

    def rules_for(self, messages: Sequence[ChatMessage]) -> tuple[tuple[ScriptedRule, ...], str]:
        """The rule set and default that would answer ``messages``."""
        if self.mode == "memory" or (self.mode == "auto" and _is_memory_request(messages)):
            return self.memory_rules, self.memory_default
        return self.rules, self.default

    def answer(self, messages: Sequence[ChatMessage]) -> str:
        """The reply for ``messages``, without recording the call or sleeping."""
        rules, default = self.rules_for(messages)
        text = _last_user_message(messages)
        for rule in rules:
            if rule.matches(text):
                return rule.reply
        return default

    # -- the ChatLLM protocol -----------------------------------------------------

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the scripted completion, after the configured artificial latency."""
        self.calls.append(tuple(messages))
        self._maybe_fail()
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        return self.answer(messages)

    async def aclose(self) -> None:
        """No-op; present so the scripted model is drop-in for the real client."""
