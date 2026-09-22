"""``ReplyGenerator`` and the cleanup that stands between a chat model and a speaker.

A model that ignores "plain spoken text only" is not a bug we can prompt away, so the
cleanup is tested as a contract in its own right: whatever goes in, what comes out is
speakable.
"""

from __future__ import annotations

import pytest

from rt_agent.llm import LLMEmptyResponseError, ReplyGenerator, ScriptedLLM, clean_reply
from rt_agent.llm.reply import split_sentences
from tests.conftest import make_context, make_memory, make_utterance


class TestCleanReply:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Hello there.  ", "Hello there."),
            ("**Hello** _there_.", "Hello there."),
            ("`code` and ```fence```", "code and fence"),
            ("- Hello there.", "Hello there."),
            ("# Hello there.", "Hello there."),
            ("> Hello there.", "Hello there."),
            ("[tea](https://example.com) is ready.", "tea is ready."),
            ('"Hello there."', "Hello there."),
            ("Alice: Hello there.", "Hello there."),
            ("ROBOT: Hello there.", "Hello there."),
            ("Hello\n\nthere.", "Hello there."),
            ("<think>hmm</think>Hello there.", "Hello there."),
            ("Hello there.<|im_end|>", "Hello there."),
        ],
    )
    def test_cleanup_cases(self, raw: str, expected: str) -> None:
        assert clean_reply(raw) == expected

    def test_a_lowercase_colon_opener_is_not_a_speaker_label(self) -> None:
        assert clean_reply("Yes: it is four o'clock.") == "Yes: it is four o'clock."

    def test_at_most_two_sentences(self) -> None:
        cleaned = clean_reply("One. Two. Three. Four.")
        assert cleaned == "One. Two."

    def test_sentence_cap_is_configurable(self) -> None:
        assert clean_reply("One. Two. Three.", max_sentences=1) == "One."

    def test_length_is_capped_on_a_word_boundary(self) -> None:
        cleaned = clean_reply("word " * 200, max_chars=60)
        assert len(cleaned) <= 61
        assert cleaned.endswith(".")
        assert "wor." not in cleaned

    def test_empty_input_raises(self) -> None:
        with pytest.raises(LLMEmptyResponseError):
            clean_reply("   \n  ")

    def test_reasoning_only_output_raises(self) -> None:
        with pytest.raises(LLMEmptyResponseError):
            clean_reply("<think>still thinking</think>")

    def test_speaker_prefix_can_be_kept(self) -> None:
        assert clean_reply("S2 likes tea.", strip_speaker_prefix=False) == "S2 likes tea."


class TestSplitSentences:
    def test_keeps_punctuation(self) -> None:
        assert split_sentences("Hi. How are you?") == ["Hi.", "How are you?"]

    def test_unpunctuated_text_is_one_sentence(self) -> None:
        assert split_sentences("no punctuation here") == ["no punctuation here"]


class TestReplyGenerator:
    async def test_it_prompts_and_cleans(self) -> None:
        llm = ScriptedLLM([{"match": r"\btime\b", "reply": "**It is four.** And warm outside."}])
        ctx = make_context(current=make_utterance("Alice, what time is it?"))
        assert await ReplyGenerator(llm).reply(ctx) == "It is four. And warm outside."

    async def test_the_model_sees_the_retrieved_memories(self) -> None:
        llm = ScriptedLLM([{"match": "Pixel", "reply": "How is Pixel doing?"}])
        ctx = make_context(current=make_utterance("Alice, guess what happened."))
        reply = await ReplyGenerator(llm).reply(ctx, [make_memory("S1 has a cat called Pixel.")])
        assert reply == "How is Pixel doing?"

    async def test_context_memories_are_used_by_default(self) -> None:
        llm = ScriptedLLM([{"match": "Pixel", "reply": "How is Pixel doing?"}])
        ctx = make_context(memories=(make_memory("S1 has a cat called Pixel."),))
        assert await ReplyGenerator(llm).reply(ctx) == "How is Pixel doing?"

    async def test_model_errors_are_not_swallowed_on_the_speaking_path(self) -> None:
        llm = ScriptedLLM(fail_with=RuntimeError("model is down"))
        with pytest.raises(RuntimeError, match="model is down"):
            await ReplyGenerator(llm).reply(make_context())

    async def test_an_empty_completion_raises(self) -> None:
        llm = ScriptedLLM(default="   ")
        with pytest.raises(LLMEmptyResponseError):
            await ReplyGenerator(llm).reply(make_context())

    async def test_length_budget_is_enforced(self) -> None:
        llm = ScriptedLLM(default="word " * 300)
        reply = await ReplyGenerator(llm, max_chars=80).reply(make_context())
        assert len(reply) <= 81
