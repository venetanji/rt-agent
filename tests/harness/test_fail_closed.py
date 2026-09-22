"""The harness under a backend that is slow, broken, or answering nonsense.

Every one of these ends the same way: the robot says nothing, names a reason, and the
run keeps going. A listening robot that crashes when its classifier is down is worse
than one that stays quiet, and a listening robot that *speaks* when its classifier is
down is worse still.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rt_agent.contracts import BundleAnswers
from rt_agent.llm import LLMUnavailableError, ScriptedLLM
from rt_agent.memory import MemoryOutcome
from rt_agent.systemone import (
    QUESTION_BUNDLE_V1,
    MockSystemOne,
    SystemOneRateLimitError,
    SystemOneUnavailableError,
)
from rt_agent.systemone.mock import MockAnswerNotFound
from tests.harness.conftest import kitchen_utterances, make_agent


async def test_a_backend_that_never_answers_produces_waits_not_a_crash(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> None:
    """Every call sleeps past the deadline: 28 turns, 28 WAITs, no exception."""
    kitchen_backend.latency_ms = 200.0
    agent = make_agent(tmp_path, kitchen_backend, scripted_llm, deadline_ms=20)
    summary = await agent.run(kitchen_utterances(agent.session_id))
    await agent.aclose()

    assert summary.utterances == 28
    assert summary.speak_count == 0
    assert summary.wait_reasons == {"backend_unavailable": 28}
    assert summary.backend_failures == 28
    assert all(turn.decision.answers is None for turn in agent.turns)
    assert all("deadline" in (turn.backend_error or "") for turn in agent.turns)
    # A fail-closed decision leaves the face exactly where it was.
    assert all(turn.emitted_cue is None for turn in agent.turns)
    assert summary.emotion_changes == 0


@pytest.mark.parametrize(
    "error",
    [
        SystemOneUnavailableError("transport failure"),
        SystemOneRateLimitError("HTTP 429"),
        MockAnswerNotFound("no fixture"),
        RuntimeError("something nobody predicted"),
    ],
    ids=["unavailable", "rate_limited", "no_answer", "unexpected"],
)
async def test_every_backend_failure_is_a_wait(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM, error: Exception
) -> None:
    kitchen_backend.arm_failure(error)
    agent = make_agent(tmp_path, kitchen_backend, scripted_llm)
    utterances = kitchen_utterances(agent.session_id)
    record = await agent.handle(utterances[11])  # line 12: the plainest SPEAK there is
    await agent.aclose()

    assert record.decision.speak is False
    assert record.decision.wait_reason == "backend_unavailable"
    assert record.decision.answers is None
    assert record.backend_error is not None
    assert type(error).__name__ in record.backend_error
    assert record.spoke is False
    assert record.reply is None


async def test_a_backend_that_recovers_speaks_again(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> None:
    """One failure is not a session: the very next utterance is decided normally."""
    kitchen_backend.arm_failure(SystemOneUnavailableError("one blip"), count=1)
    agent = make_agent(tmp_path, kitchen_backend, scripted_llm)
    summary = await agent.run(kitchen_utterances(agent.session_id))
    await agent.aclose()

    assert summary.backend_failures == 1
    assert agent.turns[0].decision.wait_reason == "backend_unavailable"
    assert summary.speak_count == 2
    assert summary.spoken_count == 2


async def test_malformed_answers_are_a_wait_not_an_exception(
    tmp_path: Path, scripted_llm: ScriptedLLM
) -> None:
    """A backend that answers, but not the questions we asked, must not admit a reply."""

    def half_an_answer(state: str, questions: object) -> BundleAnswers:
        # Only the first admission question comes back. The policy needs all of them.
        return BundleAnswers(
            bundle_version=QUESTION_BUNDLE_V1.version,
            model="jev-1.13.0",
            latency_ms=10.0,
            nouls={"intelligible_complete": 0.99},
        )

    backend = MockSystemOne(rule=half_an_answer)
    agent = make_agent(tmp_path, backend, scripted_llm)
    utterances = kitchen_utterances(agent.session_id)
    record = await agent.handle(utterances[11])
    await agent.aclose()

    assert record.decision.speak is False
    assert record.decision.wait_reason == "malformed_answers"
    # The answers are still kept on the decision: the log has to show what came back.
    assert record.decision.answers is not None


async def test_a_late_but_successful_answer_is_refused(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> None:
    """Inside the wait_for, over the policy's deadline: a named refusal, not a reply."""
    kitchen_backend.latency_ms = 60.0
    agent = make_agent(tmp_path, kitchen_backend, scripted_llm, deadline_ms=5000)
    agent.policy.config = agent.policy.config.model_copy(update={"deadline_ms": 10})
    utterances = kitchen_utterances(agent.session_id)
    record = await agent.handle(utterances[11])
    await agent.aclose()

    assert record.decision.speak is False
    assert record.decision.wait_reason == "deadline_exceeded"


async def test_a_broken_reply_model_costs_the_reply_and_nothing_else(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> None:
    """The decision to speak stands; the words do not arrive; the run continues."""
    scripted_llm.arm_failure(LLMUnavailableError("the reply model is down"))
    agent = make_agent(tmp_path, kitchen_backend, scripted_llm)
    summary = await agent.run(kitchen_utterances(agent.session_id))
    await agent.aclose()

    assert summary.speak_count == 2
    assert summary.spoken_count == 0
    assert summary.reply_failures == 2
    assert summary.speech_plans == ()
    spoke = [turn for turn in agent.turns if turn.decision.speak]
    for turn in spoke:
        assert turn.reply is None
        assert turn.reply_error is not None
        assert "LLMUnavailableError" in turn.reply_error
    # No ROBOT turn was logged for a reply that was never spoken.
    assert not any(item.is_robot for item in agent.transcript.read_session(agent.session_id))
    # The memory path is independent of the reply path and still ran.
    assert summary.memory_outcomes.get("failed") == 2


async def test_a_face_bridge_that_throws_does_not_stop_the_conversation(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> None:
    class BrokenFace:
        def emit_affect(self, cue: object) -> None:
            raise OSError("the preview socket went away")

        def emit_clauses(self, clauses: object) -> None:
            raise OSError("the preview socket went away")

        def close(self) -> None:
            raise OSError("and it stayed away")

    agent = make_agent(tmp_path, kitchen_backend, scripted_llm)
    agent.face = BrokenFace()  # type: ignore[assignment]
    summary = await agent.run(kitchen_utterances(agent.session_id))
    await agent.aclose()

    assert summary.utterances == 28
    assert summary.speak_count == 2
    assert summary.spoken_count == 2


async def test_a_memory_write_that_fails_is_recorded_not_raised(
    tmp_path: Path, kitchen_backend: MockSystemOne
) -> None:
    """``MemoryWriter.write`` never raises, and a failed write is still an outcome."""
    llm = ScriptedLLM.from_file(
        Path(__file__).resolve().parents[2] / "examples" / "scripted_llm.json"
    )
    agent = make_agent(tmp_path, kitchen_backend, llm)
    utterances = kitchen_utterances(agent.session_id)
    llm.arm_failure(LLMUnavailableError("down"), count=1)
    await agent.handle(utterances[4])  # line 5: the decaf preference
    await agent.drain_memories()
    await agent.aclose()

    assert [outcome.outcome for outcome in agent.memory_outcomes] == [MemoryOutcome.FAILED]
    assert agent.memory_outcomes[0].reason == "llm_failed"
    lines = (agent.run_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["outcome"] == "failed"


async def test_background_memory_writes_are_awaited_before_shutdown(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> None:
    """Nothing is left running when the agent closes, and nothing is lost."""
    agent = make_agent(tmp_path, kitchen_backend, scripted_llm)
    await agent.run(kitchen_utterances(agent.session_id))
    assert len(agent.memory_outcomes) == 2
    await agent.aclose()
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("memory-")]


async def test_closing_twice_is_harmless(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> None:
    agent = make_agent(tmp_path, kitchen_backend, scripted_llm)
    await agent.run(kitchen_utterances(agent.session_id)[:2])
    await agent.aclose()
    await agent.aclose()
