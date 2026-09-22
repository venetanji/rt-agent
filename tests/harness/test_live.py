"""Opt-in live checks of the whole harness against a real System One endpoint.

    RT_AGENT_LIVE=1 TYPESAFE_API_KEY=... uv run pytest -q -m live

Skipped by default: the rest of the harness suite replays recordings of exactly these
calls. What this adds is the one thing a recording cannot prove — that the loop still
works against the service as it is today, with a real deadline over a real network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rt_agent.harness import AgentConfig, ListeningAgent
from tests.harness.conftest import SCRIPTED_RULES, kitchen_utterances

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RT_AGENT_LIVE") != "1",
        reason="live System One tests are opt-in; set RT_AGENT_LIVE=1",
    ),
]

#: Lines 11-13: the unintelligible fragment, the direct question, the third-person
#: mention. Three calls is enough to show admission working in both directions.
LIVE_LINES = (11, 12, 13)


@pytest.fixture
def live_config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        backend="jev",
        llm="scripted",
        scripted_rules=SCRIPTED_RULES,
        face="jsonl",
        out_dir=tmp_path / "out",
        session_id="live",
        # Generous next to the 1500 ms production deadline: this asserts the loop, not
        # the p95 of a shared sandbox's network.
        deadline_ms=20000,
    ).validated()


async def test_the_loop_admits_and_refuses_against_live_jev(live_config: AgentConfig) -> None:
    agent = ListeningAgent.from_config(live_config, session_id="live")
    utterances = kitchen_utterances("live")
    try:
        records = [await agent.handle(utterances[number - 1]) for number in LIVE_LINES]
        summary = agent.write_summary()
    finally:
        await agent.aclose()

    fragment, question, mention = records

    assert summary.backend_model_resolved is not None
    assert summary.backend_model_resolved.startswith("jev-")
    assert all(record.backend_error is None for record in records)

    # The recorded values are stable to about +/- 0.02; keep the assertions coarse.
    assert fragment.decision.speak is False
    assert fragment.decision.wait_reason == "not_intelligible"

    assert question.decision.speak is True
    assert question.spoke is True
    assert question.decision.addressee == "robot"
    assert question.clause_count >= 1
    assert question.speech_plan is not None
    assert Path(question.speech_plan).is_file()

    # Said *about* the robot, to the other person: the floor is not the robot's.
    assert mention.decision.speak is False
    assert mention.decision.answers is not None
    assert mention.decision.answers.noul("addressed_to_robot") < 0.7


async def test_a_live_memory_is_written_and_gated(live_config: AgentConfig) -> None:
    """Line 5 is durable and not sensitive: the writer runs and the gate passes it."""
    agent = ListeningAgent.from_config(live_config, session_id="live-memory")
    utterances = kitchen_utterances("live-memory")
    try:
        record = await agent.handle(utterances[4])
        await agent.drain_memories()
    finally:
        await agent.aclose()

    assert record.memory_scheduled is True
    assert len(agent.memory_outcomes) == 1
    outcome = agent.memory_outcomes[0]
    assert outcome.faithfulness_p is not None
    assert outcome.outcome.value in ("stored", "discarded_unfaithful")
    assert outcome.systemone_latency_ms is not None
