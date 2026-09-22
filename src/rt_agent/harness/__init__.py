"""The harness: the orchestrator that wires every other package into one listening robot.

:class:`~rt_agent.harness.agent.ListeningAgent` is the loop of design brief §7 —
utterance in, transcript log, decision context, one frozen System One bundle under a
hard deadline, policy, face, background memory, and a reply only when the answers say
so. :class:`~rt_agent.harness.config.AgentConfig` chooses which backend, text generator
and face it runs with; :mod:`rt_agent.harness.cli` is the ``rt-agent`` command.
"""

from __future__ import annotations

from rt_agent.harness.agent import (
    DECISION_RECORD_VERSION,
    ROBOT_WORDS_PER_SECOND,
    SUMMARY_VERSION,
    ListeningAgent,
    RunSummary,
    TurnRecord,
    percentile,
)
from rt_agent.harness.config import (
    AgentConfig,
    BackendName,
    FaceName,
    LlmName,
    new_session_id,
)
from rt_agent.harness.record import CURRENT_UTTERANCE, FixtureRecorder

__all__ = [
    "CURRENT_UTTERANCE",
    "DECISION_RECORD_VERSION",
    "ROBOT_WORDS_PER_SECOND",
    "SUMMARY_VERSION",
    "AgentConfig",
    "BackendName",
    "FaceName",
    "FixtureRecorder",
    "ListeningAgent",
    "LlmName",
    "RunSummary",
    "TurnRecord",
    "new_session_id",
    "percentile",
]
