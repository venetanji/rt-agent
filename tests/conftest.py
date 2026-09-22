"""Shared builders for the rt-agent test suite. Nothing here touches a network."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rt_agent.contracts import (
    AudioEvidence,
    BundleAnswers,
    ChoiceAnswer,
    DecisionContext,
    MemoryRecord,
    RobotState,
    ScoreAnswer,
    Utterance,
)
from rt_agent.systemone import BUNDLE_VERSION

FIXTURES = Path(__file__).parent / "fixtures"
SYSTEMONE_FIXTURES = FIXTURES / "systemone"
STATE_FIXTURES = FIXTURES / "state"

NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def now() -> datetime:
    """A fixed, timezone-aware clock so every decision is reproducible."""
    return NOW


def make_audio(**overrides: float | None) -> AudioEvidence:
    """Plausible audio evidence for a clean utterance."""
    values: dict[str, float | None] = {
        "vad_mean_prob": 0.92,
        "voiced_fraction": 0.71,
        "duration_s": 2.0,
        "rms": 0.06,
        "clipped_fraction": 0.0,
        "overlap_fraction": 0.0,
    }
    values.update(overrides)
    return AudioEvidence.model_validate(values)


def make_utterance(
    text: str = "Alice, what time is it?",
    speaker_label: str = "S1",
    utterance_id: str = "u1",
    t_start_s: float = 10.0,
    t_end_s: float = 12.0,
    **overrides: object,
) -> Utterance:
    """One diarized turn."""
    values: dict[str, object] = {
        "utterance_id": utterance_id,
        "session_id": "test-session",
        "speaker_label": speaker_label,
        "text": text,
        "t_start_s": t_start_s,
        "t_end_s": t_end_s,
        "audio": make_audio(),
    }
    values.update(overrides)
    return Utterance.model_validate(values)


def make_memory(text: str = "S1 drinks tea, never coffee.") -> MemoryRecord:
    """One stored memory."""
    return MemoryRecord(
        memory_id="m1",
        session_id="test-session",
        speaker_label="S1",
        kind="preference",
        text=text,
        source_utterance_ids=("a1",),
        worth_p=0.9,
        faithfulness_p=0.95,
        sensitive=False,
        created_at=datetime(2026, 9, 20, 9, 0, tzinfo=UTC),
    )


def make_context(
    current: Utterance | None = None,
    recent: tuple[Utterance, ...] = (),
    robot_state: RobotState | None = None,
    memories: tuple[MemoryRecord, ...] = (),
) -> DecisionContext:
    """A decision context with sensible defaults."""
    return DecisionContext(
        session_id="test-session",
        recent=recent,
        current=current or make_utterance(),
        retrieved_memories=memories,
        robot_state=robot_state or RobotState(),
    )


@pytest.fixture
def context() -> DecisionContext:
    """The default decision context: one direct question to the robot."""
    return make_context()


def make_answers(
    *,
    intelligible: float = 0.95,
    addressed: float = 0.95,
    invites: float = 0.9,
    stay_quiet: float = 0.05,
    addressee: str = "robot",
    addressee_conf: float = 0.95,
    emotion: str = "attentive",
    emotion_conf: float = 0.9,
    intensity_level: float = 2.0,
    worth: float = 0.1,
    sensitive: float = 0.05,
    memory_kind: str = "other",
    memory_kind_conf: float = 0.8,
    latency_ms: float = 250.0,
    model: str = "jev-1.13.0",
) -> BundleAnswers:
    """Build a well-formed BundleAnswers without calling anything."""

    def spread(winner: str, options: tuple[str, ...], peak: float) -> dict[str, float]:
        if winner not in options:
            # Deliberately malformed cases substitute an option the bundle never offered.
            options = (winner, *options[1:])
        rest = round((1.0 - peak) / (len(options) - 1), 2)
        distribution = {option: rest for option in options if option != winner}
        distribution[winner] = round(1.0 - rest * (len(options) - 1), 2)
        return distribution

    addressee_options = ("robot", "other_human", "self_or_group", "unclear")
    kind_options = (
        "personal_fact",
        "preference",
        "event",
        "plan",
        "relationship",
        "health",
        "other",
    )
    emotion_options = (
        "neutral",
        "attentive",
        "warm",
        "happy",
        "amused",
        "curious",
        "surprised",
        "concerned",
        "sympathetic",
        "sad",
    )
    levels = {str(index): 0.0 for index in range(5)}
    levels[str(int(intensity_level))] = 1.0

    return BundleAnswers(
        bundle_version=BUNDLE_VERSION,
        model=model,
        request_id="req_test",
        latency_ms=latency_ms,
        nouls={
            "intelligible_complete": intelligible,
            "addressed_to_robot": addressed,
            "invites_response_now": invites,
            "robot_should_stay_quiet_safety": stay_quiet,
            "worth_remembering": worth,
            "sensitive_personal": sensitive,
        },
        choices={
            "addressee": ChoiceAnswer(
                chosen=addressee,
                probabilities=spread(addressee, addressee_options, 0.9),
                confidence=addressee_conf,
            ),
            "emotion": ChoiceAnswer(
                chosen=emotion,
                probabilities=spread(emotion, emotion_options, 0.9),
                confidence=emotion_conf,
            ),
            "memory_kind": ChoiceAnswer(
                chosen=memory_kind,
                probabilities=spread(memory_kind, kind_options, 0.9),
                confidence=memory_kind_conf,
            ),
        },
        scores={
            "emotion_intensity": ScoreAnswer(
                value=intensity_level,
                probabilities=levels,
                confidence=0.8,
                legend={str(index): f"level {index}" for index in range(5)},
            )
        },
    )
