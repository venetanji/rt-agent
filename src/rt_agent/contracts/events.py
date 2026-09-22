"""Transcript-side contracts: audio evidence, utterances and the decision context.

Speaker labels are session-local anonymous labels (``S1``, ``S2``, ... and ``ROBOT``
for the robot's own speech). Real names never enter these models, and voice
embeddings are never persisted in the PoC.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from rt_agent.contracts.base import FrozenModel, Probability, Unit
from rt_agent.contracts.memory import MemoryRecord

__all__ = [
    "MAX_RETRIEVED_MEMORIES",
    "RECENT_WINDOW",
    "ROBOT_SPEAKER_LABEL",
    "AudioEvidence",
    "DecisionContext",
    "RobotState",
    "Transcription",
    "Utterance",
]

#: The speaker label reserved for the robot's own turns.
ROBOT_SPEAKER_LABEL = "ROBOT"

#: N in "the last N turns" — the rolling window handed to System One.
RECENT_WINDOW = 6

#: How many retrieved memories may accompany a decision.
MAX_RETRIEVED_MEMORIES = 3

_SPEAKER_LABEL_PATTERN = r"^[A-Z][A-Z0-9_-]{0,31}$"


class AudioEvidence(FrozenModel):
    """Acoustic evidence for one utterance. Unknown values stay ``None``, never fabricated."""

    vad_mean_prob: Probability
    voiced_fraction: Unit
    duration_s: float = Field(ge=0.0, le=3600.0)
    rms: float = Field(ge=0.0, le=10.0)
    clipped_fraction: Unit
    overlap_fraction: Unit | None = None


class Utterance(FrozenModel):
    """``utterance/v1`` — one diarized, VAD-segmented turn."""

    schema_version: Literal["utterance/v1"] = "utterance/v1"
    utterance_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    speaker_label: str = Field(pattern=_SPEAKER_LABEL_PATTERN)
    text: str = Field(min_length=1, max_length=1000)
    t_start_s: float = Field(ge=0.0)
    t_end_s: float = Field(ge=0.0)
    asr_confidence: Probability | None = None
    language: str = Field(default="en", pattern=r"^[a-z]{2}(-[A-Za-z]{2,8})?$")
    audio: AudioEvidence | None = None
    is_final: bool = True

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        if self.t_end_s < self.t_start_s:
            raise ValueError("t_end_s must not be earlier than t_start_s")
        return self

    @property
    def is_robot(self) -> bool:
        """True for the robot's own speech."""
        return self.speaker_label == ROBOT_SPEAKER_LABEL

    @property
    def duration_s(self) -> float:
        """Wall-clock length of the turn."""
        return self.t_end_s - self.t_start_s


class RobotState(FrozenModel):
    """What the robot itself is doing at decision time."""

    speaking: bool = False
    last_spoke_age_s: float | None = Field(default=None, ge=0.0)
    current_emotion: str = Field(default="neutral", min_length=1, max_length=64)


class DecisionContext(FrozenModel):
    """Everything one System One bundle call is allowed to see.

    This is deliberately *not* a transcript log: it is a small rolling window plus at
    most three retrieved memories. The persistent store is a separate layer.
    """

    robot_name: str = Field(default="Alice", min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=128)
    recent: tuple[Utterance, ...] = ()
    current: Utterance
    retrieved_memories: tuple[MemoryRecord, ...] = ()
    robot_state: RobotState = RobotState()

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if len(self.recent) > RECENT_WINDOW:
            raise ValueError(f"recent holds at most {RECENT_WINDOW} utterances")
        if len(self.retrieved_memories) > MAX_RETRIEVED_MEMORIES:
            raise ValueError(f"retrieved_memories holds at most {MAX_RETRIEVED_MEMORIES} records")
        return self


class Transcription(FrozenModel):
    """Raw ASR output for one audio segment, before diarization and session bookkeeping."""

    text: str = Field(min_length=1, max_length=1000)
    language: str = Field(default="en", pattern=r"^[a-z]{2}(-[A-Za-z]{2,8})?$")
    confidence: Probability | None = None
