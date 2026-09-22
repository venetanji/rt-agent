"""Typed System One answers (``noul`` / ``choice`` / ``score``) and ``decision/v1``.

The answer models carry the fail-closed validation that the wire protocol needs:
distributions must be complete, finite, bounded and sum to 1 within
``PROBABILITY_SUM_TOLERANCE`` (the server rounds every reported probability to two
decimals, so an exact sum of 1.0 is not achievable), and a ``choice`` winner must
agree with the argmax of its own distribution. Anything else raises, and the policy
turns the raise into a WAIT.
"""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from rt_agent.contracts.affect import AffectCue
from rt_agent.contracts.base import FrozenModel, Probability, Unit
from rt_agent.contracts.memory import MemoryKind

__all__ = [
    "PROBABILITY_SUM_TOLERANCE",
    "Addressee",
    "BundleAnswers",
    "ChoiceAnswer",
    "Decision",
    "NoulAnswer",
    "ScoreAnswer",
    "Usage",
]

#: Upstream rounds each reported probability to 2 decimals; a 4-option distribution
#: can therefore legitimately sum to 0.99 or 1.01. Anything outside is malformed.
PROBABILITY_SUM_TOLERANCE = 0.011

Addressee = Literal["robot", "other_human", "self_or_group", "unclear"]


def _check_distribution(probabilities: dict[str, float]) -> None:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    total = 0.0
    for key, value in probabilities.items():
        if not math.isfinite(value):
            raise ValueError(f"probability for {key!r} is not finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"probability for {key!r} is outside [0, 1]")
        total += value
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            f"probabilities sum to {total!r}, expected 1.0 +/- {PROBABILITY_SUM_TOLERANCE}"
        )


class NoulAnswer(FrozenModel):
    """``noul``: the calibrated probability that the statement is true. No confidence field."""

    type: Literal["noul"] = "noul"
    probability: Probability


class ChoiceAnswer(FrozenModel):
    """``choice``: a winner plus the full distribution over the frozen option set."""

    type: Literal["choice"] = "choice"
    chosen: str = Field(min_length=1)
    probabilities: dict[str, float]
    confidence: Unit

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_distribution(self.probabilities)
        if self.chosen not in self.probabilities:
            raise ValueError(f"chosen option {self.chosen!r} is absent from probabilities")
        best = max(self.probabilities.values())
        if self.probabilities[self.chosen] < best - PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"chosen option {self.chosen!r} is not the argmax of probabilities")
        return self


class ScoreAnswer(FrozenModel):
    """``score``: a probability-weighted position on an ordered rubric."""

    type: Literal["score"] = "score"
    value: float = Field(ge=0.0)
    probabilities: dict[str, float]
    confidence: Unit
    legend: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_distribution(self.probabilities)
        levels: list[int] = []
        for key in self.probabilities:
            if not key.isdigit():
                raise ValueError(f"score probability key {key!r} is not a level index")
            levels.append(int(key))
        if sorted(levels) != list(range(len(levels))):
            raise ValueError("score probability keys must be a contiguous range starting at 0")
        if self.value > max(levels):
            raise ValueError(f"score {self.value!r} exceeds the top rubric level {max(levels)}")
        return self

    @property
    def max_level(self) -> int:
        """Index of the highest rubric level (levels are 0-based)."""
        return max(int(key) for key in self.probabilities)

    @property
    def normalized(self) -> float:
        """The score mapped onto [0, 1] — ``value / max_level``."""
        top = self.max_level
        return 0.0 if top == 0 else self.value / top


class Usage(FrozenModel):
    """Billed input tokens and (free) output tokens for one bundle call."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class BundleAnswers(FrozenModel):
    """A fully parsed, validated response to one frozen question bundle."""

    bundle_version: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    request_id: str | None = None
    latency_ms: float = Field(ge=0.0)
    usage: Usage | None = None
    nouls: dict[str, float] = Field(default_factory=dict)
    choices: dict[str, ChoiceAnswer] = Field(default_factory=dict)
    scores: dict[str, ScoreAnswer] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name, value in self.nouls.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"noul {name!r} is outside [0, 1]")
        overlap = (
            set(self.nouls) & set(self.choices)
            | set(self.nouls) & set(self.scores)
            | set(self.choices) & set(self.scores)
        )
        if overlap:
            raise ValueError(f"question ids answered twice: {sorted(overlap)}")
        return self

    def noul(self, name: str) -> float:
        """Probability for a ``noul`` question. Raises ``KeyError`` when absent."""
        return self.nouls[name]

    def noul_or(self, name: str, default: float) -> float:
        """Probability for a ``noul`` question, or ``default`` when the answer is missing."""
        return self.nouls.get(name, default)

    def choice(self, name: str) -> ChoiceAnswer:
        """Answer to a ``choice`` question. Raises ``KeyError`` when absent."""
        return self.choices[name]

    def score(self, name: str) -> ScoreAnswer:
        """Answer to a ``score`` question. Raises ``KeyError`` when absent."""
        return self.scores[name]

    @property
    def answered(self) -> frozenset[str]:
        """Every question id present in the response."""
        return frozenset(self.nouls) | frozenset(self.choices) | frozenset(self.scores)


class Decision(FrozenModel):
    """``decision/v1`` — the policy's verdict for one utterance.

    ``answers`` is ``None`` exactly when the backend failed or timed out; in that case
    ``speak`` is ``False`` and ``wait_reason`` says why.
    """

    schema_version: Literal["decision/v1"] = "decision/v1"
    utterance_id: str = Field(min_length=1, max_length=128)
    speak: bool
    wait_reason: str | None = None
    addressee: Addressee
    emotion_preset: str = Field(min_length=1, max_length=64)
    emotion_conf: Unit
    intensity: Unit
    affect: AffectCue
    remember: bool = False
    memory_kind: MemoryKind | None = None
    sensitive: bool = False
    answers: BundleAnswers | None = None
    policy_version: str = Field(min_length=1, max_length=64)
    bundle_version: str = Field(min_length=1, max_length=64)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.speak and self.wait_reason is not None:
            raise ValueError("wait_reason must be None when speak is True")
        if not self.speak and self.wait_reason is None:
            raise ValueError("wait_reason is required when speak is False")
        if self.remember and self.memory_kind is None:
            raise ValueError("memory_kind is required when remember is True")
        return self
