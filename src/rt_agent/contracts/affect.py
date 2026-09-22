"""``affect-cue/v1`` — the face-facing affect contract.

The cue describes *the robot's intended delivery*, never a diagnosis of a human's
emotion. Today's alice face only reacts to |valence| > 0.2; the full vector is
carried anyway so a richer face can use it without a schema change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from rt_agent.contracts.base import AffectVector, FrozenModel, Unit

__all__ = ["AffectCue"]


class AffectCue(FrozenModel):
    """One expression target for the face, valid for a bounded window."""

    schema_version: Literal["affect-cue/v1"] = "affect-cue/v1"
    vector: AffectVector
    intensity: Unit
    preset: str = Field(min_length=1, max_length=64)
    source_id: str = Field(default="rt-agent", min_length=1, max_length=64)
    source_confidence: Unit
    valid_for_ms: int = Field(default=1500, ge=0, le=600_000)
    issued_at: AwareDatetime

    @property
    def valence(self) -> float:
        """Axis 0 of the affect vector."""
        return self.vector[0]

    @property
    def arousal(self) -> float:
        """Axis 1 of the affect vector."""
        return self.vector[1]

    @property
    def dominance(self) -> float:
        """Axis 2 of the affect vector."""
        return self.vector[2]

    @property
    def visible_on_alice_face(self) -> bool:
        """True when today's alice face would show anything but ``neutral``."""
        return abs(self.vector[0]) > 0.2
