"""``speech-clause/v1`` — one face-ready spoken clause, shape-compatible with alice.

A response is split into at most 32 clauses on sentence punctuation; each clause
carries the affect vector that should be showing while it is spoken, so the face and
the voice never drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from rt_agent.contracts.base import (
    AffectVector,
    ClauseText,
    FrozenModel,
    Seed32,
    Sequence32,
    Unit,
)

__all__ = ["MAX_CLAUSES", "SpeechClauseOut"]

#: alice accepts sequence 0-31, i.e. at most 32 clauses per response.
MAX_CLAUSES = 32


class SpeechClauseOut(FrozenModel):
    """One clause of the robot's reply, with the affect it should be delivered with."""

    schema_version: Literal["speech-clause/v1"] = "speech-clause/v1"
    generation_id: str = Field(min_length=1, max_length=128)
    clause_id: str = Field(min_length=1, max_length=128)
    sequence: Sequence32
    text: ClauseText
    vector: AffectVector
    intensity: Unit
    seed: Seed32
    end_of_response: bool = False
