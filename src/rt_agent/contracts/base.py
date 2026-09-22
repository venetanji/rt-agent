"""Shared base model and constrained scalar aliases for every rt-agent contract.

Every contract model is frozen (immutable after construction) and rejects unknown
fields, so a schema drift on either side of a seam fails loudly at parse time
rather than silently dropping data.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AffectVector",
    "ClauseText",
    "FrozenModel",
    "Probability",
    "Seed32",
    "Sequence32",
    "Signed",
    "Unit",
]


class FrozenModel(BaseModel):
    """Immutable, strictly-validated base for all rt-agent contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")


#: A value in [0, 1] — intensities, fractions, confidences.
Unit = Annotated[float, Field(ge=0.0, le=1.0)]

#: A calibrated probability in [0, 1].
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

#: A signed value in [-1, 1] — one axis of an affect vector.
Signed = Annotated[float, Field(ge=-1.0, le=1.0)]

#: ``affect-vector/v1``: (valence, arousal, dominance), each in [-1, 1].
AffectVector = Annotated[tuple[Signed, Signed, Signed], Field()]

#: Clause index inside one response: alice accepts 0-31 (32 clauses maximum).
Sequence32 = Annotated[int, Field(ge=0, le=31)]

#: Spoken clause text: alice accepts 1-1000 characters.
ClauseText = Annotated[str, Field(min_length=1, max_length=1000)]

#: An unsigned 32-bit TTS seed.
Seed32 = Annotated[int, Field(ge=0, le=2**32 - 1)]
