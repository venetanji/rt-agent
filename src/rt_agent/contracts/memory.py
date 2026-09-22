"""``memory/v1`` — one durable fact the robot may still know next week."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from rt_agent.contracts.base import FrozenModel, Probability

__all__ = ["MEMORY_KINDS", "MemoryKind", "MemoryRecord"]

MemoryKind = Literal[
    "personal_fact",
    "preference",
    "event",
    "plan",
    "relationship",
    "health",
    "other",
]

#: The frozen option list, in the order the question bundle presents it.
MEMORY_KINDS: tuple[MemoryKind, ...] = (
    "personal_fact",
    "preference",
    "event",
    "plan",
    "relationship",
    "health",
    "other",
)


class MemoryRecord(FrozenModel):
    """A stored memory. ``text`` is written by a ChatLLM and gated by a faithfulness check."""

    schema_version: Literal["memory/v1"] = "memory/v1"
    memory_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    speaker_label: str = Field(min_length=1, max_length=32)
    kind: MemoryKind
    text: str = Field(min_length=1, max_length=300)
    source_utterance_ids: tuple[str, ...] = ()
    worth_p: Probability
    faithfulness_p: Probability
    sensitive: bool = False
    created_at: AwareDatetime
