"""Runtime-checkable protocols for every replaceable seam in the harness.

Each protocol is the entire contract between two packages: audio, systemone, llm,
memory and face are written against these and never against each other.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from rt_agent.contracts.affect import AffectCue
from rt_agent.contracts.base import FrozenModel
from rt_agent.contracts.decision import BundleAnswers
from rt_agent.contracts.events import Transcription
from rt_agent.contracts.memory import MemoryRecord
from rt_agent.contracts.speech import SpeechClauseOut

__all__ = [
    "AudioSource",
    "ChatLLM",
    "ChatMessage",
    "Diarizer",
    "FaceBridge",
    "MemoryStore",
    "QuestionsWire",
    "SystemOneBackend",
    "Transcriber",
    "VoiceActivityDetector",
]

#: One question as it goes on the wire: ``{"type": ..., "instructions": ..., "criteria": ...}``.
QuestionWire = Mapping[str, Any]

#: The whole bundle on the wire, keyed by question id.
QuestionsWire = Mapping[str, QuestionWire]

#: Mono float32 PCM.
Pcm = NDArray[np.float32]


class ChatMessage(FrozenModel):
    """One message in a chat completion request."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


@runtime_checkable
class AudioSource(Protocol):
    """A stream of fixed-size mono float32 frames."""

    sample_rate: int
    frame_samples: int

    def frames(self) -> AsyncIterator[Pcm]:
        """Yield frames until the source is exhausted or closed."""
        ...

    async def aclose(self) -> None:
        """Release the underlying device or file."""
        ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Frame-level speech probability, stateful across a session."""

    def probability(self, frame: Pcm) -> float:
        """Probability that this frame contains speech."""
        ...

    def reset(self) -> None:
        """Drop internal state at a session or utterance boundary."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Speech-to-text for one endpointed segment."""

    async def transcribe(self, pcm: Pcm, sample_rate: int) -> Transcription:
        """Transcribe one complete utterance."""
        ...


@runtime_checkable
class Diarizer(Protocol):
    """Assigns a session-local anonymous speaker label to a segment."""

    def label(self, pcm: Pcm, sample_rate: int) -> str:
        """Return a label such as ``S1``; never a real name."""
        ...

    def reset(self) -> None:
        """Forget all speakers at a session boundary."""
        ...


@runtime_checkable
class SystemOneBackend(Protocol):
    """A System One endpoint: one rendered state plus a frozen bundle in, typed answers out."""

    def ask(self, state: str, questions: QuestionsWire) -> BundleAnswers:
        """Blocking bundle call."""
        ...

    async def aask(self, state: str, questions: QuestionsWire) -> BundleAnswers:
        """Awaitable bundle call; the harness wraps it in a hard deadline."""
        ...


@runtime_checkable
class ChatLLM(Protocol):
    """A remote text generator — replies and memory sentences. Never on the admission path."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant's completion as plain text."""
        ...


@runtime_checkable
class FaceBridge(Protocol):
    """Where affect cues and spoken clauses go: JSONL, UDP, or a ROS 2 topic."""

    def emit_affect(self, cue: AffectCue) -> None:
        """Publish one expression target."""
        ...

    def emit_clauses(self, clauses: Sequence[SpeechClauseOut]) -> None:
        """Publish an ordered run of clauses for one response."""
        ...

    def close(self) -> None:
        """Flush and release the transport."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Persistent memory layer, separate from the rolling decision window."""

    def add(self, record: MemoryRecord) -> None:
        """Persist one memory."""
        ...

    def search(self, session_id: str, query: str, limit: int = 3) -> tuple[MemoryRecord, ...]:
        """Best-effort retrieval by keyword and recency."""
        ...

    def all(self) -> tuple[MemoryRecord, ...]:
        """Every stored memory, newest first."""
        ...

    def close(self) -> None:
        """Close the underlying database."""
        ...
