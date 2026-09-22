"""Speaker labelling — session-local, anonymous, and never persisted.

The privacy rule this package is built around (alice ``AGENTS.md``, restated in the
PoC brief): **no voice embeddings on disk, ever, and no real names.** A label is a
within-session pointer that lets the decision bundle say "the same voice as two
turns ago"; it carries no identity across sessions and is thrown away with
:meth:`reset`. Downstream consumers must treat acoustic labels as *advisory* — a
diarizer confusing two similar voices should cost the robot a slightly wrong
context line, never a wrong person's memory.

Three implementations, in increasing order of ambition:

``SingleSpeakerDiarizer``
    One microphone, one person. The honest default for the PoC.
``ScriptedDiarizer``
    Deterministic labels from a fixed list, round-robin. For tests and for replaying
    an already-diarized transcript.
``EmbeddingDiarizer``
    Placeholder for real online diarization; raises on construction.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rt_agent.audio.sources import Pcm

__all__ = ["EmbeddingDiarizer", "ScriptedDiarizer", "SingleSpeakerDiarizer", "validate_label"]

#: ``Utterance.speaker_label`` accepts ``S1``, ``S2``, ``ROBOT`` — upper case, no names.
SPEAKER_LABEL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]{0,31}$")


def validate_label(label: str) -> str:
    """Return ``label`` if it is a legal anonymous speaker label, else raise."""
    if not SPEAKER_LABEL_PATTERN.match(label):
        raise ValueError(
            f"speaker label {label!r} must match {SPEAKER_LABEL_PATTERN.pattern} — "
            "labels are anonymous session-local tokens such as 'S1', never names"
        )
    return label


class SingleSpeakerDiarizer:
    """Every segment is the same speaker. Satisfies :class:`rt_agent.contracts.Diarizer`."""

    def __init__(self, label: str = "S1") -> None:
        self.label_value = validate_label(label)
        self.segments = 0

    def label(self, pcm: Pcm, sample_rate: int) -> str:
        """Return the fixed label, ignoring the audio entirely."""
        del pcm, sample_rate
        self.segments += 1
        return self.label_value

    def reset(self) -> None:
        """Forget the segment counter; the label itself never changes."""
        self.segments = 0


class ScriptedDiarizer:
    """Hand out labels from a fixed list, round-robin, in segment order.

    Useful in two places: tests that need a deterministic two-speaker conversation
    without a diarization model, and replaying audio whose speaker order is already
    known from a transcript.
    """

    def __init__(self, labels: Sequence[str]) -> None:
        if not labels:
            raise ValueError("ScriptedDiarizer needs at least one label")
        self.labels: tuple[str, ...] = tuple(validate_label(label) for label in labels)
        self.segments = 0

    def label(self, pcm: Pcm, sample_rate: int) -> str:
        """Return the next scripted label, wrapping around when the list runs out."""
        del pcm, sample_rate
        label = self.labels[self.segments % len(self.labels)]
        self.segments += 1
        return label

    def reset(self) -> None:
        """Restart the script from its first label."""
        self.segments = 0


class EmbeddingDiarizer:
    """Online speaker diarization by voice embedding — **not implemented in the PoC**.

    The intended implementation is `Diart <https://github.com/juanmc2005/diart>`_
    (incremental clustering over pyannote segmentation + embedding models, which
    alice already ships as sherpa-onnx exports for overlap detection) or
    `pyannote.audio <https://github.com/pyannote/pyannote-audio>`_ run offline on a
    recorded session.

    It is stubbed rather than half-built on purpose. Two design notes constrain any
    real version:

    * **Labels are advisory.** They are acoustic clusters, not identities. Nothing
      downstream may gate a memory, an address decision, or a safety behaviour on a
      label alone.
    * **Labels are session-local and embeddings never touch disk.** Clusters live in
      memory for the length of one session and are dropped on :meth:`reset`.
      Persisting a centroid would turn an anonymous label into a biometric
      identifier, which the privacy policy forbids.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError(
            "EmbeddingDiarizer is a placeholder. Use SingleSpeakerDiarizer (one "
            "speaker) or ScriptedDiarizer (known order); a real implementation would "
            "wrap Diart or pyannote.audio, keep clusters in memory only, and treat "
            "the labels as advisory."
        )

    def label(self, pcm: Pcm, sample_rate: int) -> str:  # pragma: no cover - unreachable
        """Never reached: the constructor raises."""
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - unreachable
        """Never reached: the constructor raises."""
        raise NotImplementedError
