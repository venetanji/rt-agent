"""Bounded endpointing: turn a frame + probability stream into whole utterances.

The state machine is a port of alice's ``Endpoint``
(``alice-workspace/src/alice/conversation/vad.py:33-127``), thresholds and all:

===========================  =========================  ==========================
alice constant               value                      here
===========================  =========================  ==========================
start (``probability >=``)   0.5                        ``start_probability``
release (``probability <``)  0.35                       ``release_probability``
``_ENDPOINT_SILENCE_SAMPLES`` ``FRAME * 10`` = 320 ms   ``endpoint_silence_s``
preroll ``[-3200:]``         200 ms                     ``preroll_s``
``_MAX_UTTERANCE_SAMPLES``   ``RATE * 15`` = 15 s       ``max_segment_s``
re-arm after overlong        320 ms quiet               ``endpoint_silence_s``
minimum voiced audio         1536 samples = 96 ms       ``min_voiced_s``
===========================  =========================  ==========================

Two behaviours are worth keeping straight because they are easy to get wrong:

* The **preroll is in the audio but not in the statistics.** Frames captured before
  the trigger are prepended to the waveform so the first phoneme survives, but they
  are excluded from ``vad_mean_prob`` and ``voiced_fraction``, whose denominator is
  the post-trigger frame count. This is alice's rule, documented at
  ``vad.py:34-39``.
* An **overlong utterance is discarded, not truncated**, and the detector then
  refuses to start a new segment until it has seen 320 ms of continuous quiet. A
  15-second monologue is a microphone problem or a television, not a turn.

A file or a stream that stops while somebody is still talking never reaches the
320 ms tail, so :meth:`Endpointer.flush` exists to close the last segment at
end-of-stream. Alice avoids the question by zero-padding its replay WAV; we make it
explicit instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rt_agent.audio.sources import DEFAULT_FRAME_SAMPLES, TARGET_SAMPLE_RATE, Pcm
from rt_agent.contracts import AudioEvidence

__all__ = [
    "ENDPOINT_SILENCE_S",
    "MAX_SEGMENT_S",
    "MIN_VOICED_S",
    "PREROLL_S",
    "RELEASE_PROBABILITY",
    "START_PROBABILITY",
    "AudioSegment",
    "Endpointer",
]

#: Speech starts at probability >= 0.5 (alice ``vad.py:90``).
START_PROBABILITY = 0.5
#: Speech releases below 0.35 — hysteresis keeps mid-word dips from cutting a turn.
RELEASE_PROBABILITY = 0.35
#: 320 ms of released audio closes the turn (alice ``FRAME * 10``).
ENDPOINT_SILENCE_S = 0.32
#: 200 ms kept ahead of the trigger so the first phoneme is not clipped.
PREROLL_S = 0.2
#: 15 s hard cap; longer is discarded, never truncated.
MAX_SEGMENT_S = 15.0
#: 96 ms of voiced audio minimum, else the segment is dropped as a click or a cough.
MIN_VOICED_S = 0.096

#: ``|x| >= 0.99`` counts as clipped, as in alice's audio evidence.
CLIPPING_THRESHOLD = 0.99


@dataclass(frozen=True)
class AudioSegment:
    """One endpointed utterance: the waveform, where it sat in the stream, its evidence.

    ``t_start_s`` and ``t_end_s`` are stream-relative seconds (0 = first sample the
    endpointer ever saw) and include the preroll, so
    ``t_end_s - t_start_s == len(samples) / sample_rate``.
    """

    samples: Pcm
    sample_rate: int
    t_start_s: float
    t_end_s: float
    evidence: AudioEvidence

    @property
    def duration_s(self) -> float:
        """Length of the waveform in seconds."""
        return self.samples.size / self.sample_rate


def _evidence(
    samples: Pcm,
    *,
    probability_sum: float,
    probability_frames: int,
    voiced_samples: int,
    frame_samples: int,
    sample_rate: int,
) -> AudioEvidence:
    """Build ``AudioEvidence`` for one segment. Never invents an unknown field."""
    mean_probability = probability_sum / probability_frames if probability_frames else 0.0
    denominator = probability_frames * frame_samples
    voiced_fraction = voiced_samples / denominator if denominator else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    clipped = float(np.mean(np.abs(samples) >= CLIPPING_THRESHOLD).item()) if samples.size else 0.0
    return AudioEvidence(
        vad_mean_prob=min(1.0, max(0.0, mean_probability)),
        voiced_fraction=min(1.0, max(0.0, voiced_fraction)),
        duration_s=samples.size / sample_rate,
        rms=min(10.0, rms),
        clipped_fraction=clipped,
        # Overlap needs a second model (alice runs pyannote-segmentation-3.0); the
        # PoC front end does not, and an unknown value stays None rather than 0.0.
        overlap_fraction=None,
    )


class Endpointer:
    """Alice's bounded endpointer, emitting :class:`AudioSegment` instead of raw PCM."""

    def __init__(
        self,
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
        frame_samples: int = DEFAULT_FRAME_SAMPLES,
        start_probability: float = START_PROBABILITY,
        release_probability: float = RELEASE_PROBABILITY,
        endpoint_silence_s: float = ENDPOINT_SILENCE_S,
        preroll_s: float = PREROLL_S,
        max_segment_s: float = MAX_SEGMENT_S,
        min_voiced_s: float = MIN_VOICED_S,
    ) -> None:
        if sample_rate <= 0 or frame_samples <= 0:
            raise ValueError("sample_rate and frame_samples must be positive")
        if not 0.0 <= release_probability <= start_probability <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= release <= start <= 1")
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.start_probability = start_probability
        self.release_probability = release_probability
        self.endpoint_silence_samples = round(endpoint_silence_s * sample_rate)
        self.preroll_samples = round(preroll_s * sample_rate)
        self.max_segment_samples = round(max_segment_s * sample_rate)
        self.min_voiced_samples = round(min_voiced_s * sample_rate)
        #: Total samples pushed through the endpointer; the stream clock.
        self.position = 0
        #: Segments thrown away for exceeding ``max_segment_s``.
        self.discarded_overlong = 0
        #: Segments thrown away for holding less than ``min_voiced_s`` of speech.
        self.dropped_short = 0
        self.last_discard_reason: str | None = None
        self._reset_segment()

    def _reset_segment(self) -> None:
        self.preroll: Pcm = np.zeros(0, dtype=np.float32)
        self._chunks: list[Pcm] = []
        self._samples = 0
        self._voiced = 0
        self._silence = 0
        self._probability_sum = 0.0
        self._probability_frames = 0
        self.speaking = False
        self._discarding = False
        self._recovery_silence = 0

    def reset(self) -> None:
        """Forget everything, including the stream clock. Use at a session boundary."""
        self.position = 0
        self.last_discard_reason = None
        self._reset_segment()

    @property
    def silence_ms(self) -> int:
        """Milliseconds of released audio accumulated inside the current segment."""
        return self._silence * 1000 // self.sample_rate

    def push(self, frame: Pcm, probability: float) -> AudioSegment | None:
        """Feed one frame and its speech probability; return a segment when one closes."""
        array = np.asarray(frame, dtype=np.float32)
        if (
            array.shape != (self.frame_samples,)
            or not np.isfinite(array).all()
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("invalid endpointer frame or probability")
        self.position += self.frame_samples

        if self._discarding:
            # Re-arm only after 320 ms of continuous quiet (alice vad.py:81-89).
            if probability < self.release_probability:
                self._recovery_silence += self.frame_samples
            else:
                self._recovery_silence = 0
            if self._recovery_silence >= self.endpoint_silence_samples:
                self._discarding = False
                self._recovery_silence = 0
            return None

        if probability >= self.start_probability and not self.speaking:
            self.speaking = True
            self._chunks = [self.preroll.copy()]
            self._samples = self.preroll.size

        if not self.speaking:
            self.preroll = np.concatenate((self.preroll, array))[-self.preroll_samples :]
            return None

        if self._samples + self.frame_samples > self.max_segment_samples:
            self._reset_segment()
            self._discarding = True
            self.discarded_overlong += 1
            self.last_discard_reason = "overlong_utterance"
            return None

        self._chunks.append(array.copy())
        self._samples += self.frame_samples
        self._probability_sum += probability
        self._probability_frames += 1
        if probability >= self.start_probability:
            self._voiced += self.frame_samples
        self._silence = (
            self._silence + self.frame_samples if probability < self.release_probability else 0
        )
        if self._silence >= self.endpoint_silence_samples:
            return self._close()
        return None

    def flush(self) -> AudioSegment | None:
        """Close an open segment at end-of-stream, without waiting for the silence tail."""
        if not self.speaking:
            self._reset_segment()
            return None
        return self._close()

    def _close(self) -> AudioSegment | None:
        if self._voiced < self.min_voiced_samples:
            self.dropped_short += 1
            self.last_discard_reason = "too_little_voiced_audio"
            self._reset_segment()
            return None
        samples = np.concatenate(self._chunks).astype(np.float32)
        evidence = _evidence(
            samples,
            probability_sum=self._probability_sum,
            probability_frames=self._probability_frames,
            voiced_samples=self._voiced,
            frame_samples=self.frame_samples,
            sample_rate=self.sample_rate,
        )
        t_end_s = self.position / self.sample_rate
        t_start_s = max(0.0, t_end_s - samples.size / self.sample_rate)
        self.last_discard_reason = None
        self._reset_segment()
        return AudioSegment(
            samples=samples,
            sample_rate=self.sample_rate,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            evidence=evidence,
        )
