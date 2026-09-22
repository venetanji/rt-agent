"""Local speech-to-text with faster-whisper (CTranslate2), off the event loop.

Alice calls a hosted Qwen ASR over HTTP; the robot host should not depend on a
server for its own ears, so the PoC runs Whisper locally. ``base.en`` int8 on CPU
is the smallest model that reliably transcribes conversational English, and the
weights come from Hugging Face (``Systran/faster-whisper-<size>``) on first use.

Two settings are deliberate, not defaults:

* ``vad_filter=False`` — the audio handed to :meth:`FasterWhisperTranscriber.transcribe`
  has already been endpointed by :mod:`rt_agent.audio.endpoint`. Letting Whisper's
  own VAD re-segment it would silently contradict the evidence attached to the
  utterance.
* ``beam_size=1`` — greedy decoding. On short endpointed turns the accuracy
  difference against beam 5 is small and the latency difference is not.

**Confidence.** Whisper reports ``avg_logprob`` per segment: the mean natural log
probability of the decoded tokens. ``exp(avg_logprob)`` is therefore the geometric
mean per-token probability, a number in (0, 1] that behaves like a confidence — 0.8
for clean speech, 0.3-0.5 when the decoder is guessing. We take the
duration-weighted mean of ``avg_logprob`` across segments, exponentiate once, and
clamp to [0, 1]. It is a monotone re-scaling of the model's own likelihood, **not a
calibrated probability of being correct**; the policy must treat it as evidence, not
as accuracy.
"""

from __future__ import annotations

import asyncio
import math
import threading
from typing import Any

import numpy as np

from rt_agent.audio.sources import TARGET_SAMPLE_RATE, Pcm, resample_linear
from rt_agent.contracts import Transcription

__all__ = ["ASR_LANGUAGE", "EmptyTranscriptError", "FasterWhisperTranscriber"]

#: The PoC is English-only: ``base.en`` cannot decode anything else, and the
#: question bundle is written in English.
ASR_LANGUAGE = "en"

#: ``Utterance.text`` and ``Transcription.text`` accept at most 1000 characters.
MAX_TEXT_CHARS = 1000


class EmptyTranscriptError(ValueError):
    """Whisper returned no text for a segment.

    The contract forbids an empty ``Transcription.text``, and a silent segment is a
    normal event rather than a failure, so callers (see
    :class:`rt_agent.audio.pipeline.AudioFrontEnd`) catch this and drop the segment.
    """


def _confidence_from_avg_logprob(avg_logprob: float) -> float:
    """Map Whisper's mean token log-probability onto [0, 1]; see the module docstring."""
    if not math.isfinite(avg_logprob):
        return 0.0
    return min(1.0, max(0.0, math.exp(avg_logprob)))


class FasterWhisperTranscriber:
    """A :class:`rt_agent.contracts.Transcriber` backed by faster-whisper on CPU.

    The model is loaded lazily on the first transcription (a ``base.en`` int8 load is
    ~6 s cold, including the Hugging Face download) and every blocking decode runs in
    the default thread executor, so an ``asyncio`` harness keeps servicing audio
    frames while Whisper works.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        compute_type: str = "int8",
        device: str = "cpu",
        *,
        beam_size: int = 1,
        cpu_threads: int = 0,
        download_root: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.download_root = download_root
        self.language = ASR_LANGUAGE
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        """``faster-whisper/<size>@<compute_type>`` — safe to log next to a decision."""
        return f"faster-whisper/{self.model_size}@{self.compute_type}"

    def load(self) -> Any:
        """Load (and cache) the CTranslate2 model. Blocking; safe to call twice."""
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as error:  # pragma: no cover - depends on the host
                    raise RuntimeError(
                        "FasterWhisperTranscriber needs faster-whisper: install rt-agent[audio]"
                    ) from error
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    download_root=self.download_root,
                )
            return self._model

    async def transcribe(self, pcm: Pcm, sample_rate: int) -> Transcription:
        """Transcribe one endpointed segment. Raises :class:`EmptyTranscriptError` on silence."""
        audio = np.asarray(pcm, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("transcribe expects finite 1-D mono float32 PCM")
        if sample_rate != TARGET_SAMPLE_RATE:
            audio = resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_blocking, audio)

    def _transcribe_blocking(self, audio: Pcm) -> Transcription:
        model = self.load()
        segments, _info = model.transcribe(
            audio,
            language=ASR_LANGUAGE,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
            without_timestamps=True,
        )
        pieces: list[str] = []
        weighted_logprob = 0.0
        weight_total = 0.0
        for segment in segments:
            text = str(segment.text).strip()
            if text:
                pieces.append(text)
            weight = max(float(segment.end) - float(segment.start), 1e-3)
            weighted_logprob += float(segment.avg_logprob) * weight
            weight_total += weight
        joined = " ".join(pieces).strip()
        if not joined:
            raise EmptyTranscriptError("whisper produced no text for this segment")
        confidence = (
            _confidence_from_avg_logprob(weighted_logprob / weight_total)
            if weight_total > 0
            else None
        )
        return Transcription(
            text=joined[:MAX_TEXT_CHARS],
            language=ASR_LANGUAGE,
            confidence=confidence,
        )
