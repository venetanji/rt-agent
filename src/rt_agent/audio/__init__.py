"""The audio front end: microphone or file in, diarized final utterances out.

Import from here; the module split (:mod:`~rt_agent.audio.sources`,
:mod:`~rt_agent.audio.vad`, :mod:`~rt_agent.audio.endpoint`,
:mod:`~rt_agent.audio.asr`, :mod:`~rt_agent.audio.diarize`,
:mod:`~rt_agent.audio.replay`, :mod:`~rt_agent.audio.pipeline`) is an implementation
detail.

Two paths, one output type:

* **Live/recorded audio** — ``WavFileSource``/``MicSource`` → ``SileroVad`` →
  ``Endpointer`` → ``FasterWhisperTranscriber`` → a ``Diarizer``, composed by
  ``AudioFrontEnd``.
* **Replay** — ``ReplaySource`` reads a diarized transcript JSONL and yields the same
  ``Utterance`` objects with no models and no hardware. This is the default path for
  the harness and for every test that is not specifically about audio.

Nothing here imports torch, and nothing imports ``soundfile``, ``sounddevice``,
``onnxruntime`` or ``faster_whisper`` at module scope, so ``import rt_agent.audio``
works on a host with none of the extras installed.
"""

from __future__ import annotations

from rt_agent.audio.asr import ASR_LANGUAGE, EmptyTranscriptError, FasterWhisperTranscriber
from rt_agent.audio.diarize import (
    EmbeddingDiarizer,
    ScriptedDiarizer,
    SingleSpeakerDiarizer,
    validate_label,
)
from rt_agent.audio.endpoint import (
    ENDPOINT_SILENCE_S,
    MAX_SEGMENT_S,
    MIN_VOICED_S,
    PREROLL_S,
    RELEASE_PROBABILITY,
    START_PROBABILITY,
    AudioSegment,
    Endpointer,
)
from rt_agent.audio.pipeline import AudioFrontEnd, transcribe_wav
from rt_agent.audio.replay import ReplayFormatError, ReplaySource
from rt_agent.audio.sources import (
    DEFAULT_FRAME_SAMPLES,
    TARGET_SAMPLE_RATE,
    MicrophoneUnavailableError,
    MicSource,
    Pcm,
    WavFileSource,
    resample_linear,
    to_mono,
)
from rt_agent.audio.vad import (
    QUALIFIED_SILERO_SHA256,
    SILERO_FRAME_SAMPLES,
    SILERO_SAMPLE_RATE,
    SileroVad,
    find_silero_model,
)

__all__ = [
    "ASR_LANGUAGE",
    "DEFAULT_FRAME_SAMPLES",
    "ENDPOINT_SILENCE_S",
    "MAX_SEGMENT_S",
    "MIN_VOICED_S",
    "PREROLL_S",
    "QUALIFIED_SILERO_SHA256",
    "RELEASE_PROBABILITY",
    "SILERO_FRAME_SAMPLES",
    "SILERO_SAMPLE_RATE",
    "START_PROBABILITY",
    "TARGET_SAMPLE_RATE",
    "AudioFrontEnd",
    "AudioSegment",
    "EmbeddingDiarizer",
    "EmptyTranscriptError",
    "Endpointer",
    "FasterWhisperTranscriber",
    "MicSource",
    "MicrophoneUnavailableError",
    "Pcm",
    "ReplayFormatError",
    "ReplaySource",
    "ScriptedDiarizer",
    "SileroVad",
    "SingleSpeakerDiarizer",
    "WavFileSource",
    "find_silero_model",
    "resample_linear",
    "to_mono",
    "transcribe_wav",
    "validate_label",
]
