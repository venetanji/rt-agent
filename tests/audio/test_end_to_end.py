"""The real thing: a public-domain speech clip through the whole front end.

Marked ``models`` because it loads Silero (ONNX, ~2 MB, shipped with the wheel) and
faster-whisper ``base.en`` (~140 MB, downloaded from Hugging Face on first run).
Skipped entirely when ``RT_AGENT_SKIP_MODELS=1``.

Audio: LibriSpeech dev-clean utterance 1272-128104-0000, CC BY 4.0 — see
``tests/fixtures/audio/LICENSE.md``.
"""

from __future__ import annotations

import time

import pytest

from rt_agent.audio.asr import FasterWhisperTranscriber
from rt_agent.audio.diarize import SingleSpeakerDiarizer
from rt_agent.audio.endpoint import Endpointer
from rt_agent.audio.pipeline import AudioFrontEnd, transcribe_wav
from rt_agent.audio.sources import WavFileSource
from rt_agent.audio.vad import QUALIFIED_SILERO_SHA256, SileroVad
from tests.audio.conftest import ensure_speech_clip, skip_without_models

pytestmark = pytest.mark.models

#: Words the reference transcript contains, chosen to be robust to punctuation and case.
EXPECTED_WORDS = ("quilter", "apostle", "middle", "classes", "gospel")


async def test_wav_to_utterance_through_silero_and_whisper() -> None:
    skip_without_models()
    clip = ensure_speech_clip()

    started = time.perf_counter()
    vad = SileroVad()
    vad_load_s = time.perf_counter() - started

    front_end = AudioFrontEnd(
        source=WavFileSource(clip),
        vad=vad,
        endpointer=Endpointer(),
        transcriber=FasterWhisperTranscriber(model_size="base.en", compute_type="int8"),
        diarizer=SingleSpeakerDiarizer("S1"),
        session_id="e2e",
    )
    started = time.perf_counter()
    utterances = [utterance async for utterance in front_end.utterances()]
    wall_s = time.perf_counter() - started

    assert front_end.segments_seen >= 1
    assert len(utterances) >= 1
    utterance = utterances[0]
    lowered = utterance.text.lower()
    missing = [word for word in EXPECTED_WORDS if word not in lowered]
    assert not missing, f"missing {missing} in {utterance.text!r}"

    assert utterance.utterance_id == "u0001"
    assert utterance.session_id == "e2e"
    assert utterance.speaker_label == "S1"
    assert utterance.language == "en"
    assert utterance.is_final
    assert utterance.asr_confidence is not None and utterance.asr_confidence > 0.5

    evidence = utterance.audio
    assert evidence is not None
    assert evidence.vad_mean_prob > 0.7
    assert evidence.voiced_fraction > 0.7
    assert 3.0 < evidence.duration_s < 6.0
    assert 0.0 < evidence.rms < 1.0
    assert evidence.clipped_fraction == 0.0
    assert evidence.overlap_fraction is None

    # Timings, printed with -s so the numbers in the report can be reproduced.
    print(
        f"\nsilero load {vad_load_s * 1000:.0f} ms; "
        f"{front_end.frames_seen} frames; "
        f"vad {front_end.vad_seconds * 1000:.0f} ms "
        f"({front_end.vad_seconds / max(utterance.t_end_s, 1e-9):.4f} x realtime); "
        f"asr {front_end.asr_seconds:.2f} s "
        f"({front_end.asr_seconds / evidence.duration_s:.2f} x realtime); "
        f"wall {wall_s:.2f} s\ntranscript: {utterance.text}"
    )


async def test_transcribe_wav_is_the_same_path_in_one_call() -> None:
    skip_without_models()
    clip = ensure_speech_clip()
    utterances = await transcribe_wav(clip, session_id="convenience")
    assert len(utterances) >= 1
    assert "quilter" in utterances[0].text.lower()
    assert utterances[0].session_id == "convenience"


def test_the_vad_runs_on_onnxruntime_without_importing_torch() -> None:
    skip_without_models()
    import subprocess
    import sys

    # A fresh interpreter: if anything on the VAD path imported torch, it would be
    # in sys.modules afterwards. The robot host must not need torch to listen.
    script = (
        "import sys; import numpy as np;"
        "from rt_agent.audio.vad import SileroVad;"
        "v = SileroVad();"
        "p = v.probability(np.zeros(512, dtype=np.float32));"
        "print(v.model_sha256, 'torch' in sys.modules, p)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    digest, torch_imported, _probability = result.stdout.split()
    assert torch_imported == "False"
    assert digest == QUALIFIED_SILERO_SHA256


def test_silero_reports_the_qualified_model_digest() -> None:
    skip_without_models()
    vad = SileroVad()
    assert vad.model_sha256 == QUALIFIED_SILERO_SHA256
    assert vad.is_qualified
    assert vad.model_id == f"silero_vad.onnx@sha256:{QUALIFIED_SILERO_SHA256}"
