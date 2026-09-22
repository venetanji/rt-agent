"""Fixtures for the audio tests, plus the ``models`` marker and the speech clip.

Everything in ``tests/audio`` runs offline and in well under a second except the
one end-to-end test, which is marked ``models``: it loads Silero and Whisper and is
skipped when ``RT_AGENT_SKIP_MODELS=1``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

AUDIO_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"

#: LibriSpeech dev-clean 1272-128104-0000, committed under tests/fixtures/audio.
SPEECH_CLIP = AUDIO_FIXTURES / "librispeech-1272-128104-0000.wav"

#: The reference transcript that ships with the LibriSpeech utterance.
SPEECH_CLIP_TEXT = (
    "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"
)

#: Regenerates the clip if it is ever missing; see tests/fixtures/audio/LICENSE.md.
_ROWS_API = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=hf-internal-testing%2Flibrispeech_asr_dummy&config=clean&split=validation"
)


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``models`` marker without touching the shared pyproject config."""
    config.addinivalue_line(
        "markers",
        "models: tests that load real ASR/VAD models (skipped when RT_AGENT_SKIP_MODELS=1)",
    )


def skip_without_models() -> None:
    """Skip the calling test when the environment opts out of model downloads."""
    if os.environ.get("RT_AGENT_SKIP_MODELS") == "1":
        pytest.skip("RT_AGENT_SKIP_MODELS=1")


def ensure_speech_clip() -> Path:
    """Return the committed LibriSpeech clip, downloading it once if it is absent."""
    if SPEECH_CLIP.is_file():
        return SPEECH_CLIP
    with urllib.request.urlopen(_ROWS_API, timeout=60) as response:
        payload = json.load(response)
    row = payload["rows"][0]["row"]
    with urllib.request.urlopen(row["audio"][0]["src"], timeout=120) as response:
        data = response.read()
    AUDIO_FIXTURES.mkdir(parents=True, exist_ok=True)
    SPEECH_CLIP.write_bytes(data)
    return SPEECH_CLIP
