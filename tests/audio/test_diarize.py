"""Speaker labelling: anonymous, session-local, and advisory by construction."""

from __future__ import annotations

import numpy as np
import pytest

from rt_agent.audio.diarize import (
    EmbeddingDiarizer,
    ScriptedDiarizer,
    SingleSpeakerDiarizer,
    validate_label,
)
from rt_agent.contracts import Diarizer

PCM = np.zeros(1600, dtype=np.float32)


def test_the_shipped_diarizers_satisfy_the_protocol() -> None:
    assert isinstance(SingleSpeakerDiarizer(), Diarizer)
    assert isinstance(ScriptedDiarizer(["S1", "S2"]), Diarizer)


def test_single_speaker_always_returns_the_same_label() -> None:
    diarizer = SingleSpeakerDiarizer()
    assert [diarizer.label(PCM, 16_000) for _ in range(3)] == ["S1", "S1", "S1"]
    assert diarizer.segments == 3
    diarizer.reset()
    assert diarizer.segments == 0


def test_scripted_labels_go_round_robin_and_reset() -> None:
    diarizer = ScriptedDiarizer(["S1", "S2", "S3"])
    assert [diarizer.label(PCM, 16_000) for _ in range(5)] == ["S1", "S2", "S3", "S1", "S2"]
    diarizer.reset()
    assert diarizer.label(PCM, 16_000) == "S1"


@pytest.mark.parametrize("label", ["Alice", "s1", "", "S" * 40, "S 1"])
def test_names_and_malformed_labels_are_refused(label: str) -> None:
    with pytest.raises(ValueError, match="anonymous session-local"):
        validate_label(label)


def test_scripted_needs_at_least_one_label() -> None:
    with pytest.raises(ValueError, match="at least one label"):
        ScriptedDiarizer([])


def test_the_embedding_diarizer_is_an_explicit_placeholder() -> None:
    with pytest.raises(NotImplementedError, match="Diart or pyannote"):
        EmbeddingDiarizer()
    assert "session-local" in (EmbeddingDiarizer.__doc__ or "")
    assert "advisory" in (EmbeddingDiarizer.__doc__ or "")
