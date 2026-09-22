"""Builders shared by the face tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rt_agent.contracts import AffectCue
from rt_agent.policy.presets import EMOTION_PRESETS

ALICE_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "alice"

ISSUED_AT = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)


def make_cue(
    preset: str = "warm",
    intensity: float = 0.5,
    *,
    vector: tuple[float, float, float] | None = None,
    source_confidence: float = 0.9,
    valid_for_ms: int = 1500,
) -> AffectCue:
    """One affect cue, by preset name unless an explicit vector is given."""
    return AffectCue(
        vector=vector if vector is not None else EMOTION_PRESETS[preset],
        intensity=intensity,
        preset=preset,
        source_confidence=source_confidence,
        valid_for_ms=valid_for_ms,
        issued_at=ISSUED_AT,
    )


@pytest.fixture
def cue() -> AffectCue:
    """The default cue: the ``warm`` preset at half intensity."""
    return make_cue()


@pytest.fixture
def schema() -> dict:
    """alice's ``speech-plan/v1`` JSON Schema, copied verbatim at commit 8ddcb6a."""
    import json

    return json.loads((ALICE_FIXTURES / "speech-plan.schema.json").read_text())
