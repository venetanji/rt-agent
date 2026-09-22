"""The preset table is authored data; these tests pin the parts other code depends on."""

from __future__ import annotations

import pytest

from rt_agent.policy import DEFAULT_PRESET, EMOTION_PRESETS, PRESETS_VERSION, preset_vector


def test_version_and_default() -> None:
    assert PRESETS_VERSION == "emotion-presets/v1"
    assert DEFAULT_PRESET == "neutral"
    assert EMOTION_PRESETS[DEFAULT_PRESET] == (0.0, 0.0, 0.0)


def test_every_axis_is_inside_the_affect_vector_range() -> None:
    for name, vector in EMOTION_PRESETS.items():
        assert len(vector) == 3, name
        assert all(-1.0 <= axis <= 1.0 for axis in vector), name


def test_the_table_is_read_only() -> None:
    with pytest.raises(TypeError):
        EMOTION_PRESETS["neutral"] = (1.0, 1.0, 1.0)  # type: ignore[index]


def test_surprised_clears_the_alice_smile_threshold() -> None:
    # alice's anchor selector tests valence strictly (> 0.2), so 0.20 would render
    # as neutral and the surprise would be invisible.
    assert preset_vector("surprised")[0] == pytest.approx(0.25)
    assert preset_vector("surprised")[0] > 0.2


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("attentive", (0.1, 0.2, 0.0)),
        ("warm", (0.5, 0.2, 0.1)),
        ("happy", (0.65, 0.4, 0.1)),
        ("amused", (0.7, 0.55, 0.1)),
        ("curious", (0.3, 0.4, -0.1)),
        ("concerned", (-0.3, 0.35, -0.1)),
        ("sympathetic", (-0.3, -0.1, 0.0)),
        ("sad", (-0.6, -0.4, -0.3)),
    ],
)
def test_authored_vectors(name: str, expected: tuple[float, float, float]) -> None:
    assert preset_vector(name) == expected


def test_an_unknown_preset_raises() -> None:
    with pytest.raises(KeyError):
        preset_vector("smug")
