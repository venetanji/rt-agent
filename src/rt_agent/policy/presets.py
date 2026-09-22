"""``emotion-presets/v1`` — the authored preset table.

Each preset maps a name the model can choose to an ``affect-vector/v1``
(valence, arousal, dominance). The names are exactly the option set of the ``emotion``
question, so the bundle and the presets must be changed together.

Today's alice face reads only valence: > +0.2 opens a smile, < -0.2 a closed frown,
anything between is neutral, and the amplitude is ``intensity * 0.5``. The arousal and
dominance axes are carried anyway so a richer face needs no schema change.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "DEFAULT_PRESET",
    "EMOTION_PRESETS",
    "PRESETS_VERSION",
    "preset_names",
    "preset_vector",
]

PRESETS_VERSION = "emotion-presets/v1"

#: The preset the robot rests in, and the one every decay path ends at.
DEFAULT_PRESET = "neutral"

EMOTION_PRESETS: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "neutral": (0.0, 0.0, 0.0),
        "attentive": (0.1, 0.2, 0.0),
        "warm": (0.5, 0.2, 0.1),
        "happy": (0.65, 0.4, 0.1),
        "amused": (0.7, 0.55, 0.1),
        "curious": (0.3, 0.4, -0.1),
        # 0.25, not the authored 0.2: alice's anchor selector tests valence strictly
        # (> 0.2), so a surprised face at exactly 0.2 would render as neutral.
        "surprised": (0.25, 0.8, -0.2),
        "concerned": (-0.3, 0.35, -0.1),
        "sympathetic": (-0.3, -0.1, 0.0),
        "sad": (-0.6, -0.4, -0.3),
    }
)


def preset_names() -> tuple[str, ...]:
    """Every preset name, in authored order."""
    return tuple(EMOTION_PRESETS)


def preset_vector(name: str) -> tuple[float, float, float]:
    """The affect vector for a preset. Raises ``KeyError`` for an unknown name."""
    return EMOTION_PRESETS[name]
