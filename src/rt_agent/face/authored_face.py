"""A simulator of alice's authored expression policy — *not* the authority.

alice decides what her face does; this module only predicts it, so a hardware-free run
can log "what the face would have done". The policy mirrored here is
``config/speech/authored-expression-v1.json`` and the anchor table in
``config/models/procedural-motion-v1.yaml``, read through
``_AuthoredAnchorPlanner._interpolated_target`` (``src/alice/speech/expression_bridge.py``):

.. code-block:: python

    name = "neutral"
    if intent.vector[0] >  valence_threshold: name = positive_anchor   # smile_open
    elif intent.vector[0] < -valence_threshold: name = negative_anchor # frown_closed
    return self._scaled_anchor(name, intent.intensity * anchor_scale)

Only valence selects the anchor. Arousal and dominance are carried through alice's whole
stack but reach nothing except blink/gaze hazard rates, so a cue that differs from another
only in arousal produces an identical face here — and on the robot.

What this simulator deliberately does **not** model, because it cannot be checked
hardware-free: the 400 ms accepted motion prefix, per-channel rate/acceleration caps
(mouth corners are limited to 0.8/s), procedural blink and gaze, the speech envelope's
ownership of the jaw while audio plays, and the ``sad_hold_ms`` post-speech pose. Real
onset is therefore *slower* than what ``advance`` reports, never faster. alice's own
documentation forbids claiming instantaneous expression onset.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import Field

from rt_agent.contracts import AffectCue, FrozenModel, Unit

__all__ = [
    "ANCHORS",
    "BENCH_PROFILE",
    "FACE_PROFILES",
    "VISIBLE_PROFILE",
    "AuthoredFaceModel",
    "FaceProfile",
    "FaceState",
    "FaceTarget",
    "ProfileName",
]

#: ``config/models/procedural-motion-v1.yaml``. Right corner polarity is inverted in
#: hardware, hence the opposite signs — this is not a typo.
ANCHORS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "neutral": MappingProxyType(
            {"mouth_open": 0.0, "left_mouth_corner": 0.0, "right_mouth_corner": 0.0}
        ),
        "smile_open": MappingProxyType(
            {"mouth_open": 1.0, "left_mouth_corner": 1.0, "right_mouth_corner": -1.0}
        ),
        "frown_closed": MappingProxyType(
            {"mouth_open": -1.0, "left_mouth_corner": -1.0, "right_mouth_corner": 1.0}
        ),
    }
)

#: The channels the authored anchors touch. The other eight stay at Home (0.0).
ANCHOR_CHANNELS: tuple[str, ...] = ("mouth_open", "left_mouth_corner", "right_mouth_corner")


class FaceProfile(FrozenModel):
    """One ``authored-expression/v1`` policy file, as a value."""

    name: str = Field(min_length=1, max_length=64)
    valence_threshold: float = Field(gt=0.0, lt=1.0)
    anchor_scale: float = Field(gt=0.0, le=1.0)
    transition_s: float = Field(ge=0.8, le=5.0)
    positive_anchor: str = "smile_open"
    negative_anchor: str = "frown_closed"
    source: str = Field(min_length=1)


#: ``config/speech/authored-expression-v1.json`` — the conservative bench policy.
BENCH_PROFILE = FaceProfile(
    name="bench",
    valence_threshold=0.2,
    anchor_scale=0.5,
    transition_s=1.6,
    source="alice config/speech/authored-expression-v1.json",
)

#: ``config/speech/authored-expression-visible-v1.json`` — the visible-face profile.
VISIBLE_PROFILE = FaceProfile(
    name="visible",
    valence_threshold=0.2,
    anchor_scale=1.0,
    transition_s=0.8,
    source="alice config/speech/authored-expression-visible-v1.json",
)

FACE_PROFILES: Mapping[str, FaceProfile] = MappingProxyType(
    {"bench": BENCH_PROFILE, "visible": VISIBLE_PROFILE}
)

ProfileName = Literal["bench", "visible"]


class FaceTarget(FrozenModel):
    """Where the simulated face is heading for one cue.

    Not to be confused with ``alice_interfaces/msg/FaceTarget``, which is the ROS message
    the motion node publishes to the Maestro. This is a prediction, that is a command.
    """

    anchor: str = Field(min_length=1, max_length=64)
    amplitude: Unit
    channels: dict[str, float]
    profile: str = Field(min_length=1, max_length=64)
    visible: bool


class FaceState(FrozenModel):
    """The simulated face at one instant, part-way through a transition."""

    t: float = Field(ge=0.0)
    anchor: str = Field(min_length=1, max_length=64)
    amplitude: Unit
    channels: dict[str, float]
    progress: Unit
    settled: bool


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


class AuthoredFaceModel:
    """Predict alice's authored face for a stream of cues, on a simple linear clock.

    Usage: ``apply(cue, now)`` whenever the decision changes, ``advance(now)`` as often as
    the log wants a sample. Times are seconds on any monotonic clock the caller likes;
    only differences matter.
    """

    def __init__(self, profile: ProfileName | FaceProfile = "bench") -> None:
        if isinstance(profile, FaceProfile):
            self.profile = profile
        elif profile in FACE_PROFILES:
            self.profile = FACE_PROFILES[profile]
        else:
            known = ", ".join(sorted(FACE_PROFILES))
            raise ValueError(f"unknown face profile {profile!r}; known profiles: {known}")
        self._target = FaceTarget(
            anchor="neutral",
            amplitude=0.0,
            channels=dict.fromkeys(ANCHOR_CHANNELS, 0.0),
            profile=self.profile.name,
            visible=False,
        )
        self._start: dict[str, float] = dict.fromkeys(ANCHOR_CHANNELS, 0.0)
        self._current: dict[str, float] = dict.fromkeys(ANCHOR_CHANNELS, 0.0)
        self._started_at = 0.0
        self._now = 0.0

    def visibility(self, cue: AffectCue) -> bool:
        """True when this cue crosses the valence threshold, i.e. moves the face at all.

        The comparison is strict, exactly as alice's planner writes it: a cue at valence
        0.2 renders as ``neutral``; 0.25 (the ``surprised`` preset) renders as a smile.
        """
        return abs(cue.vector[0]) > self.profile.valence_threshold

    def anchor_for(self, cue: AffectCue) -> str:
        """The anchor name alice's planner would select for this cue."""
        valence = cue.vector[0]
        if valence > self.profile.valence_threshold:
            return self.profile.positive_anchor
        if valence < -self.profile.valence_threshold:
            return self.profile.negative_anchor
        return "neutral"

    def target_for(self, cue: AffectCue) -> FaceTarget:
        """The scaled anchor this cue asks for. Pure: it does not touch the simulation."""
        anchor = self.anchor_for(cue)
        amplitude = _clamp(cue.intensity * self.profile.anchor_scale, 0.0, 1.0)
        unit = ANCHORS[anchor]
        return FaceTarget(
            anchor=anchor,
            amplitude=amplitude,
            channels={name: unit[name] * amplitude for name in ANCHOR_CHANNELS},
            profile=self.profile.name,
            visible=self.visibility(cue),
        )

    def apply(self, cue: AffectCue, now: float) -> FaceTarget:
        """Adopt a new target and start a fresh transition from wherever the face is now."""
        state = self.advance(now)
        self._start = dict(state.channels)
        self._target = self.target_for(cue)
        self._started_at = now
        return self._target

    def advance(self, now: float) -> FaceState:
        """Step the simulation to ``now`` and report the face.

        Interpolation is linear over ``transition_s``; alice's generator is smoother and
        rate-limited, so treat this as an upper bound on how fast the face moves.
        """
        if now < self._now:
            raise ValueError("time must not go backwards")
        self._now = now
        span = self.profile.transition_s
        elapsed = now - self._started_at
        progress = 1.0 if span <= 0.0 else _clamp(elapsed / span, 0.0, 1.0)
        self._current = {
            name: self._start[name] + (self._target.channels[name] - self._start[name]) * progress
            for name in ANCHOR_CHANNELS
        }
        return FaceState(
            t=now,
            anchor=self._target.anchor,
            amplitude=self._target.amplitude,
            channels=dict(self._current),
            progress=progress,
            settled=progress >= 1.0,
        )

    def reset(self, now: float = 0.0) -> None:
        """Return the face to Home and restart the clock.

        alice does the same thing at the end of every run: ``FaceRuntime`` drives every
        channel to 0 whenever no source is proposing anything.
        """
        self._target = FaceTarget(
            anchor="neutral",
            amplitude=0.0,
            channels=dict.fromkeys(ANCHOR_CHANNELS, 0.0),
            profile=self.profile.name,
            visible=False,
        )
        self._start = dict.fromkeys(ANCHOR_CHANNELS, 0.0)
        self._current = dict.fromkeys(ANCHOR_CHANNELS, 0.0)
        self._started_at = now
        self._now = now
