"""``EmotionController`` — hysteresis between what the model picks and what the face shows.

A face that follows every utterance twitches. This controller adds three guards, and
nothing else:

* a **confidence gate** — a cue below ``emotion_confidence_min`` never moves the face;
* a **refractory period** — after a switch, the preset is held for
  ``emotion_refractory_s`` no matter what arrives; a repeat of the *same* preset is not
  a switch and may refresh the cue at any time;
* an **expiry and decay** — once a cue outlives ``valid_for_ms`` its intensity is
  halved for each elapsed window, and once it falls below
  ``emotion_min_visible_intensity`` the face returns to neutral.

The controller is pure: it never reads a clock. The caller passes ``now``, which is
what makes the timeline testable.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from rt_agent.contracts.affect import AffectCue
from rt_agent.contracts.decision import Decision
from rt_agent.policy.policy import PolicyConfig
from rt_agent.policy.presets import DEFAULT_PRESET, EMOTION_PRESETS

__all__ = ["EmotionController"]


class EmotionController:
    """Holds the expression the face is currently showing and decides when it may change."""

    def __init__(
        self,
        config: PolicyConfig | None = None,
        *,
        started_at: datetime,
        initial_preset: str = DEFAULT_PRESET,
    ) -> None:
        self.config = config or PolicyConfig()
        self._cue = self._neutral(started_at, preset=initial_preset)
        # No switch has happened yet, so the very first expression is not held back by
        # the refractory period.
        self._last_switch_at: datetime | None = None

    # -- construction helpers -----------------------------------------------------

    def _neutral(self, now: datetime, preset: str = DEFAULT_PRESET) -> AffectCue:
        name = preset if preset in EMOTION_PRESETS else DEFAULT_PRESET
        return AffectCue(
            vector=EMOTION_PRESETS[name],
            intensity=0.0,
            preset=name,
            source_confidence=0.0,
            valid_for_ms=self.config.cue_valid_for_ms,
            issued_at=now,
        )

    def _reissue(self, cue: AffectCue, *, intensity: float, issued_at: datetime) -> AffectCue:
        """Re-stamp a cue. The controller, not the proposal, owns how long a cue lives."""
        return AffectCue(
            vector=cue.vector,
            intensity=intensity,
            preset=cue.preset,
            source_id=cue.source_id,
            source_confidence=cue.source_confidence,
            valid_for_ms=self.config.cue_valid_for_ms,
            issued_at=issued_at,
        )

    # -- state --------------------------------------------------------------------

    def current(self) -> AffectCue:
        """The cue the face should be showing, as of the last ``tick`` or ``apply``."""
        return self._cue

    @property
    def last_switch_at(self) -> datetime | None:
        """When the preset last changed, or ``None`` before the first switch."""
        return self._last_switch_at

    def expires_at(self) -> datetime:
        """When the current cue stops being valid."""
        return self._cue.issued_at + timedelta(milliseconds=self._cue.valid_for_ms)

    # -- the two operations -------------------------------------------------------

    def tick(self, now: datetime) -> AffectCue | None:
        """Advance expiry and decay. Returns a new cue to emit, or ``None`` if unchanged."""
        emitted: AffectCue | None = None
        window = timedelta(milliseconds=self.config.cue_valid_for_ms)
        if window <= timedelta(0):
            return None
        while now >= self._cue.issued_at + window:
            if self._cue.preset == DEFAULT_PRESET and self._cue.intensity == 0.0:
                break
            boundary = self._cue.issued_at + window
            decayed = self._cue.intensity * self.config.emotion_decay_factor
            if decayed < self.config.emotion_min_visible_intensity:
                self._cue = self._neutral(boundary)
            else:
                self._cue = self._reissue(self._cue, intensity=decayed, issued_at=boundary)
            emitted = self._cue
        return emitted

    def apply(self, decision: Decision, now: datetime) -> AffectCue | None:
        """Fold one decision into the face state.

        Returns the cue that should be emitted now, or ``None`` when nothing changed.
        """
        emitted = self.tick(now)
        proposed = decision.affect

        if proposed.source_confidence < self.config.emotion_confidence_min:
            return emitted

        if proposed.preset == self._cue.preset:
            # Not a switch: hold the same expression and push its expiry out.
            self._cue = self._reissue(proposed, intensity=proposed.intensity, issued_at=now)
            return self._cue

        refractory = timedelta(seconds=self.config.emotion_refractory_s)
        if self._last_switch_at is not None and now - self._last_switch_at < refractory:
            return emitted

        self._cue = self._reissue(proposed, intensity=proposed.intensity, issued_at=now)
        self._last_switch_at = now
        return self._cue
