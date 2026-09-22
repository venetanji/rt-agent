"""The face-side hysteresis: confidence gate, refractory hold, expiry and decay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rt_agent.contracts import AffectCue, Decision
from rt_agent.policy import EmotionController, PolicyConfig
from rt_agent.policy.presets import EMOTION_PRESETS

T0 = datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)


def at(seconds: float) -> datetime:
    """A moment on the test timeline."""
    return T0 + timedelta(seconds=seconds)


def proposal(preset: str, confidence: float, intensity: float, when: datetime) -> Decision:
    """A decision that proposes one expression, with everything else held constant."""
    return Decision(
        utterance_id="u1",
        speak=False,
        wait_reason="no_invitation",
        addressee="robot",
        emotion_preset=preset,
        emotion_conf=confidence,
        intensity=intensity,
        affect=AffectCue(
            vector=EMOTION_PRESETS[preset],
            intensity=intensity,
            preset=preset,
            source_confidence=confidence,
            valid_for_ms=1500,
            issued_at=when,
        ),
        policy_version="policy/v1",
        bundle_version="decision-bundle/v1",
        created_at=when,
    )


@pytest.fixture
def controller() -> EmotionController:
    return EmotionController(PolicyConfig(), started_at=T0)


def test_the_face_starts_neutral(controller: EmotionController) -> None:
    assert controller.current().preset == "neutral"
    assert controller.current().intensity == 0.0
    assert controller.last_switch_at is None


def test_the_first_expression_is_not_held_back_by_the_refractory_period(
    controller: EmotionController,
) -> None:
    emitted = controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    assert emitted is not None
    assert emitted.preset == "warm"
    assert controller.last_switch_at == at(0)


def test_a_low_confidence_proposal_never_moves_the_face(controller: EmotionController) -> None:
    assert controller.apply(proposal("happy", 0.49, 1.0, at(0)), at(0)) is None
    assert controller.current().preset == "neutral"


def test_a_switch_inside_the_refractory_window_is_refused(
    controller: EmotionController,
) -> None:
    controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    assert controller.apply(proposal("happy", 0.95, 0.9, at(0.5)), at(0.5)) is None
    assert controller.current().preset == "warm"


def test_the_same_preset_is_a_hold_not_a_switch(controller: EmotionController) -> None:
    controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    emitted = controller.apply(proposal("warm", 0.9, 0.6, at(1.0)), at(1.0))
    assert emitted is not None
    assert emitted.preset == "warm"
    assert emitted.intensity == pytest.approx(0.6)
    assert emitted.issued_at == at(1.0)
    # A hold does not restart the refractory clock.
    assert controller.last_switch_at == at(0)


def test_a_switch_after_the_refractory_period_is_allowed(
    controller: EmotionController,
) -> None:
    controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    emitted = controller.apply(proposal("happy", 0.9, 0.9, at(2.0)), at(2.0))
    assert emitted is not None
    assert emitted.preset == "happy"
    assert controller.last_switch_at == at(2.0)


def test_an_expired_cue_decays_by_half(controller: EmotionController) -> None:
    controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    assert controller.tick(at(1.4)) is None
    decayed = controller.tick(at(1.6))
    assert decayed is not None
    assert decayed.preset == "warm"
    assert decayed.intensity == pytest.approx(0.4)
    assert decayed.issued_at == at(1.5)


def test_several_missed_windows_decay_several_times(controller: EmotionController) -> None:
    controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    decayed = controller.tick(at(4.6))
    assert decayed is not None
    assert decayed.intensity == pytest.approx(0.1)


def test_decay_ends_at_neutral_and_then_stops(controller: EmotionController) -> None:
    controller.apply(proposal("sad", 0.9, 0.8, at(0)), at(0))
    settled = controller.tick(at(60.0))
    assert settled is not None
    assert settled.preset == "neutral"
    assert settled.intensity == 0.0
    assert controller.tick(at(120.0)) is None


def test_the_full_timeline(controller: EmotionController) -> None:
    # t=0 first expression, t=0.5 blocked by refractory, t=1.0 hold, t=1.2 low
    # confidence, t=2.9 decay then switch.
    assert controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0)) is not None
    assert controller.apply(proposal("happy", 0.95, 0.9, at(0.5)), at(0.5)) is None
    assert controller.apply(proposal("warm", 0.9, 0.6, at(1.0)), at(1.0)) is not None
    assert controller.apply(proposal("sad", 0.3, 1.0, at(1.2)), at(1.2)) is None
    assert controller.current().intensity == pytest.approx(0.6)

    switched = controller.apply(proposal("happy", 0.9, 0.9, at(2.9)), at(2.9))
    assert switched is not None
    assert switched.preset == "happy"
    assert switched.intensity == pytest.approx(0.9)
    assert controller.expires_at() == at(2.9) + timedelta(milliseconds=1500)


def test_apply_also_decays_before_it_considers_a_proposal(
    controller: EmotionController,
) -> None:
    controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    # A low-confidence proposal three seconds later: the face should have decayed twice.
    emitted = controller.apply(proposal("happy", 0.2, 0.9, at(3.1)), at(3.1))
    assert emitted is not None
    assert emitted.preset == "warm"
    assert emitted.intensity == pytest.approx(0.2)


def test_the_controller_never_reads_a_clock(controller: EmotionController) -> None:
    # Every observable transition is a function of the injected `now`.
    controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    before = controller.current()
    assert controller.tick(at(0.1)) is None
    assert controller.current() is before


def test_a_zero_length_validity_disables_decay() -> None:
    controller = EmotionController(PolicyConfig(cue_valid_for_ms=0), started_at=T0)
    controller.apply(proposal("warm", 0.9, 0.8, at(0)), at(0))
    assert controller.tick(at(99.0)) is None


def test_an_unknown_initial_preset_falls_back_to_neutral() -> None:
    controller = EmotionController(started_at=T0, initial_preset="smug")
    assert controller.current().preset == "neutral"
