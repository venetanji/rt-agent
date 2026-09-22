"""The simulator must reproduce alice's authored anchor policy exactly, threshold included."""

from __future__ import annotations

import pytest

from rt_agent.face.authored_face import (
    ANCHORS,
    BENCH_PROFILE,
    VISIBLE_PROFILE,
    AuthoredFaceModel,
)
from rt_agent.policy.presets import EMOTION_PRESETS
from tests.face.conftest import make_cue


def test_profiles_match_the_alice_config_files():
    assert (BENCH_PROFILE.anchor_scale, BENCH_PROFILE.transition_s) == (0.5, 1.6)
    assert (VISIBLE_PROFILE.anchor_scale, VISIBLE_PROFILE.transition_s) == (1.0, 0.8)
    assert BENCH_PROFILE.valence_threshold == VISIBLE_PROFILE.valence_threshold == 0.2
    assert BENCH_PROFILE.positive_anchor == "smile_open"
    assert BENCH_PROFILE.negative_anchor == "frown_closed"


def test_anchor_table_matches_procedural_motion_v1():
    assert ANCHORS["smile_open"] == {
        "mouth_open": 1.0,
        "left_mouth_corner": 1.0,
        "right_mouth_corner": -1.0,
    }
    assert ANCHORS["frown_closed"] == {
        "mouth_open": -1.0,
        "left_mouth_corner": -1.0,
        "right_mouth_corner": 1.0,
    }
    assert set(ANCHORS["neutral"].values()) == {0.0}


@pytest.mark.parametrize(
    ("valence", "anchor"),
    [
        (0.21, "smile_open"),
        (0.2, "neutral"),
        (0.0, "neutral"),
        (-0.2, "neutral"),
        (-0.21, "frown_closed"),
        (1.0, "smile_open"),
        (-1.0, "frown_closed"),
    ],
)
def test_valence_alone_selects_the_anchor(valence, anchor):
    model = AuthoredFaceModel("bench")
    cue = make_cue(vector=(valence, 0.0, 0.0))
    assert model.anchor_for(cue) == anchor


def test_the_threshold_is_strict_which_is_why_surprised_is_0_25():
    """The authored preset table uses 0.25, not 0.2, and this is the reason."""
    model = AuthoredFaceModel("bench")
    surprised = EMOTION_PRESETS["surprised"]
    assert surprised[0] == 0.25

    visible = make_cue("surprised", 0.8)
    assert model.visibility(visible) is True
    assert model.target_for(visible).anchor == "smile_open"

    at_threshold = make_cue("surprised", 0.8, vector=(0.2, surprised[1], surprised[2]))
    assert model.visibility(at_threshold) is False
    assert model.target_for(at_threshold).anchor == "neutral"


def test_arousal_and_dominance_do_not_change_the_face():
    model = AuthoredFaceModel("bench")
    calm = model.target_for(make_cue(vector=(0.5, -0.9, -0.9), intensity=0.6))
    excited = model.target_for(make_cue(vector=(0.5, 0.9, 0.9), intensity=0.6))
    assert calm.channels == excited.channels


def test_amplitude_is_intensity_times_anchor_scale():
    cue = make_cue("happy", 0.8)
    assert AuthoredFaceModel("bench").target_for(cue).amplitude == pytest.approx(0.4)
    assert AuthoredFaceModel("visible").target_for(cue).amplitude == pytest.approx(0.8)


def test_channels_are_the_anchor_scaled_by_amplitude():
    target = AuthoredFaceModel("visible").target_for(make_cue("sad", 1.0))
    assert target.anchor == "frown_closed"
    assert target.channels == {
        "mouth_open": -1.0,
        "left_mouth_corner": -1.0,
        "right_mouth_corner": 1.0,
    }


def test_neutral_target_is_home_whatever_the_intensity():
    target = AuthoredFaceModel("visible").target_for(make_cue("attentive", 1.0))
    assert target.anchor == "neutral"
    assert set(target.channels.values()) == {0.0}
    assert target.visible is False


def test_advance_interpolates_over_transition_s_and_settles():
    model = AuthoredFaceModel("visible")  # transition_s == 0.8
    model.apply(make_cue("happy", 1.0), now=0.0)

    half = model.advance(0.4)
    assert half.progress == pytest.approx(0.5)
    assert half.settled is False
    assert half.channels["mouth_open"] == pytest.approx(0.5)
    assert half.channels["right_mouth_corner"] == pytest.approx(-0.5)

    done = model.advance(0.8)
    assert done.settled is True
    assert done.channels["mouth_open"] == pytest.approx(1.0)

    assert model.advance(5.0).channels["mouth_open"] == pytest.approx(1.0)


def test_a_new_cue_starts_from_where_the_face_actually_is():
    model = AuthoredFaceModel("visible")
    model.apply(make_cue("happy", 1.0), now=0.0)
    model.advance(0.4)  # half way to a full smile

    model.apply(make_cue("sad", 1.0), now=0.4)
    assert model.advance(0.4).channels["mouth_open"] == pytest.approx(0.5)
    midway = model.advance(0.8)
    assert midway.anchor == "frown_closed"
    assert midway.channels["mouth_open"] == pytest.approx(-0.25)


def test_bench_profile_is_slower_than_visible_for_the_same_cue():
    cue = make_cue("happy", 1.0)
    bench, visible = AuthoredFaceModel("bench"), AuthoredFaceModel("visible")
    bench.apply(cue, now=0.0)
    visible.apply(cue, now=0.0)
    assert bench.advance(0.8).progress == pytest.approx(0.5)
    assert visible.advance(0.8).progress == pytest.approx(1.0)


def test_reset_returns_the_face_to_home():
    model = AuthoredFaceModel("visible")
    model.apply(make_cue("happy", 1.0), now=0.0)
    model.advance(1.0)
    model.reset(now=2.0)
    state = model.advance(2.0)
    assert state.anchor == "neutral"
    assert set(state.channels.values()) == {0.0}


def test_time_must_not_go_backwards():
    model = AuthoredFaceModel()
    model.advance(1.0)
    with pytest.raises(ValueError, match="backwards"):
        model.advance(0.5)


def test_unknown_profile_names_are_refused():
    with pytest.raises(ValueError, match="unknown face profile"):
        AuthoredFaceModel("theatrical")
