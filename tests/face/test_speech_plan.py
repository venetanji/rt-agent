"""speech-plan/v1 export, validated against alice's own JSON Schema."""

from __future__ import annotations

import json

import jsonschema
import pytest

from rt_agent.face.clauses import build_speech_clauses, split_clauses
from rt_agent.face.speech_plan import (
    MAX_SEGMENTS,
    VOICES,
    export_speech_plan,
    validate_speech_plan,
    write_speech_plan,
)
from tests.face.conftest import ALICE_FIXTURES, make_cue


def test_alices_own_plan_validates_positive_control(schema):
    plan = json.loads((ALICE_FIXTURES / "alice-introduction.json").read_text())
    jsonschema.validate(plan, schema)
    validate_speech_plan(plan)


def test_exported_plan_validates_against_the_alice_schema(schema):
    plan = export_speech_plan(
        "utt-0042",
        "rt-agent/poc",
        ["Ten past nine.", "You have about twenty minutes."],
        make_cue("warm", 0.6),
        seed=29,
    )
    jsonschema.validate(plan, schema)

    assert plan["schema_version"] == "speech-plan/v1"
    assert plan["affect_schema_id"] == "affect-vector/v1"
    assert plan["utterance_id"] == "utt-0042"
    assert plan["source_id"] == "rt-agent/poc"
    assert plan["voice"] == "azelma"
    assert plan["seed"] == 29
    assert [segment["text"] for segment in plan["segments"]] == [
        "Ten past nine.",
        "You have about twenty minutes.",
    ]


def test_speech_clauses_can_be_exported_directly(schema):
    clauses = build_speech_clauses(
        "utt-0042", split_clauses("Hello there. How are you?"), make_cue(), 7
    )
    plan = export_speech_plan("utt-0042", "rt-agent/poc", clauses, make_cue())
    jsonschema.validate(plan, schema)
    assert [segment["text"] for segment in plan["segments"]] == [
        "Hello there.",
        "How are you?",
    ]


def test_first_cue_is_at_progress_zero_and_progress_strictly_increases(schema):
    cues = [make_cue("warm", 0.4), make_cue("happy", 0.9)]
    plan = export_speech_plan("u", "s", ["First.", "Second."], cues)
    jsonschema.validate(plan, schema)

    for segment in plan["segments"]:
        progress = [cue["progress"] for cue in segment["cues"]]
        assert progress[0] == 0.0
        assert progress == sorted(set(progress))

    # blend_to_next: segment 1 ends where segment 2 begins, as alice-introduction does.
    assert plan["segments"][0]["cues"][1]["vector"] == plan["segments"][1]["cues"][0]["vector"]
    # The last segment holds its own affect to the end.
    assert plan["segments"][1]["cues"][0]["vector"] == plan["segments"][1]["cues"][1]["vector"]


def test_blend_to_next_off_gives_one_onset_cue_per_segment(schema):
    plan = export_speech_plan("u", "s", ["A.", "B."], make_cue(), blend_to_next=False)
    jsonschema.validate(plan, schema)
    assert all(len(segment["cues"]) == 1 for segment in plan["segments"])
    assert all(segment["cues"][0]["progress"] == 0.0 for segment in plan["segments"])


def test_pause_after_s_is_per_segment_or_shared(schema):
    plan = export_speech_plan("u", "s", ["A.", "B."], make_cue(), pause_after_s=[0.3, 0.0])
    jsonschema.validate(plan, schema)
    assert [segment["pause_after_s"] for segment in plan["segments"]] == [0.3, 0.0]

    shared = export_speech_plan("u", "s", ["A.", "B."], make_cue(), pause_after_s=0.5)
    assert [segment["pause_after_s"] for segment in shared["segments"]] == [0.5, 0.5]


@pytest.mark.parametrize("voice", VOICES)
def test_every_alice_voice_is_accepted(voice, schema):
    jsonschema.validate(export_speech_plan("u", "s", ["A."], make_cue(), voice), schema)


def test_an_unknown_voice_is_refused_before_it_reaches_the_robot():
    with pytest.raises(ValueError, match="voice must be one of"):
        export_speech_plan("u", "s", ["A."], make_cue(), "gruff")


def test_bounds_are_enforced_locally():
    cue = make_cue()
    with pytest.raises(ValueError, match="at least one segment"):
        export_speech_plan("u", "s", [], cue)
    with pytest.raises(ValueError, match="1-32 segments"):
        export_speech_plan("u", "s", [f"S{i}." for i in range(MAX_SEGMENTS + 1)], cue)
    with pytest.raises(ValueError, match="4000"):
        export_speech_plan("u", "s", ["x" * 900] * 6, cue)
    with pytest.raises(ValueError, match="must not be empty"):
        export_speech_plan("u", "s", ["   "], cue)
    with pytest.raises(ValueError, match="pause_after_s"):
        export_speech_plan("u", "s", ["A."], cue, pause_after_s=9.0)
    with pytest.raises(ValueError, match="seed"):
        export_speech_plan("u", "s", ["A."], cue, seed=2**32)
    with pytest.raises(ValueError, match="cues for 2 segments"):
        export_speech_plan("u", "s", ["A.", "B."], [cue])


def test_validate_rejects_a_hand_edited_plan_the_schema_would_miss():
    plan = export_speech_plan("u", "s", ["A.", "B."], make_cue())
    plan["segments"][0]["cues"][0]["progress"] = 0.5
    with pytest.raises(ValueError, match="progress 0"):
        validate_speech_plan(plan)

    plan = export_speech_plan("u", "s", ["A."], make_cue())
    plan["segments"][0]["cues"].append(dict(plan["segments"][0]["cues"][-1]))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_speech_plan(plan)


def test_a_too_long_plan_is_caught_even_though_the_schema_allows_it(schema):
    plan = export_speech_plan("u", "s", ["x" * 500] * 8, make_cue())
    jsonschema.validate(plan, schema)  # the schema has no total-length rule
    plan["segments"].append(
        {"text": "y" * 500, "cues": [{"progress": 0.0, "vector": [0, 0, 0], "intensity": 0.0}]}
    )
    with pytest.raises(ValueError, match="4000"):
        validate_speech_plan(plan)


def test_write_speech_plan_round_trips_and_refuses_to_clobber(tmp_path, schema):
    plan = export_speech_plan("utt-0042", "rt-agent/poc", ["Hello."], make_cue())
    path = write_speech_plan(tmp_path / "plans" / "utt-0042.json", plan)

    assert path.exists()
    written = json.loads(path.read_text())
    assert written == plan
    jsonschema.validate(written, schema)
    assert path.read_text().endswith("\n")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_speech_plan(path, plan)
    assert write_speech_plan(path, plan, overwrite=True) == path


def test_write_refuses_an_invalid_plan(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        write_speech_plan(tmp_path / "bad.json", {"schema_version": "speech-plan/v2"})
    assert not (tmp_path / "bad.json").exists()
