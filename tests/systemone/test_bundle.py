"""The frozen bundle: shape, escape hatches, and agreement with the preset table."""

from __future__ import annotations

import pytest

from rt_agent.contracts import MEMORY_KINDS
from rt_agent.policy import EMOTION_PRESETS
from rt_agent.systemone import (
    BUNDLE_VERSION,
    MEMORY_FAITHFUL_QUESTION,
    QUESTION_BUNDLE_V1,
    ChoiceQuestion,
    NoulQuestion,
    QuestionBundle,
    ScoreQuestion,
)

EXPECTED_IDS = (
    "intelligible_complete",
    "addressed_to_robot",
    "invites_response_now",
    "robot_should_stay_quiet_safety",
    "addressee",
    "emotion",
    "emotion_intensity",
    "worth_remembering",
    "sensitive_personal",
    "memory_kind",
)


def test_the_bundle_is_the_frozen_question_set() -> None:
    assert QUESTION_BUNDLE_V1.version == BUNDLE_VERSION == "decision-bundle/v1"
    assert QUESTION_BUNDLE_V1.ids == EXPECTED_IDS


def test_wire_shape_matches_the_api_contract() -> None:
    wire = QUESTION_BUNDLE_V1.to_wire()
    assert list(wire) == list(EXPECTED_IDS)
    for question_id, spec in wire.items():
        assert spec["type"] in {"noul", "choice", "score"}
        assert isinstance(spec["instructions"], str) and spec["instructions"]
        if spec["type"] == "noul":
            assert set(spec["criteria"]) == {"true", "false"}
        elif spec["type"] == "choice":
            assert len(spec["criteria"]) >= 2, question_id
            assert all(isinstance(value, str) and value for value in spec["criteria"].values())
        else:
            assert isinstance(spec["criteria"], list)
            assert 2 <= len(spec["criteria"]) <= 10


def test_every_choice_has_an_escape_hatch() -> None:
    addressee = QUESTION_BUNDLE_V1.get("addressee")
    memory_kind = QUESTION_BUNDLE_V1.get("memory_kind")
    assert isinstance(addressee, ChoiceQuestion)
    assert isinstance(memory_kind, ChoiceQuestion)
    assert "unclear" in addressee.option_names
    assert "other" in memory_kind.option_names


def test_emotion_options_are_exactly_the_preset_table() -> None:
    emotion = QUESTION_BUNDLE_V1.get("emotion")
    assert isinstance(emotion, ChoiceQuestion)
    assert emotion.option_names == tuple(EMOTION_PRESETS)


def test_memory_kind_options_are_exactly_the_contract_literal() -> None:
    memory_kind = QUESTION_BUNDLE_V1.get("memory_kind")
    assert isinstance(memory_kind, ChoiceQuestion)
    assert memory_kind.option_names == MEMORY_KINDS


def test_intensity_rubric_is_a_five_level_scale() -> None:
    intensity = QUESTION_BUNDLE_V1.get("emotion_intensity")
    assert isinstance(intensity, ScoreQuestion)
    assert intensity.max_level == 4
    assert len({level for level in intensity.levels}) == 5


def test_every_question_stands_alone_without_its_id() -> None:
    # Question ids are never sent, so no criterion may be a bare label.
    for question in QUESTION_BUNDLE_V1.questions:
        if isinstance(question, NoulQuestion):
            texts = [question.when_true, question.when_false]
        elif isinstance(question, ChoiceQuestion):
            texts = [description for _, description in question.options]
        else:
            texts = list(question.levels)
        for text in texts:
            assert len(text.split()) >= 3, (question.id, text)


def test_the_memory_faithfulness_question_is_separate() -> None:
    assert MEMORY_FAITHFUL_QUESTION.id == "memory_faithful"
    assert MEMORY_FAITHFUL_QUESTION.id not in QUESTION_BUNDLE_V1.ids
    assert MEMORY_FAITHFUL_QUESTION.to_wire()["type"] == "noul"


def test_bundles_reject_duplicate_ids() -> None:
    question = NoulQuestion(id="a", instructions="i", when_true="t", when_false="f")
    with pytest.raises(ValueError, match="unique"):
        QuestionBundle(version="v", questions=(question, question))


def test_choices_need_at_least_two_options() -> None:
    with pytest.raises(ValueError, match="two options"):
        ChoiceQuestion(id="a", instructions="i", options=(("only", "one"),))
