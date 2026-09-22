"""Validation bounds on the frozen contracts: every out-of-range value must be refused."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rt_agent.contracts import (
    PROBABILITY_SUM_TOLERANCE,
    AffectCue,
    AudioEvidence,
    ChoiceAnswer,
    Decision,
    DecisionContext,
    MemoryRecord,
    RobotState,
    ScoreAnswer,
    SpeechClauseOut,
)
from tests.conftest import make_context, make_memory, make_utterance

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def cue(**overrides: object) -> AffectCue:
    values: dict[str, object] = {
        "vector": (0.5, 0.2, 0.1),
        "intensity": 0.5,
        "preset": "warm",
        "source_confidence": 0.8,
        "issued_at": NOW,
    }
    values.update(overrides)
    return AffectCue.model_validate(values)


class TestFrozenness:
    def test_models_are_immutable(self) -> None:
        with pytest.raises(ValidationError):
            make_utterance().text = "changed"  # type: ignore[misc]

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            make_utterance(extra_field="nope")

    def test_schema_versions_are_literal(self) -> None:
        assert make_utterance().schema_version == "utterance/v1"
        assert cue().schema_version == "affect-cue/v1"
        assert make_memory().schema_version == "memory/v1"


class TestAffectCue:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("bad", [-1.01, 1.01])
    def test_vector_coordinates_are_bounded(self, axis: int, bad: float) -> None:
        vector = [0.0, 0.0, 0.0]
        vector[axis] = bad
        with pytest.raises(ValidationError):
            cue(vector=tuple(vector))

    @pytest.mark.parametrize("edge", [-1.0, 1.0])
    def test_vector_edges_are_accepted(self, edge: float) -> None:
        assert cue(vector=(edge, edge, edge)).vector == (edge, edge, edge)

    def test_vector_needs_three_axes(self) -> None:
        with pytest.raises(ValidationError):
            cue(vector=(0.1, 0.2))

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_intensity_is_bounded(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            cue(intensity=bad)

    def test_issued_at_must_be_timezone_aware(self) -> None:
        with pytest.raises(ValidationError):
            cue(issued_at=datetime(2026, 9, 22, 12, 0))

    def test_visibility_follows_the_alice_valence_rule(self) -> None:
        assert cue(vector=(0.25, 0.8, -0.2)).visible_on_alice_face
        assert not cue(vector=(0.2, 0.8, -0.2)).visible_on_alice_face


class TestSpeechClause:
    def clause(self, **overrides: object) -> SpeechClauseOut:
        values: dict[str, object] = {
            "generation_id": "g1",
            "clause_id": "g1-0",
            "sequence": 0,
            "text": "Hello there.",
            "vector": (0.5, 0.2, 0.1),
            "intensity": 0.4,
            "seed": 7,
        }
        values.update(overrides)
        return SpeechClauseOut.model_validate(values)

    @pytest.mark.parametrize("sequence", [0, 31])
    def test_sequence_range_is_accepted(self, sequence: int) -> None:
        assert self.clause(sequence=sequence).sequence == sequence

    @pytest.mark.parametrize("sequence", [-1, 32])
    def test_sequence_outside_range_is_refused(self, sequence: int) -> None:
        with pytest.raises(ValidationError):
            self.clause(sequence=sequence)

    @pytest.mark.parametrize("length", [1, 1000])
    def test_text_length_is_accepted(self, length: int) -> None:
        assert len(self.clause(text="x" * length).text) == length

    @pytest.mark.parametrize("length", [0, 1001])
    def test_text_outside_range_is_refused(self, length: int) -> None:
        with pytest.raises(ValidationError):
            self.clause(text="x" * length)

    def test_seed_is_unsigned_32_bit(self) -> None:
        assert self.clause(seed=2**32 - 1).seed == 2**32 - 1
        with pytest.raises(ValidationError):
            self.clause(seed=2**32)


class TestUtterance:
    def test_text_bounds(self) -> None:
        assert len(make_utterance(text="x" * 1000).text) == 1000
        with pytest.raises(ValidationError):
            make_utterance(text="")
        with pytest.raises(ValidationError):
            make_utterance(text="x" * 1001)

    def test_speaker_labels_stay_anonymous(self) -> None:
        assert make_utterance(speaker_label="ROBOT").is_robot
        with pytest.raises(ValidationError):
            make_utterance(speaker_label="Marta")

    def test_interval_must_not_run_backwards(self) -> None:
        with pytest.raises(ValidationError):
            make_utterance(t_start_s=5.0, t_end_s=4.0)

    def test_asr_confidence_is_a_probability_or_unknown(self) -> None:
        assert make_utterance(asr_confidence=None).asr_confidence is None
        with pytest.raises(ValidationError):
            make_utterance(asr_confidence=1.2)


class TestAudioEvidence:
    def test_fractions_are_bounded(self) -> None:
        with pytest.raises(ValidationError):
            AudioEvidence(
                vad_mean_prob=1.2,
                voiced_fraction=0.5,
                duration_s=1.0,
                rms=0.1,
                clipped_fraction=0.0,
            )

    def test_overlap_may_be_unknown(self) -> None:
        evidence = AudioEvidence(
            vad_mean_prob=0.9,
            voiced_fraction=0.5,
            duration_s=1.0,
            rms=0.1,
            clipped_fraction=0.0,
            overlap_fraction=None,
        )
        assert evidence.overlap_fraction is None


class TestDecisionContext:
    def test_window_is_bounded(self) -> None:
        recent = tuple(
            make_utterance(utterance_id=f"u{index}", t_start_s=index, t_end_s=index + 0.5)
            for index in range(7)
        )
        with pytest.raises(ValidationError):
            make_context(recent=recent)
        assert len(make_context(recent=recent[:6]).recent) == 6

    def test_memories_are_bounded(self) -> None:
        with pytest.raises(ValidationError):
            make_context(memories=tuple(make_memory() for _ in range(4)))

    def test_defaults(self) -> None:
        ctx = make_context()
        assert ctx.robot_name == "Alice"
        assert ctx.robot_state == RobotState()


class TestMemoryRecord:
    def test_text_is_capped_at_300_characters(self) -> None:
        assert len(make_memory(text="x" * 300).text) == 300
        with pytest.raises(ValidationError):
            make_memory(text="x" * 301)

    def test_kind_is_closed(self) -> None:
        with pytest.raises(ValidationError):
            MemoryRecord(
                memory_id="m",
                session_id="s",
                speaker_label="S1",
                kind="gossip",  # type: ignore[arg-type]
                text="x",
                worth_p=0.9,
                faithfulness_p=0.9,
                created_at=NOW,
            )


class TestAnswerDistributions:
    def test_two_decimal_rounding_is_tolerated(self) -> None:
        for total in (0.99, 1.0, 1.01):
            answer = ChoiceAnswer(
                chosen="a",
                probabilities={"a": total - 0.2, "b": 0.2},
                confidence=0.6,
            )
            assert answer.chosen == "a"

    def test_a_distribution_that_misses_by_more_than_the_tolerance_is_refused(self) -> None:
        assert pytest.approx(0.011) == PROBABILITY_SUM_TOLERANCE
        with pytest.raises(ValidationError):
            ChoiceAnswer(chosen="a", probabilities={"a": 0.8, "b": 0.15}, confidence=0.6)

    def test_choice_must_agree_with_its_own_argmax(self) -> None:
        with pytest.raises(ValidationError):
            ChoiceAnswer(chosen="b", probabilities={"a": 0.8, "b": 0.2}, confidence=0.6)

    def test_chosen_option_must_exist(self) -> None:
        with pytest.raises(ValidationError):
            ChoiceAnswer(chosen="c", probabilities={"a": 0.8, "b": 0.2}, confidence=0.6)

    def test_score_levels_must_be_contiguous_from_zero(self) -> None:
        with pytest.raises(ValidationError):
            ScoreAnswer(value=1.0, probabilities={"1": 0.5, "2": 0.5}, confidence=0.5)

    def test_score_normalizes_onto_the_unit_interval(self) -> None:
        answer = ScoreAnswer(
            value=2.86,
            probabilities={"0": 0.0, "1": 0.02, "2": 0.14, "3": 0.8, "4": 0.04},
            confidence=0.81,
        )
        assert answer.max_level == 4
        assert answer.normalized == pytest.approx(0.715)

    def test_score_may_not_exceed_the_top_level(self) -> None:
        with pytest.raises(ValidationError):
            ScoreAnswer(value=2.5, probabilities={"0": 0.5, "1": 0.5}, confidence=0.5)


class TestDecisionInvariants:
    def base(self, **overrides: object) -> Decision:
        values: dict[str, object] = {
            "utterance_id": "u1",
            "speak": True,
            "wait_reason": None,
            "addressee": "robot",
            "emotion_preset": "attentive",
            "emotion_conf": 0.9,
            "intensity": 0.5,
            "affect": cue(),
            "remember": False,
            "memory_kind": None,
            "sensitive": False,
            "policy_version": "policy/v1",
            "bundle_version": "decision-bundle/v1",
            "created_at": NOW,
        }
        values.update(overrides)
        return Decision.model_validate(values)

    def test_speaking_decisions_carry_no_wait_reason(self) -> None:
        with pytest.raises(ValidationError):
            self.base(speak=True, wait_reason="not_addressed")

    def test_waiting_decisions_must_name_a_reason(self) -> None:
        with pytest.raises(ValidationError):
            self.base(speak=False, wait_reason=None)

    def test_remembering_requires_a_kind(self) -> None:
        with pytest.raises(ValidationError):
            self.base(remember=True, memory_kind=None)

    def test_a_withheld_memory_keeps_its_kind(self) -> None:
        decision = self.base(remember=False, memory_kind="health", sensitive=True)
        assert decision.memory_kind == "health"


class TestDecisionContextIsNotATranscriptLog:
    def test_context_only_carries_the_window_and_three_memories(self) -> None:
        ctx = make_context(
            recent=(make_utterance(utterance_id="a", t_start_s=1, t_end_s=2),),
            memories=(make_memory(), make_memory(), make_memory()),
        )
        assert isinstance(ctx, DecisionContext)
        assert len(ctx.retrieved_memories) == 3
