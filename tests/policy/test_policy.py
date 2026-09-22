"""Policy v1 truth table. Everything that is not a confident yes must end in silence."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from rt_agent.contracts import DecisionContext, RobotState
from rt_agent.policy import Policy, PolicyConfig
from rt_agent.systemone import MockSystemOne
from tests.conftest import SYSTEMONE_FIXTURES, make_answers, make_context, make_utterance

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


@pytest.fixture
def policy() -> Policy:
    return Policy(PolicyConfig())


class TestAdmissionTruthTable:
    def test_all_gates_open_means_speak(self, policy: Policy) -> None:
        decision = policy.decide(make_context(), make_answers(), 250.0, now=NOW)
        assert decision.speak is True
        assert decision.wait_reason is None

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({"intelligible": 0.59}, "not_intelligible"),
            ({"addressed": 0.69}, "not_addressed"),
            ({"invites": 0.59}, "no_invitation"),
            ({"stay_quiet": 0.5}, "safety_quiet"),
        ],
    )
    def test_each_gate_closes_on_its_own(
        self, policy: Policy, overrides: dict[str, float], reason: str
    ) -> None:
        decision = policy.decide(make_context(), make_answers(**overrides), 250.0, now=NOW)
        assert decision.speak is False
        assert decision.wait_reason == reason

    @pytest.mark.parametrize(
        ("overrides"),
        [
            {"intelligible": 0.6},
            {"addressed": 0.7},
            {"invites": 0.6},
            {"stay_quiet": 0.49},
        ],
    )
    def test_the_threshold_values_themselves_still_speak(
        self, policy: Policy, overrides: dict[str, float]
    ) -> None:
        assert policy.decide(make_context(), make_answers(**overrides), 250.0, now=NOW).speak

    def test_the_robot_never_talks_over_itself(self, policy: Policy) -> None:
        ctx = make_context(robot_state=RobotState(speaking=True))
        decision = policy.decide(ctx, make_answers(), 250.0, now=NOW)
        assert decision.speak is False
        assert decision.wait_reason == "robot_speaking"

    def test_reasons_are_reported_in_a_fixed_precedence(self, policy: Policy) -> None:
        # Everything fails at once: the most serious reason wins, deterministically.
        ctx = make_context(robot_state=RobotState(speaking=True))
        answers = make_answers(intelligible=0.0, addressed=0.0, invites=0.0, stay_quiet=1.0)
        assert policy.decide(ctx, answers, 250.0, now=NOW).wait_reason == "robot_speaking"

        idle = make_context()
        assert policy.decide(idle, answers, 250.0, now=NOW).wait_reason == "safety_quiet"
        assert (
            policy.decide(
                idle, make_answers(intelligible=0.0, addressed=0.0), 250.0, now=NOW
            ).wait_reason
            == "not_intelligible"
        )
        assert (
            policy.decide(
                idle, make_answers(addressed=0.0, invites=0.0), 250.0, now=NOW
            ).wait_reason
            == "not_addressed"
        )


class TestFailClosed:
    def test_no_answers_means_silence(self, policy: Policy) -> None:
        decision = policy.decide(make_context(), None, 1500.0, now=NOW)
        assert decision.speak is False
        assert decision.wait_reason == "backend_unavailable"
        assert decision.answers is None
        assert decision.remember is False

    def test_a_blown_deadline_discards_a_perfectly_good_answer(self, policy: Policy) -> None:
        answers = make_answers()
        decision = policy.decide(make_context(), answers, 1500.1, now=NOW)
        assert decision.speak is False
        assert decision.wait_reason == "deadline_exceeded"
        # The answers are still attached for the log.
        assert decision.answers is answers

    def test_a_deadline_met_exactly_is_still_accepted(self, policy: Policy) -> None:
        assert policy.decide(make_context(), make_answers(), 1500.0, now=NOW).speak is True

    def test_a_missing_question_is_malformed_not_a_default(self, policy: Policy) -> None:
        answers = make_answers()
        stripped = answers.model_copy(
            update={"nouls": {k: v for k, v in answers.nouls.items() if k != "addressed_to_robot"}}
        )
        decision = policy.decide(make_context(), stripped, 250.0, now=NOW)
        assert decision.wait_reason == "malformed_answers"

    def test_an_unknown_emotion_preset_is_malformed(self, policy: Policy) -> None:
        decision = policy.decide(make_context(), make_answers(emotion="smug"), 250.0, now=NOW)
        assert decision.wait_reason == "malformed_answers"

    def test_an_unknown_memory_kind_is_malformed(self, policy: Policy) -> None:
        decision = policy.decide(make_context(), make_answers(memory_kind="gossip"), 250.0, now=NOW)
        assert decision.wait_reason == "malformed_answers"

    def test_a_fail_closed_decision_leaves_the_face_alone(self, policy: Policy) -> None:
        ctx = make_context(robot_state=RobotState(current_emotion="warm"))
        decision = policy.decide(ctx, None, 0.0, now=NOW)
        assert decision.emotion_preset == "warm"
        assert decision.emotion_conf == 0.0
        assert decision.intensity == 0.0
        # Below the controller's confidence gate, so the face does not move.
        assert decision.affect.source_confidence == 0.0

    def test_an_unknown_held_emotion_falls_back_to_neutral(self, policy: Policy) -> None:
        ctx = make_context(robot_state=RobotState(current_emotion="smug"))
        assert policy.decide(ctx, None, 0.0, now=NOW).emotion_preset == "neutral"


class TestEmotion:
    def test_the_chosen_preset_and_its_vector_are_carried(self, policy: Policy) -> None:
        decision = policy.decide(
            make_context(), make_answers(emotion="happy", emotion_conf=0.8), 250.0, now=NOW
        )
        assert decision.emotion_preset == "happy"
        assert decision.affect.vector == (0.65, 0.4, 0.1)
        assert decision.affect.preset == "happy"
        assert decision.affect.source_id == "rt-agent"
        assert decision.affect.issued_at == NOW

    def test_intensity_is_the_rubric_position_over_four(self, policy: Policy) -> None:
        for level in range(5):
            decision = policy.decide(
                make_context(), make_answers(intensity_level=float(level)), 250.0, now=NOW
            )
            assert decision.intensity == pytest.approx(level / 4)

    def test_the_safety_hold_overrides_the_chosen_expression(self, policy: Policy) -> None:
        decision = policy.decide(
            make_context(),
            make_answers(stay_quiet=0.92, emotion="amused", emotion_conf=0.6, intensity_level=0),
            250.0,
            now=NOW,
        )
        assert decision.speak is False
        assert decision.wait_reason == "safety_quiet"
        assert decision.emotion_preset == "concerned"
        assert decision.emotion_conf == pytest.approx(0.92)
        assert decision.intensity >= 0.5

    def test_a_low_confidence_emotion_is_still_reported(self, policy: Policy) -> None:
        # The policy reports; the EmotionController is what gates the face.
        decision = policy.decide(
            make_context(), make_answers(emotion="sad", emotion_conf=0.1), 250.0, now=NOW
        )
        assert decision.emotion_preset == "sad"
        assert decision.emotion_conf == pytest.approx(0.1)


class TestMemory:
    def test_an_ordinary_durable_fact_is_remembered(self, policy: Policy) -> None:
        decision = policy.decide(
            make_context(),
            make_answers(worth=0.75, sensitive=0.1, memory_kind="event"),
            250.0,
            now=NOW,
        )
        assert decision.remember is True
        assert decision.memory_kind == "event"
        assert decision.sensitive is False

    def test_below_the_threshold_nothing_is_stored(self, policy: Policy) -> None:
        decision = policy.decide(make_context(), make_answers(worth=0.74), 250.0, now=NOW)
        assert decision.remember is False
        assert decision.memory_kind is None

    def test_a_sensitive_fact_is_withheld_but_still_classified(self, policy: Policy) -> None:
        decision = policy.decide(
            make_context(),
            make_answers(worth=0.9, sensitive=0.5, memory_kind="health"),
            250.0,
            now=NOW,
        )
        assert decision.sensitive is True
        assert decision.remember is False
        assert decision.memory_kind == "health"

    def test_sensitive_facts_are_stored_only_when_configured(self) -> None:
        policy = Policy(PolicyConfig(allow_sensitive=True))
        decision = policy.decide(
            make_context(),
            make_answers(worth=0.9, sensitive=0.9, memory_kind="health"),
            250.0,
            now=NOW,
        )
        assert decision.remember is True
        assert decision.sensitive is True

    def test_the_faithfulness_gate_is_a_separate_decision(self, policy: Policy) -> None:
        assert policy.memory_is_faithful(0.7) is True
        assert policy.memory_is_faithful(0.69) is False


class TestProvenance:
    def test_every_decision_names_its_versions_and_utterance(self, policy: Policy) -> None:
        ctx = make_context(current=make_utterance(utterance_id="u-42"))
        decision = policy.decide(ctx, make_answers(), 250.0, now=NOW)
        assert decision.utterance_id == "u-42"
        assert decision.policy_version == "policy/v1"
        assert decision.bundle_version == "decision-bundle/v1"
        assert decision.schema_version == "decision/v1"
        assert decision.created_at == NOW

    def test_a_decision_serializes_to_one_jsonl_line(self, policy: Policy) -> None:
        decision = policy.decide(make_context(), make_answers(), 250.0, now=NOW)
        line = json.dumps(decision.model_dump(mode="json"))
        assert "\n" not in line
        assert json.loads(line)["utterance_id"] == "u1"

    def test_thresholds_come_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RT_AGENT_ADDRESSED_MIN", "0.95")
        config = PolicyConfig()
        assert config.addressed_min == pytest.approx(0.95)
        decision = Policy(config).decide(
            make_context(), make_answers(addressed=0.9), 250.0, now=NOW
        )
        assert decision.wait_reason == "not_addressed"

    def test_thresholds_are_frozen(self) -> None:
        with pytest.raises(Exception, match=r"frozen"):
            PolicyConfig().deadline_ms = 10  # type: ignore[misc]


class TestRecordedScenarios:
    """The six recorded live calls, replayed through the policy exactly as the harness will."""

    EXPECTED = {
        "two_humans_weekend": (False, "not_addressed", "attentive", False),
        "direct_question_time": (True, None, "attentive", False),
        "daughter_birthday": (False, "no_invitation", "happy", True),
        "medical_appointment": (True, None, "attentive", False),
        "distressed_person": (False, "safety_quiet", "concerned", False),
        "unintelligible_fragment": (False, "not_intelligible", "attentive", False),
    }

    @pytest.mark.parametrize("scenario", sorted(EXPECTED))
    def test_recorded_scenario(self, policy: Policy, scenario: str) -> None:
        backend = MockSystemOne.from_fixtures(SYSTEMONE_FIXTURES)
        recording = backend.recording(scenario)
        ctx = DecisionContext.model_validate(recording.raw["context"])
        answers = backend.ask_context(ctx)
        decision = policy.decide(ctx, answers, answers.latency_ms, now=NOW)
        speak, reason, preset, remember = self.EXPECTED[scenario]
        assert (decision.speak, decision.wait_reason) == (speak, reason)
        assert decision.emotion_preset == preset
        assert decision.remember is remember

    def test_the_medical_scenario_is_classified_but_withheld(self, policy: Policy) -> None:
        backend = MockSystemOne.from_fixtures(SYSTEMONE_FIXTURES)
        recording = backend.recording("medical_appointment")
        ctx = DecisionContext.model_validate(recording.raw["context"])
        decision = policy.decide(ctx, recording.answers, 250.0, now=NOW)
        assert decision.sensitive is True
        assert decision.remember is False
        assert decision.memory_kind == "health"

    def test_every_recorded_call_fits_the_deadline(self) -> None:
        backend = MockSystemOne.from_fixtures(SYSTEMONE_FIXTURES)
        for scenario in backend.scenarios:
            assert backend.recording(scenario).answers.latency_ms < 1500.0
