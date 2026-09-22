"""The fixture-driven backend: it must behave like the real one, including when it fails."""

from __future__ import annotations

import json
import time

import pytest

from rt_agent.contracts import BundleAnswers, DecisionContext, SystemOneBackend
from rt_agent.systemone import (
    QUESTION_BUNDLE_V1,
    MockAnswerNotFound,
    MockSystemOne,
    SystemOneTimeoutError,
    render_state,
)
from rt_agent.systemone.mock import load_fixture
from tests.conftest import SYSTEMONE_FIXTURES, make_answers, make_context, make_utterance

SCENARIOS = (
    "daughter_birthday",
    "direct_question_time",
    "distressed_person",
    "medical_appointment",
    "two_humans_weekend",
    "unintelligible_fragment",
)


@pytest.fixture
def backend() -> MockSystemOne:
    return MockSystemOne.from_fixtures(SYSTEMONE_FIXTURES)


def test_all_six_recordings_load_and_revalidate(backend: MockSystemOne) -> None:
    assert backend.scenarios == SCENARIOS


def test_the_mock_satisfies_the_backend_protocol(backend: MockSystemOne) -> None:
    assert isinstance(backend, SystemOneBackend)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_recorded_state_gets_its_recorded_answers(backend: MockSystemOne, scenario: str) -> None:
    recording = backend.recording(scenario)
    answers = backend.ask(recording.state, QUESTION_BUNDLE_V1.to_wire())
    assert answers is recording.answers
    assert answers.latency_ms > 0.0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_recorded_request_still_matches_the_frozen_bundle(scenario: str) -> None:
    document = json.loads((SYSTEMONE_FIXTURES / f"{scenario}.json").read_text(encoding="utf-8"))
    assert document["bundle_version"] == QUESTION_BUNDLE_V1.version
    assert document["questions"] == QUESTION_BUNDLE_V1.to_wire(), (
        "the bundle wording changed; re-run scripts/record_systemone_fixtures.py"
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_recorded_state_still_matches_the_renderer(scenario: str) -> None:
    document = json.loads((SYSTEMONE_FIXTURES / f"{scenario}.json").read_text(encoding="utf-8"))
    ctx = DecisionContext.model_validate(document["context"])
    assert render_state(ctx) == document["state"], (
        "the state renderer changed; re-run scripts/record_systemone_fixtures.py"
    )


def test_a_context_can_be_replayed_end_to_end(backend: MockSystemOne) -> None:
    recording = backend.recording("direct_question_time")
    ctx = DecisionContext.model_validate(recording.raw["context"])
    answers = backend.ask_context(ctx)
    assert answers.choice("addressee").chosen == "robot"


@pytest.mark.asyncio
async def test_the_async_path_serves_the_same_answers(backend: MockSystemOne) -> None:
    recording = backend.recording("two_humans_weekend")
    ctx = DecisionContext.model_validate(recording.raw["context"])
    assert await backend.aask_context(ctx) is recording.answers


def test_an_unrelated_part_of_the_context_still_matches_by_current_utterance(
    backend: MockSystemOne,
) -> None:
    recording = backend.recording("unintelligible_fragment")
    ctx = DecisionContext.model_validate(recording.raw["context"])
    moved = ctx.model_copy(update={"robot_name": "Alice", "recent": ()})
    assert backend.ask_context(moved) is recording.answers


def test_an_unknown_state_fails_closed_rather_than_guessing(backend: MockSystemOne) -> None:
    ctx = make_context(current=make_utterance(text="something never recorded"))
    with pytest.raises(MockAnswerNotFound):
        backend.ask_context(ctx)


def test_a_rule_function_answers_synthetic_cases() -> None:
    def rule(state: str, questions: object) -> BundleAnswers:
        return make_answers(addressed=0.99 if "Alice," in state else 0.01)

    backend = MockSystemOne(rule=rule)
    spoken = backend.ask_context(make_context(current=make_utterance(text="Alice, hello there")))
    ignored = backend.ask_context(make_context(current=make_utterance(text="pass the salt")))
    assert spoken.noul("addressed_to_robot") == 0.99
    assert ignored.noul("addressed_to_robot") == 0.01


def test_fixtures_take_precedence_over_the_rule(backend: MockSystemOne) -> None:
    backend.rule = lambda state, questions: make_answers()
    recording = backend.recording("medical_appointment")
    assert backend.ask(recording.state, QUESTION_BUNDLE_V1.to_wire()) is recording.answers


def test_calls_are_recorded_for_assertions(backend: MockSystemOne) -> None:
    recording = backend.recording("direct_question_time")
    backend.ask(recording.state, QUESTION_BUNDLE_V1.to_wire())
    assert len(backend.calls) == 1
    assert backend.calls[0][0] == recording.state


def test_artificial_latency_is_applied(backend: MockSystemOne) -> None:
    backend.latency_ms = 25.0
    recording = backend.recording("direct_question_time")
    started = time.perf_counter()
    backend.ask(recording.state, QUESTION_BUNDLE_V1.to_wire())
    assert (time.perf_counter() - started) * 1000.0 >= 20.0


def test_failure_injection_can_be_permanent(backend: MockSystemOne) -> None:
    backend.arm_failure(SystemOneTimeoutError("injected"))
    recording = backend.recording("direct_question_time")
    for _ in range(3):
        with pytest.raises(SystemOneTimeoutError):
            backend.ask(recording.state, QUESTION_BUNDLE_V1.to_wire())


def test_failure_injection_can_be_bounded(backend: MockSystemOne) -> None:
    backend.arm_failure(SystemOneTimeoutError("injected"), count=2)
    recording = backend.recording("direct_question_time")
    wire = QUESTION_BUNDLE_V1.to_wire()
    for _ in range(2):
        with pytest.raises(SystemOneTimeoutError):
            backend.ask(recording.state, wire)
    assert backend.ask(recording.state, wire) is recording.answers


def test_loading_a_single_fixture_exposes_the_raw_document() -> None:
    recording = load_fixture(SYSTEMONE_FIXTURES / "distressed_person.json")
    assert recording.scenario == "distressed_person"
    assert recording.raw["model_requested"] == "jev-latest"
    assert recording.raw["request_id"].startswith("req_")
    assert repr(recording).startswith("RecordedCall(")
