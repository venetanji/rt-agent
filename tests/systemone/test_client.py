"""Response parsing is the fail-closed boundary: every malformed shape must raise."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest

from rt_agent.systemone import (
    QUESTION_BUNDLE_V1,
    SystemOneAuthError,
    SystemOneProtocolError,
    SystemOneRateLimitError,
    SystemOneTimeoutError,
    SystemOneUnavailableError,
    SystemOneUsageError,
    parse_response,
)
from rt_agent.systemone.client import JEV_BASE_URL, KEV_BASE_URL, SystemOneClient
from tests.conftest import SYSTEMONE_FIXTURES

QUESTIONS = QUESTION_BUNDLE_V1.to_wire()


def fixture_payload(scenario: str = "direct_question_time") -> dict[str, Any]:
    document = json.loads((SYSTEMONE_FIXTURES / f"{scenario}.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = document["response"]
    return payload


def parse(payload: dict[str, Any], **kwargs: Any) -> Any:
    return parse_response(payload, QUESTIONS, latency_ms=250.0, **kwargs)


class TestParsingRecordedResponses:
    @pytest.mark.parametrize(
        "scenario",
        [
            "two_humans_weekend",
            "direct_question_time",
            "daughter_birthday",
            "medical_appointment",
            "distressed_person",
            "unintelligible_fragment",
        ],
    )
    def test_every_fixture_parses(self, scenario: str) -> None:
        answers = parse(fixture_payload(scenario))
        assert answers.answered == set(QUESTIONS)
        assert answers.model == "jev-1.13.0"
        assert answers.usage is not None and answers.usage.input_tokens > 0

    def test_typed_accessors(self) -> None:
        answers = parse(fixture_payload())
        assert 0.0 <= answers.noul("addressed_to_robot") <= 1.0
        assert answers.choice("addressee").chosen == "robot"
        assert 0.0 <= answers.score("emotion_intensity").normalized <= 1.0
        assert answers.noul_or("not_asked", 0.0) == 0.0

    def test_the_resolved_model_version_is_kept_not_the_alias(self) -> None:
        assert parse(fixture_payload()).model == "jev-1.13.0"


class TestFailClosedParsing:
    def test_a_missing_answer_is_refused(self) -> None:
        payload = fixture_payload()
        del payload["answers"]["addressed_to_robot"]
        with pytest.raises(SystemOneProtocolError, match="missing answers"):
            parse(payload)

    def test_an_answer_we_did_not_ask_for_is_refused(self) -> None:
        payload = fixture_payload()
        payload["answers"]["smuggled"] = {"type": "noul", "noul": 1.0}
        with pytest.raises(SystemOneProtocolError, match="did not ask"):
            parse(payload)

    def test_a_type_mismatch_is_refused(self) -> None:
        payload = fixture_payload()
        payload["answers"]["addressed_to_robot"] = {"type": "choice", "choice": "yes"}
        with pytest.raises(SystemOneProtocolError, match="has type"):
            parse(payload)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), "0.5", None, True])
    def test_a_noul_outside_the_unit_interval_is_refused(self, bad: object) -> None:
        payload = fixture_payload()
        payload["answers"]["addressed_to_robot"]["noul"] = bad
        with pytest.raises(SystemOneProtocolError):
            parse(payload)

    def test_a_distribution_over_the_wrong_options_is_refused(self) -> None:
        payload = fixture_payload()
        probabilities = payload["answers"]["addressee"]["probabilities"]
        probabilities["martian"] = probabilities.pop("unclear")
        with pytest.raises(SystemOneProtocolError, match="returned options"):
            parse(payload)

    def test_a_distribution_that_does_not_sum_to_one_is_refused(self) -> None:
        payload = fixture_payload()
        payload["answers"]["addressee"]["probabilities"]["robot"] = 0.5
        with pytest.raises(SystemOneProtocolError, match="sum"):
            parse(payload)

    def test_two_decimal_rounding_is_still_accepted(self) -> None:
        payload = fixture_payload("two_humans_weekend")
        probabilities = payload["answers"]["addressee"]["probabilities"]
        total = round(sum(probabilities.values()), 2)
        assert abs(total - 1.0) <= 0.011
        assert parse(payload) is not None

    def test_a_winner_that_is_not_the_argmax_is_refused(self) -> None:
        payload = fixture_payload()
        payload["answers"]["addressee"]["choice"] = "unclear"
        with pytest.raises(SystemOneProtocolError, match="argmax"):
            parse(payload)

    def test_a_score_with_the_wrong_number_of_levels_is_refused(self) -> None:
        payload = fixture_payload()
        payload["answers"]["emotion_intensity"]["probabilities"].pop("4")
        with pytest.raises(SystemOneProtocolError, match="levels"):
            parse(payload)

    def test_a_missing_model_id_is_refused(self) -> None:
        payload = fixture_payload()
        del payload["model"]
        with pytest.raises(SystemOneProtocolError, match="model id"):
            parse(payload)

    def test_an_unexpected_model_echo_is_refused_when_pinned(self) -> None:
        payload = fixture_payload()
        with pytest.raises(SystemOneProtocolError, match="does not match"):
            parse(payload, expected_model="kev-latest")

    def test_a_non_object_body_is_refused(self) -> None:
        with pytest.raises(SystemOneProtocolError, match="JSON object"):
            parse_response(["nope"], QUESTIONS, latency_ms=1.0)

    def test_the_original_payload_is_not_mutated_by_parsing(self) -> None:
        payload = fixture_payload()
        before = copy.deepcopy(payload)
        parse(payload)
        assert payload == before


def client_with(handler: Any, **kwargs: Any) -> SystemOneClient:
    client = SystemOneClient.jev(**kwargs)
    client._sync = httpx.Client(
        base_url=JEV_BASE_URL,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer test"},
    )
    return client


class TestHttpErrorMapping:
    @pytest.mark.parametrize(
        ("status", "error"),
        [
            (401, SystemOneAuthError),
            (403, SystemOneAuthError),
            (400, SystemOneUsageError),
            (422, SystemOneUsageError),
            (429, SystemOneRateLimitError),
            (529, SystemOneRateLimitError),
            (500, SystemOneUnavailableError),
            (503, SystemOneUnavailableError),
        ],
    )
    def test_status_codes_become_typed_errors(self, status: int, error: type[Exception]) -> None:
        client = client_with(lambda request: httpx.Response(status, json={"detail": "no"}))
        with pytest.raises(error):
            client.ask("state", QUESTIONS)

    def test_rate_limit_carries_retry_after(self) -> None:
        client = client_with(
            lambda request: httpx.Response(429, json={}, headers={"retry-after": "2.5"})
        )
        with pytest.raises(SystemOneRateLimitError) as caught:
            client.ask("state", QUESTIONS)
        assert caught.value.retry_after_s == 2.5

    def test_a_timeout_is_its_own_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(SystemOneTimeoutError):
            client_with(handler).ask("state", QUESTIONS)

    def test_a_transport_failure_is_unavailability(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(SystemOneUnavailableError):
            client_with(handler).ask("state", QUESTIONS)

    def test_a_successful_call_is_parsed_and_timed(self) -> None:
        payload = fixture_payload()
        client = client_with(
            lambda request: httpx.Response(
                200, json=payload, headers={"X-Typesafe-Request-Id": "req_abc"}
            )
        )
        answers = client.ask("state", QUESTIONS)
        assert answers.request_id == "req_abc"
        assert answers.latency_ms >= 0.0

    def test_an_empty_bundle_never_reaches_the_network(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("should not be called")

        with pytest.raises(SystemOneUsageError):
            client_with(handler).ask("state", {})


class TestModelListing:
    LISTING = {
        "models": [
            {"name": "jev-latest", "description": "latest"},
            {"name": "jev-preview", "description": "preview"},
        ]
    }

    def test_check_model_accepts_an_advertised_model(self) -> None:
        client = client_with(lambda request: httpx.Response(200, json=self.LISTING))
        assert client.check_model() == ("jev-latest", "jev-preview")

    def test_check_model_refuses_an_unadvertised_model(self) -> None:
        client = client_with(
            lambda request: httpx.Response(200, json=self.LISTING), model="jev-9.9.9"
        )
        with pytest.raises(SystemOneProtocolError, match="not advertised"):
            client.check_model()

    def test_an_empty_listing_is_a_protocol_error(self) -> None:
        client = client_with(lambda request: httpx.Response(200, json={"models": []}))
        with pytest.raises(SystemOneProtocolError):
            client.check_model()


class TestCredentialsAndPresets:
    def test_a_missing_credential_fails_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(SystemOneAuthError, match="TYPESAFE_API_KEY"):
            SystemOneClient.jev().ask("state", QUESTIONS)

    def test_the_jev_preset_targets_the_hosted_api_through_the_proxy(self) -> None:
        client = SystemOneClient.jev()
        assert client.base_url == JEV_BASE_URL
        assert client.model == "jev-latest"
        assert client.trust_env is True
        assert client.expected_model is None

    def test_the_kev_preset_is_loopback_only_and_needs_no_credential(self) -> None:
        client = SystemOneClient.kev()
        assert client.base_url == KEV_BASE_URL
        assert client.model == "kev-latest"
        assert client.trust_env is False
        assert client.require_api_key is False
        assert client.expected_model == "kev-latest"

    def test_the_kev_preset_accepts_a_different_host(self) -> None:
        client = SystemOneClient.kev(base_url="http://127.0.0.1:9100/")
        assert client.base_url == "http://127.0.0.1:9100"


@pytest.mark.asyncio
async def test_the_async_path_parses_the_same_way() -> None:
    payload = fixture_payload()
    client = SystemOneClient.jev()
    client._async = httpx.AsyncClient(
        base_url=JEV_BASE_URL,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    answers = await client.aask("state", QUESTIONS)
    assert answers.choice("addressee").chosen == "robot"
    await client.aclose()
