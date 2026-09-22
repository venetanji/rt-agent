"""``SystemOneClient`` — one HTTP client for hosted Jev and local Kev.

Both speak the same wire protocol (``POST /v1/systemone``, ``GET /v1/models``), so the
endpoint is a config knob and the adapter is written once. Two things are deliberate:

* **No retries.** 429/529 are real, but for the admission path a late SPEAK is worse
  than a WAIT; the caller fails closed instead. Retry only off the critical path.
* **Strict, fail-closed parsing.** Every requested question must come back, with the
  type we asked for, a complete distribution over exactly the option set we sent, sums
  within :data:`~rt_agent.contracts.decision.PROBABILITY_SUM_TOLERANCE` (the server
  rounds probabilities to two decimals) and a winner that agrees with its own argmax.
  Anything else raises :class:`SystemOneProtocolError`.
"""

from __future__ import annotations

import math
import os
import time
from types import TracebackType
from typing import Any, Self

import httpx

from rt_agent.contracts.decision import BundleAnswers, ChoiceAnswer, ScoreAnswer, Usage
from rt_agent.contracts.events import DecisionContext
from rt_agent.contracts.protocols import QuestionsWire
from rt_agent.systemone.bundle import BUNDLE_VERSION, QUESTION_BUNDLE_V1, QuestionBundle
from rt_agent.systemone.errors import (
    SystemOneAuthError,
    SystemOneProtocolError,
    SystemOneRateLimitError,
    SystemOneTimeoutError,
    SystemOneUnavailableError,
    SystemOneUsageError,
)
from rt_agent.systemone.state import render_state

__all__ = [
    "JEV_BASE_URL",
    "JEV_MODEL",
    "KEV_BASE_URL",
    "KEV_MODEL",
    "REQUEST_ID_HEADER",
    "SystemOneClient",
    "parse_response",
]

JEV_BASE_URL = "https://api.typesafe.ai"
JEV_MODEL = "jev-latest"
KEV_BASE_URL = "http://127.0.0.1:8009"
KEV_MODEL = "kev-latest"

REQUEST_ID_HEADER = "X-Typesafe-Request-Id"

_SYSTEMONE_PATH = "/v1/systemone"
_MODELS_PATH = "/v1/models"


# --------------------------------------------------------------------------------------
# Response parsing (shared by the live client and the fixture-driven mock)
# --------------------------------------------------------------------------------------


def _as_probability(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemOneProtocolError(f"{where} is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise SystemOneProtocolError(f"{where} is outside [0, 1]: {number!r}")
    return number


def _distribution(raw: Any, where: str) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise SystemOneProtocolError(f"{where} is not a non-empty object")
    return {str(key): _as_probability(value, f"{where}[{key!r}]") for key, value in raw.items()}


def _expected_choice_options(spec: Any, question_id: str) -> frozenset[str]:
    criteria = spec.get("criteria") if isinstance(spec, dict) else None
    if not isinstance(criteria, dict) or not criteria:
        raise SystemOneProtocolError(f"choice question {question_id!r} has no option set")
    return frozenset(str(key) for key in criteria)


def _expected_score_levels(spec: Any, question_id: str) -> int:
    criteria = spec.get("criteria") if isinstance(spec, dict) else None
    if not isinstance(criteria, list) or not criteria:
        raise SystemOneProtocolError(f"score question {question_id!r} has no rubric")
    return len(criteria)


def parse_response(
    payload: Any,
    questions: QuestionsWire,
    *,
    bundle_version: str = BUNDLE_VERSION,
    latency_ms: float,
    request_id: str | None = None,
    expected_model: str | None = None,
) -> BundleAnswers:
    """Validate a raw ``/v1/systemone`` payload against the questions we asked.

    Raises :class:`SystemOneProtocolError` on anything unexpected — the caller turns
    that into a WAIT.
    """
    if not isinstance(payload, dict):
        raise SystemOneProtocolError("response body is not a JSON object")

    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise SystemOneProtocolError("response is missing a model id")
    if expected_model is not None and model != expected_model:
        raise SystemOneProtocolError(
            f"response model {model!r} does not match the expected {expected_model!r}"
        )

    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise SystemOneProtocolError("response is missing an answers object")

    unexpected = sorted(set(answers) - set(questions))
    if unexpected:
        raise SystemOneProtocolError(f"response answers questions we did not ask: {unexpected}")
    missing = sorted(set(questions) - set(answers))
    if missing:
        raise SystemOneProtocolError(f"response is missing answers for: {missing}")

    nouls: dict[str, float] = {}
    choices: dict[str, ChoiceAnswer] = {}
    scores: dict[str, ScoreAnswer] = {}

    for question_id, spec in questions.items():
        answer = answers[question_id]
        if not isinstance(answer, dict):
            raise SystemOneProtocolError(f"answer for {question_id!r} is not an object")
        asked_type = spec.get("type") if isinstance(spec, dict) else None
        if answer.get("type") != asked_type:
            raise SystemOneProtocolError(
                f"answer for {question_id!r} has type {answer.get('type')!r}, asked {asked_type!r}"
            )

        if asked_type == "noul":
            nouls[question_id] = _as_probability(answer.get("noul"), f"{question_id}.noul")
        elif asked_type == "choice":
            probabilities = _distribution(
                answer.get("probabilities"), f"{question_id}.probabilities"
            )
            expected_options = _expected_choice_options(spec, question_id)
            if frozenset(probabilities) != expected_options:
                raise SystemOneProtocolError(
                    f"{question_id!r} returned options {sorted(probabilities)}, "
                    f"asked {sorted(expected_options)}"
                )
            chosen = answer.get("choice")
            if not isinstance(chosen, str):
                raise SystemOneProtocolError(f"{question_id!r} returned no choice")
            try:
                choices[question_id] = ChoiceAnswer(
                    chosen=chosen,
                    probabilities=probabilities,
                    confidence=_as_probability(
                        answer.get("confidence"), f"{question_id}.confidence"
                    ),
                )
            except ValueError as error:
                raise SystemOneProtocolError(f"{question_id!r}: {error}") from error
        elif asked_type == "score":
            probabilities = _distribution(
                answer.get("probabilities"), f"{question_id}.probabilities"
            )
            level_count = _expected_score_levels(spec, question_id)
            if frozenset(probabilities) != frozenset(str(index) for index in range(level_count)):
                raise SystemOneProtocolError(
                    f"{question_id!r} returned levels {sorted(probabilities)}, "
                    f"asked for {level_count} levels"
                )
            raw_score = answer.get("score")
            if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
                raise SystemOneProtocolError(f"{question_id!r} returned no numeric score")
            legend_raw = answer.get("legend") or {}
            if not isinstance(legend_raw, dict):
                raise SystemOneProtocolError(f"{question_id!r} returned a malformed legend")
            try:
                scores[question_id] = ScoreAnswer(
                    value=float(raw_score),
                    probabilities=probabilities,
                    confidence=_as_probability(
                        answer.get("confidence"), f"{question_id}.confidence"
                    ),
                    legend={str(key): str(value) for key, value in legend_raw.items()},
                )
            except ValueError as error:
                raise SystemOneProtocolError(f"{question_id!r}: {error}") from error
        else:
            raise SystemOneProtocolError(
                f"question {question_id!r} has unknown type {asked_type!r}"
            )

    usage_raw = payload.get("usage")
    usage: Usage | None = None
    if isinstance(usage_raw, dict):
        input_tokens = usage_raw.get("input_tokens")
        output_tokens = usage_raw.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

    return BundleAnswers(
        bundle_version=bundle_version,
        model=model,
        request_id=request_id,
        latency_ms=latency_ms,
        usage=usage,
        nouls=nouls,
        choices=choices,
        scores=scores,
    )


# --------------------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------------------


class SystemOneClient:
    """A long-lived System One client. Reuse one instance; the first call is the slow one.

    ``trust_env`` is ``True`` for hosted endpoints so that ``HTTPS_PROXY`` and
    ``SSL_CERT_FILE`` are honoured. TLS verification is never disabled.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "TYPESAFE_API_KEY",
        timeout_s: float = 1.5,
        *,
        connect_timeout_s: float | None = None,
        trust_env: bool = True,
        require_api_key: bool = True,
        expected_model: str | None = None,
        bundle_version: str = BUNDLE_VERSION,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.connect_timeout_s = (
            min(timeout_s, 2.0) if connect_timeout_s is None else connect_timeout_s
        )
        self.trust_env = trust_env
        self.require_api_key = require_api_key
        self.expected_model = expected_model
        self.bundle_version = bundle_version
        self._sync: httpx.Client | None = None
        self._async: httpx.AsyncClient | None = None

    # -- presets ------------------------------------------------------------------

    @classmethod
    def jev(
        cls,
        base_url: str = JEV_BASE_URL,
        model: str = JEV_MODEL,
        timeout_s: float = 1.5,
        **kwargs: Any,
    ) -> SystemOneClient:
        """Hosted TypeSafe Jev. Sending room transcripts here is a deliberate decision."""
        return cls(base_url=base_url, model=model, timeout_s=timeout_s, **kwargs)

    @classmethod
    def kev(
        cls,
        base_url: str = KEV_BASE_URL,
        model: str = KEV_MODEL,
        timeout_s: float = 3.0,
        **kwargs: Any,
    ) -> SystemOneClient:
        """Local Kev server: same protocol, loopback only, no credentials, no proxy.

        The response's model echo is checked against ``kev-latest`` as a cheap sanity
        check only; a name echo is not evidence of model identity. Use
        :meth:`check_model` (and the server's ``run`` field) for that.
        """
        kwargs.setdefault("trust_env", False)
        kwargs.setdefault("require_api_key", False)
        kwargs.setdefault("expected_model", model)
        return cls(base_url=base_url, model=model, timeout_s=timeout_s, **kwargs)

    # -- plumbing -----------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = os.environ.get(self.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.require_api_key:
            raise SystemOneAuthError(
                f"environment variable {self.api_key_env} is not set; "
                "configure a System One credential before calling the hosted API"
            )
        return headers

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s)

    def _sync_client(self) -> httpx.Client:
        if self._sync is None:
            self._sync = httpx.Client(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self._timeout(),
                trust_env=self.trust_env,
                follow_redirects=False,
            )
        return self._sync

    def _async_client(self) -> httpx.AsyncClient:
        if self._async is None:
            self._async = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self._timeout(),
                trust_env=self.trust_env,
                follow_redirects=False,
            )
        return self._async

    def _body(self, state: str, questions: QuestionsWire) -> dict[str, Any]:
        if not questions:
            raise SystemOneUsageError("a bundle call needs at least one question")
        return {"state": state, "model": self.model, "questions": dict(questions)}

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        request_id = response.headers.get(REQUEST_ID_HEADER)
        detail = response.text[:500]
        message = f"HTTP {response.status_code} from System One (request_id={request_id}): {detail}"
        status = response.status_code
        if status in (401, 403):
            raise SystemOneAuthError(message)
        if status in (429, 529):
            retry_after = response.headers.get("retry-after")
            try:
                seconds = float(retry_after) if retry_after is not None else None
            except ValueError:
                seconds = None
            raise SystemOneRateLimitError(message, retry_after_s=seconds)
        if 400 <= status < 500:
            raise SystemOneUsageError(message)
        raise SystemOneUnavailableError(message)

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as error:
            raise SystemOneProtocolError(f"response body is not JSON: {error}") from error

    def _parse(
        self, response: httpx.Response, questions: QuestionsWire, latency_ms: float
    ) -> BundleAnswers:
        return parse_response(
            self._json(response),
            questions,
            bundle_version=self.bundle_version,
            latency_ms=latency_ms,
            request_id=response.headers.get(REQUEST_ID_HEADER),
            expected_model=self.expected_model,
        )

    # -- the bundle call ----------------------------------------------------------

    def ask(self, state: str, questions: QuestionsWire) -> BundleAnswers:
        """Blocking bundle call. Raises a :class:`SystemOneError` subclass on any failure."""
        body = self._body(state, questions)
        started = time.perf_counter()
        try:
            response = self._sync_client().post(_SYSTEMONE_PATH, json=body)
        except httpx.TimeoutException as error:
            raise SystemOneTimeoutError(
                f"System One call timed out after {self.timeout_s}s"
            ) from error
        except httpx.HTTPError as error:
            raise SystemOneUnavailableError(f"System One transport failure: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._raise_for_status(response)
        return self._parse(response, questions, latency_ms)

    async def aask(self, state: str, questions: QuestionsWire) -> BundleAnswers:
        """Awaitable bundle call; wrap it in ``asyncio.wait_for`` for a hard deadline."""
        body = self._body(state, questions)
        started = time.perf_counter()
        try:
            response = await self._async_client().post(_SYSTEMONE_PATH, json=body)
        except httpx.TimeoutException as error:
            raise SystemOneTimeoutError(
                f"System One call timed out after {self.timeout_s}s"
            ) from error
        except httpx.HTTPError as error:
            raise SystemOneUnavailableError(f"System One transport failure: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._raise_for_status(response)
        return self._parse(response, questions, latency_ms)

    def ask_context(
        self, ctx: DecisionContext, bundle: QuestionBundle = QUESTION_BUNDLE_V1
    ) -> BundleAnswers:
        """Render a decision context and ask the frozen bundle about it."""
        return self.ask(render_state(ctx), bundle.to_wire())

    async def aask_context(
        self, ctx: DecisionContext, bundle: QuestionBundle = QUESTION_BUNDLE_V1
    ) -> BundleAnswers:
        """Async :meth:`ask_context`."""
        return await self.aask(render_state(ctx), bundle.to_wire())

    # -- model identity -----------------------------------------------------------

    @staticmethod
    def _model_ids(payload: Any) -> tuple[str, ...]:
        entries: Any
        if isinstance(payload, dict):
            entries = payload.get("data") or payload.get("models") or []
        elif isinstance(payload, list):
            entries = payload
        else:
            raise SystemOneProtocolError("model listing is neither an object nor an array")
        if not isinstance(entries, list):
            raise SystemOneProtocolError("model listing does not contain an array of models")
        ids: list[str] = []
        for entry in entries:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, dict):
                identifier = entry.get("id") or entry.get("model") or entry.get("name")
                if isinstance(identifier, str):
                    ids.append(identifier)
        if not ids:
            raise SystemOneProtocolError("model listing is empty")
        return tuple(ids)

    def check_model(self) -> tuple[str, ...]:
        """``GET /v1/models``; raise unless this client's model is advertised."""
        try:
            response = self._sync_client().get(_MODELS_PATH)
        except httpx.TimeoutException as error:
            raise SystemOneTimeoutError("model listing timed out") from error
        except httpx.HTTPError as error:
            raise SystemOneUnavailableError(f"model listing transport failure: {error}") from error
        self._raise_for_status(response)
        ids = self._model_ids(self._json(response))
        if self.model not in ids:
            raise SystemOneProtocolError(
                f"model {self.model!r} is not advertised by {self.base_url}; available: {list(ids)}"
            )
        return ids

    async def acheck_model(self) -> tuple[str, ...]:
        """Async :meth:`check_model`."""
        try:
            response = await self._async_client().get(_MODELS_PATH)
        except httpx.TimeoutException as error:
            raise SystemOneTimeoutError("model listing timed out") from error
        except httpx.HTTPError as error:
            raise SystemOneUnavailableError(f"model listing transport failure: {error}") from error
        self._raise_for_status(response)
        ids = self._model_ids(self._json(response))
        if self.model not in ids:
            raise SystemOneProtocolError(
                f"model {self.model!r} is not advertised by {self.base_url}; available: {list(ids)}"
            )
        return ids

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        """Close the synchronous transport."""
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        """Close the asynchronous transport."""
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
