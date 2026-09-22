"""``MockSystemOne`` — a network-free System One backend for tests and replays.

It serves answers three ways, tried in order:

1. an exact match on the rendered state (what the live recorder captured);
2. a match on the current-utterance line of the state, so a fixture keeps working when
   an unrelated part of the context changes;
3. a rule function ``(state, questions) -> BundleAnswers`` for synthetic cases.

Fixtures are parsed through the *same* strict validator the live client uses, so a
fixture that the real policy would reject cannot silently pass a test. Artificial
latency and failure injection let the harness's deadline and fail-closed paths be
exercised without a server.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from rt_agent.contracts.decision import BundleAnswers
from rt_agent.contracts.events import DecisionContext
from rt_agent.contracts.protocols import QuestionsWire
from rt_agent.systemone.bundle import BUNDLE_VERSION, QUESTION_BUNDLE_V1, QuestionBundle
from rt_agent.systemone.client import parse_response
from rt_agent.systemone.errors import SystemOneError
from rt_agent.systemone.state import render_state

__all__ = ["FIXTURE_SCHEMA_VERSION", "MockAnswerNotFound", "MockSystemOne", "RecordedCall"]

#: Version tag written into every recorded fixture file.
FIXTURE_SCHEMA_VERSION = "systemone-fixture/v1"

_CURRENT_MARKER = "Current utterance:"

Rule = Callable[[str, QuestionsWire], BundleAnswers]


class MockAnswerNotFound(SystemOneError):
    """No fixture and no rule could answer this state — fail closed, like the real thing."""


class RecordedCall:
    """One recorded live call: the exact request that was sent and what came back."""

    __slots__ = ("answers", "path", "raw", "scenario", "state", "utterance_id")

    def __init__(
        self,
        scenario: str,
        utterance_id: str,
        state: str,
        answers: BundleAnswers,
        raw: Mapping[str, Any],
        path: Path | None = None,
    ) -> None:
        self.scenario = scenario
        self.utterance_id = utterance_id
        self.state = state
        self.answers = answers
        self.raw = raw
        self.path = path

    def __repr__(self) -> str:
        return f"RecordedCall(scenario={self.scenario!r}, utterance_id={self.utterance_id!r})"


def _current_line(state: str) -> str | None:
    lines = state.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == _CURRENT_MARKER and index + 1 < len(lines):
            return lines[index + 1].strip()
    return None


def load_fixture(path: Path) -> RecordedCall:
    """Load and re-validate one recorded call."""
    document = json.loads(path.read_text(encoding="utf-8"))
    state = document["state"]
    questions = document["questions"]
    answers = parse_response(
        document["response"],
        questions,
        bundle_version=document.get("bundle_version", BUNDLE_VERSION),
        latency_ms=float(document.get("latency_ms", 0.0)),
        request_id=document.get("request_id"),
    )
    return RecordedCall(
        scenario=document.get("scenario", path.stem),
        utterance_id=document.get("utterance_id", path.stem),
        state=state,
        answers=answers,
        raw=document,
        path=path,
    )


class MockSystemOne:
    """A :class:`~rt_agent.contracts.protocols.SystemOneBackend` that never touches a network."""

    def __init__(
        self,
        recordings: Iterable[RecordedCall] = (),
        *,
        rule: Rule | None = None,
        latency_ms: float = 0.0,
        fail_with: Exception | None = None,
        fail_count: int | None = None,
        sleep: bool = True,
    ) -> None:
        self._by_state: dict[str, RecordedCall] = {}
        self._by_current: dict[str, RecordedCall] = {}
        self._by_utterance: dict[str, RecordedCall] = {}
        self._by_scenario: dict[str, RecordedCall] = {}
        for recording in recordings:
            self.add(recording)
        self.rule = rule
        self.latency_ms = latency_ms
        self.sleep = sleep
        self._fail_with = fail_with
        self._fail_remaining = fail_count
        self.calls: list[tuple[str, QuestionsWire]] = []

    # -- construction -------------------------------------------------------------

    @classmethod
    def from_fixtures(
        cls, directory: str | Path, pattern: str = "*.json", **kwargs: Any
    ) -> MockSystemOne:
        """Load every recorded call in a directory (``tests/fixtures/systemone`` by default)."""
        root = Path(directory)
        recordings = [load_fixture(path) for path in sorted(root.glob(pattern))]
        return cls(recordings, **kwargs)

    def add(self, recording: RecordedCall) -> None:
        """Index one recorded call."""
        self._by_state[recording.state] = recording
        current = _current_line(recording.state)
        if current is not None:
            self._by_current[current] = recording
        self._by_utterance[recording.utterance_id] = recording
        self._by_scenario[recording.scenario] = recording

    # -- failure injection --------------------------------------------------------

    def arm_failure(self, error: Exception, count: int | None = None) -> None:
        """Make the next ``count`` calls (or all of them) raise ``error``."""
        self._fail_with = error
        self._fail_remaining = count

    def clear_failure(self) -> None:
        """Stop injecting failures."""
        self._fail_with = None
        self._fail_remaining = None

    def _maybe_fail(self) -> None:
        if self._fail_with is None:
            return
        if self._fail_remaining is None:
            raise self._fail_with
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            if self._fail_remaining == 0:
                error, self._fail_with = self._fail_with, None
                raise error
            raise self._fail_with

    # -- lookup -------------------------------------------------------------------

    @property
    def scenarios(self) -> tuple[str, ...]:
        """Names of every loaded recording."""
        return tuple(sorted(self._by_scenario))

    def recording(self, name: str) -> RecordedCall:
        """One recording by scenario name or utterance id."""
        if name in self._by_scenario:
            return self._by_scenario[name]
        return self._by_utterance[name]

    def _lookup(self, state: str, questions: QuestionsWire) -> BundleAnswers:
        recording = self._by_state.get(state)
        if recording is None:
            current = _current_line(state)
            if current is not None:
                recording = self._by_current.get(current)
        if recording is not None:
            return recording.answers
        if self.rule is not None:
            return self.rule(state, questions)
        raise MockAnswerNotFound(
            f"no fixture matches this state and no rule is configured "
            f"(current utterance: {_current_line(state)!r})"
        )

    # -- the backend protocol -----------------------------------------------------

    def ask(self, state: str, questions: QuestionsWire) -> BundleAnswers:
        """Blocking lookup, with the configured artificial latency."""
        self.calls.append((state, dict(questions)))
        self._maybe_fail()
        if self.latency_ms and self.sleep:
            time.sleep(self.latency_ms / 1000.0)
        return self._lookup(state, questions)

    async def aask(self, state: str, questions: QuestionsWire) -> BundleAnswers:
        """Awaitable lookup, with the configured artificial latency."""
        self.calls.append((state, dict(questions)))
        self._maybe_fail()
        if self.latency_ms and self.sleep:
            await asyncio.sleep(self.latency_ms / 1000.0)
        return self._lookup(state, questions)

    def ask_context(
        self, ctx: DecisionContext, bundle: QuestionBundle = QUESTION_BUNDLE_V1
    ) -> BundleAnswers:
        """Convenience mirror of :meth:`SystemOneClient.ask_context`."""
        return self.ask(render_state(ctx), bundle.to_wire())

    async def aask_context(
        self, ctx: DecisionContext, bundle: QuestionBundle = QUESTION_BUNDLE_V1
    ) -> BundleAnswers:
        """Convenience mirror of :meth:`SystemOneClient.aask_context`."""
        return await self.aask(render_state(ctx), bundle.to_wire())

    def close(self) -> None:
        """No-op; present so the mock is drop-in for the real client."""

    async def aclose(self) -> None:
        """No-op; present so the mock is drop-in for the real client."""
