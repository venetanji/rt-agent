"""Record live System One calls made during a replay, as replayable test fixtures.

A fixture is only worth having if it proves something about the real service, so what
lands on disk here is the exact request that was sent and the exact body that came
back — never a reconstruction from parsed answers.
:func:`~rt_agent.systemone.load_fixture` re-validates each file with the same strict
parser the live client uses, so a fixture the policy would reject cannot pass a test.

Two kinds of call happen during one replay and both are recorded:

* the frozen ``decision-bundle/v1`` call for an utterance, written to
  ``<utterance_id>.json``;
* the single ``memory_faithful`` gate call the background
  :class:`~rt_agent.memory.writer.MemoryWriter` makes, written to
  ``<utterance_id>.memory_faithful.json``.

The second one happens in a background task, long after the harness has moved on to a
later utterance, so "which utterance is this?" cannot be answered by a mutable
attribute. It is answered by a :class:`~contextvars.ContextVar`: ``asyncio.create_task``
copies the context at creation time, so a memory task created while handling ``u0005``
still reports ``u0005`` when it finally runs.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rt_agent.systemone.bundle import BUNDLE_VERSION, MEMORY_FAITHFUL_VERSION, Q_MEMORY_FAITHFUL
from rt_agent.systemone.client import SystemOneCall
from rt_agent.systemone.mock import FIXTURE_SCHEMA_VERSION

__all__ = ["CURRENT_UTTERANCE", "FixtureRecorder"]

#: The utterance the harness is currently deciding about. Background tasks inherit the
#: value that was set when they were created.
CURRENT_UTTERANCE: ContextVar[str] = ContextVar("rt_agent_current_utterance", default="call")


class FixtureRecorder:
    """Write one JSON fixture per live System One call into a directory."""

    def __init__(self, out_dir: Path | str, *, scenario_prefix: str = "kitchen_chat") -> None:
        self.out_dir = Path(out_dir)
        self.scenario_prefix = scenario_prefix
        self.written: list[Path] = []
        self._counts: dict[str, int] = {}

    def _name_for(self, call: SystemOneCall) -> str:
        utterance_id = CURRENT_UTTERANCE.get()
        stem = (
            f"{utterance_id}.memory_faithful"
            if Q_MEMORY_FAITHFUL in call.questions
            else utterance_id
        )
        seen = self._counts.get(stem, 0)
        self._counts[stem] = seen + 1
        # A second call with the same stem would otherwise silently replace the first.
        return stem if seen == 0 else f"{stem}.{seen + 1}"

    def __call__(self, call: SystemOneCall) -> None:
        """Write one recorded call. Attached as ``SystemOneClient.on_call``."""
        is_memory = Q_MEMORY_FAITHFUL in call.questions
        name = self._name_for(call)
        document: dict[str, Any] = {
            "fixture_version": FIXTURE_SCHEMA_VERSION,
            "scenario": f"{self.scenario_prefix}:{name}",
            "utterance_id": name,
            "recorded_at": datetime.now(UTC).isoformat(),
            "model_resolved": call.model,
            "bundle_version": MEMORY_FAITHFUL_VERSION if is_memory else BUNDLE_VERSION,
            "state": call.state,
            "questions": call.questions,
            "response": call.payload,
            "latency_ms": round(call.latency_ms, 1),
            "request_id": call.request_id,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{name}.json"
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.written.append(path)
