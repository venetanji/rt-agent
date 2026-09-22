"""Builders for the harness tests. Network-free: every answer is a recorded live call."""

from __future__ import annotations

from pathlib import Path

import pytest

from rt_agent.audio import ReplaySource
from rt_agent.contracts import Utterance
from rt_agent.face import JsonlFaceBridge
from rt_agent.harness import AgentConfig, ListeningAgent
from rt_agent.llm import ScriptedLLM
from rt_agent.systemone import MockSystemOne

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"
KITCHEN_CHAT = EXAMPLES / "kitchen_chat.jsonl"
SCRIPTED_RULES = EXAMPLES / "scripted_llm.json"
KITCHEN_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "systemone" / "kitchen_chat"

#: The session id the fixtures were recorded under. Nothing depends on it — the state
#: render never carries a session id — but keeping it steady makes a diff readable.
SESSION = "kitchen-test"


@pytest.fixture
def kitchen_backend() -> MockSystemOne:
    """Every live Jev call of the recorded kitchen-chat run, replayed offline."""
    return MockSystemOne.from_fixtures(KITCHEN_FIXTURES)


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    """The demo rules file: the two spoken replies and the memory sentences."""
    return ScriptedLLM.from_file(SCRIPTED_RULES)


def kitchen_utterances(session_id: str = SESSION) -> list[Utterance]:
    """The 28 lines of the example transcript, parsed but not yet decided about."""
    return list(ReplaySource(KITCHEN_CHAT, session_id=session_id).read())


def line(utterances: list[Utterance], number: int) -> Utterance:
    """The utterance on 1-based line ``number`` of the transcript file."""
    return utterances[number - 1]


def make_agent(
    tmp_path: Path,
    backend: MockSystemOne,
    llm: ScriptedLLM,
    *,
    session_id: str = SESSION,
    **overrides: object,
) -> ListeningAgent:
    """A fully wired agent writing into ``tmp_path``, with no network and no hardware."""
    settings: dict[str, object] = {
        "backend": "mock",
        "llm": "scripted",
        "face": "jsonl",
        "out_dir": tmp_path / "out",
        "session_id": session_id,
        "fixtures_dir": KITCHEN_FIXTURES,
    }
    settings.update(overrides)
    config = AgentConfig(**settings).validated()
    run_dir = config.run_dir(session_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return ListeningAgent(
        backend,
        llm,
        JsonlFaceBridge(run_dir, face=config.build_face_model()),
        session_id=session_id,
        run_dir=run_dir,
        config=config,
    )
