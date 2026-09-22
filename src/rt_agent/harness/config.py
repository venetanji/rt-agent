"""``AgentConfig`` — every knob the harness has, in one frozen settings object.

Selection, not construction, is what this module is about: which System One backend
answers the bundle, which text generator writes the replies, where the face listens,
where the run's artifacts land. The factory methods build exactly one object each and
nothing else, so the CLI stays a thin argument parser and a test can build the same
agent without going through it.

Every field reads ``RT_AGENT_<FIELD>`` from the environment, and so does the nested
:class:`~rt_agent.policy.PolicyConfig` — the thresholds keep their own names
(``RT_AGENT_ADDRESSED_MIN`` and friends) rather than being re-declared here, so there is
exactly one spelling of each threshold in the system.

One deliberate overlap: ``deadline_ms`` exists on both this object and ``PolicyConfig``.
Both read the same ``RT_AGENT_DEADLINE_MS``, and :meth:`AgentConfig.policy_config` makes
the agent's value authoritative, so the hard deadline the ``asyncio.wait_for`` enforces
and the one the policy checks can never drift apart.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rt_agent.contracts import ChatLLM, FaceBridge, SystemOneBackend
from rt_agent.face import VOICES, AuthoredFaceModel, JsonlFaceBridge, NullFaceBridge
from rt_agent.face.authored_face import ProfileName
from rt_agent.llm import OpenAICompatibleLLM, ScriptedLLM
from rt_agent.policy import PolicyConfig
from rt_agent.systemone import JEV_MODEL, KEV_MODEL, MockSystemOne, SystemOneClient

__all__ = [
    "AgentConfig",
    "BackendName",
    "FaceName",
    "LlmName",
    "new_session_id",
]

BackendName = Literal["jev", "kev", "mock"]
LlmName = Literal["scripted", "openai"]
FaceName = Literal["jsonl", "udp", "null", "ros"]


def new_session_id(prefix: str = "s") -> str:
    """A session id that sorts by time and never collides: ``s-20260922-120000-1a2b3c``."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


class AgentConfig(BaseSettings):
    """Everything the harness needs to build itself. Frozen; override via env or kwargs."""

    model_config = SettingsConfigDict(env_prefix="RT_AGENT_", frozen=True, extra="forbid")

    # -- identity and output ------------------------------------------------------

    #: Session id for this run. Empty means "make one" (see :meth:`resolved_session_id`).
    session_id: str = ""
    #: The robot's name, as it appears in the rendered state and the reply persona.
    robot_name: str = Field(default="Alice", min_length=1, max_length=64)
    #: Root output directory; the run writes into ``out_dir / session_id``.
    out_dir: Path = Path("out")

    # -- System One backend -------------------------------------------------------

    #: ``jev`` (hosted), ``kev`` (local server) or ``mock`` (recorded fixtures).
    backend: BackendName = "jev"
    #: Override the backend's base URL; ``None`` uses the preset for ``backend``.
    backend_base_url: str | None = None
    #: Override the backend's model id; ``None`` uses the preset for ``backend``.
    backend_model: str | None = None
    #: Where ``backend="mock"`` loads its recorded calls from.
    fixtures_dir: Path | None = None
    #: Hard budget for one bundle call, enforced by ``asyncio.wait_for``.
    deadline_ms: int = Field(default=1500, ge=1)
    #: HTTP timeout for the backend transport. Kept a little above the deadline so the
    #: harness's own ``wait_for`` is what cuts a slow call off, with a named reason.
    backend_timeout_s: float = Field(default=0.0, ge=0.0)

    # -- text generation ----------------------------------------------------------

    #: ``scripted`` (deterministic, offline) or ``openai`` (an OpenAI-compatible server).
    llm: LlmName = "scripted"
    #: Rules file for ``llm="scripted"``; ``None`` falls back to bare defaults.
    scripted_rules: Path | None = None

    # -- face ---------------------------------------------------------------------

    #: ``jsonl`` (files under the run directory), ``udp``, ``null`` or ``ros``.
    face: FaceName = "jsonl"
    udp_host: str = "127.0.0.1"
    udp_port: int = Field(default=9917, ge=1, le=65535)
    #: Which alice face profile the simulator mirrors: ``bench`` or ``visible``.
    face_profile: ProfileName = "bench"
    #: ``face="ros"`` only: the ``RunSpeech`` epoch. alice's ``SessionNode`` mints the
    #: real one, so a genuine run must pass the value read off the first action feedback
    #: (``RunSpeechClient.await_identity``); the generated default is a placeholder that
    #: lets the bridge be constructed for a preview.
    ros_epoch: str = ""
    #: ``face="ros"`` only: the generation id stamped onto every published clause.
    ros_generation_id: str = "rt-agent"

    # -- speech -------------------------------------------------------------------

    #: alice TTS voice written into every exported ``speech-plan/v1``.
    voice: str = "azelma"
    #: Base TTS seed; each clause gets ``seed + sequence``.
    seed: int = Field(default=0, ge=0, le=2**32 - 1)

    # -- memory -------------------------------------------------------------------

    #: SQLite file for the memory store; ``None`` puts it in the run directory.
    memory_db: Path | None = None
    #: How many memory writes may be in flight at once.
    memory_concurrency: int = Field(default=2, ge=1, le=32)
    #: Budget for one memory write (LLM sentence plus the faithfulness call).
    memory_deadline_ms: int = Field(default=15000, ge=1)

    # -- pacing -------------------------------------------------------------------

    #: Replay in the transcript's own time instead of as fast as possible.
    realtime: bool = False
    #: Realtime speed multiplier; 2.0 replays a five-minute scene in two and a half.
    speed: float = Field(default=1.0, gt=0.0)

    # -- thresholds ---------------------------------------------------------------

    #: Policy v1 thresholds. Reads ``RT_AGENT_*`` itself, so nothing is re-declared here.
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    # -- derived ------------------------------------------------------------------

    def resolved_session_id(self) -> str:
        """The configured session id, or a fresh time-ordered one."""
        return self.session_id or new_session_id()

    def run_dir(self, session_id: str) -> Path:
        """Where this run's artifacts go: ``out_dir / session_id``."""
        return self.out_dir / session_id

    def policy_config(self) -> PolicyConfig:
        """Thresholds with the agent's deadline folded in, so the two cannot drift."""
        if self.policy.deadline_ms == self.deadline_ms:
            return self.policy
        return self.policy.model_copy(update={"deadline_ms": self.deadline_ms})

    def resolved_backend_timeout_s(self) -> float:
        """Transport timeout: the configured one, or a little more than the deadline."""
        if self.backend_timeout_s > 0.0:
            return self.backend_timeout_s
        return max(1.0, self.deadline_ms / 1000.0 + 0.5)

    def memory_db_path(self, session_id: str) -> Path:
        """Where the SQLite memory store lives for this run."""
        if self.memory_db is not None:
            return self.memory_db
        return self.run_dir(session_id) / "memories.sqlite3"

    # -- factories ----------------------------------------------------------------

    def build_backend(self) -> SystemOneBackend:
        """Build the System One backend named by :attr:`backend`.

        ``jev`` and ``kev`` are the same client with different presets; ``mock`` needs
        :attr:`fixtures_dir` and never touches a network.
        """
        if self.backend == "mock":
            if self.fixtures_dir is None:
                raise ValueError(
                    "backend='mock' needs fixtures_dir (a directory of recorded calls)"
                )
            return MockSystemOne.from_fixtures(self.fixtures_dir)

        timeout_s = self.resolved_backend_timeout_s()
        options: dict[str, Any] = {"timeout_s": timeout_s}
        if self.backend_base_url is not None:
            options["base_url"] = self.backend_base_url
        if self.backend_model is not None:
            options["model"] = self.backend_model
        if self.backend == "kev":
            # kev() pins expected_model to its own model id; keep that in step with an
            # explicit override so a mismatch is still caught.
            if self.backend_model is not None:
                options["expected_model"] = self.backend_model
            return SystemOneClient.kev(**options)
        return SystemOneClient.jev(**options)

    def backend_model_id(self) -> str:
        """The model id this run asks for, before the server echoes what it actually ran."""
        if self.backend_model is not None:
            return self.backend_model
        return {"jev": JEV_MODEL, "kev": KEV_MODEL, "mock": "mock"}[self.backend]

    def build_llm(self) -> ChatLLM:
        """Build the text generator named by :attr:`llm`."""
        if self.llm == "openai":
            return OpenAICompatibleLLM()
        if self.scripted_rules is not None:
            return ScriptedLLM.from_file(self.scripted_rules)
        return ScriptedLLM()

    def build_face_model(self) -> AuthoredFaceModel:
        """The alice face simulator this run logs predictions from."""
        return AuthoredFaceModel(self.face_profile)

    def build_face(self, run_dir: Path) -> FaceBridge:
        """Build the face bridge named by :attr:`face`.

        ``ros`` is constructed only when rclpy and ``alice_interfaces`` are importable;
        it raises :class:`~rt_agent.face.RosUnavailableError` otherwise, at construction
        time, so a run never believes it has a face it does not have.
        """
        if self.face == "null":
            return NullFaceBridge()
        if self.face == "udp":
            from rt_agent.face import UdpJsonFaceBridge

            return UdpJsonFaceBridge(self.udp_host, self.udp_port, face=self.build_face_model())
        if self.face == "ros":
            from rt_agent.face import RosFaceBridge, RunIdentity

            identity = RunIdentity(
                run_id=f"rt-agent-{self.session_id or 'run'}",
                epoch=self.ros_epoch or uuid.uuid4().hex,
                generation_id=self.ros_generation_id,
            )
            return RosFaceBridge(identity)
        return JsonlFaceBridge(run_dir, face=self.build_face_model())

    def validated(self) -> AgentConfig:
        """Check the cross-field rules a single field validator cannot see."""
        if self.voice not in VOICES:
            raise ValueError(f"unknown voice {self.voice!r}; alice accepts {', '.join(VOICES)}")
        if self.backend == "mock" and self.fixtures_dir is None:
            raise ValueError("backend='mock' needs fixtures_dir")
        return self
