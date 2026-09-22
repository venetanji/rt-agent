"""``ListeningAgent`` — the loop that turns heard turns into silence, a face, or a reply.

This is design brief §7 in code, and the order of operations is the design:

1. a finalized ``utterance/v1`` arrives and goes straight to the ``TranscriptLog``;
2. a :class:`~rt_agent.contracts.DecisionContext` is assembled — the last six turns
   *including the robot's own*, the three best memories, and what the robot itself is
   doing;
3. that context renders to ``state-render/v1`` and is asked the one frozen bundle,
   under a hard :func:`asyncio.wait_for` deadline;
4. :class:`~rt_agent.policy.Policy` turns the answers into one ``decision/v1``;
5. the :class:`~rt_agent.policy.EmotionController` decides whether the face may move,
   and the cue goes to the :class:`~rt_agent.contracts.FaceBridge`;
6. a memory-worthy turn schedules a **background** write, never on the critical path;
7. only then, and only if the decision says so, is a reply generated, split into
   clauses, emitted, exported as a ``speech-plan/v1`` and appended to the log as a
   ``ROBOT`` turn.

**Fail-closed everywhere.** Nothing in this module may raise into the caller. A backend
that times out or errors produces answers of ``None``, which the policy turns into
``speak=False`` with ``wait_reason="backend_unavailable"``. A face bridge that throws
costs its record, not the conversation. A reply that fails to generate means the robot
says nothing, and the run log says why.

**The clock is the transcript's, not the wall's.** Every decision, cue and refractory
check is stamped with ``session_epoch + t_end_s``. A replay that runs in 8 seconds and
the 96-second conversation it replays therefore make the *same* decisions, which is what
makes the emotion timeline reproducible and the hysteresis meaningful offline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from pydantic import AwareDatetime, Field

from rt_agent.contracts import (
    RECENT_WINDOW,
    ROBOT_SPEAKER_LABEL,
    AffectCue,
    ChatLLM,
    Decision,
    DecisionContext,
    FaceBridge,
    FrozenModel,
    MemoryRecord,
    RobotState,
    SpeechClauseOut,
    SystemOneBackend,
    Utterance,
)
from rt_agent.face import (
    AuthoredFaceModel,
    FaceTarget,
    build_speech_clauses,
    export_speech_plan,
    split_clauses,
    write_speech_plan,
)
from rt_agent.harness.config import AgentConfig
from rt_agent.harness.record import CURRENT_UTTERANCE, FixtureRecorder
from rt_agent.llm import ReplyGenerator
from rt_agent.memory import (
    MemoryOutcome,
    MemoryRetriever,
    MemoryWriteOutcome,
    MemoryWriter,
    SqliteMemoryStore,
    TranscriptLog,
)
from rt_agent.policy import POLICY_VERSION, EmotionController, Policy
from rt_agent.policy.presets import PRESETS_VERSION
from rt_agent.systemone import BUNDLE_VERSION, QUESTION_BUNDLE_V1, render_state
from rt_agent.systemone.client import SystemOneClient

__all__ = [
    "DECISION_RECORD_VERSION",
    "ROBOT_WORDS_PER_SECOND",
    "SUMMARY_VERSION",
    "ListeningAgent",
    "RunSummary",
    "TurnRecord",
    "percentile",
]

_LOGGER = logging.getLogger(__name__)

#: Version tag of one line of ``decisions.jsonl``.
DECISION_RECORD_VERSION = "decision-record/v1"

#: Version tag of ``summary.json``.
SUMMARY_VERSION = "run-summary/v1"

#: The harness does not synthesise audio, so the length of a ``ROBOT`` turn is an
#: estimate at an unhurried speaking rate — enough to place the turn on the timeline,
#: never evidence of how long the robot actually spoke.
ROBOT_WORDS_PER_SECOND = 2.8

_MIN_ROBOT_TURN_S = 0.4


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of ``values``; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


class TurnRecord(FrozenModel):
    """``decision-record/v1`` — one line of ``decisions.jsonl``.

    The decision itself is embedded whole (including the raw ``BundleAnswers``), so the
    file is a complete account of why the robot did what it did, not a summary of it.
    """

    schema_version: Literal["decision-record/v1"] = "decision-record/v1"
    seq: int = Field(ge=0)
    session_id: str
    utterance: Utterance
    decision: Decision
    spoke: bool = False
    reply: str | None = None
    clause_count: int = 0
    speech_plan: str | None = None
    emotion_changed: bool = False
    emitted_cue: AffectCue | None = None
    face_target: FaceTarget | None = None
    retrieved_memory_ids: tuple[str, ...] = ()
    memory_scheduled: bool = False
    bundle_ms: float = Field(default=0.0, ge=0.0)
    policy_ms: float = Field(default=0.0, ge=0.0)
    llm_ms: float | None = None
    total_ms: float = Field(default=0.0, ge=0.0)
    backend_error: str | None = None
    reply_error: str | None = None
    recorded_at: AwareDatetime


class RunSummary(FrozenModel):
    """``run-summary/v1`` — what a whole replay or listening session came to."""

    schema_version: Literal["run-summary/v1"] = "run-summary/v1"
    session_id: str
    started_at: AwareDatetime
    finished_at: AwareDatetime
    wall_seconds: float = Field(ge=0.0)
    backend: str
    backend_model_requested: str
    backend_model_resolved: str | None = None
    llm: str
    face: str
    policy_version: str = POLICY_VERSION
    bundle_version: str = BUNDLE_VERSION
    presets_version: str = PRESETS_VERSION
    utterances: int = Field(default=0, ge=0)
    speak_count: int = Field(default=0, ge=0)
    spoken_count: int = Field(default=0, ge=0)
    wait_count: int = Field(default=0, ge=0)
    wait_reasons: dict[str, int] = Field(default_factory=dict)
    backend_failures: int = Field(default=0, ge=0)
    reply_failures: int = Field(default=0, ge=0)
    emotion_changes: int = Field(default=0, ge=0)
    emotion_timeline: tuple[tuple[str, str], ...] = ()
    emotion_counts: dict[str, int] = Field(default_factory=dict)
    memory_outcomes: dict[str, int] = Field(default_factory=dict)
    memories_stored: int = Field(default=0, ge=0)
    speech_plans: tuple[str, ...] = ()
    bundle_ms_p50: float = 0.0
    bundle_ms_p95: float = 0.0
    bundle_ms_max: float = 0.0
    total_ms_p50: float = 0.0
    total_ms_p95: float = 0.0
    total_ms_max: float = 0.0


class _Jsonl:
    """A tiny append-and-flush JSONL writer, so a killed run still leaves a readable log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def append(self, record: dict[str, Any]) -> None:
        """Write one object and flush it."""
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the file if it was ever opened."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class ListeningAgent:
    """One listening session: utterances in, decisions, a face, memories and replies out.

    Build it with :meth:`from_config` (the CLI's path) or hand it components directly
    (what the tests do). Either way it owns the pieces it was given and closes them in
    :meth:`aclose`.
    """

    def __init__(
        self,
        backend: SystemOneBackend,
        llm: ChatLLM,
        face: FaceBridge,
        *,
        session_id: str,
        run_dir: Path,
        config: AgentConfig | None = None,
        store: SqliteMemoryStore | None = None,
        transcript_log: TranscriptLog | None = None,
        policy: Policy | None = None,
        emotion: EmotionController | None = None,
        retriever: MemoryRetriever | None = None,
        reply_generator: ReplyGenerator | None = None,
        memory_writer: MemoryWriter | None = None,
        face_model: AuthoredFaceModel | None = None,
        recorder: FixtureRecorder | None = None,
        session_epoch: datetime | None = None,
    ) -> None:
        self.config = (config or AgentConfig()).validated()
        self.session_id = session_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.backend = backend
        self.llm = llm
        self.face = face
        self.recorder = recorder

        policy_config = self.config.policy_config()
        self.policy = policy or Policy(policy_config)
        self.session_epoch = session_epoch or datetime.now(UTC)
        self.emotion = emotion or EmotionController(policy_config, started_at=self.session_epoch)
        self.face_model = face_model or self.config.build_face_model()

        self.store = (
            store
            if store is not None
            else SqliteMemoryStore(self.config.memory_db_path(session_id))
        )
        self.transcript = transcript_log or TranscriptLog(self.run_dir / "transcript.jsonl")
        self.retriever = retriever or MemoryRetriever(self.store)
        self.replies = reply_generator or ReplyGenerator(llm)
        self.memory_writer = memory_writer or MemoryWriter(
            llm,
            backend,
            self.store,
            policy_config,
            deadline_ms=self.config.memory_deadline_ms,
        )

        self._decisions = _Jsonl(self.run_dir / "decisions.jsonl")
        self._memories = _Jsonl(self.run_dir / "memories.jsonl")

        self._window: deque[Utterance] = deque(maxlen=RECENT_WINDOW)
        self._speaking = False
        self._last_robot_end_s: float | None = None
        self._speak_index = 0
        self._seq = 0

        self._memory_tasks: set[asyncio.Task[MemoryWriteOutcome]] = set()
        self._memory_gate = asyncio.Semaphore(self.config.memory_concurrency)

        #: Everything the run produced, kept for the summary and for tests.
        self.turns: list[TurnRecord] = []
        self.memory_outcomes: list[MemoryWriteOutcome] = []
        self.speech_plans: list[Path] = []
        self.emotion_timeline: list[tuple[str, str]] = []
        self.started_at = datetime.now(UTC)
        self.finished_at: datetime | None = None
        self.resolved_model: str | None = None
        self.skipped_partials = 0
        self._closed = False

    # -- construction -----------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        session_id: str | None = None,
        backend: SystemOneBackend | None = None,
        llm: ChatLLM | None = None,
        face: FaceBridge | None = None,
        recorder: FixtureRecorder | None = None,
    ) -> ListeningAgent:
        """Build every component the config names. Overrides are for tests and the CLI."""
        config = config.validated()
        resolved_session = session_id or config.resolved_session_id()
        run_dir = config.run_dir(resolved_session)
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_backend = backend if backend is not None else config.build_backend()
        if recorder is not None and isinstance(resolved_backend, SystemOneClient):
            resolved_backend.on_call = recorder
        return cls(
            resolved_backend,
            llm if llm is not None else config.build_llm(),
            face if face is not None else config.build_face(run_dir),
            session_id=resolved_session,
            run_dir=run_dir,
            config=config,
            recorder=recorder,
        )

    # -- the session clock ------------------------------------------------------------

    def moment(self, utterance: Utterance) -> datetime:
        """The session-timeline instant an utterance ends, as an aware datetime."""
        return self.session_epoch + timedelta(seconds=utterance.t_end_s)

    # -- context assembly -------------------------------------------------------------

    def _robot_state(self, utterance: Utterance) -> RobotState:
        """What the robot itself is doing, as the state render will report it.

        ``speaking`` is the real flag, not an estimate: it is true only while this agent
        is inside the reply path. One utterance is handled at a time, so in a replay it
        is never true at decision time — which is honest, because the harness synthesises
        no audio and cannot know when speech would really have ended. ``last_spoke_age_s``
        is measured from the *estimated* end of the last ``ROBOT`` turn (see
        :data:`ROBOT_WORDS_PER_SECOND`) and clamped at zero while that turn is still
        notionally running.
        """
        age: float | None = None
        if self._last_robot_end_s is not None:
            age = max(0.0, utterance.t_end_s - self._last_robot_end_s)
        return RobotState(
            speaking=self._speaking,
            last_spoke_age_s=age,
            current_emotion=self.emotion.current().preset,
        )

    def build_context(self, utterance: Utterance) -> DecisionContext:
        """Assemble the one thing a bundle call is allowed to see."""
        try:
            memories: tuple[MemoryRecord, ...] = self.retriever.retrieve(utterance)
        # Retrieval is a convenience, not a gate: a broken store must not cost a decision.
        except Exception as error:
            _LOGGER.warning("memory retrieval failed for %s: %s", utterance.utterance_id, error)
            memories = ()
        return DecisionContext(
            robot_name=self.config.robot_name,
            session_id=self.session_id,
            recent=tuple(self._window),
            current=utterance,
            retrieved_memories=memories,
            robot_state=self._robot_state(utterance),
        )

    # -- the bundle call --------------------------------------------------------------

    async def _ask(self, ctx: DecisionContext) -> tuple[Any, float, str | None]:
        """Ask the frozen bundle under the deadline. Never raises; ``None`` means WAIT."""
        state = render_state(ctx)
        questions = QUESTION_BUNDLE_V1.to_wire()
        started = time.perf_counter()
        try:
            answers = await asyncio.wait_for(
                self.backend.aask(state, questions),
                timeout=self.config.deadline_ms / 1000.0,
            )
        except TimeoutError:
            elapsed = (time.perf_counter() - started) * 1000.0
            _LOGGER.warning(
                "System One deadline of %d ms exceeded for %s",
                self.config.deadline_ms,
                ctx.current.utterance_id,
            )
            return None, elapsed, f"deadline of {self.config.deadline_ms} ms exceeded"
        # Anything at all from the backend means silence, not a crash: this is the
        # fail-closed seam the whole policy is built around.
        except Exception as error:
            elapsed = (time.perf_counter() - started) * 1000.0
            _LOGGER.warning(
                "System One call failed for %s: %s: %s",
                ctx.current.utterance_id,
                type(error).__name__,
                error,
            )
            return None, elapsed, f"{type(error).__name__}: {error}"
        elapsed = (time.perf_counter() - started) * 1000.0
        self.resolved_model = answers.model
        return answers, elapsed, None

    # -- the face ---------------------------------------------------------------------

    def _emit_affect(
        self, decision: Decision, utterance: Utterance
    ) -> tuple[AffectCue | None, FaceTarget | None, bool]:
        previous = self.emotion.current().preset
        cue = self.emotion.apply(decision, self.moment(utterance))
        if cue is None:
            return None, None, False
        changed = cue.preset != previous
        if changed:
            self.emotion_timeline.append((f"{utterance.t_end_s:.2f}", cue.preset))
        target = self.face_model.apply(cue, utterance.t_end_s)
        try:
            self.face.emit_affect(cue)
        # A broken preview must not cost the conversation.
        except Exception as error:
            _LOGGER.warning("face bridge refused a cue: %s", error)
        return cue, target, changed

    # -- memory -----------------------------------------------------------------------

    def _schedule_memory(self, decision: Decision, ctx: DecisionContext) -> bool:
        """Start the background write for a memory-worthy turn. Returns whether it ran.

        The policy expresses "worth remembering" by naming a ``memory_kind``; a
        sensitive one that config does not allow arrives here as ``remember=False`` with
        the kind still set. Both are scheduled, because a withheld memory is an outcome
        the run has to account for — and the writer makes no model call for it.
        """
        if decision.memory_kind is None:
            return False

        async def run() -> MemoryWriteOutcome:
            async with self._memory_gate:
                outcome = await self.memory_writer.write(decision, ctx)
            self.memory_outcomes.append(outcome)
            self._memories.append(outcome.model_dump(mode="json"))
            _LOGGER.info(
                "memory %s for %s (%s)",
                outcome.outcome.value,
                outcome.utterance_id,
                outcome.reason or "-",
            )
            return outcome

        task = asyncio.create_task(run(), name=f"memory-{decision.utterance_id}")
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_tasks.discard)
        return True

    async def drain_memories(self) -> None:
        """Wait for every scheduled memory write. Called before shutdown, and by tests."""
        while self._memory_tasks:
            pending = tuple(self._memory_tasks)
            await asyncio.gather(*pending, return_exceptions=True)

    # -- speaking ---------------------------------------------------------------------

    def _robot_turn(self, trigger: Utterance, text: str) -> Utterance:
        words = max(1, len(text.split()))
        duration = max(_MIN_ROBOT_TURN_S, words / ROBOT_WORDS_PER_SECOND)
        start = trigger.t_end_s
        return Utterance(
            utterance_id=f"robot-{self._speak_index:04d}",
            session_id=self.session_id,
            speaker_label=ROBOT_SPEAKER_LABEL,
            text=text,
            t_start_s=start,
            t_end_s=start + duration,
            audio=None,
            is_final=True,
        )

    async def _speak(
        self, ctx: DecisionContext, cue: AffectCue
    ) -> tuple[str | None, list[SpeechClauseOut], Path | None, float, str | None]:
        """Generate, emit and export one reply. Returns ``(text, clauses, plan, ms, error)``."""
        started = time.perf_counter()
        try:
            reply = await self.replies.reply(ctx, ctx.retrieved_memories)
        except Exception as error:
            elapsed = (time.perf_counter() - started) * 1000.0
            _LOGGER.warning(
                "reply generation failed for %s: %s: %s",
                ctx.current.utterance_id,
                type(error).__name__,
                error,
            )
            return None, [], None, elapsed, f"{type(error).__name__}: {error}"
        elapsed = (time.perf_counter() - started) * 1000.0

        self._speak_index += 1
        generation_id = f"robot-{self._speak_index:04d}"
        try:
            texts = split_clauses(reply)
            clauses = build_speech_clauses(generation_id, texts, cue, self.config.seed)
        except Exception as error:
            _LOGGER.warning("clause split failed for %s: %s", ctx.current.utterance_id, error)
            self._speak_index -= 1
            return None, [], None, elapsed, f"{type(error).__name__}: {error}"

        try:
            self.face.emit_clauses(clauses)
        except Exception as error:
            _LOGGER.warning("face bridge refused the clauses: %s", error)

        plan_path: Path | None = None
        try:
            plan = export_speech_plan(
                ctx.current.utterance_id,
                "rt-agent",
                clauses,
                cue,
                self.config.voice,
                self.config.seed,
            )
            plan_path = write_speech_plan(
                self.run_dir / f"speech-plan-{self._speak_index}.json", plan, overwrite=True
            )
            self.speech_plans.append(plan_path)
        except Exception as error:
            _LOGGER.warning("speech-plan export failed for %s: %s", ctx.current.utterance_id, error)

        return reply, clauses, plan_path, elapsed, None

    # -- one turn ---------------------------------------------------------------------

    async def handle(self, utterance: Utterance) -> TurnRecord:
        """Run the whole decision path for one finalized utterance. Never raises."""
        token = CURRENT_UTTERANCE.set(utterance.utterance_id)
        try:
            return await self._handle(utterance)
        finally:
            CURRENT_UTTERANCE.reset(token)

    async def _handle(self, utterance: Utterance) -> TurnRecord:
        turn_started = time.perf_counter()
        self.transcript.append(utterance)

        ctx = self.build_context(utterance)
        answers, bundle_ms, backend_error = await self._ask(ctx)

        policy_started = time.perf_counter()
        decision = self.policy.decide(ctx, answers, bundle_ms, now=self.moment(utterance))
        policy_ms = (time.perf_counter() - policy_started) * 1000.0

        cue, target, changed = self._emit_affect(decision, utterance)
        scheduled = self._schedule_memory(decision, ctx)

        reply: str | None = None
        clauses: list[SpeechClauseOut] = []
        plan_path: Path | None = None
        llm_ms: float | None = None
        reply_error: str | None = None
        if decision.speak:
            self._speaking = True
            try:
                reply, clauses, plan_path, llm_ms, reply_error = await self._speak(
                    ctx, self.emotion.current()
                )
            finally:
                self._speaking = False

        self._window.append(utterance)
        if reply is not None:
            robot_turn = self._robot_turn(utterance, reply)
            self.transcript.append(robot_turn)
            self._window.append(robot_turn)
            self._last_robot_end_s = robot_turn.t_end_s

        record = TurnRecord(
            seq=self._seq,
            session_id=self.session_id,
            utterance=utterance,
            decision=decision,
            spoke=reply is not None,
            reply=reply,
            clause_count=len(clauses),
            speech_plan=str(plan_path) if plan_path is not None else None,
            emotion_changed=changed,
            emitted_cue=cue,
            face_target=target,
            retrieved_memory_ids=tuple(item.memory_id for item in ctx.retrieved_memories),
            memory_scheduled=scheduled,
            bundle_ms=bundle_ms,
            policy_ms=policy_ms,
            llm_ms=llm_ms,
            total_ms=(time.perf_counter() - turn_started) * 1000.0,
            backend_error=backend_error,
            reply_error=reply_error,
            recorded_at=datetime.now(UTC),
        )
        self._seq += 1
        self.turns.append(record)
        self._decisions.append(record.model_dump(mode="json"))
        _LOGGER.info(
            "%s %s %s (%s, %.0f ms)",
            utterance.utterance_id,
            "SPEAK" if decision.speak else "WAIT",
            decision.wait_reason or "-",
            decision.emotion_preset,
            record.total_ms,
        )
        return record

    # -- the whole session ------------------------------------------------------------

    async def run(self, source: AsyncIterator[Utterance] | Iterable[Utterance]) -> RunSummary:
        """Consume a stream of utterances to exhaustion and write ``summary.json``."""
        if isinstance(source, AsyncIterator):
            async for utterance in source:
                await self._maybe_handle(utterance)
        else:
            for utterance in source:
                await self._maybe_handle(utterance)
        await self.drain_memories()
        return self.write_summary()

    async def _maybe_handle(self, utterance: Utterance) -> None:
        if not utterance.is_final:
            # Partial hypotheses are for a UI, never for a decision.
            self.skipped_partials += 1
            return
        await self.handle(utterance)

    # -- the summary ------------------------------------------------------------------

    def summary(self) -> RunSummary:
        """Everything the run came to, as a ``run-summary/v1``."""
        finished = self.finished_at or datetime.now(UTC)
        wait_reasons = Counter(
            turn.decision.wait_reason
            for turn in self.turns
            if turn.decision.wait_reason is not None
        )
        outcomes = Counter(outcome.outcome.value for outcome in self.memory_outcomes)
        bundle = [turn.bundle_ms for turn in self.turns]
        total = [turn.total_ms for turn in self.turns]
        return RunSummary(
            session_id=self.session_id,
            started_at=self.started_at,
            finished_at=finished,
            wall_seconds=max(0.0, (finished - self.started_at).total_seconds()),
            backend=self.config.backend,
            backend_model_requested=self.config.backend_model_id(),
            backend_model_resolved=self.resolved_model,
            llm=self.config.llm,
            face=self.config.face,
            utterances=len(self.turns),
            speak_count=sum(1 for turn in self.turns if turn.decision.speak),
            spoken_count=sum(1 for turn in self.turns if turn.spoke),
            wait_count=sum(1 for turn in self.turns if not turn.decision.speak),
            wait_reasons=dict(sorted(wait_reasons.items())),
            backend_failures=sum(1 for turn in self.turns if turn.backend_error is not None),
            reply_failures=sum(1 for turn in self.turns if turn.reply_error is not None),
            emotion_changes=sum(1 for turn in self.turns if turn.emotion_changed),
            emotion_timeline=tuple(self.emotion_timeline),
            emotion_counts=dict(
                sorted(Counter(turn.decision.emotion_preset for turn in self.turns).items())
            ),
            memory_outcomes=dict(sorted(outcomes.items())),
            memories_stored=sum(
                1 for outcome in self.memory_outcomes if outcome.outcome is MemoryOutcome.STORED
            ),
            speech_plans=tuple(path.name for path in self.speech_plans),
            bundle_ms_p50=round(percentile(bundle, 0.5), 1),
            bundle_ms_p95=round(percentile(bundle, 0.95), 1),
            bundle_ms_max=round(max(bundle, default=0.0), 1),
            total_ms_p50=round(percentile(total, 0.5), 1),
            total_ms_p95=round(percentile(total, 0.95), 1),
            total_ms_max=round(max(total, default=0.0), 1),
        )

    def write_summary(self) -> RunSummary:
        """Write ``summary.json`` into the run directory and return what it says."""
        self.finished_at = datetime.now(UTC)
        summary = self.summary()
        path = self.run_dir / "summary.json"
        path.write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary

    # -- lifecycle --------------------------------------------------------------------

    async def aclose(self) -> None:
        """Drain background work, then release every resource this agent owns."""
        if self._closed:
            return
        self._closed = True
        await self.drain_memories()
        for close in (self._decisions.close, self._memories.close, self.transcript.close):
            try:
                close()
            except Exception as error:  # a close that fails must not hide the others
                _LOGGER.warning("close failed: %s", error)
        try:
            self.face.close()
        except Exception as error:
            _LOGGER.warning("face close failed: %s", error)
        try:
            self.store.close()
        except Exception as error:
            _LOGGER.warning("store close failed: %s", error)
        aclose = getattr(self.backend, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception as error:
                _LOGGER.warning("backend close failed: %s", error)
        llm_close = getattr(self.llm, "aclose", None)
        if llm_close is not None:
            try:
                await llm_close()
            except Exception as error:
                _LOGGER.warning("llm close failed: %s", error)

    async def __aenter__(self) -> Self:
        """Enter an ``async with`` block."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close everything on the way out."""
        await self.aclose()
