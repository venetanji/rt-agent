"""Policy v1 — calibrated answers in, one ``decision/v1`` out. Fail-closed by construction.

Every path that is not a confident, complete, well-formed YES ends in
``speak=False`` with a named ``wait_reason``. There is no "probably fine" branch: a
missing answer, a malformed distribution, a blown deadline or an unknown preset all
produce the same silence as an explicit "do not speak".

Memory withholding is expressed as ``remember=False`` with ``sensitive=True`` and a
``memory_kind`` still set: that combination is what the harness logs as "withheld".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rt_agent.contracts.affect import AffectCue
from rt_agent.contracts.decision import Addressee, BundleAnswers, Decision
from rt_agent.contracts.events import DecisionContext
from rt_agent.contracts.memory import MEMORY_KINDS, MemoryKind
from rt_agent.policy.presets import DEFAULT_PRESET, EMOTION_PRESETS, PRESETS_VERSION
from rt_agent.systemone.bundle import (
    BUNDLE_VERSION,
    Q_ADDRESSED_TO_ROBOT,
    Q_ADDRESSEE,
    Q_EMOTION,
    Q_EMOTION_INTENSITY,
    Q_INTELLIGIBLE_COMPLETE,
    Q_INVITES_RESPONSE_NOW,
    Q_MEMORY_KIND,
    Q_SENSITIVE_PERSONAL,
    Q_STAY_QUIET_SAFETY,
    Q_WORTH_REMEMBERING,
)

__all__ = [
    "POLICY_VERSION",
    "Policy",
    "PolicyConfig",
    "WaitReason",
]

POLICY_VERSION = "policy/v1"

#: Every reason the robot can stay quiet. These strings end up in out/decisions.jsonl.
WaitReason = Literal[
    "backend_unavailable",
    "deadline_exceeded",
    "malformed_answers",
    "robot_speaking",
    "safety_quiet",
    "not_intelligible",
    "not_addressed",
    "no_invitation",
]

WAIT_BACKEND_UNAVAILABLE: WaitReason = "backend_unavailable"
WAIT_DEADLINE_EXCEEDED: WaitReason = "deadline_exceeded"
WAIT_MALFORMED_ANSWERS: WaitReason = "malformed_answers"
WAIT_ROBOT_SPEAKING: WaitReason = "robot_speaking"
WAIT_SAFETY_QUIET: WaitReason = "safety_quiet"
WAIT_NOT_INTELLIGIBLE: WaitReason = "not_intelligible"
WAIT_NOT_ADDRESSED: WaitReason = "not_addressed"
WAIT_NO_INVITATION: WaitReason = "no_invitation"

_VALID_ADDRESSEES = frozenset({"robot", "other_human", "self_or_group", "unclear"})


class PolicyConfig(BaseSettings):
    """Thresholds for policy v1. Every field is overridable via ``RT_AGENT_*`` env vars.

    The thresholds are bench values, not calibrated accuracy guarantees, and System One
    answers are only stable to about +/- 0.02 — so none of them sits on a knife edge
    next to a measured value.
    """

    model_config = SettingsConfigDict(env_prefix="RT_AGENT_", frozen=True, extra="forbid")

    #: Hard budget for one bundle call. Anything slower is treated as unavailable.
    deadline_ms: int = Field(default=1500, ge=1)

    # Admission
    intelligible_min: float = Field(default=0.6, ge=0.0, le=1.0)
    addressed_min: float = Field(default=0.7, ge=0.0, le=1.0)
    invites_min: float = Field(default=0.6, ge=0.0, le=1.0)
    safety_quiet_max: float = Field(default=0.5, ge=0.0, le=1.0)

    # Emotion
    emotion_confidence_min: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion_refractory_s: float = Field(default=2.0, ge=0.0)
    cue_valid_for_ms: int = Field(default=1500, ge=0)
    emotion_decay_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    emotion_min_visible_intensity: float = Field(default=0.05, ge=0.0, le=1.0)

    # Memory
    worth_remembering_min: float = Field(default=0.75, ge=0.0, le=1.0)
    sensitive_min: float = Field(default=0.5, ge=0.0, le=1.0)
    memory_faithful_min: float = Field(default=0.7, ge=0.0, le=1.0)
    allow_sensitive: bool = False


class Policy:
    """Turns one bundle of answers into one decision. Pure: no clock, no I/O, no state."""

    def __init__(
        self,
        config: PolicyConfig | None = None,
        *,
        policy_version: str = POLICY_VERSION,
        bundle_version: str = BUNDLE_VERSION,
    ) -> None:
        self.config = config or PolicyConfig()
        self.policy_version = policy_version
        self.bundle_version = bundle_version
        self.presets_version = PRESETS_VERSION

    # -- helpers ------------------------------------------------------------------

    def _cue(
        self,
        preset: str,
        intensity: float,
        confidence: float,
        now: datetime,
    ) -> AffectCue:
        return AffectCue(
            vector=EMOTION_PRESETS[preset],
            intensity=intensity,
            preset=preset,
            source_confidence=confidence,
            valid_for_ms=self.config.cue_valid_for_ms,
            issued_at=now,
        )

    def _fail_closed(
        self,
        ctx: DecisionContext,
        reason: WaitReason,
        now: datetime,
        answers: BundleAnswers | None,
    ) -> Decision:
        """Silence, with the robot's current expression left exactly as it is."""
        held = ctx.robot_state.current_emotion
        preset = held if held in EMOTION_PRESETS else DEFAULT_PRESET
        return Decision(
            utterance_id=ctx.current.utterance_id,
            speak=False,
            wait_reason=reason,
            addressee="unclear",
            emotion_preset=preset,
            emotion_conf=0.0,
            intensity=0.0,
            affect=self._cue(preset, intensity=0.0, confidence=0.0, now=now),
            remember=False,
            memory_kind=None,
            sensitive=False,
            answers=answers,
            policy_version=self.policy_version,
            bundle_version=self.bundle_version,
            created_at=now,
        )

    # -- the decision -------------------------------------------------------------

    def decide(
        self,
        ctx: DecisionContext,
        answers: BundleAnswers | None,
        elapsed_ms: float,
        *,
        now: datetime | None = None,
    ) -> Decision:
        """Decide speak/wait, the expression and memory-worthiness for one utterance.

        ``answers`` is ``None`` when the backend errored, timed out or was never called.
        ``elapsed_ms`` is measured by the caller and includes everything the deadline
        covers, so a late-but-successful answer is still refused. ``now`` is the clock
        stamped onto the decision and its cue; tests pass it explicitly.
        """
        moment = now if now is not None else datetime.now(UTC)
        if answers is None:
            return self._fail_closed(ctx, WAIT_BACKEND_UNAVAILABLE, moment, None)
        if elapsed_ms > self.config.deadline_ms:
            return self._fail_closed(ctx, WAIT_DEADLINE_EXCEEDED, moment, answers)

        try:
            return self._decide_checked(ctx, answers, moment)
        except (KeyError, ValueError, TypeError):
            return self._fail_closed(ctx, WAIT_MALFORMED_ANSWERS, moment, answers)

    def _decide_checked(
        self, ctx: DecisionContext, answers: BundleAnswers, now: datetime
    ) -> Decision:
        config = self.config

        intelligible = answers.noul(Q_INTELLIGIBLE_COMPLETE)
        addressed = answers.noul(Q_ADDRESSED_TO_ROBOT)
        invites = answers.noul(Q_INVITES_RESPONSE_NOW)
        stay_quiet = answers.noul(Q_STAY_QUIET_SAFETY)

        addressee_answer = answers.choice(Q_ADDRESSEE)
        if addressee_answer.chosen not in _VALID_ADDRESSEES:
            raise ValueError(f"unknown addressee {addressee_answer.chosen!r}")
        addressee = cast(Addressee, addressee_answer.chosen)

        emotion_answer = answers.choice(Q_EMOTION)
        if emotion_answer.chosen not in EMOTION_PRESETS:
            raise ValueError(f"unknown emotion preset {emotion_answer.chosen!r}")
        preset = emotion_answer.chosen
        emotion_conf = emotion_answer.confidence
        intensity = answers.score(Q_EMOTION_INTENSITY).normalized

        # -- speak / wait, in a fixed order so the reason is deterministic ----------
        wait_reason: WaitReason | None = None
        if ctx.robot_state.speaking:
            wait_reason = WAIT_ROBOT_SPEAKING
        elif stay_quiet >= config.safety_quiet_max:
            wait_reason = WAIT_SAFETY_QUIET
        elif intelligible < config.intelligible_min:
            wait_reason = WAIT_NOT_INTELLIGIBLE
        elif addressed < config.addressed_min:
            wait_reason = WAIT_NOT_ADDRESSED
        elif invites < config.invites_min:
            wait_reason = WAIT_NO_INVITATION

        # A safety hold overrides the chosen expression: the robot looks concerned and
        # says nothing.
        if wait_reason == WAIT_SAFETY_QUIET:
            preset = "concerned"
            emotion_conf = max(emotion_conf, stay_quiet)
            intensity = max(intensity, 0.5)

        # -- memory -----------------------------------------------------------------
        worth = answers.noul(Q_WORTH_REMEMBERING)
        sensitive_p = answers.noul(Q_SENSITIVE_PERSONAL)
        kind_answer = answers.choice(Q_MEMORY_KIND)
        if kind_answer.chosen not in MEMORY_KINDS:
            raise ValueError(f"unknown memory kind {kind_answer.chosen!r}")
        worth_enough = worth >= config.worth_remembering_min
        sensitive = sensitive_p >= config.sensitive_min
        memory_kind: MemoryKind | None = kind_answer.chosen if worth_enough else None
        remember = worth_enough and (not sensitive or config.allow_sensitive)

        return Decision(
            utterance_id=ctx.current.utterance_id,
            speak=wait_reason is None,
            wait_reason=wait_reason,
            addressee=addressee,
            emotion_preset=preset,
            emotion_conf=emotion_conf,
            intensity=intensity,
            affect=self._cue(preset, intensity, emotion_conf, now),
            remember=remember,
            memory_kind=memory_kind,
            sensitive=sensitive,
            answers=answers,
            policy_version=self.policy_version,
            bundle_version=self.bundle_version,
            created_at=now,
        )

    # -- memory gate --------------------------------------------------------------

    def memory_is_faithful(self, faithfulness_p: float) -> bool:
        """Second-call gate: a written memory is kept only if it is supported."""
        return faithfulness_p >= self.config.memory_faithful_min
