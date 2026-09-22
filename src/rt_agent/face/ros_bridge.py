"""The ROS 2 seam to alice — importable without ROS, unusable without it.

This sandbox has no ``rclpy`` and no ``alice_interfaces``. Importing this module must
therefore never fail: every ROS import is lazy and guarded, and ``RosFaceBridge`` /
``RunSpeechClient`` raise :class:`RosUnavailableError` **at construction** with a message
that says exactly what is missing and where to get it.

What *is* usable everywhere is the message-dict layer: :func:`speech_clause_msg_dict` and
:func:`affect_cue_msg_dict` are pure functions producing the exact field sets of the
corresponding ``.msg`` definitions, so the wire shape can be unit-tested, diffed and
reviewed on a laptop. :func:`apply_msg_fields` then pours such a dict into a real ROS
message object, and is itself testable against any plain object with the right attributes.

Two ``.msg`` types are involved:

* ``alice_interfaces/msg/SpeechClause`` — **exists today**, the designed external entry
  point on ``/alice/run/committed_clauses``.
* ``alice_interfaces/msg/AffectCue`` — **proposed**, does not exist in alice yet. See
  :data:`AFFECT_CUE_MSG` and ``docs/alice-integration.md`` for the four ROS changes it
  needs. Nothing here will work against today's alice until those land.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

from pydantic import Field

from rt_agent.contracts import AffectCue, FrozenModel, SpeechClauseOut

__all__ = [
    "ACTION_RUN_SPEECH",
    "AFFECT_CUE_FIELDS",
    "AFFECT_CUE_MSG",
    "MAX_AFFECT_VALID_NS",
    "MAX_TRANSPORT_AGE_NS",
    "SPEECH_CLAUSE_FIELDS",
    "STREAM_HEADER_FIELDS",
    "TOPIC_AFFECT_CUE",
    "TOPIC_COMMITTED_CLAUSES",
    "RosFaceBridge",
    "RosUnavailableError",
    "RunIdentity",
    "RunSpeechClient",
    "affect_cue_msg_dict",
    "apply_msg_fields",
    "speech_clause_msg_dict",
    "stream_header_dict",
]

#: The external committed-clause seam that exists in alice today (RELIABLE, depth 128).
TOPIC_COMMITTED_CLAUSES = "/alice/run/committed_clauses"

#: The proposed idle-affect topic. Not present in alice today.
TOPIC_AFFECT_CUE = "/alice/affect/cue"

#: The one action server, owned by ``SessionNode``.
ACTION_RUN_SPEECH = "/alice/run_speech"

#: ``MAX_TRANSPORT_AGE_NS`` in ``alice_nodes/transport.py``: 250 ms, checked twice — once
#: at receipt and again when the session relays the clause onward.
MAX_TRANSPORT_AGE_NS = 250_000_000

#: The proposal bounds an affect cue's lifetime at 2 s.
MAX_AFFECT_VALID_NS = 2_000_000_000

#: ``RunSpeech.Goal.COMMITTED_CLAUSES``.
RUN_SPEECH_SOURCE_FIXTURE = 0
RUN_SPEECH_SOURCE_COMMITTED_CLAUSES = 1

#: Field order of ``alice_interfaces/msg/StreamHeader``.
STREAM_HEADER_FIELDS: tuple[str, ...] = (
    "identity",
    "sequence",
    "source_monotonic_ns",
    "publisher_incarnation",
)

#: Field order of ``alice_interfaces/msg/SpeechClause`` — verified against the .msg file.
SPEECH_CLAUSE_FIELDS: tuple[str, ...] = (
    "header",
    "schema_version",
    "clause_id",
    "clause_sequence",
    "text",
    "affect_vector",
    "intensity",
    "seed",
    "end_of_response",
)

#: Field order of the **proposed** ``alice_interfaces/msg/AffectCue``.
AFFECT_CUE_FIELDS: tuple[str, ...] = (
    "header",
    "schema_version",
    "affect_vector",
    "intensity",
    "valid_for_ns",
    "source_id",
    "source_confidence",
)

#: The proposed ``.msg`` file, verbatim, so the text lives next to the builder that emits it.
AFFECT_CUE_MSG = """\
StreamHeader header
string<=32 schema_version   # "affect-cue/v1"
float64[3] affect_vector    # each in [-1, 1]
float64 intensity           # [0, 1]
uint64 valid_for_ns         # bounded lifetime, <= 2e9
string<=64 source_id        # e.g. "rt-agent/jev-emotion-v1"
float64 source_confidence   # [0, 1]
"""


class RosUnavailableError(RuntimeError):
    """Raised when a ROS-only object is constructed without ``rclpy``/``alice_interfaces``."""


class RunIdentity(FrozenModel):
    """``alice_interfaces/msg/RunIdentity`` — the triple every message must carry.

    ``epoch`` is **not** yours to choose: ``SessionNode.execute`` mints a fresh ``uuid4``
    and discards whatever the goal asked for, so this must be built from the epoch read off
    the first action feedback. See :meth:`RunSpeechClient.await_identity`.
    """

    run_id: str = Field(min_length=1, max_length=128)
    epoch: str = Field(min_length=1, max_length=128)
    generation_id: str = Field(min_length=1, max_length=128)


def stream_header_dict(
    identity: RunIdentity,
    sequence: int,
    source_monotonic_ns: int,
    publisher_incarnation: str,
) -> dict[str, Any]:
    """Build ``StreamHeader`` as a plain dict.

    Guards mirrored from ``SequenceGuard`` (``alice_nodes/transport.py``): the stream is
    *exact*, so ``sequence`` must start at 0 and increase by exactly 1; the timestamp must
    be a real ``time.monotonic_ns()`` reading, never in the future and never older than
    250 ms at receipt; ``publisher_incarnation`` must equal the goal's
    ``requester_incarnation`` for the whole run, or ``SessionNode.external_clause`` raises
    "external source incarnation mismatch" and faults the run.
    """
    if sequence < 0:
        raise ValueError("stream sequence must be non-negative and start at 0")
    if source_monotonic_ns < 0:
        raise ValueError("source_monotonic_ns must be a non-negative monotonic reading")
    if not 1 <= len(publisher_incarnation) <= 128:
        raise ValueError("publisher_incarnation must be 1-128 characters")
    return {
        "identity": identity.model_dump(),
        "sequence": sequence,
        "source_monotonic_ns": source_monotonic_ns,
        "publisher_incarnation": publisher_incarnation,
    }


def speech_clause_msg_dict(
    clause: SpeechClauseOut,
    run_identity: RunIdentity,
    *,
    sequence: int | None = None,
    source_monotonic_ns: int | None = None,
    publisher_incarnation: str = "rt-agent",
) -> dict[str, Any]:
    """``alice_interfaces/msg/SpeechClause`` as a dict, field for field.

    ``sequence`` defaults to ``clause.sequence`` because the external clause stream carries
    exactly one clause per message, so the stream sequence and the clause sequence advance
    together. ``source_monotonic_ns`` defaults to ``time.monotonic_ns()`` **at call time**,
    which is the rule the whole seam turns on: stamp at publish, never at decision. A
    decision that takes 300 ms between stamping and publishing does not merely drop the
    clause — it faults the entire run at the session's relay re-check.

    Guards that reject this message once it reaches alice:

    * ``publisher_incarnation`` different from the goal's ``requester_incarnation``;
    * a ``run_id``/``epoch``/``generation_id`` that is not the current admitted identity
      (silently dropped, not faulted);
    * ``sequence`` that is not exactly the next one, a duplicate, or backwards;
    * ``source_monotonic_ns`` in the future, or older than 250 ms at receipt **or** at relay;
    * ``ClauseSequence``: out-of-order ``clause_sequence``, duplicate ``clause_id``,
      a changed ``generation_id``, more than 4000 characters in the response, or anything
      at all after ``end_of_response``;
    * pydantic bounds: ``clause_sequence`` above 31, ``affect_vector`` not three finite
      values in [-1, 1], ``intensity`` outside [0, 1], empty or >1000-character ``text``,
      ``seed`` outside 0…2³²-1, an unknown ``schema_version``.
    """
    if run_identity.generation_id != clause.generation_id:
        raise ValueError(
            f"clause generation_id {clause.generation_id!r} does not match the run identity "
            f"{run_identity.generation_id!r}; alice takes generation_id from the header"
        )
    header = stream_header_dict(
        run_identity,
        clause.sequence if sequence is None else sequence,
        time.monotonic_ns() if source_monotonic_ns is None else source_monotonic_ns,
        publisher_incarnation,
    )
    return {
        "header": header,
        "schema_version": clause.schema_version,
        "clause_id": clause.clause_id,
        "clause_sequence": clause.sequence,
        "text": clause.text,
        "affect_vector": [float(axis) for axis in clause.vector],
        "intensity": float(clause.intensity),
        "seed": int(clause.seed),
        "end_of_response": bool(clause.end_of_response),
    }


def affect_cue_msg_dict(
    cue: AffectCue,
    run_identity: RunIdentity,
    *,
    sequence: int = 0,
    source_monotonic_ns: int | None = None,
    publisher_incarnation: str = "rt-agent",
) -> dict[str, Any]:
    """The **proposed** ``alice_interfaces/msg/AffectCue`` as a dict.

    Nothing in alice subscribes to this today; see :data:`AFFECT_CUE_MSG`. Once the
    proposal lands, the cue is routed through ``session`` like the committed-clause topic
    and admitted by the unchanged ``admit_header(header, "session", "affect", sparse=True)``,
    so the same 250 ms freshness, identity, incarnation and ordering guards apply — plus
    the proposal's own bound that ``valid_for_ns`` may not exceed 2 s.

    ``AffectCue.valid_for_ms`` is converted to nanoseconds here; a cue asking to be valid
    for longer than 2 s is rejected locally rather than faulting a run on the robot.
    """
    valid_for_ns = int(cue.valid_for_ms) * 1_000_000
    if valid_for_ns > MAX_AFFECT_VALID_NS:
        raise ValueError(
            f"valid_for_ns {valid_for_ns} exceeds the proposed {MAX_AFFECT_VALID_NS} ns bound"
        )
    if len(cue.source_id) > 64:
        raise ValueError("source_id is capped at 64 characters on the wire")
    header = stream_header_dict(
        run_identity,
        sequence,
        time.monotonic_ns() if source_monotonic_ns is None else source_monotonic_ns,
        publisher_incarnation,
    )
    return {
        "header": header,
        "schema_version": "affect-cue/v1",
        "affect_vector": [float(axis) for axis in cue.vector],
        "intensity": float(cue.intensity),
        "valid_for_ns": valid_for_ns,
        "source_id": cue.source_id,
        "source_confidence": float(cue.source_confidence),
    }


def apply_msg_fields(msg: Any, payload: Mapping[str, Any]) -> Any:
    """Pour a message dict into a ROS message object, recursing into nested messages.

    Pure with respect to this package: it only reads ``payload`` and writes attributes, so
    it can be exercised against a stub object with the same attribute names.
    """
    for name, value in payload.items():
        if not hasattr(msg, name):
            raise AttributeError(f"{type(msg).__name__} has no field {name!r}")
        if isinstance(value, Mapping):
            apply_msg_fields(getattr(msg, name), value)
        else:
            setattr(msg, name, value)
    return msg


_MISSING_ROS = (
    "ROS 2 is not available in this process: {reason}. "
    "rt_agent.face.ros_bridge needs 'rclpy' and the generated 'alice_interfaces' package "
    "on PYTHONPATH, which means running inside alice's ROS container "
    "(infra/ros2/deploy.py up, then the 'tools' service) or an environment that has "
    "sourced ros2_ws/install/setup.bash. Everywhere else, use JsonlFaceBridge or "
    "UdpJsonFaceBridge, or export a speech-plan/v1 file with "
    "rt_agent.face.speech_plan.export_speech_plan."
)


def _require_ros() -> tuple[Any, Any]:
    """Import ``rclpy`` and ``alice_interfaces`` or raise a message that says what to do."""
    try:
        import rclpy
        from alice_interfaces import msg as alice_msg
    except ImportError as exc:
        raise RosUnavailableError(_MISSING_ROS.format(reason=exc)) from exc
    return rclpy, alice_msg


def _require_action() -> Any:
    """Import the ``RunSpeech`` action type, or raise the same guided error."""
    try:
        from alice_interfaces.action import RunSpeech
    except ImportError as exc:
        raise RosUnavailableError(_MISSING_ROS.format(reason=exc)) from exc
    return RunSpeech


class RosFaceBridge:
    """A ``FaceBridge`` that publishes to alice's ROS topics.

    Construction fails immediately with :class:`RosUnavailableError` when ROS is missing —
    never at import, and never silently later while a run is live.

    ``emit_clauses`` publishes ``SpeechClause`` on ``/alice/run/committed_clauses`` with a
    freshly stamped ``source_monotonic_ns`` per message. ``emit_affect`` publishes the
    **proposed** ``AffectCue`` on ``/alice/affect/cue`` and therefore reaches nothing on
    today's alice; it is a no-op unless ``affect_enabled=True`` is passed explicitly, so a
    harness cannot quietly believe it has an idle face when it does not.
    """

    def __init__(
        self,
        run_identity: RunIdentity,
        *,
        node: Any = None,
        node_name: str = "rt_agent_face",
        publisher_incarnation: str = "rt-agent",
        affect_enabled: bool = False,
        clause_topic: str = TOPIC_COMMITTED_CLAUSES,
        affect_topic: str = TOPIC_AFFECT_CUE,
    ) -> None:
        rclpy, alice_msg = _require_ros()
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

        self.run_identity = run_identity
        self.publisher_incarnation = publisher_incarnation
        self.affect_enabled = affect_enabled
        self._owns_node = node is None
        if self._owns_node and not rclpy.ok():
            rclpy.init()
        self._node = node or rclpy.create_node(node_name)
        reliable = QoSProfile(
            depth=128,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._clause_msg = alice_msg.SpeechClause
        self._clause_pub = self._node.create_publisher(
            alice_msg.SpeechClause, clause_topic, reliable
        )
        self._affect_pub: Any = None
        self._affect_msg: Any = None
        if affect_enabled:
            self._affect_msg = getattr(alice_msg, "AffectCue", None)
            if self._affect_msg is None:
                raise RosUnavailableError(
                    "alice_interfaces has no AffectCue message: the idle-affect path is a "
                    "proposal, see docs/alice-integration.md for the four ROS changes it needs."
                )
            self._affect_pub = self._node.create_publisher(self._affect_msg, affect_topic, reliable)
        self._affect_sequence = 0

    def emit_affect(self, cue: AffectCue) -> None:
        """Publish one proposed ``AffectCue``; a no-op unless ``affect_enabled``."""
        if self._affect_pub is None:
            return
        payload = affect_cue_msg_dict(
            cue,
            self.run_identity,
            sequence=self._affect_sequence,
            publisher_incarnation=self.publisher_incarnation,
        )
        self._affect_pub.publish(apply_msg_fields(self._affect_msg(), payload))
        self._affect_sequence += 1

    def emit_clauses(self, clauses: Sequence[SpeechClauseOut]) -> None:
        """Publish an ordered run of committed clauses, stamping each at publish time."""
        for clause in clauses:
            payload = speech_clause_msg_dict(
                clause,
                self.run_identity,
                publisher_incarnation=self.publisher_incarnation,
            )
            self._clause_pub.publish(apply_msg_fields(self._clause_msg(), payload))

    def close(self) -> None:
        """Destroy the node if this bridge created it."""
        if self._owns_node and self._node is not None:
            self._node.destroy_node()
            self._node = None


class RunSpeechClient:
    """Drive one bounded ``RunSpeech`` run with externally committed clauses.

    The sequence, and the reason each step exists:

    1. :meth:`send_goal` sends a ``RunSpeech`` goal with ``source=COMMITTED_CLAUSES (1)``
       and an empty ``fixture_name``. ``config_sha256`` must equal
       ``config_digest(config_root, profile)``, ``calibration_sha256`` the face manifest
       hash, and ``clock_domain_fingerprint`` must be computed **inside the same container
       and kernel namespace** as the nodes — the ``host-monotonic-zero/v2`` proof is checked
       before any state mutates. A second concurrent goal is REJECTed: ``SessionNode`` holds
       a non-blocking admission lock and there is exactly one run at a time.
    2. :meth:`await_identity` blocks on the **first action feedback message** to learn the
       authoritative epoch. This is the gotcha of the whole integration: the session mints a
       fresh ``uuid4`` epoch and throws the client's away, and it publishes one feedback
       message before it starts accepting external clauses. An epoch can never be reused.
    3. :meth:`publish_clauses` publishes each clause on ``/alice/run/committed_clauses``
       with ``sequence`` 0,1,2,… and ``source_monotonic_ns`` stamped immediately before
       ``publish``, honouring the exact-sequence rule and the 250 ms freshness rule — which
       is re-checked a second time when the session relays the clause onward
       ("external clause original source expired at relay").
    4. :meth:`result` waits for the terminal outcome.

    Every guard that can reject a message, and what it costs: an incarnation mismatch, a
    sequence that is not exactly the next one, a duplicate or backwards sequence, a
    timestamp in the future or older than 250 ms at receipt or at relay, a duplicate
    ``clause_id``, a changed ``generation_id``, a response over 4000 characters, anything
    after ``end_of_response``, ``clause_sequence`` over 31, a malformed vector or intensity,
    or a full 32-slot internal queue — all of these **fault the whole run**, because
    ``SessionNode.external_clause`` wraps the body in a bare ``except: self.fail(...)``.
    Only a wrong identity is treated gently: the message is dropped silently. The external
    phase must also complete within 10 s of START, total audio must stay under 10 s
    including the 0.3 s tail, and the run is hard-bounded at 25 s.
    """

    def __init__(
        self,
        *,
        run_id: str,
        generation_id: str,
        requester_incarnation: str,
        selected_profile: str,
        config_sha256: str,
        calibration_sha256: str,
        clock_domain_fingerprint: str,
        seed: int = 0,
        sad_hold_ms: int = 0,
        node: Any = None,
        node_name: str = "rt_agent_run_speech",
    ) -> None:
        rclpy, alice_msg = _require_ros()
        self._action_type = _require_action()
        from rclpy.action import ActionClient

        if not 0 <= sad_hold_ms <= 2000:
            raise ValueError("sad_hold_ms must be in 0..2000")
        self.run_id = run_id
        self.generation_id = generation_id
        self.requester_incarnation = requester_incarnation
        self.selected_profile = selected_profile
        self.config_sha256 = config_sha256
        self.calibration_sha256 = calibration_sha256
        self.clock_domain_fingerprint = clock_domain_fingerprint
        self.seed = seed
        self.sad_hold_ms = sad_hold_ms
        self.identity: RunIdentity | None = None

        self._rclpy = rclpy
        self._alice_msg = alice_msg
        self._owns_node = node is None
        if self._owns_node and not rclpy.ok():
            rclpy.init()
        self._node = node or rclpy.create_node(node_name)
        self._client = ActionClient(self._node, self._action_type, ACTION_RUN_SPEECH)
        self._goal_handle: Any = None
        self._bridge: RosFaceBridge | None = None

    def goal_dict(self) -> dict[str, Any]:
        """The ``RunSpeech.Goal`` field set, for review and tests without ROS.

        ``identity.epoch`` is a placeholder: the session discards it. ``hardware`` is always
        ``False`` because ROS hardware admission rejects unconditionally
        (``require_ros_hardware_visibility``) and re-rejects at the Maestro adapter factory.
        """
        return {
            "schema_version": "run-speech/v1",
            "identity": {
                "run_id": self.run_id,
                "epoch": "requested",
                "generation_id": self.generation_id,
            },
            "source": RUN_SPEECH_SOURCE_COMMITTED_CLAUSES,
            "fixture_name": "",
            "selected_profile": self.selected_profile,
            "seed": self.seed,
            "hardware": False,
            "sad_hold_ms": self.sad_hold_ms,
            "config_sha256": self.config_sha256,
            "calibration_sha256": self.calibration_sha256,
            "clock_domain_fingerprint": self.clock_domain_fingerprint,
            "requester_incarnation": self.requester_incarnation,
        }

    def send_goal(self, *, server_timeout_s: float = 5.0) -> Any:
        """Send the goal and subscribe to feedback. Returns the goal-handle future."""
        if not self._client.wait_for_server(timeout_sec=server_timeout_s):
            raise TimeoutError(f"no RunSpeech action server on {ACTION_RUN_SPEECH}")
        goal = apply_msg_fields(self._action_type.Goal(), self.goal_dict())
        return self._client.send_goal_async(goal, feedback_callback=self._on_feedback)

    def _on_feedback(self, feedback: Any) -> None:
        """Latch the authoritative epoch from the first feedback message and never again."""
        if self.identity is not None:
            return
        identity = feedback.feedback.header.identity
        self.identity = RunIdentity(
            run_id=identity.run_id,
            epoch=identity.epoch,
            generation_id=identity.generation_id,
        )

    def await_identity(self, *, timeout_s: float = 10.0) -> RunIdentity:
        """Spin until the first feedback message reveals the authoritative epoch.

        Nothing may be published before this returns: a clause addressed to the client's own
        epoch does not fault the run, it is simply dropped, and the run then fails for
        missing clauses instead — a confusing failure that costs a whole 25 s run.
        """
        deadline = time.monotonic() + timeout_s
        while self.identity is None:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "no RunSpeech feedback within "
                    f"{timeout_s}s: the authoritative epoch is only published there"
                )
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        return self.identity

    def bridge(self, **kwargs: Any) -> RosFaceBridge:
        """A :class:`RosFaceBridge` bound to the authoritative identity and this node."""
        identity = self.identity
        if identity is None:
            raise RuntimeError("call await_identity() before publishing anything")
        if self._bridge is None:
            self._bridge = RosFaceBridge(
                identity,
                node=self._node,
                publisher_incarnation=self.requester_incarnation,
                **kwargs,
            )
        return self._bridge

    def publish_clauses(self, clauses: Sequence[SpeechClauseOut]) -> None:
        """Publish the whole response, validating the local half of alice's ledger first.

        Checked here so a mistake costs a local exception rather than a faulted run:
        sequences exactly 0,1,2,…, exactly one ``end_of_response`` and it on the last
        clause, unique ``clause_id`` values, one constant ``generation_id`` equal to the
        run's, at most 32 clauses and at most 4000 characters in total.
        """
        if not clauses:
            raise ValueError("a response needs at least one clause")
        if len(clauses) > 32:
            raise ValueError("alice admits at most 32 clauses per response")
        if sum(len(clause.text) for clause in clauses) > 4000:
            raise ValueError("response text exceeds alice's 4000 character ledger bound")
        if [clause.sequence for clause in clauses] != list(range(len(clauses))):
            raise ValueError("clause sequences must be exactly 0, 1, 2, ...")
        if len({clause.clause_id for clause in clauses}) != len(clauses):
            raise ValueError("clause_id values must be unique within a response")
        if not clauses[-1].end_of_response:
            raise ValueError("the last clause must set end_of_response")
        if any(clause.end_of_response for clause in clauses[:-1]):
            raise ValueError("nothing may follow end_of_response")
        self.bridge().emit_clauses(clauses)

    def result(self, goal_handle: Any, *, timeout_s: float = 30.0) -> Any:
        """Spin until the run reaches a terminal outcome and return the ``RunSpeech.Result``.

        ``terminal_outcome`` is SUCCESS / CANCELLED / FAULT. A fault here almost always
        traces back to one rejected clause, because ``SessionNode.external_clause`` turns
        any exception into ``fail()`` for the whole run.
        """
        future = goal_handle.get_result_async()
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() > deadline:
                raise TimeoutError(f"RunSpeech did not finish within {timeout_s}s")
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        return future.result().result

    def close(self) -> None:
        """Release the action client and the node this client created."""
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        self._client.destroy()
        if self._owns_node and self._node is not None:
            self._node.destroy_node()
            self._node = None

    def __enter__(self) -> RunSpeechClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
