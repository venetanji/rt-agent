"""The face package: everything rt-agent sends to alice, plus a simulator of what she does.

Three integration paths live here, in increasing order of how much alice has to change:

* :mod:`rt_agent.face.speech_plan` — export ``speech-plan/v1`` JSON and play it with the
  host ``alice-speak`` CLI. **Zero alice changes**, and the only path that can move a real
  servo today.
* :mod:`rt_agent.face.ros_bridge` — publish ``SpeechClause`` on
  ``/alice/run/committed_clauses`` during a bounded ``RunSpeech`` run. **The seam already
  exists**; the same module sketches the *proposed* idle ``AffectCue`` path, which needs
  four small ROS changes described in ``docs/alice-integration.md``.
* :mod:`rt_agent.face.bridge` — JSONL and UDP bridges for hardware-free runs and previews.

:mod:`rt_agent.face.authored_face` predicts what alice's face would do; it is a simulator,
never the authority. :mod:`rt_agent.face.clauses` splits a reply into clauses that fit
alice's ledger bounds.
"""

from __future__ import annotations

from rt_agent.face.authored_face import (
    ANCHORS,
    BENCH_PROFILE,
    FACE_PROFILES,
    VISIBLE_PROFILE,
    AuthoredFaceModel,
    FaceProfile,
    FaceState,
    FaceTarget,
)
from rt_agent.face.bridge import (
    CompositeFaceBridge,
    JsonlFaceBridge,
    NullFaceBridge,
    UdpJsonFaceBridge,
)
from rt_agent.face.clauses import (
    MAX_CLAUSE_CHARS,
    MAX_RESPONSE_CHARS,
    build_speech_clauses,
    split_clauses,
)
from rt_agent.face.ros_bridge import (
    ACTION_RUN_SPEECH,
    AFFECT_CUE_MSG,
    TOPIC_AFFECT_CUE,
    TOPIC_COMMITTED_CLAUSES,
    RosFaceBridge,
    RosUnavailableError,
    RunIdentity,
    RunSpeechClient,
    affect_cue_msg_dict,
    speech_clause_msg_dict,
)
from rt_agent.face.speech_plan import (
    SPEECH_PLAN_SCHEMA_VERSION,
    VOICES,
    export_speech_plan,
    validate_speech_plan,
    write_speech_plan,
)

__all__ = [
    "ACTION_RUN_SPEECH",
    "AFFECT_CUE_MSG",
    "ANCHORS",
    "BENCH_PROFILE",
    "FACE_PROFILES",
    "MAX_CLAUSE_CHARS",
    "MAX_RESPONSE_CHARS",
    "SPEECH_PLAN_SCHEMA_VERSION",
    "TOPIC_AFFECT_CUE",
    "TOPIC_COMMITTED_CLAUSES",
    "VISIBLE_PROFILE",
    "VOICES",
    "AuthoredFaceModel",
    "CompositeFaceBridge",
    "FaceProfile",
    "FaceState",
    "FaceTarget",
    "JsonlFaceBridge",
    "NullFaceBridge",
    "RosFaceBridge",
    "RosUnavailableError",
    "RunIdentity",
    "RunSpeechClient",
    "UdpJsonFaceBridge",
    "affect_cue_msg_dict",
    "build_speech_clauses",
    "export_speech_plan",
    "speech_clause_msg_dict",
    "split_clauses",
    "validate_speech_plan",
    "write_speech_plan",
]
