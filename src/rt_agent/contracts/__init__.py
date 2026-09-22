"""Every cross-package contract in rt-agent: frozen pydantic v2 models and protocols.

Import from here (``from rt_agent.contracts import Utterance``); the module split is an
implementation detail.
"""

from __future__ import annotations

from rt_agent.contracts.affect import AffectCue
from rt_agent.contracts.base import (
    AffectVector,
    ClauseText,
    FrozenModel,
    Probability,
    Seed32,
    Sequence32,
    Signed,
    Unit,
)
from rt_agent.contracts.decision import (
    PROBABILITY_SUM_TOLERANCE,
    Addressee,
    BundleAnswers,
    ChoiceAnswer,
    Decision,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)
from rt_agent.contracts.events import (
    MAX_RETRIEVED_MEMORIES,
    RECENT_WINDOW,
    ROBOT_SPEAKER_LABEL,
    AudioEvidence,
    DecisionContext,
    RobotState,
    Transcription,
    Utterance,
)
from rt_agent.contracts.memory import MEMORY_KINDS, MemoryKind, MemoryRecord
from rt_agent.contracts.protocols import (
    AudioSource,
    ChatLLM,
    ChatMessage,
    Diarizer,
    FaceBridge,
    MemoryStore,
    QuestionsWire,
    SystemOneBackend,
    Transcriber,
    VoiceActivityDetector,
)
from rt_agent.contracts.speech import MAX_CLAUSES, SpeechClauseOut

__all__ = [
    "MAX_CLAUSES",
    "MAX_RETRIEVED_MEMORIES",
    "MEMORY_KINDS",
    "PROBABILITY_SUM_TOLERANCE",
    "RECENT_WINDOW",
    "ROBOT_SPEAKER_LABEL",
    "Addressee",
    "AffectCue",
    "AffectVector",
    "AudioEvidence",
    "AudioSource",
    "BundleAnswers",
    "ChatLLM",
    "ChatMessage",
    "ChoiceAnswer",
    "ClauseText",
    "Decision",
    "DecisionContext",
    "Diarizer",
    "FaceBridge",
    "FrozenModel",
    "MemoryKind",
    "MemoryRecord",
    "MemoryStore",
    "NoulAnswer",
    "Probability",
    "QuestionsWire",
    "RobotState",
    "ScoreAnswer",
    "Seed32",
    "Sequence32",
    "Signed",
    "SpeechClauseOut",
    "SystemOneBackend",
    "Transcriber",
    "Transcription",
    "Unit",
    "Usage",
    "Utterance",
    "VoiceActivityDetector",
]
