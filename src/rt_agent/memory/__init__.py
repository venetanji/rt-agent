"""The persistent layer: the transcript log, the memory store, the writer and retrieval.

alice has no memory at all; this package is the whole of it. Three pieces, deliberately
separate: every turn goes to the :class:`~rt_agent.memory.transcript_log.TranscriptLog`,
a very small number of gated sentences go to a
:class:`~rt_agent.memory.store.SqliteMemoryStore`, and at most three of those come back
through the :class:`~rt_agent.memory.retrieval.MemoryRetriever` to accompany a decision.
"""

from __future__ import annotations

from rt_agent.memory.retrieval import (
    DEFAULT_HALF_LIFE_S,
    STOPWORDS,
    CandidateSource,
    MemoryRetriever,
    RetrievalWeights,
    keyword_overlap,
    rank_memories,
    recency_score,
    score_memory,
    tokenize,
)
from rt_agent.memory.store import InMemoryStore, SqliteMemoryStore
from rt_agent.memory.transcript_log import TranscriptLog, TranscriptLogError
from rt_agent.memory.writer import (
    DEFAULT_EXCERPT_TURNS,
    MEMORY_WRITE_VERSION,
    MemoryOutcome,
    MemoryWriteOutcome,
    MemoryWriter,
    excerpt_for,
    render_faithfulness_state,
)

__all__ = [
    "DEFAULT_EXCERPT_TURNS",
    "DEFAULT_HALF_LIFE_S",
    "MEMORY_WRITE_VERSION",
    "STOPWORDS",
    "CandidateSource",
    "InMemoryStore",
    "MemoryOutcome",
    "MemoryRetriever",
    "MemoryWriteOutcome",
    "MemoryWriter",
    "RetrievalWeights",
    "SqliteMemoryStore",
    "TranscriptLog",
    "TranscriptLogError",
    "excerpt_for",
    "keyword_overlap",
    "rank_memories",
    "recency_score",
    "render_faithfulness_state",
    "score_memory",
    "tokenize",
]
