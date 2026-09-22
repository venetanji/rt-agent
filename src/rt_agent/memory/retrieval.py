"""Retrieval: which at most three stored facts may accompany one decision.

There are no embeddings here on purpose. The PoC stores tens of short sentences, not
millions, so a transparent score beats an opaque one: keyword overlap with the current
utterance, a gentle recency term, and a boost for facts about the person who is
speaking. Every weight is visible, every tie is broken deterministically, and the same
scoring function ranks results inside the SQLite store, so FTS5 and the LIKE fallback
return the same order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from rt_agent.contracts.events import MAX_RETRIEVED_MEMORIES, DecisionContext, Utterance
from rt_agent.contracts.memory import MemoryRecord

__all__ = [
    "DEFAULT_HALF_LIFE_S",
    "STOPWORDS",
    "CandidateSource",
    "MemoryRetriever",
    "RetrievalWeights",
    "keyword_overlap",
    "rank_memories",
    "recency_score",
    "score_memory",
    "tokenize",
]


def _words(block: str) -> frozenset[str]:
    """Whitespace-separated words as a set — keeps the list below readable."""
    return frozenset(block.split())


#: Words that carry no retrieval signal in short spoken turns.
STOPWORDS = _words(
    """
    a about all am an and any are as at be been being but by can cant could did do does
    doing dont for from get got had has have he her hers him his how i if im in into is
    it its just like me might must my no nor not of off on or our out over own re said
    say see she should so some such than that the their them then there these they this
    those to too up us very was we were what when where which while who whom why will
    with would you your yours
    """
)

#: A memory loses half of its recency score after a week.
DEFAULT_HALF_LIFE_S = 7 * 24 * 3600.0

_WORD = re.compile(r"[a-z0-9']+")
_MIN_TOKEN_CHARS = 2


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase word tokens, minus stopwords, very short words and duplicates."""
    seen: dict[str, None] = {}
    for match in _WORD.finditer(text.lower()):
        token = match.group(0).strip("'")
        if len(token) < _MIN_TOKEN_CHARS or token in STOPWORDS:
            continue
        seen.setdefault(token, None)
    return tuple(seen)


def keyword_overlap(query_tokens: Sequence[str], text: str) -> float:
    """Fraction of the query's distinct keywords that appear in ``text`` (0 when empty)."""
    if not query_tokens:
        return 0.0
    tokens = set(tokenize(text))
    if not tokens:
        return 0.0
    hits = sum(1 for token in query_tokens if token in tokens)
    return hits / len(query_tokens)


def recency_score(created_at: datetime, now: datetime, half_life_s: float) -> float:
    """Exponential decay in [0, 1]: 1.0 when just written, 0.5 after one half-life."""
    if half_life_s <= 0:
        return 0.0
    age_s = max((now - created_at).total_seconds(), 0.0)
    return float(math.pow(0.5, age_s / half_life_s))


@dataclass(frozen=True)
class RetrievalWeights:
    """How the three signals are combined. Keyword overlap dominates by design."""

    keyword: float = 1.0
    recency: float = 0.3
    same_speaker: float = 0.25
    half_life_s: float = DEFAULT_HALF_LIFE_S


DEFAULT_WEIGHTS = RetrievalWeights()


def score_memory(
    record: MemoryRecord,
    query_tokens: Sequence[str],
    *,
    now: datetime,
    speaker_label: str | None = None,
    weights: RetrievalWeights = DEFAULT_WEIGHTS,
) -> float:
    """Score one memory against a tokenized query. Higher is more relevant."""
    overlap = keyword_overlap(query_tokens, record.text)
    recency = recency_score(record.created_at, now, weights.half_life_s)
    boost = (
        weights.same_speaker
        if speaker_label is not None and record.speaker_label == speaker_label
        else 0.0
    )
    return weights.keyword * overlap + weights.recency * recency + boost


def rank_memories(
    records: Iterable[MemoryRecord],
    query: str,
    *,
    now: datetime | None = None,
    speaker_label: str | None = None,
    weights: RetrievalWeights = DEFAULT_WEIGHTS,
    require_overlap: bool = True,
) -> tuple[tuple[float, MemoryRecord], ...]:
    """Score and order memories, best first.

    Ordering is fully deterministic: score (rounded to 6 decimals so float noise cannot
    reorder equals), then newest first, then ``memory_id``. With ``require_overlap``
    only memories sharing at least one keyword with the query survive — an irrelevant
    "known fact" in the state costs accuracy, so an empty result is the better answer.
    """
    moment = now if now is not None else datetime.now(UTC)
    tokens = tokenize(query)
    scored: list[tuple[float, MemoryRecord]] = []
    for record in records:
        if require_overlap and tokens and keyword_overlap(tokens, record.text) <= 0.0:
            continue
        if require_overlap and not tokens:
            continue
        scored.append(
            (
                score_memory(
                    record, tokens, now=moment, speaker_label=speaker_label, weights=weights
                ),
                record,
            )
        )
    scored.sort(
        key=lambda item: (
            -round(item[0], 6),
            -item[1].created_at.timestamp(),
            item[1].memory_id,
        )
    )
    return tuple(scored)


@runtime_checkable
class CandidateSource(Protocol):
    """A store that can pre-filter memories cheaply (FTS5 or LIKE) before ranking."""

    def candidates(
        self,
        query: str,
        *,
        session_id: str | None = None,
        speaker_label: str | None = None,
        limit: int = 64,
    ) -> tuple[MemoryRecord, ...]:
        """Unranked memories that might match ``query``."""
        ...


class _AllSource(Protocol):
    def all(self) -> tuple[MemoryRecord, ...]: ...


class MemoryRetriever:
    """Picks the at most three memories a decision context is allowed to carry.

    The store is used through :class:`CandidateSource` when it offers one (both stores
    in this package do) and through ``all()`` otherwise, so any
    :class:`~rt_agent.contracts.protocols.MemoryStore` works.
    """

    def __init__(
        self,
        store: CandidateSource | _AllSource,
        *,
        weights: RetrievalWeights = DEFAULT_WEIGHTS,
        session_scoped: bool = False,
        require_overlap: bool = True,
        candidate_pool: int = 64,
    ) -> None:
        self.store = store
        self.weights = weights
        self.session_scoped = session_scoped
        self.require_overlap = require_overlap
        self.candidate_pool = candidate_pool

    # -- query shaping ------------------------------------------------------------

    @staticmethod
    def query_text(source: DecisionContext | Utterance | str) -> str:
        """The text a retrieval query is built from: the current utterance."""
        if isinstance(source, str):
            return source
        if isinstance(source, Utterance):
            return source.text
        return source.current.text

    @staticmethod
    def _session_of(source: DecisionContext | Utterance | str) -> str | None:
        if isinstance(source, DecisionContext | Utterance):
            return source.session_id
        return None

    @staticmethod
    def _speaker_of(source: DecisionContext | Utterance | str) -> str | None:
        if isinstance(source, DecisionContext):
            return source.current.speaker_label
        if isinstance(source, Utterance):
            return source.speaker_label
        return None

    def _candidates(
        self, query: str, session_id: str | None, speaker_label: str | None
    ) -> tuple[MemoryRecord, ...]:
        store = self.store
        if isinstance(store, CandidateSource):
            return store.candidates(
                query,
                session_id=session_id,
                speaker_label=speaker_label,
                limit=self.candidate_pool,
            )
        return store.all()

    # -- the public call ----------------------------------------------------------

    def retrieve(
        self,
        source: DecisionContext | Utterance | str,
        speaker_label: str | None = None,
        k: int = MAX_RETRIEVED_MEMORIES,
        *,
        now: datetime | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Return at most ``k`` memories for this utterance, best first.

        ``speaker_label`` defaults to the current speaker when a context or utterance is
        passed; it only boosts facts about that speaker, it never filters them out.
        """
        if k <= 0:
            return ()
        query = self.query_text(source)
        label = speaker_label if speaker_label is not None else self._speaker_of(source)
        session = self._session_of(source) if self.session_scoped else None
        candidates = self._candidates(query, session, None)
        ranked = rank_memories(
            candidates,
            query,
            now=now,
            speaker_label=label,
            weights=self.weights,
            require_overlap=self.require_overlap,
        )
        return tuple(record for _, record in ranked[:k])

    def scored(
        self,
        source: DecisionContext | Utterance | str,
        speaker_label: str | None = None,
        k: int = MAX_RETRIEVED_MEMORIES,
        *,
        now: datetime | None = None,
    ) -> tuple[tuple[float, MemoryRecord], ...]:
        """Same as :meth:`retrieve`, but keeps the scores — for logs and debugging."""
        if k <= 0:
            return ()
        query = self.query_text(source)
        label = speaker_label if speaker_label is not None else self._speaker_of(source)
        session = self._session_of(source) if self.session_scoped else None
        candidates = self._candidates(query, session, None)
        ranked = rank_memories(
            candidates,
            query,
            now=now,
            speaker_label=label,
            weights=self.weights,
            require_overlap=self.require_overlap,
        )
        return ranked[:k]
