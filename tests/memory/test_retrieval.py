"""Retrieval ranking: keywords first, then recency, then the same-speaker boost.

Determinism is a requirement, not a nicety: the retrieved memories go into the System
One state, and a state that shuffles between runs makes every measured probability
unreproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rt_agent.contracts.memory import MemoryRecord
from rt_agent.memory import (
    InMemoryStore,
    MemoryRetriever,
    RetrievalWeights,
    SqliteMemoryStore,
    keyword_overlap,
    rank_memories,
    recency_score,
    tokenize,
)
from tests.conftest import make_context, make_utterance

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def record(
    memory_id: str,
    text: str,
    *,
    speaker_label: str = "S1",
    days_old: float = 0.0,
    session_id: str = "test-session",
) -> MemoryRecord:
    """One memory, ``days_old`` days before :data:`NOW`."""
    return MemoryRecord.model_validate(
        {
            "memory_id": memory_id,
            "session_id": session_id,
            "speaker_label": speaker_label,
            "kind": "other",
            "text": text,
            "source_utterance_ids": ("u0",),
            "worth_p": 0.9,
            "faithfulness_p": 0.9,
            "sensitive": False,
            "created_at": NOW - timedelta(days=days_old),
        }
    )


class TestTokenize:
    def test_stopwords_and_short_words_go(self) -> None:
        assert tokenize("Is the tea in a pot?") == ("tea", "pot")

    def test_case_and_punctuation_do_not_matter(self) -> None:
        assert tokenize("PEANUTS!!!") == ("peanuts",)

    def test_duplicates_collapse_in_order(self) -> None:
        assert tokenize("tea tea pot tea") == ("tea", "pot")


class TestSignals:
    def test_keyword_overlap_is_a_fraction_of_the_query(self) -> None:
        assert keyword_overlap(("tea", "pot"), "S1 drinks tea.") == 0.5
        assert keyword_overlap((), "anything") == 0.0

    def test_recency_halves_after_a_half_life(self) -> None:
        week = 7 * 24 * 3600.0
        assert recency_score(NOW, NOW, week) == 1.0
        assert recency_score(NOW - timedelta(days=7), NOW, week) == pytest.approx(0.5)
        assert recency_score(NOW - timedelta(days=70), NOW, week) < 0.01


class TestRanking:
    def test_keyword_beats_recency(self) -> None:
        old_hit = record("old", "S1 drinks tea every morning.", days_old=30)
        fresh_miss = record("fresh", "S1 repaired the bicycle.", days_old=0)
        ranked = rank_memories([fresh_miss, old_hit], "is there tea?", now=NOW)
        assert [entry[1].memory_id for entry in ranked] == ["old"]

    def test_recency_breaks_a_keyword_tie(self) -> None:
        older = record("older", "S1 drinks tea.", days_old=30)
        newer = record("newer", "S1 drinks tea.", days_old=1)
        ranked = rank_memories([older, newer], "tea", now=NOW)
        assert [entry[1].memory_id for entry in ranked] == ["newer", "older"]

    def test_same_speaker_boost(self) -> None:
        mine = record("mine", "tea is good", speaker_label="S2", days_old=10)
        theirs = record("theirs", "tea is good", speaker_label="S1", days_old=9)
        ranked = rank_memories([mine, theirs], "tea", now=NOW, speaker_label="S2")
        assert [entry[1].memory_id for entry in ranked] == ["mine", "theirs"]

    def test_identical_scores_break_ties_deterministically(self) -> None:
        first = record("b", "S1 drinks tea.", days_old=1)
        second = record("a", "S1 drinks tea.", days_old=1)
        ranked = rank_memories([first, second], "tea", now=NOW)
        assert [entry[1].memory_id for entry in ranked] == ["a", "b"]
        assert rank_memories([second, first], "tea", now=NOW) == ranked

    def test_zero_overlap_is_dropped(self) -> None:
        assert rank_memories([record("m", "S1 likes tea.")], "submarines", now=NOW) == ()

    def test_overlap_can_be_made_optional(self) -> None:
        ranked = rank_memories(
            [record("m", "S1 likes tea.")], "submarines", now=NOW, require_overlap=False
        )
        assert [entry[1].memory_id for entry in ranked] == ["m"]

    def test_weights_are_configurable(self) -> None:
        old_hit = record("old", "S1 drinks tea every morning.", days_old=30)
        fresh_hit = record("fresh", "S1 drinks tea and coffee.", days_old=0)
        keyword_only = RetrievalWeights(keyword=1.0, recency=0.0, same_speaker=0.0)
        ranked = rank_memories([fresh_hit, old_hit], "tea", now=NOW, weights=keyword_only)
        assert [entry[0] for entry in ranked] == [1.0, 1.0]


class TestRetriever:
    def test_top_k_from_a_decision_context(self) -> None:
        store = InMemoryStore(
            [
                record("m1", "S1 drinks tea, never coffee."),
                record("m2", "S2 is allergic to peanuts.", speaker_label="S2"),
                record("m3", "S1 keeps the tea in the blue tin.", days_old=2),
                record("m4", "S1 broke the tea pot last week.", days_old=20),
            ]
        )
        ctx = make_context(current=make_utterance("Alice, is there any tea?"))
        found = MemoryRetriever(store).retrieve(ctx, now=NOW)
        assert [memory.memory_id for memory in found] == ["m1", "m3", "m4"]
        assert len(found) <= 3

    def test_k_is_respected(self) -> None:
        store = InMemoryStore([record("m1", "tea one"), record("m2", "tea two", days_old=1)])
        assert len(MemoryRetriever(store).retrieve("tea", k=1, now=NOW)) == 1
        assert MemoryRetriever(store).retrieve("tea", k=0, now=NOW) == ()

    def test_plain_text_queries_work(self) -> None:
        store = InMemoryStore([record("m1", "S1 drinks tea, never coffee.")])
        assert MemoryRetriever(store).retrieve("tea", now=NOW)[0].memory_id == "m1"

    def test_speaker_is_taken_from_the_context(self) -> None:
        store = InMemoryStore(
            [
                record("s1fact", "the tea is cold", speaker_label="S1", days_old=5),
                record("s2fact", "the tea is cold", speaker_label="S2", days_old=6),
            ]
        )
        ctx = make_context(current=make_utterance("is the tea cold?", speaker_label="S2"))
        assert MemoryRetriever(store).retrieve(ctx, now=NOW)[0].memory_id == "s2fact"

    def test_an_explicit_label_overrides_the_context(self) -> None:
        store = InMemoryStore(
            [
                record("s1fact", "the tea is cold", speaker_label="S1", days_old=5),
                record("s2fact", "the tea is cold", speaker_label="S2", days_old=6),
            ]
        )
        ctx = make_context(current=make_utterance("is the tea cold?", speaker_label="S2"))
        found = MemoryRetriever(store).retrieve(ctx, "S1", now=NOW)
        assert found[0].memory_id == "s1fact"

    def test_nothing_relevant_returns_nothing(self) -> None:
        store = InMemoryStore([record("m1", "S1 drinks tea.")])
        assert MemoryRetriever(store).retrieve("tell me about submarines", now=NOW) == ()

    def test_session_scoping_is_opt_in(self) -> None:
        store = InMemoryStore(
            [
                record("here", "S1 drinks tea.", session_id="test-session"),
                record("elsewhere", "S1 drinks tea.", session_id="other", days_old=1),
            ]
        )
        ctx = make_context(current=make_utterance("any tea?"))
        assert len(MemoryRetriever(store).retrieve(ctx, now=NOW)) == 2
        scoped = MemoryRetriever(store, session_scoped=True).retrieve(ctx, now=NOW)
        assert [memory.memory_id for memory in scoped] == ["here"]

    def test_it_works_over_the_sqlite_store(self) -> None:
        with SqliteMemoryStore(":memory:") as store:
            store.add(record("m1", "S1 drinks tea, never coffee."))
            store.add(record("m2", "S2 is allergic to peanuts.", speaker_label="S2"))
            found = MemoryRetriever(store).retrieve("is there tea?", now=NOW)
            assert [memory.memory_id for memory in found] == ["m1"]

    def test_a_store_without_candidates_falls_back_to_all(self) -> None:
        class MinimalStore:
            """Only the MemoryStore protocol: no candidate pre-filter."""

            def __init__(self) -> None:
                self._records = (record("m1", "S1 drinks tea."),)

            def all(self) -> tuple[MemoryRecord, ...]:
                return self._records

        found = MemoryRetriever(MinimalStore()).retrieve("tea", now=NOW)
        assert [memory.memory_id for memory in found] == ["m1"]

    def test_scores_are_available_for_logging(self) -> None:
        store = InMemoryStore([record("m1", "S1 drinks tea.")])
        scored = MemoryRetriever(store).scored("tea", now=NOW)
        assert scored[0][0] > 0.0
        assert scored[0][1].memory_id == "m1"
