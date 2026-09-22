"""The memory store: CRUD, round-tripping, persistence and the FTS5/LIKE equivalence.

The FTS5 path and the LIKE fallback are tested side by side on the same corpus, because
the fallback only earns its place if a deployment without FTS5 behaves the same.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rt_agent.contracts import MemoryStore
from rt_agent.contracts.memory import MemoryRecord
from rt_agent.memory import InMemoryStore, SqliteMemoryStore

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)

CORPUS = (
    ("m1", "S1", "preference", "S1 drinks tea, never coffee.", 0),
    ("m2", "S2", "health", "S2 is allergic to peanuts.", 1),
    ("m3", "S1", "personal_fact", "S1 works as a teacher in Lisbon.", 2),
    ("m4", "S3", "plan", "S3 is flying to Porto on Friday.", 3),
)


def make_record(
    memory_id: str = "m1",
    speaker_label: str = "S1",
    kind: str = "preference",
    text: str = "S1 drinks tea, never coffee.",
    days_old: int = 0,
    *,
    session_id: str = "s-1",
    sensitive: bool = False,
) -> MemoryRecord:
    """One stored memory, ``days_old`` days before :data:`NOW`."""
    return MemoryRecord.model_validate(
        {
            "memory_id": memory_id,
            "session_id": session_id,
            "speaker_label": speaker_label,
            "kind": kind,
            "text": text,
            "source_utterance_ids": (f"u{memory_id}", "u0"),
            "worth_p": 0.9,
            "faithfulness_p": 0.88,
            "sensitive": sensitive,
            "created_at": datetime(2026, 9, 22 - days_old, 9, 0, tzinfo=UTC),
        }
    )


def populated(store: SqliteMemoryStore | InMemoryStore) -> SqliteMemoryStore | InMemoryStore:
    """Fill a store with :data:`CORPUS`."""
    for memory_id, speaker, kind, text, days in CORPUS:
        store.add(make_record(memory_id, speaker, kind, text, days))
    return store


@pytest.fixture(params=["sqlite-fts", "sqlite-like", "memory"])
def store(request: pytest.FixtureRequest) -> SqliteMemoryStore | InMemoryStore:
    """Every store implementation, so the shared behaviour is tested three times."""
    if request.param == "memory":
        return InMemoryStore()
    return SqliteMemoryStore(":memory:", use_fts=request.param == "sqlite-fts")


class TestProtocol:
    def test_every_store_satisfies_the_protocol(
        self, store: SqliteMemoryStore | InMemoryStore
    ) -> None:
        assert isinstance(store, MemoryStore)


class TestCrud:
    def test_add_and_get(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        record = make_record()
        store.add(record)
        assert store.get("m1") == record

    def test_get_missing_is_none(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        assert store.get("nope") is None

    def test_every_field_round_trips(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        record = make_record(sensitive=True)
        store.add(record)
        loaded = store.get("m1")
        assert loaded is not None
        assert loaded.model_dump() == record.model_dump()
        assert loaded.source_utterance_ids == ("um1", "u0")
        assert loaded.sensitive is True
        assert loaded.created_at.tzinfo is not None

    def test_re_adding_replaces(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        store.add(make_record())
        store.add(make_record(text="S1 switched to coffee."))
        assert store.count() == 1
        loaded = store.get("m1")
        assert loaded is not None
        assert loaded.text == "S1 switched to coffee."

    def test_delete(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        store.add(make_record())
        assert store.delete("m1") is True
        assert store.delete("m1") is False
        assert store.count() == 0

    def test_list_is_newest_first(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        assert [record.memory_id for record in store.list()] == ["m1", "m2", "m3", "m4"]
        assert store.all() == store.list()

    def test_list_filters(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        store.add(make_record("m5", "S1", "other", "S1 left early.", 4, session_id="s-2"))
        assert [record.memory_id for record in store.list(speaker_label="S1")] == [
            "m1",
            "m3",
            "m5",
        ]
        assert [record.memory_id for record in store.list("s-2")] == ["m5"]
        assert [record.memory_id for record in store.list("s-2", "S2")] == []

    def test_recent_and_count(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        assert [record.memory_id for record in store.recent(2)] == ["m1", "m2"]
        assert store.count() == 4
        assert len(store) == 4
        assert store.count("s-1") == 4
        assert store.count("other-session") == 0

    def test_iteration(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        assert [record.memory_id for record in store] == ["m1", "m2", "m3", "m4"]


class TestSearch:
    def test_keyword_hit(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        found = store.search(None, "is there any tea left?", now=NOW)
        assert [record.memory_id for record in found] == ["m1"]

    def test_limit_is_respected(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        found = store.search(None, "S1 tea teacher peanuts Porto", limit=2, now=NOW)
        assert len(found) == 2

    def test_no_match_is_empty(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        assert store.search(None, "submarine navigation") == ()

    def test_a_stopword_only_query_is_empty(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        assert store.search(None, "and the of it") == ()

    def test_session_scoping(self, store: SqliteMemoryStore | InMemoryStore) -> None:
        populated(store)
        store.add(make_record("m9", "S9", "preference", "S9 likes tea too.", 0, session_id="s-2"))
        found = store.search("s-2", "tea", now=NOW)
        assert [record.memory_id for record in found] == ["m9"]

    def test_punctuation_and_case_do_not_matter(
        self, store: SqliteMemoryStore | InMemoryStore
    ) -> None:
        populated(store)
        found = store.search(None, "PEANUTS!!!", now=NOW)
        assert [record.memory_id for record in found] == ["m2"]


class TestFtsAndFallbackAgree:
    @pytest.mark.parametrize(
        "query",
        [
            "tea",
            "is there tea or coffee",
            "who is the teacher",
            "peanuts",
            "flying to Porto on Friday",
            "nothing relevant at all",
        ],
    )
    def test_same_results(self, query: str) -> None:
        with (
            SqliteMemoryStore(":memory:", use_fts=True) as fts,
            SqliteMemoryStore(":memory:", use_fts=False) as like,
            InMemoryStore() as memory,
        ):
            populated(fts)
            populated(like)
            populated(memory)
            assert fts.fts_enabled is True
            assert like.fts_enabled is False
            expected = fts.search(None, query, now=NOW)
            assert like.search(None, query, now=NOW) == expected
            assert memory.search(None, query, now=NOW) == expected

    def test_fts_availability_is_detected(self) -> None:
        # This interpreter has FTS5; auto-detection must therefore turn it on, and
        # forcing it off must still produce a working store.
        with SqliteMemoryStore(":memory:") as auto:
            assert auto.fts_enabled is True
        with SqliteMemoryStore(":memory:", use_fts=False) as forced:
            populated(forced)
            assert forced.search(None, "tea", now=NOW)[0].memory_id == "m1"


class TestPersistence:
    def test_rows_survive_a_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "memories.sqlite3"
        with SqliteMemoryStore(path) as store:
            populated(store)
        with SqliteMemoryStore(path) as reopened:
            assert reopened.count() == 4
            assert reopened.search(None, "peanuts", now=NOW)[0].memory_id == "m2"

    def test_an_index_written_without_fts_is_backfilled(self, tmp_path: Path) -> None:
        path = tmp_path / "memories.sqlite3"
        with SqliteMemoryStore(path, use_fts=False) as without:
            populated(without)
        with SqliteMemoryStore(path, use_fts=True) as with_fts:
            assert with_fts.fts_enabled is True
            assert with_fts.search(None, "tea", now=NOW)[0].memory_id == "m1"
