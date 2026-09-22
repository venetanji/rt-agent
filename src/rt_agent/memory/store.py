"""The persistent memory layer: a single SQLite file, plus an in-process twin.

This is the layer the design brief keeps separate from the rolling System One window —
"classifier context is not a transcript log". What lands here is a handful of short,
LLM-written, faithfulness-gated sentences, so the schema is one table and the search is
honest keyword matching: FTS5 when the interpreter's SQLite has it, a LIKE scan when it
does not. Both paths hand their candidates to the same ranking function
(:func:`~rt_agent.memory.retrieval.rank_memories`), which tokenizes the query the same
way in both cases — so the fallback is slower, not different.

Privacy, per the alice AGENTS.md rules: speaker labels are session-local anonymous
labels, no voice embeddings and no raw audio are ever written here, and a memory marked
``sensitive`` only reaches this layer when the policy config explicitly allows it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from rt_agent.contracts.events import MAX_RETRIEVED_MEMORIES
from rt_agent.contracts.memory import MemoryRecord
from rt_agent.memory.retrieval import rank_memories, tokenize

__all__ = ["SCHEMA_VERSION", "InMemoryStore", "SqliteMemoryStore"]

# ``MemoryStore.list`` shadows the builtin inside these class bodies, so annotations
# below use these aliases instead of ``list[...]``.
_StrList = list[str]
_AnyList = list[Any]
_RecordList = list[MemoryRecord]

#: Bump when the table layout changes in a way an existing file cannot satisfy.
SCHEMA_VERSION = 1

_COLUMNS = (
    "memory_id",
    "schema_version",
    "session_id",
    "speaker_label",
    "kind",
    "text",
    "source_utterance_ids",
    "worth_p",
    "faithfulness_p",
    "sensitive",
    "created_at",
    "created_at_ts",
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id            TEXT PRIMARY KEY,
    schema_version       TEXT NOT NULL,
    session_id           TEXT NOT NULL,
    speaker_label        TEXT NOT NULL,
    kind                 TEXT NOT NULL,
    text                 TEXT NOT NULL,
    source_utterance_ids TEXT NOT NULL,
    worth_p              REAL NOT NULL,
    faithfulness_p       REAL NOT NULL,
    sensitive            INTEGER NOT NULL,
    created_at           TEXT NOT NULL,
    created_at_ts        REAL NOT NULL
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS memories_session_idx ON memories(session_id)",
    "CREATE INDEX IF NOT EXISTS memories_speaker_idx ON memories(speaker_label)",
    "CREATE INDEX IF NOT EXISTS memories_created_idx ON memories(created_at_ts DESC)",
)

_CREATE_FTS = "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(memory_id, text)"

_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM memories"


def _row_to_record(row: tuple[Any, ...]) -> MemoryRecord:
    values = dict(zip(_COLUMNS, row, strict=True))
    return MemoryRecord(
        schema_version="memory/v1",
        memory_id=str(values["memory_id"]),
        session_id=str(values["session_id"]),
        speaker_label=str(values["speaker_label"]),
        kind=values["kind"],
        text=str(values["text"]),
        source_utterance_ids=tuple(json.loads(values["source_utterance_ids"])),
        worth_p=float(values["worth_p"]),
        faithfulness_p=float(values["faithfulness_p"]),
        sensitive=bool(values["sensitive"]),
        created_at=datetime.fromisoformat(str(values["created_at"])),
    )


def _record_to_row(record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.memory_id,
        record.schema_version,
        record.session_id,
        record.speaker_label,
        record.kind,
        record.text,
        json.dumps(list(record.source_utterance_ids)),
        record.worth_p,
        record.faithfulness_p,
        int(record.sensitive),
        record.created_at.isoformat(),
        record.created_at.timestamp(),
    )


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression (quoted terms, OR-joined)."""
    return " OR ".join(f'"{token}"' for token in tokenize(query))


class SqliteMemoryStore:
    """A :class:`~rt_agent.contracts.protocols.MemoryStore` backed by one SQLite file.

    ``path`` may be ``":memory:"`` for a throwaway database. ``use_fts`` defaults to
    auto-detection: FTS5 is used when this interpreter's SQLite provides it, and the
    LIKE scan is used otherwise; pass ``False`` to exercise the fallback deliberately.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        use_fts: bool | None = None,
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser())
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(_CREATE_TABLE)
        for statement in _INDEXES:
            self._connection.execute(statement)
        self.fts_enabled = self._setup_fts(use_fts)
        self._connection.commit()

    def _setup_fts(self, use_fts: bool | None) -> bool:
        if use_fts is False:
            return False
        try:
            self._connection.execute(_CREATE_FTS)
        except sqlite3.OperationalError:
            if use_fts:
                raise
            return False
        self._sync_fts()
        return True

    def _sync_fts(self) -> None:
        """Backfill the index when a file written without FTS5 is reopened with it."""
        indexed = self._connection.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
        stored = self._connection.execute("SELECT count(*) FROM memories").fetchone()[0]
        if indexed == stored:
            return
        self._connection.execute("DELETE FROM memories_fts")
        self._connection.executemany(
            "INSERT INTO memories_fts(memory_id, text) VALUES (?, ?)",
            self._connection.execute("SELECT memory_id, text FROM memories").fetchall(),
        )

    # -- writes -------------------------------------------------------------------

    def add(self, record: MemoryRecord) -> None:
        """Persist one memory; re-adding the same ``memory_id`` replaces it."""
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._lock, self._connection:
            self._connection.execute(
                f"INSERT OR REPLACE INTO memories ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                _record_to_row(record),
            )
            if self.fts_enabled:
                self._connection.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?", (record.memory_id,)
                )
                self._connection.execute(
                    "INSERT INTO memories_fts(memory_id, text) VALUES (?, ?)",
                    (record.memory_id, record.text),
                )

    def add_many(self, records: Iterable[MemoryRecord]) -> None:
        """Persist several memories in one transaction."""
        for record in records:
            self.add(record)

    def delete(self, memory_id: str) -> bool:
        """Remove one memory. Returns True when a row was actually deleted."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
            )
            if self.fts_enabled:
                self._connection.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
                )
            return cursor.rowcount > 0

    # -- reads --------------------------------------------------------------------

    def _query(self, where: str, params: tuple[Any, ...], limit: int | None) -> _RecordList:
        sql = f"{_SELECT}{where} ORDER BY created_at_ts DESC, memory_id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params = (*params, limit)
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def get(self, memory_id: str) -> MemoryRecord | None:
        """One memory by id, or ``None``."""
        found = self._query(" WHERE memory_id = ?", (memory_id,), 1)
        return found[0] if found else None

    def list(
        self,
        session_id: str | None = None,
        speaker_label: str | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Stored memories, newest first, optionally filtered by session and speaker."""
        clauses: _StrList = []
        params: _AnyList = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if speaker_label is not None:
            clauses.append("speaker_label = ?")
            params.append(speaker_label)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return tuple(self._query(where, tuple(params), limit))

    def all(self) -> tuple[MemoryRecord, ...]:
        """Every stored memory, newest first."""
        return self.list()

    def recent(
        self, limit: int = MAX_RETRIEVED_MEMORIES, session_id: str | None = None
    ) -> tuple[MemoryRecord, ...]:
        """The ``limit`` most recently written memories."""
        return self.list(session_id, limit=limit)

    def count(self, session_id: str | None = None) -> int:
        """How many memories are stored (in one session, if given)."""
        sql = "SELECT count(*) FROM memories"
        params: tuple[Any, ...] = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        with self._lock:
            row = self._connection.execute(sql, params).fetchone()
        return int(row[0])

    def candidates(
        self,
        query: str,
        *,
        session_id: str | None = None,
        speaker_label: str | None = None,
        limit: int = 64,
    ) -> tuple[MemoryRecord, ...]:
        """Unranked memories that might match ``query``.

        FTS5 matches whole tokens; the LIKE fallback matches substrings, so it returns a
        superset (a query for "tea" also finds "teacher"). The extra rows are dropped by
        the ranking step, which tokenizes both sides, so :meth:`search` agrees either way.
        """
        tokens = tokenize(query)
        if not tokens:
            return ()
        filters: _StrList = []
        params: _AnyList = []
        if session_id is not None:
            filters.append("session_id = ?")
            params.append(session_id)
        if speaker_label is not None:
            filters.append("speaker_label = ?")
            params.append(speaker_label)

        if self.fts_enabled:
            matched = self._fts_ids(query, limit * 4)
            if not matched:
                return ()
            placeholders = ", ".join("?" for _ in matched)
            filters.append(f"memory_id IN ({placeholders})")
            params.extend(matched)
        else:
            likes = " OR ".join("lower(text) LIKE ?" for _ in tokens)
            filters.append(f"({likes})")
            params.extend(f"%{token}%" for token in tokens)

        where = f" WHERE {' AND '.join(filters)}" if filters else ""
        return tuple(self._query(where, tuple(params), limit))

    def _fts_ids(self, query: str, limit: int) -> _StrList:
        expression = _fts_query(query)
        if not expression:
            return []
        with self._lock:
            try:
                rows = self._connection.execute(
                    "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (expression, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                # A malformed MATCH expression must never take the harness down.
                return []
        return [str(row[0]) for row in rows]

    def search(
        self,
        session_id: str | None,
        query: str,
        limit: int = MAX_RETRIEVED_MEMORIES,
        *,
        now: datetime | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Best-effort retrieval by keyword and recency, best first.

        ``session_id`` may be ``None`` to search every session — memories are meant to
        outlive the conversation that produced them.
        """
        candidates = self.candidates(query, session_id=session_id, limit=max(limit * 8, 16))
        ranked = rank_memories(candidates, query, now=now)
        return tuple(record for _, record in ranked[:limit])

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        """Enter a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the database on the way out."""
        self.close()

    def __len__(self) -> int:
        """How many memories are stored."""
        return self.count()

    def __iter__(self) -> Iterator[MemoryRecord]:
        """Iterate over every memory, newest first."""
        return iter(self.all())


class InMemoryStore:
    """The same store, kept in a dict. For tests, replays and ``--no-persist`` runs."""

    def __init__(self, records: Iterable[MemoryRecord] = ()) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._lock = threading.RLock()
        self.fts_enabled = False
        for record in records:
            self.add(record)

    # -- writes -------------------------------------------------------------------

    def add(self, record: MemoryRecord) -> None:
        """Persist one memory; re-adding the same ``memory_id`` replaces it."""
        with self._lock:
            self._records[record.memory_id] = record

    def add_many(self, records: Iterable[MemoryRecord]) -> None:
        """Persist several memories."""
        for record in records:
            self.add(record)

    def delete(self, memory_id: str) -> bool:
        """Remove one memory. Returns True when it was present."""
        with self._lock:
            return self._records.pop(memory_id, None) is not None

    # -- reads --------------------------------------------------------------------

    def get(self, memory_id: str) -> MemoryRecord | None:
        """One memory by id, or ``None``."""
        with self._lock:
            return self._records.get(memory_id)

    def list(
        self,
        session_id: str | None = None,
        speaker_label: str | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Stored memories, newest first, optionally filtered by session and speaker."""
        with self._lock:
            found = [
                record
                for record in self._records.values()
                if (session_id is None or record.session_id == session_id)
                and (speaker_label is None or record.speaker_label == speaker_label)
            ]
        found.sort(key=lambda record: (-record.created_at.timestamp(), record.memory_id))
        return tuple(found if limit is None else found[:limit])

    def all(self) -> tuple[MemoryRecord, ...]:
        """Every stored memory, newest first."""
        return self.list()

    def recent(
        self, limit: int = MAX_RETRIEVED_MEMORIES, session_id: str | None = None
    ) -> tuple[MemoryRecord, ...]:
        """The ``limit`` most recently written memories."""
        return self.list(session_id, limit=limit)

    def count(self, session_id: str | None = None) -> int:
        """How many memories are stored (in one session, if given)."""
        return len(self.list(session_id))

    def candidates(
        self,
        query: str,
        *,
        session_id: str | None = None,
        speaker_label: str | None = None,
        limit: int = 64,
    ) -> tuple[MemoryRecord, ...]:
        """Unranked memories that share at least one keyword with ``query``."""
        tokens = tokenize(query)
        if not tokens:
            return ()
        found = [
            record
            for record in self.list(session_id, speaker_label)
            if any(token in record.text.lower() for token in tokens)
        ]
        return tuple(found[:limit])

    def search(
        self,
        session_id: str | None,
        query: str,
        limit: int = MAX_RETRIEVED_MEMORIES,
        *,
        now: datetime | None = None,
    ) -> tuple[MemoryRecord, ...]:
        """Best-effort retrieval by keyword and recency, best first."""
        candidates = self.candidates(query, session_id=session_id, limit=max(limit * 8, 16))
        ranked = rank_memories(candidates, query, now=now)
        return tuple(record for _, record in ranked[:limit])

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        """No-op; present so the in-memory store is drop-in for the SQLite one."""

    def __enter__(self) -> Self:
        """Enter a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Nothing to release."""
        self.close()

    def __len__(self) -> int:
        """How many memories are stored."""
        return self.count()

    def __iter__(self) -> Iterator[MemoryRecord]:
        """Iterate over every memory, newest first."""
        return iter(self.all())
