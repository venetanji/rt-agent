"""``TranscriptLog`` — every finalized turn, appended as one JSON line.

This is the durable record of what was heard, including the robot's own turns
(``speaker_label == "ROBOT"``). It is explicitly *not* what a System One call reads:
the classifier gets a small rolling window, the log gets everything, and the memory
store gets the handful of sentences worth keeping. Sessions share one file and are
separated by ``session_id`` on read, so a replay and a live run can be compared without
moving files around.

The format is JSON Lines of ``utterance/v1``: one object per line, UTF-8, no trailing
commas, append-only. A line that does not parse is skipped with a warning rather than
taking the harness down mid-conversation; pass ``strict=True`` to read to turn that
into an error instead.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from pydantic import ValidationError

from rt_agent.contracts.events import RECENT_WINDOW, Utterance

__all__ = ["TranscriptLog", "TranscriptLogError"]

_LOGGER = logging.getLogger(__name__)


class TranscriptLogError(RuntimeError):
    """A transcript line could not be parsed while reading in strict mode."""


class TranscriptLog:
    """Append-only JSONL transcript over one file path."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._handle: IO[str] | None = None

    # -- writing ------------------------------------------------------------------

    def _open(self) -> IO[str]:
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("a", encoding="utf-8")
        return self._handle

    def append(self, utterance: Utterance) -> None:
        """Append one turn and flush, so a crash cannot lose the line just written."""
        line = json.dumps(
            utterance.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        with self._lock:
            handle = self._open()
            handle.write(line + "\n")
            handle.flush()

    def extend(self, utterances: Iterable[Utterance]) -> None:
        """Append several turns in order."""
        for utterance in utterances:
            self.append(utterance)

    # -- reading ------------------------------------------------------------------

    def read_all(self, *, strict: bool = False) -> tuple[Utterance, ...]:
        """Every turn in the file, in the order it was appended."""
        return tuple(self.iter_all(strict=strict))

    def iter_all(self, *, strict: bool = False) -> Iterator[Utterance]:
        """Stream the file, one ``utterance/v1`` at a time."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield Utterance.model_validate_json(line)
                except (ValidationError, ValueError) as error:
                    message = f"{self.path}:{number} is not a valid utterance/v1: {error}"
                    if strict:
                        raise TranscriptLogError(message) from error
                    _LOGGER.warning("%s", message)

    def read_session(self, session_id: str, *, strict: bool = False) -> tuple[Utterance, ...]:
        """Every turn of one session, in order."""
        return tuple(
            utterance
            for utterance in self.iter_all(strict=strict)
            if utterance.session_id == session_id
        )

    def recent(
        self, session_id: str, limit: int = RECENT_WINDOW, *, strict: bool = False
    ) -> tuple[Utterance, ...]:
        """The last ``limit`` turns of one session, oldest first — the rolling window."""
        if limit <= 0:
            return ()
        return self.read_session(session_id, strict=strict)[-limit:]

    def sessions(self, *, strict: bool = False) -> tuple[str, ...]:
        """Session ids present in the file, in first-seen order."""
        seen: dict[str, None] = {}
        for utterance in self.iter_all(strict=strict):
            seen.setdefault(utterance.session_id, None)
        return tuple(seen)

    def count(self, session_id: str | None = None) -> int:
        """How many turns are logged (in one session, if given)."""
        if session_id is None:
            return sum(1 for _ in self.iter_all())
        return len(self.read_session(session_id))

    # -- lifecycle ----------------------------------------------------------------

    def close(self) -> None:
        """Close the append handle. Reading reopens the file on demand."""
        with self._lock:
            if self._handle is not None and not self._handle.closed:
                self._handle.close()
            self._handle = None

    def __enter__(self) -> Self:
        """Enter a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the append handle on the way out."""
        self.close()
