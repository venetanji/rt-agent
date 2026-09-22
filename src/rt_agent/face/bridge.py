"""Hardware-free ``FaceBridge`` implementations: JSONL on disk, JSON over UDP, fan-out, null.

Every bridge here satisfies ``rt_agent.contracts.FaceBridge`` (``emit_affect``,
``emit_clauses``, ``close``) and none of them needs ROS, a robot, or a network peer that
actually exists — a UDP datagram to nowhere is not an error, which is exactly the
behaviour a demo wants.

Record shapes are documented on each class and are stable; they are what the
``docs/alice-integration.md`` tables describe.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from rt_agent.contracts import AffectCue, FaceBridge, SpeechClauseOut
from rt_agent.face.authored_face import AuthoredFaceModel, ProfileName

__all__ = [
    "MAX_DATAGRAM_BYTES",
    "CompositeFaceBridge",
    "JsonlFaceBridge",
    "NullFaceBridge",
    "UdpJsonFaceBridge",
]

#: Refuse to build a datagram that a default-MTU path would fragment or drop.
MAX_DATAGRAM_BYTES = 8192


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _affect_payload(cue: AffectCue, face: AuthoredFaceModel | None) -> dict[str, Any]:
    """``{"cue": affect-cue/v1, "face": simulated target | None}``."""
    target = face.target_for(cue) if face is not None else None
    return {
        "cue": cue.model_dump(mode="json"),
        "face": None if target is None else target.model_dump(mode="json"),
    }


class JsonlFaceBridge:
    """Append cues to ``affect.jsonl`` and clauses to ``clauses.jsonl`` under ``out_dir``.

    ``affect.jsonl`` — one object per cue::

        {"kind": "affect", "seq": 0, "sent_at": "2026-09-22T12:00:00+00:00",
         "cue": {"schema_version": "affect-cue/v1", "vector": [0.5, 0.2, 0.1],
                 "intensity": 0.5, "preset": "warm", "source_id": "rt-agent",
                 "source_confidence": 0.9, "valid_for_ms": 1500,
                 "issued_at": "2026-09-22T12:00:00+00:00"},
         "face": {"anchor": "smile_open", "amplitude": 0.25,
                  "channels": {"mouth_open": 0.25, "left_mouth_corner": 0.25,
                               "right_mouth_corner": -0.25},
                  "profile": "bench", "visible": true}}

    ``clauses.jsonl`` — one object per clause, with **no envelope**, because the line is
    then byte-compatible with alice's own clause fixtures and can be replayed straight
    through ``alice-speak --clauses`` or ``alice.contracts.speech_stream.jsonl_clauses``::

        {"schema_version": "speech-clause/v1", "generation_id": "utt-0042",
         "clause_id": "utt-0042-00", "sequence": 0, "text": "Ten past nine.",
         "vector": [0.5, 0.2, 0.1], "intensity": 0.5, "seed": 29,
         "end_of_response": true}

    Files are opened lazily in append mode and flushed after every record, so a killed
    process still leaves a readable log.
    """

    def __init__(
        self,
        out_dir: Path | str,
        *,
        face: AuthoredFaceModel | ProfileName | None = "bench",
        affect_name: str = "affect.jsonl",
        clauses_name: str = "clauses.jsonl",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.affect_path = self.out_dir / affect_name
        self.clauses_path = self.out_dir / clauses_name
        self.face = AuthoredFaceModel(face) if isinstance(face, str) else face
        self._affect: TextIO | None = None
        self._clauses: TextIO | None = None
        self._affect_seq = 0
        self._clause_seq = 0

    def _open(self, path: Path) -> TextIO:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8")

    def _write(self, handle: TextIO, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
        handle.flush()

    def emit_affect(self, cue: AffectCue) -> None:
        """Append one cue (plus the simulated face target) to ``affect.jsonl``."""
        if self._affect is None:
            self._affect = self._open(self.affect_path)
        record = {"kind": "affect", "seq": self._affect_seq, "sent_at": _now_iso()}
        record.update(_affect_payload(cue, self.face))
        self._write(self._affect, record)
        self._affect_seq += 1

    def emit_clauses(self, clauses: Sequence[SpeechClauseOut]) -> None:
        """Append one ``speech-clause/v1`` object per clause to ``clauses.jsonl``."""
        if not clauses:
            return
        if self._clauses is None:
            self._clauses = self._open(self.clauses_path)
        for clause in clauses:
            self._write(self._clauses, clause.model_dump(mode="json"))
            self._clause_seq += 1

    def close(self) -> None:
        """Close both handles; closing twice is harmless."""
        for handle in (self._affect, self._clauses):
            if handle is not None:
                handle.close()
        self._affect = self._clauses = None

    def __enter__(self) -> JsonlFaceBridge:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class UdpJsonFaceBridge:
    """Send one JSON datagram per cue and per clause to ``(host, port)``.

    Fire-and-forget: there is no peer handshake and no delivery guarantee, which is the
    right trade for a face preview but the wrong one for alice's committed-clause topic —
    that seam is RELIABLE and is served by ``rt_agent.face.ros_bridge`` instead.

    Every datagram is a single JSON object carrying a ``kind`` discriminator::

        {"kind": "affect", "seq": 0, "sent_at": "...", "cue": {...}, "face": {...}}
        {"kind": "clause", "seq": 0, "sent_at": "...",
         "clause": {"schema_version": "speech-clause/v1", ...},
         "index": 0, "count": 2}

    ``seq`` counts every datagram this bridge has sent, of either kind; ``index``/``count``
    locate the clause inside its response so a receiver can detect a lost one.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9917,
        *,
        face: AuthoredFaceModel | ProfileName | None = "bench",
        sock: socket.socket | None = None,
    ) -> None:
        if not 0 < port < 65536:
            raise ValueError("port must be in 1..65535")
        self.host = host
        self.port = port
        self.face = AuthoredFaceModel(face) if isinstance(face, str) else face
        self._sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._owns_sock = sock is None
        self._seq = 0
        self.sent = 0

    @property
    def address(self) -> tuple[str, int]:
        """The destination this bridge sends to."""
        return (self.host, self.port)

    def _send(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
        if len(payload) > MAX_DATAGRAM_BYTES:
            raise ValueError(
                f"datagram is {len(payload)} bytes, over the {MAX_DATAGRAM_BYTES} byte limit"
            )
        self._sock.sendto(payload, self.address)
        self.sent += 1

    def _envelope(self, kind: str) -> dict[str, Any]:
        record = {"kind": kind, "seq": self._seq, "sent_at": _now_iso()}
        self._seq += 1
        return record

    def emit_affect(self, cue: AffectCue) -> None:
        """Send one ``kind="affect"`` datagram."""
        record = self._envelope("affect")
        record.update(_affect_payload(cue, self.face))
        self._send(record)

    def emit_clauses(self, clauses: Sequence[SpeechClauseOut]) -> None:
        """Send one ``kind="clause"`` datagram per clause, in order."""
        count = len(clauses)
        for index, clause in enumerate(clauses):
            record = self._envelope("clause")
            record.update(
                {"clause": clause.model_dump(mode="json"), "index": index, "count": count}
            )
            self._send(record)

    def close(self) -> None:
        """Close the socket if this bridge created it."""
        if self._owns_sock:
            self._sock.close()

    def __enter__(self) -> UdpJsonFaceBridge:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class CompositeFaceBridge:
    """Fan one decision out to several bridges, e.g. a JSONL log plus a live UDP preview.

    Every bridge is always called, even if an earlier one raised: a broken preview must not
    cost the run its log. The first exception is re-raised once all bridges have been
    given their turn.
    """

    def __init__(self, *bridges: FaceBridge) -> None:
        self.bridges: tuple[FaceBridge, ...] = bridges

    def _fanout(self, name: str, *args: Any) -> None:
        first: BaseException | None = None
        for bridge in self.bridges:
            try:
                getattr(bridge, name)(*args)
            # A broken preview must not cost the run its log: remember and carry on.
            except Exception as exc:
                first = first or exc
        if first is not None:
            raise first

    def emit_affect(self, cue: AffectCue) -> None:
        """Forward the cue to every bridge."""
        self._fanout("emit_affect", cue)

    def emit_clauses(self, clauses: Sequence[SpeechClauseOut]) -> None:
        """Forward the clauses to every bridge."""
        self._fanout("emit_clauses", clauses)

    def close(self) -> None:
        """Close every bridge."""
        self._fanout("close")


class NullFaceBridge:
    """Discard everything, but count it — the default when no face is attached."""

    def __init__(self) -> None:
        self.affect_count = 0
        self.clause_count = 0
        self.closed = False

    def emit_affect(self, cue: AffectCue) -> None:
        """Count one cue."""
        self.affect_count += 1

    def emit_clauses(self, clauses: Sequence[SpeechClauseOut]) -> None:
        """Count the clauses."""
        self.clause_count += len(clauses)

    def close(self) -> None:
        """Mark the bridge closed."""
        self.closed = True
