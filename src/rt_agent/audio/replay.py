"""Replay a diarized transcript as :class:`~rt_agent.contracts.Utterance` events.

This is the hardware-free spine of the harness. No microphone, no VAD, no model
download: a JSONL file of already-segmented, already-diarized turns goes in and
final utterances come out, in order, with stable ids. Every integration test, every
System One fixture recording and every demo of the decision policy can run from
here, which is why it is a first-class source rather than a test helper.

Line format — one JSON object per line, UTF-8, no trailing commas::

    {"speaker": "S1", "text": "Alice, what time is it?",
     "t_start_s": 0.0, "t_end_s": 1.8,
     "audio": {"vad_mean_prob": 0.93, "voiced_fraction": 0.72,
               "rms": 0.06, "clipped_fraction": 0.0,
               "duration_s": 1.8, "overlap_fraction": null},
     "asr_confidence": 0.81, "language": "en", "utterance_id": "u0001",
     "is_final": true}

Required: ``speaker``, ``text``, ``t_start_s``, ``t_end_s``. Everything else is
optional. ``audio`` may be omitted entirely (the utterance then carries no acoustic
evidence, which is honest); when it *is* present it must carry the four measured
fields ``vad_mean_prob``, ``voiced_fraction``, ``rms`` and ``clipped_fraction``,
because a replay must not invent evidence the decision policy will read.
``duration_s`` defaults to ``t_end_s - t_start_s`` and ``overlap_fraction`` to
``null``. Unknown keys are a hard error on both objects: a typo in a fixture should
fail the run, not silently change a decision.

Ids are assigned ``u0001``, ``u0002``, ... in file order unless a line supplies its
own ``utterance_id``; ``session_id`` comes from the constructor. Blank lines and
``#`` comment lines are skipped so fixtures can be annotated.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from rt_agent.audio.diarize import validate_label
from rt_agent.contracts import AudioEvidence, Utterance

__all__ = ["ReplayFormatError", "ReplaySource"]

_REQUIRED_KEYS = frozenset({"speaker", "text", "t_start_s", "t_end_s"})
_OPTIONAL_KEYS = frozenset({"audio", "asr_confidence", "language", "utterance_id", "is_final"})
_AUDIO_REQUIRED = ("vad_mean_prob", "voiced_fraction", "rms", "clipped_fraction")
_AUDIO_OPTIONAL = ("duration_s", "overlap_fraction")


class ReplayFormatError(ValueError):
    """A transcript line is malformed, out of order, or carries unknown keys."""


def _audio_evidence(raw: Any, *, duration_s: float, line_number: int) -> AudioEvidence:
    if not isinstance(raw, dict):
        raise ReplayFormatError(f"line {line_number}: 'audio' must be an object")
    unknown = set(raw) - set(_AUDIO_REQUIRED) - set(_AUDIO_OPTIONAL)
    if unknown:
        raise ReplayFormatError(f"line {line_number}: unknown audio keys {sorted(unknown)}")
    missing = [key for key in _AUDIO_REQUIRED if key not in raw]
    if missing:
        raise ReplayFormatError(
            f"line {line_number}: 'audio' is missing measured field(s) {missing}; omit "
            "the whole 'audio' object rather than inventing evidence"
        )
    payload = dict(raw)
    payload.setdefault("duration_s", duration_s)
    payload.setdefault("overlap_fraction", None)
    try:
        return AudioEvidence.model_validate(payload)
    except Exception as error:
        raise ReplayFormatError(f"line {line_number}: invalid audio evidence: {error}") from error


class ReplaySource:
    """Read a diarized transcript JSONL file and yield final ``Utterance`` objects.

    ``realtime=False`` (the default) replays as fast as the consumer can take the
    events — what tests want. ``realtime=True`` paces the stream against the
    transcript's own timeline: utterance *n* is released when
    ``(t_start_s - first_t_start_s) / speed`` seconds of wall clock have passed, so
    ``speed=2.0`` replays a five-minute conversation in two and a half minutes.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str = "replay",
        realtime: bool = False,
        speed: float = 1.0,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.path = Path(path)
        self.session_id = session_id
        self.realtime = realtime
        self.speed = speed
        self.count = 0
        self._closed = False

    def read(self) -> Iterator[Utterance]:
        """Parse the file eagerly, line by line, without any pacing. Synchronous."""
        previous_start: float | None = None
        index = 0
        with self.path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                index += 1
                utterance = self._parse(line, line_number=line_number, index=index)
                if previous_start is not None and utterance.t_start_s < previous_start:
                    raise ReplayFormatError(
                        f"line {line_number}: t_start_s {utterance.t_start_s} goes "
                        f"backwards from {previous_start}; transcripts replay in order"
                    )
                previous_start = utterance.t_start_s
                yield utterance

    def _parse(self, line: str, *, line_number: int, index: int) -> Utterance:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReplayFormatError(f"line {line_number}: invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ReplayFormatError(f"line {line_number}: expected a JSON object")
        missing = _REQUIRED_KEYS - set(record)
        if missing:
            raise ReplayFormatError(f"line {line_number}: missing key(s) {sorted(missing)}")
        unknown = set(record) - _REQUIRED_KEYS - _OPTIONAL_KEYS
        if unknown:
            raise ReplayFormatError(f"line {line_number}: unknown key(s) {sorted(unknown)}")
        try:
            t_start_s = float(record["t_start_s"])
            t_end_s = float(record["t_end_s"])
        except (TypeError, ValueError) as error:
            raise ReplayFormatError(f"line {line_number}: timestamps must be numbers") from error
        try:
            speaker = validate_label(str(record["speaker"]))
        except ValueError as error:
            raise ReplayFormatError(f"line {line_number}: {error}") from error
        audio = (
            _audio_evidence(
                record["audio"], duration_s=max(0.0, t_end_s - t_start_s), line_number=line_number
            )
            if record.get("audio") is not None
            else None
        )
        try:
            return Utterance(
                utterance_id=str(record.get("utterance_id") or f"u{index:04d}"),
                session_id=self.session_id,
                speaker_label=speaker,
                text=str(record["text"]),
                t_start_s=t_start_s,
                t_end_s=t_end_s,
                asr_confidence=record.get("asr_confidence"),
                language=str(record.get("language", "en")),
                audio=audio,
                is_final=bool(record.get("is_final", True)),
            )
        except Exception as error:
            raise ReplayFormatError(f"line {line_number}: invalid utterance: {error}") from error

    async def utterances(self) -> AsyncIterator[Utterance]:
        """Yield every utterance in file order, paced when ``realtime`` is set."""
        started = time.monotonic()
        origin: float | None = None
        self.count = 0
        for utterance in self.read():
            if self._closed:
                return
            if self.realtime:
                if origin is None:
                    origin = utterance.t_start_s
                due = (utterance.t_start_s - origin) / self.speed
                delay = due - (time.monotonic() - started)
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
            self.count += 1
            yield utterance

    def __aiter__(self) -> AsyncIterator[Utterance]:
        """``async for utterance in source`` is the same as iterating :meth:`utterances`."""
        return self.utterances()

    async def aclose(self) -> None:
        """Stop the replay at the next utterance boundary."""
        self._closed = True
