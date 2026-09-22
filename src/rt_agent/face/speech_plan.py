"""Export a decision as ``speech-plan/v1`` JSON for alice's host ``alice-speak`` CLI.

This is the integration path that costs alice no code at all, and the only one that can
move a real servo today: ROS hardware admission rejects unconditionally, so the host
pipeline is the sole path to the Maestro.

.. code-block:: bash

    uv run --extra speech alice-speak --plan out/plans/utt-0042.json \\
        --output artifacts/speech/utt-0042

The rules enforced below are alice's own (``SpeechPlan``/``SpeechSegment``/``SpeechCue`` in
``src/alice/contracts/speech.py``, JSON Schema at ``config/speech/speech-plan.schema.json``):
1-32 segments, each 1-1000 characters and 4000 characters in total; 1-32 cues per segment;
**the first cue of every segment must sit at ``progress == 0`` and progress must be strictly
increasing**; ``pause_after_s`` in [0, 5]; ``voice`` from the enumerated set; ``seed`` in
0…2³²-1.

One asymmetry worth remembering: intra-segment cue fractions matter *only* on this offline
path. In the streaming clause path alice uses clause-onset affect alone, because a final
duration does not exist yet.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rt_agent.contracts import AffectCue, SpeechClauseOut

__all__ = [
    "MAX_PAUSE_S",
    "MAX_PLAN_CHARS",
    "MAX_SEGMENTS",
    "MAX_SEGMENT_CHARS",
    "SPEECH_PLAN_SCHEMA_VERSION",
    "VOICES",
    "export_speech_plan",
    "validate_speech_plan",
    "write_speech_plan",
]

SPEECH_PLAN_SCHEMA_VERSION = "speech-plan/v1"
AFFECT_SCHEMA_ID = "affect-vector/v1"

#: alice's ``Voice`` literal; ``azelma`` is the cached default voice of the TTS worker.
VOICES: tuple[str, ...] = (
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
)

MAX_SEGMENTS = 32
MAX_SEGMENT_CHARS = 1000
MAX_PLAN_CHARS = 4000
MAX_PAUSE_S = 5.0
MAX_SEED = 2**32 - 1
MAX_ID_CHARS = 1000


def _texts(clauses_or_texts: Sequence[SpeechClauseOut] | Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in clauses_or_texts:
        text = item.text if isinstance(item, SpeechClauseOut) else str(item)
        out.append(text.strip())
    return out


def _cue_list(
    cue_per_segment: AffectCue | Sequence[AffectCue],
    count: int,
) -> list[AffectCue]:
    if isinstance(cue_per_segment, AffectCue):
        return [cue_per_segment] * count
    cues = list(cue_per_segment)
    if len(cues) != count:
        raise ValueError(f"cue_per_segment has {len(cues)} cues for {count} segments")
    return cues


def _pauses(pause_after_s: float | Sequence[float], count: int) -> list[float]:
    if isinstance(pause_after_s, int | float):
        values = [float(pause_after_s)] * count
    else:
        values = [float(value) for value in pause_after_s]
        if len(values) != count:
            raise ValueError(f"pause_after_s has {len(values)} values for {count} segments")
    for value in values:
        if not 0.0 <= value <= MAX_PAUSE_S:
            raise ValueError(f"pause_after_s must be in [0, {MAX_PAUSE_S}], got {value}")
    return values


def _cue_dict(cue: AffectCue, progress: float) -> dict[str, Any]:
    return {
        "progress": float(progress),
        "vector": [float(axis) for axis in cue.vector],
        "intensity": float(cue.intensity),
    }


def export_speech_plan(
    utterance_id: str,
    source_id: str,
    clauses_or_texts: Sequence[SpeechClauseOut] | Sequence[str],
    cue_per_segment: AffectCue | Sequence[AffectCue],
    voice: str = "azelma",
    seed: int = 0,
    *,
    pause_after_s: float | Sequence[float] = 0.2,
    blend_to_next: bool = True,
) -> dict[str, Any]:
    """Build a valid ``speech-plan/v1`` document as a plain dict.

    ``clauses_or_texts`` may be :class:`SpeechClauseOut` objects (their ``text`` is used) or
    bare strings. ``cue_per_segment`` is either one cue for the whole plan or exactly one
    cue per segment.

    With ``blend_to_next`` (the default) each segment gets two cues: its own affect at
    ``progress 0.0`` and the *next* segment's affect at ``progress 1.0``, so the face is
    already arriving at the next clause's expression when that clause starts. This is the
    idiom of alice's own ``config/speech/alice-introduction.json``. With
    ``blend_to_next=False`` each segment carries a single cue at ``progress 0.0``, which is
    what the streaming clause path would do.

    The returned dict is validated by :func:`validate_speech_plan` before it is handed back,
    so an invalid plan raises here rather than on the robot.
    """
    texts = _texts(clauses_or_texts)
    if not texts:
        raise ValueError("a speech plan needs at least one segment")
    if any(not text for text in texts):
        raise ValueError("segment text must not be empty after stripping")

    cues = _cue_list(cue_per_segment, len(texts))
    pauses = _pauses(pause_after_s, len(texts))

    segments: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        segment_cues = [_cue_dict(cues[index], 0.0)]
        if blend_to_next:
            following = cues[min(index + 1, len(cues) - 1)]
            segment_cues.append(_cue_dict(following, 1.0))
        segments.append({"text": text, "cues": segment_cues, "pause_after_s": pauses[index]})

    plan: dict[str, Any] = {
        "schema_version": SPEECH_PLAN_SCHEMA_VERSION,
        "affect_schema_id": AFFECT_SCHEMA_ID,
        "utterance_id": utterance_id,
        "source_id": source_id,
        "seed": int(seed),
        "voice": voice,
        "segments": segments,
    }
    validate_speech_plan(plan)
    return plan


def validate_speech_plan(plan: Mapping[str, Any]) -> None:
    """Re-check a plan against alice's own bounds. Raises ``ValueError`` on the first problem.

    This duplicates the JSON Schema deliberately: ``jsonschema`` is a dev-only dependency,
    and a plan that reaches the robot malformed is a wasted trip to the bench.
    """
    if plan.get("schema_version") != SPEECH_PLAN_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SPEECH_PLAN_SCHEMA_VERSION!r}")
    if plan.get("affect_schema_id", AFFECT_SCHEMA_ID) != AFFECT_SCHEMA_ID:
        raise ValueError(f"affect_schema_id must be {AFFECT_SCHEMA_ID!r}")
    for key in ("utterance_id", "source_id"):
        value = plan.get(key)
        if not isinstance(value, str) or not 1 <= len(value) <= MAX_ID_CHARS:
            raise ValueError(f"{key} must be a string of 1-{MAX_ID_CHARS} characters")
    if plan.get("voice", "azelma") not in VOICES:
        raise ValueError(f"voice must be one of {', '.join(VOICES)}")
    seed = plan.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")

    segments = plan.get("segments")
    if (
        isinstance(segments, str)
        or not isinstance(segments, Sequence)
        or not 1 <= len(segments) <= MAX_SEGMENTS
    ):
        raise ValueError(f"a plan needs 1-{MAX_SEGMENTS} segments")

    total = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"segment {index} must be an object")
        text = segment.get("text")
        if not isinstance(text, str) or not 1 <= len(text) <= MAX_SEGMENT_CHARS:
            raise ValueError(f"segment {index} text must be 1-{MAX_SEGMENT_CHARS} characters")
        total += len(text)
        pause = segment.get("pause_after_s", 0.0)
        if not 0.0 <= float(pause) <= MAX_PAUSE_S:
            raise ValueError(f"segment {index} pause_after_s must be in [0, {MAX_PAUSE_S}]")
        cues = segment.get("cues")
        if (
            isinstance(cues, str)
            or not isinstance(cues, Sequence)
            or not 1 <= len(cues) <= MAX_SEGMENTS
        ):
            raise ValueError(f"segment {index} needs 1-{MAX_SEGMENTS} cues")
        if float(cues[0]["progress"]) != 0.0:
            raise ValueError(f"segment {index} must have a cue at progress 0")
        previous = -1.0
        for cue_index, cue in enumerate(cues):
            progress = float(cue["progress"])
            if not 0.0 <= progress <= 1.0:
                raise ValueError(f"segment {index} cue {cue_index} progress must be in [0, 1]")
            if progress <= previous:
                raise ValueError(f"segment {index} cue progress must be strictly increasing")
            previous = progress
            vector = cue["vector"]
            if len(vector) != 3 or any(not -1.0 <= float(axis) <= 1.0 for axis in vector):
                raise ValueError(
                    f"segment {index} cue {cue_index} vector must be 3 axes in [-1, 1]"
                )
            if not 0.0 <= float(cue["intensity"]) <= 1.0:
                raise ValueError(f"segment {index} cue {cue_index} intensity must be in [0, 1]")
    if total > MAX_PLAN_CHARS:
        raise ValueError(f"plan text is {total} characters, alice caps it at {MAX_PLAN_CHARS}")


def write_speech_plan(
    path: Path | str,
    plan: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``plan`` to ``path`` as pretty JSON and return the path.

    Refuses to clobber an existing file unless ``overwrite=True``, following alice's own
    "a published artifact is never overwritten" discipline: an utterance id identifies one
    plan, and silently replacing it destroys the evidence of what was actually spoken.
    """
    validate_speech_plan(plan)
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass overwrite=True to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination
