"""Turn one generated reply into alice-shaped clauses.

alice admits at most 32 clauses per response (``sequence`` 0-31), each 1-1000
characters, 4000 characters in total; anything else is rejected by
``ClauseSequence.commit`` (``src/alice/contracts/speech_stream.py``) and faults the whole
ROS run. The split below therefore has hard bounds, and the packing step is what keeps a
long reply inside them instead of letting the robot fault at relay time.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rt_agent.contracts import MAX_CLAUSES, AffectCue, SpeechClauseOut

__all__ = [
    "MAX_CLAUSE_CHARS",
    "MAX_RESPONSE_CHARS",
    "build_speech_clauses",
    "split_clauses",
]

#: alice caps one clause at 1000 characters (``Text`` in ``alice/contracts/speech.py``).
MAX_CLAUSE_CHARS = 1000

#: alice caps one whole response at 4000 characters (``ClauseSequence.commit``).
MAX_RESPONSE_CHARS = 4000

#: Typographic punctuation, spelled as code points so the source stays plain ASCII.
_ELLIPSIS = chr(0x2026)
_CLOSERS = "\"')]" + chr(0x201D) + chr(0x2019)
_DASHES = chr(0x2014) + chr(0x2013)

#: Sentence-final punctuation, optionally followed by a closing quote or bracket.
_SENTENCE_BOUNDARY = re.compile("[.!?" + _ELLIPSIS + "]+[" + re.escape(_CLOSERS) + "]*(?=\\s|$)")

#: Softer breaks used only when a sentence is longer than ``max_len``.
_SOFT_BOUNDARY = re.compile("[,;:" + _DASHES + "][" + re.escape("\"')]") + "]*(?=\\s)")

_SEED_MODULUS = 2**32


def _sentences(text: str) -> list[str]:
    """Slice on English sentence punctuation, keeping the punctuation on the clause."""
    pieces: list[str] = []
    cursor = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        piece = text[cursor : match.end()].strip()
        if piece:
            pieces.append(piece)
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _greedy(parts: Sequence[str], max_len: int) -> list[str]:
    """Join ``parts`` in order into the fewest chunks that each fit ``max_len``."""
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if current and len(candidate) > max_len:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_long(piece: str, max_len: int) -> list[str]:
    """Fallback for a sentence that is longer than one clause: commas, then words, then chars."""
    if len(piece) <= max_len:
        return [piece]

    soft: list[str] = []
    cursor = 0
    for match in _SOFT_BOUNDARY.finditer(piece):
        part = piece[cursor : match.end()].strip()
        if part:
            soft.append(part)
        cursor = match.end()
    tail = piece[cursor:].strip()
    if tail:
        soft.append(tail)

    out: list[str] = []
    for chunk in _greedy(soft or [piece], max_len):
        if len(chunk) <= max_len:
            out.append(chunk)
            continue
        for words in _greedy(chunk.split(), max_len):
            if len(words) <= max_len:
                out.append(words)
            else:
                # A single token longer than a whole clause: slice it, honestly and bluntly.
                out.extend(words[at : at + max_len] for at in range(0, len(words), max_len))
    return out


def _pack(pieces: list[str], max_len: int, max_clauses: int) -> list[str]:
    """Merge neighbours until at most ``max_clauses`` remain, then truncate if still over."""
    packed = list(pieces)
    while len(packed) > max_clauses:
        best: int | None = None
        best_len = max_len + 1
        for index in range(len(packed) - 1):
            merged = len(packed[index]) + 1 + len(packed[index + 1])
            if merged <= max_len and merged < best_len:
                best, best_len = index, merged
        if best is None:
            # Nothing can be merged without breaking max_len: the text simply does not fit.
            return packed[:max_clauses]
        packed[best : best + 2] = [f"{packed[best]} {packed[best + 1]}"]
    return packed


def split_clauses(
    text: str,
    max_len: int = MAX_CLAUSE_CHARS,
    max_clauses: int = MAX_CLAUSES,
) -> list[str]:
    """Split ``text`` into at most ``max_clauses`` clauses of at most ``max_len`` characters.

    Sentence punctuation (``.``, ``!``, ``?``, ``…``) is the primary boundary and stays on
    the clause it ends. A sentence that is still too long falls back to commas, semicolons,
    colons and dashes, then to word boundaries, then to a blunt character slice. If the
    result has more clauses than ``max_clauses`` they are merged back together where that
    fits ``max_len``; a reply that genuinely cannot fit is truncated to ``max_clauses``
    clauses, because alice rejects sequence 32 outright.

    Abbreviations are not special-cased: "Dr. Smith" splits after "Dr.". That is acceptable
    here because a clause boundary only moves an affect onset, it never drops text.
    """
    if max_len < 1:
        raise ValueError("max_len must be at least 1")
    if max_clauses < 1:
        raise ValueError("max_clauses must be at least 1")

    stripped = text.strip()
    if not stripped:
        return []

    pieces: list[str] = []
    for sentence in _sentences(stripped):
        pieces.extend(_split_long(sentence, max_len))
    return _pack(pieces, max_len, max_clauses)


def build_speech_clauses(
    generation_id: str,
    texts: Sequence[str],
    cue: AffectCue,
    seed: int,
) -> list[SpeechClauseOut]:
    """Attach one affect cue to every clause of one response.

    ``sequence`` counts from 0 exactly as alice's ledger requires, ``end_of_response`` is
    set on the last clause only, and every clause carries the *same* vector and intensity:
    the decision is made once per utterance, and alice cannot change the affect of a clause
    once it is committed. To vary affect inside a response, call this once per group of
    clauses with different cues and renumber — or simply split into more clauses.

    Each clause gets its own TTS seed ``(seed + sequence) mod 2**32`` so two clauses with
    identical text do not synthesise to byte-identical audio.
    """
    if not 0 <= seed < _SEED_MODULUS:
        raise ValueError(f"seed must be in [0, {_SEED_MODULUS - 1}]")
    if not 1 <= len(generation_id) <= 128:
        raise ValueError("generation_id must be 1-128 characters (the ROS wire caps it at 128)")

    cleaned = [item.strip() for item in texts]
    cleaned = [item for item in cleaned if item]
    if not cleaned:
        raise ValueError("at least one non-empty clause text is required")
    if len(cleaned) > MAX_CLAUSES:
        raise ValueError(f"alice admits at most {MAX_CLAUSES} clauses per response")
    total = sum(len(item) for item in cleaned)
    if total > MAX_RESPONSE_CHARS:
        raise ValueError(
            f"response text is {total} characters, alice caps it at {MAX_RESPONSE_CHARS}"
        )

    # clause_id is capped at 128 characters on the ROS wire; reserve room for the suffix.
    stem = generation_id[: 128 - 4] or "gen"
    last = len(cleaned) - 1
    return [
        SpeechClauseOut(
            generation_id=generation_id,
            clause_id=f"{stem}-{index:02d}",
            sequence=index,
            text=item,
            vector=cue.vector,
            intensity=cue.intensity,
            seed=(seed + index) % _SEED_MODULUS,
            end_of_response=index == last,
        )
        for index, item in enumerate(cleaned)
    ]
