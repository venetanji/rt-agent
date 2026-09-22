"""The replay source: strict parsing, stable ids, file order, optional pacing."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rt_agent.audio.replay import ReplayFormatError, ReplaySource

LINES = [
    {"speaker": "S1", "text": "Alice, what time is it?", "t_start_s": 0.0, "t_end_s": 1.8},
    {
        "speaker": "S2",
        "text": "She is not going to know that.",
        "t_start_s": 2.1,
        "t_end_s": 4.0,
        "audio": {
            "vad_mean_prob": 0.88,
            "voiced_fraction": 0.63,
            "rms": 0.05,
            "clipped_fraction": 0.0,
        },
    },
    {
        "speaker": "ROBOT",
        "text": "It is just after four.",
        "t_start_s": 4.2,
        "t_end_s": 5.6,
        "asr_confidence": 0.91,
        "language": "en",
        "utterance_id": "robot-1",
    },
]


def write_jsonl(path: Path, records: list[dict[str, object]], *, header: bool = False) -> Path:
    body = "\n".join(json.dumps(record) for record in records)
    prefix = "# a diarized transcript fixture\n\n" if header else ""
    path.write_text(f"{prefix}{body}\n", encoding="utf-8")
    return path


async def test_lines_become_final_utterances_in_file_order(tmp_path: Path) -> None:
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", LINES), session_id="s-42")
    utterances = [utterance async for utterance in source.utterances()]
    assert [u.utterance_id for u in utterances] == ["u0001", "u0002", "robot-1"]
    assert [u.speaker_label for u in utterances] == ["S1", "S2", "ROBOT"]
    assert {u.session_id for u in utterances} == {"s-42"}
    assert all(u.is_final for u in utterances)
    assert all(u.language == "en" for u in utterances)
    assert [u.t_start_s for u in utterances] == sorted(u.t_start_s for u in utterances)
    assert source.count == 3


async def test_audio_evidence_is_optional_and_never_invented(tmp_path: Path) -> None:
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", LINES))
    utterances = [utterance async for utterance in source.utterances()]
    assert utterances[0].audio is None
    evidence = utterances[1].audio
    assert evidence is not None
    assert evidence.vad_mean_prob == pytest.approx(0.88)
    # duration_s defaults to the line's own interval, overlap_fraction stays unknown.
    assert evidence.duration_s == pytest.approx(1.9)
    assert evidence.overlap_fraction is None
    assert utterances[2].asr_confidence == pytest.approx(0.91)
    assert utterances[0].asr_confidence is None


async def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", LINES, header=True))
    utterances = [utterance async for utterance in source.utterances()]
    # The comment must not consume an id: the first real line is still u0001.
    assert [u.utterance_id for u in utterances] == ["u0001", "u0002", "robot-1"]


def test_out_of_order_timestamps_are_rejected(tmp_path: Path) -> None:
    records = [dict(LINES[1]), dict(LINES[0])]
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", records))
    with pytest.raises(ReplayFormatError, match="goes backwards"):
        list(source.read())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("text"), "missing key"),
        (lambda r: r.update(speakr="S1"), "unknown key"),
        (lambda r: r.update(speaker="Alice"), "anonymous session-local"),
        (lambda r: r.update(t_start_s="soon"), "timestamps must be numbers"),
        (lambda r: r.update(t_end_s=-1.0), "invalid utterance"),
        (lambda r: r.update(text=""), "invalid utterance"),
    ],
)
def test_malformed_lines_fail_loudly(tmp_path: Path, mutate, message: str) -> None:
    record: dict[str, object] = dict(LINES[0])
    mutate(record)
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", [record]))
    with pytest.raises(ReplayFormatError, match=message):
        list(source.read())


def test_invalid_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(LINES[0]) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="line 2: invalid JSON"):
        list(ReplaySource(path).read())


def test_partial_audio_evidence_is_rejected_rather_than_filled_in(tmp_path: Path) -> None:
    record: dict[str, object] = dict(LINES[0])
    record["audio"] = {"vad_mean_prob": 0.9}
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", [record]))
    with pytest.raises(ReplayFormatError, match="rather than inventing evidence"):
        list(source.read())


def test_unknown_audio_keys_are_rejected(tmp_path: Path) -> None:
    record: dict[str, object] = dict(LINES[1])
    audio = dict(LINES[1]["audio"])  # type: ignore[arg-type]
    audio["snr_db"] = 12.0
    record["audio"] = audio
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", [record]))
    with pytest.raises(ReplayFormatError, match="unknown audio keys"):
        list(source.read())


async def test_realtime_pacing_follows_the_transcript_clock(tmp_path: Path) -> None:
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", LINES), realtime=True, speed=20.0)
    started = time.monotonic()
    stamps = []
    async for _utterance in source.utterances():
        stamps.append(time.monotonic() - started)
    # 4.2 s of transcript at 20x is ~0.21 s of wall clock, and the gaps are ordered.
    assert stamps == sorted(stamps)
    assert 0.15 < stamps[-1] < 1.5
    assert stamps[0] < stamps[1]


async def test_aclose_stops_the_replay(tmp_path: Path) -> None:
    source = ReplaySource(write_jsonl(tmp_path / "t.jsonl", LINES))
    seen = 0
    async for _utterance in source.utterances():
        seen += 1
        await source.aclose()
    assert seen == 1


def test_speed_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="speed must be positive"):
        ReplaySource(tmp_path / "t.jsonl", speed=0.0)
