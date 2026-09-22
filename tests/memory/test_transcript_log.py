"""The append-only transcript: one ``utterance/v1`` per line, sessions separated on read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rt_agent.contracts import ROBOT_SPEAKER_LABEL
from rt_agent.memory import TranscriptLog, TranscriptLogError
from tests.conftest import make_utterance


@pytest.fixture
def log(tmp_path: Path) -> TranscriptLog:
    """A transcript log in a directory that does not exist yet."""
    return TranscriptLog(tmp_path / "out" / "transcript.jsonl")


class TestAppend:
    def test_one_line_per_turn(self, log: TranscriptLog) -> None:
        log.append(make_utterance("Hello Alice.", utterance_id="u1"))
        log.append(
            make_utterance("Hello to you.", speaker_label=ROBOT_SPEAKER_LABEL, utterance_id="u2")
        )
        lines = log.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["utterance_id"] == "u1"
        assert json.loads(lines[1])["speaker_label"] == "ROBOT"

    def test_every_field_round_trips(self, log: TranscriptLog) -> None:
        original = make_utterance("Olá, tudo bem?", language="pt", asr_confidence=0.81)
        log.append(original)
        assert log.read_all() == (original,)

    def test_flushes_immediately(self, log: TranscriptLog) -> None:
        log.append(make_utterance())
        assert log.path.read_text(encoding="utf-8").endswith("\n")

    def test_appending_continues_an_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "transcript.jsonl"
        with TranscriptLog(path) as first:
            first.append(make_utterance(utterance_id="u1"))
        with TranscriptLog(path) as second:
            second.append(make_utterance(utterance_id="u2"))
        assert [turn.utterance_id for turn in TranscriptLog(path).read_all()] == ["u1", "u2"]

    def test_extend(self, log: TranscriptLog) -> None:
        log.extend([make_utterance(utterance_id="u1"), make_utterance(utterance_id="u2")])
        assert log.count() == 2


class TestRead:
    def test_missing_file_reads_empty(self, log: TranscriptLog) -> None:
        assert log.read_all() == ()
        assert log.read_session("test-session") == ()
        assert log.sessions() == ()

    def test_sessions_are_separated(self, log: TranscriptLog) -> None:
        log.append(make_utterance(utterance_id="a1", session_id="s-1"))
        log.append(make_utterance(utterance_id="b1", session_id="s-2"))
        log.append(make_utterance(utterance_id="a2", session_id="s-1"))
        assert [turn.utterance_id for turn in log.read_session("s-1")] == ["a1", "a2"]
        assert [turn.utterance_id for turn in log.read_session("s-2")] == ["b1"]
        assert log.sessions() == ("s-1", "s-2")
        assert log.count("s-1") == 2

    def test_recent_returns_the_tail_oldest_first(self, log: TranscriptLog) -> None:
        for index in range(8):
            log.append(make_utterance(utterance_id=f"u{index}"))
        assert [turn.utterance_id for turn in log.recent("test-session", 3)] == ["u5", "u6", "u7"]
        assert log.recent("test-session", 0) == ()

    def test_blank_lines_are_ignored(self, log: TranscriptLog) -> None:
        log.append(make_utterance())
        log.path.write_text(log.path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(log.read_all()) == 1

    def test_a_broken_line_is_skipped_by_default(
        self, log: TranscriptLog, caplog: pytest.LogCaptureFixture
    ) -> None:
        log.append(make_utterance(utterance_id="u1"))
        with log.path.open("a", encoding="utf-8") as handle:
            handle.write('{"not": "an utterance"}\n')
        log.append(make_utterance(utterance_id="u2"))
        assert [turn.utterance_id for turn in log.read_all()] == ["u1", "u2"]
        assert "not a valid utterance/v1" in caplog.text

    def test_a_broken_line_raises_in_strict_mode(self, log: TranscriptLog) -> None:
        log.append(make_utterance())
        with log.path.open("a", encoding="utf-8") as handle:
            handle.write("{oops\n")
        with pytest.raises(TranscriptLogError, match=r"transcript\.jsonl:2"):
            log.read_all(strict=True)


class TestLifecycle:
    def test_close_is_idempotent_and_reopens_on_demand(self, log: TranscriptLog) -> None:
        log.append(make_utterance(utterance_id="u1"))
        log.close()
        log.close()
        log.append(make_utterance(utterance_id="u2"))
        assert log.count() == 2
