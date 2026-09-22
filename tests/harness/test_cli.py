"""The ``rt-agent`` command line, driven through typer's own test runner.

Nothing here reaches a network: every command runs against the recorded kitchen-chat
fixtures, the scripted text generator and a temporary output directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from rt_agent.harness.cli import app
from tests.harness.conftest import KITCHEN_CHAT, KITCHEN_FIXTURES, SCRIPTED_RULES

runner = CliRunner()


def _replay(tmp_path: Path, *extra: str) -> tuple[int, str]:
    result = runner.invoke(
        app,
        [
            "replay",
            str(KITCHEN_CHAT),
            "--backend",
            "mock",
            "--fixtures",
            str(KITCHEN_FIXTURES),
            "--scripted-rules",
            str(SCRIPTED_RULES),
            "--out",
            str(tmp_path / "out"),
            "--session",
            "cli",
            *extra,
        ],
    )
    return result.exit_code, result.output


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("replay", "listen", "memories", "check-backend"):
        assert command in result.output


def test_replay_runs_the_whole_transcript_and_prints_a_summary(tmp_path: Path) -> None:
    code, output = _replay(tmp_path)
    assert code == 0, output
    assert "run summary" in output
    assert "2 of 2 admitted" in output

    run_dir = tmp_path / "out" / "cli"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["utterances"] == 28
    assert summary["speak_count"] == 2
    assert summary["memories_stored"] == 2
    for name in ("decisions.jsonl", "transcript.jsonl", "affect.jsonl", "clauses.jsonl"):
        assert (run_dir / name).exists(), name
    assert sorted(path.name for path in run_dir.glob("speech-plan-*.json")) == [
        "speech-plan-1.json",
        "speech-plan-2.json",
    ]


def test_replay_prints_the_turn_table(tmp_path: Path) -> None:
    code, output = _replay(tmp_path, "--turns")
    assert code == 0, output
    assert "u0012" in output
    assert "SPEAK" in output


def test_replay_can_run_without_a_face(tmp_path: Path) -> None:
    code, output = _replay(tmp_path, "--face", "null")
    assert code == 0, output
    assert not (tmp_path / "out" / "cli" / "affect.jsonl").exists()


def test_replay_rejects_an_unknown_backend(tmp_path: Path) -> None:
    code, output = _replay(tmp_path, "--backend", "nonesuch")
    assert code != 0
    assert "jev, kev or mock" in output


def test_replay_rejects_a_missing_transcript(tmp_path: Path) -> None:
    result = runner.invoke(app, ["replay", str(tmp_path / "nothing.jsonl")])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_listen_needs_exactly_one_source() -> None:
    both = runner.invoke(app, ["listen", "--wav", "a.wav", "--mic"])
    assert both.exit_code != 0
    assert "exactly one" in both.output
    neither = runner.invoke(app, ["listen"])
    assert neither.exit_code != 0
    assert "exactly one" in neither.output


def test_memories_reads_back_what_the_replay_stored(tmp_path: Path) -> None:
    code, output = _replay(tmp_path)
    assert code == 0, output
    db = tmp_path / "out" / "cli" / "memories.sqlite3"

    result = runner.invoke(app, ["memories", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "decaf" in result.output
    assert "inspection" in result.output

    as_json = runner.invoke(app, ["memories", "--db", str(db), "--json"])
    assert as_json.exit_code == 0
    records = json.loads(as_json.output)
    assert len(records) == 2
    assert {record["kind"] for record in records} == {"preference", "event"}
    assert all(record["schema_version"] == "memory/v1" for record in records)


def test_memories_filters_by_speaker_and_search(tmp_path: Path) -> None:
    code, _ = _replay(tmp_path)
    assert code == 0
    db = tmp_path / "out" / "cli" / "memories.sqlite3"

    only_s1 = runner.invoke(app, ["memories", "--db", str(db), "--speaker", "S1", "--json"])
    assert only_s1.exit_code == 0
    assert [record["speaker_label"] for record in json.loads(only_s1.output)] == ["S1"]

    found = runner.invoke(app, ["memories", "--db", str(db), "--search", "inspection", "--json"])
    assert found.exit_code == 0
    records = json.loads(found.output)
    assert len(records) == 1
    assert "inspection" in records[0]["text"]


def test_memories_rejects_a_missing_database(tmp_path: Path) -> None:
    result = runner.invoke(app, ["memories", "--db", str(tmp_path / "none.sqlite3")])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_check_backend_rejects_an_unknown_endpoint() -> None:
    result = runner.invoke(app, ["check-backend", "--backend", "mock"])
    assert result.exit_code != 0
    assert "jev or kev" in result.output


def test_check_backend_reports_a_dead_local_kev_cleanly() -> None:
    """No Kev is running on this host, and the command says so instead of raising.

    The port is deliberately one nothing listens on, so the failure is the connection
    refusal itself rather than a timeout.
    """
    result = runner.invoke(
        app,
        [
            "check-backend",
            "--backend",
            "kev",
            "--base-url",
            "http://127.0.0.1:9",
            "--timeout-s",
            "2",
        ],
    )
    assert result.exit_code == 1
    assert "FAILED: SystemOneUnavailableError" in result.output
    assert "http://127.0.0.1:9" in result.output
