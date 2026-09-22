"""The shipped example transcript must load as Utterances and contain what its README claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rt_agent.contracts import AudioEvidence, Utterance

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
KITCHEN_CHAT = EXAMPLES / "kitchen_chat.jsonl"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [json.loads(line) for line in KITCHEN_CHAT.read_text().splitlines() if line.strip()]


def test_the_transcript_has_the_documented_length(rows):
    assert len(rows) == 28


def test_every_line_becomes_a_valid_utterance(rows):
    for index, row in enumerate(rows):
        utterance = Utterance(
            utterance_id=f"kitchen-{index:02d}",
            session_id="kitchen-chat",
            speaker_label=row["speaker"],
            text=row["text"],
            t_start_s=row["t_start_s"],
            t_end_s=row["t_end_s"],
            audio=AudioEvidence.model_validate(row["audio"]),
        )
        assert utterance.duration_s > 0


def test_only_two_anonymous_speakers_and_no_names(rows):
    assert {row["speaker"] for row in rows} == {"S1", "S2"}


def test_timestamps_are_monotonic_and_non_overlapping(rows):
    previous_end = -1.0
    for row in rows:
        assert row["t_start_s"] >= previous_end
        assert row["t_end_s"] > row["t_start_s"]
        previous_end = row["t_end_s"]


def test_audio_duration_agrees_with_the_timestamps(rows):
    for row in rows:
        span = row["t_end_s"] - row["t_start_s"]
        assert row["audio"]["duration_s"] == pytest.approx(span, abs=0.011)


def test_the_two_direct_addresses_are_where_the_readme_says(rows):
    addresses = [index for index, row in enumerate(rows) if row["text"].startswith("Alice,")]
    assert addresses == [11, 23]  # README lines 12 and 24, 1-based
    assert rows[11]["text"].endswith("?")
    assert rows[23]["text"].endswith("?")


def test_the_third_person_mention_is_not_an_address(rows):
    mention = rows[12]["text"]
    assert mention.startswith("Alice ")
    assert not mention.startswith("Alice,")


def test_the_unintelligible_fragment_has_weak_acoustic_evidence(rows):
    fragment = rows[10]
    assert fragment["audio"]["vad_mean_prob"] < 0.5
    assert fragment["audio"]["rms"] < 0.02
    assert len(fragment["text"]) < 20


def test_the_distressed_turn_is_the_loudest_and_longest(rows):
    distressed = rows[18]
    assert distressed["audio"]["clipped_fraction"] > 0.0
    assert distressed["audio"]["rms"] == max(row["audio"]["rms"] for row in rows)
    assert distressed["audio"]["duration_s"] == max(row["audio"]["duration_s"] for row in rows)


def test_the_memory_worthy_and_sensitive_turns_are_present(rows):
    assert "decaf" in rows[4]["text"]  # preference
    assert "inspection" in rows[5]["text"]  # plan / event
    assert "blood pressure" in rows[8]["text"]  # sensitive health


def test_the_readme_documents_the_example(rows):
    readme = (EXAMPLES / "README.md").read_text()
    assert "kitchen_chat.jsonl" in readme
    for field in ("speaker", "text", "t_start_s", "t_end_s", "vad_mean_prob"):
        assert field in readme
