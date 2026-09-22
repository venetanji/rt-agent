"""Clause splitting must never hand alice something its ledger will reject."""

from __future__ import annotations

import pytest

from rt_agent.contracts import MAX_CLAUSES
from rt_agent.face.clauses import (
    MAX_RESPONSE_CHARS,
    build_speech_clauses,
    split_clauses,
)
from tests.face.conftest import make_cue


def test_splits_on_sentence_punctuation_and_keeps_it():
    clauses = split_clauses("Hello there. How are you? That's good!")
    assert clauses == ["Hello there.", "How are you?", "That's good!"]


def test_ellipsis_and_closing_quote_stay_with_the_clause():
    clauses = split_clauses('She said "later."  Then nothing… Then this.')
    assert clauses == ['She said "later."', "Then nothing…", "Then this."]


def test_empty_and_whitespace_text_produce_no_clauses():
    assert split_clauses("") == []
    assert split_clauses("   \n\t ") == []


def test_text_without_punctuation_is_one_clause():
    assert split_clauses("no punctuation at all") == ["no punctuation at all"]


def test_every_clause_respects_max_len():
    text = " ".join(f"word{index}" for index in range(400))
    clauses = split_clauses(text, max_len=60)
    assert clauses
    assert all(len(clause) <= 60 for clause in clauses)


def test_long_sentence_falls_back_to_commas():
    text = "alpha bravo, charlie delta, echo foxtrot, golf hotel, india juliet"
    clauses = split_clauses(text, max_len=30)
    assert all(len(clause) <= 30 for clause in clauses)
    assert len(clauses) > 1
    # The comma is the boundary, so it stays on the left-hand clause.
    assert clauses[0].endswith(",")


def test_a_single_token_longer_than_max_len_is_sliced():
    clauses = split_clauses("x" * 25, max_len=10)
    assert clauses == ["x" * 10, "x" * 10, "x" * 5]


def test_never_exceeds_max_clauses():
    text = " ".join(f"Sentence number {index}." for index in range(120))
    clauses = split_clauses(text)
    assert len(clauses) <= MAX_CLAUSES
    assert all(len(clause) <= 1000 for clause in clauses)


def test_packing_merges_rather_than_drops_when_it_fits():
    text = " ".join(f"S{index}." for index in range(40))
    clauses = split_clauses(text, max_clauses=4)
    assert len(clauses) == 4
    # Nothing was lost: every original sentence still appears somewhere.
    joined = " ".join(clauses)
    assert all(f"S{index}." in joined for index in range(40))


def test_truncates_only_when_the_text_cannot_possibly_fit():
    text = " ".join("x" * 10 for _ in range(20))
    clauses = split_clauses(text, max_len=10, max_clauses=3)
    assert len(clauses) == 3


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_nonsense_bounds(bad):
    with pytest.raises(ValueError):
        split_clauses("hi.", max_len=bad)
    with pytest.raises(ValueError):
        split_clauses("hi.", max_clauses=bad)


def test_build_speech_clauses_numbers_and_terminates_the_run():
    cue = make_cue("happy", 0.75)
    clauses = build_speech_clauses("utt-0042", ["One.", "Two.", "Three."], cue, seed=29)

    assert [clause.sequence for clause in clauses] == [0, 1, 2]
    assert [clause.end_of_response for clause in clauses] == [False, False, True]
    assert [clause.seed for clause in clauses] == [29, 30, 31]
    assert {clause.generation_id for clause in clauses} == {"utt-0042"}
    assert len({clause.clause_id for clause in clauses}) == 3
    assert all(clause.vector == cue.vector for clause in clauses)
    assert all(clause.intensity == cue.intensity for clause in clauses)
    assert all(clause.schema_version == "speech-clause/v1" for clause in clauses)


def test_build_speech_clauses_drops_blank_texts():
    clauses = build_speech_clauses("g", ["One.", "   ", "Two."], make_cue(), seed=0)
    assert [clause.text for clause in clauses] == ["One.", "Two."]


def test_build_speech_clauses_clause_id_fits_the_ros_wire_cap():
    clauses = build_speech_clauses("g" * 128, ["One."], make_cue(), seed=0)
    assert len(clauses[0].clause_id) <= 128


def test_build_speech_clauses_rejects_an_oversized_generation_id():
    with pytest.raises(ValueError, match="generation_id"):
        build_speech_clauses("g" * 129, ["One."], make_cue(), seed=0)


def test_build_speech_clauses_seed_wraps_at_32_bits():
    clauses = build_speech_clauses("g", ["A.", "B."], make_cue(), seed=2**32 - 1)
    assert [clause.seed for clause in clauses] == [2**32 - 1, 0]


def test_build_speech_clauses_rejects_empty_and_oversized_responses():
    cue = make_cue()
    with pytest.raises(ValueError, match="at least one"):
        build_speech_clauses("g", [], cue, seed=0)
    with pytest.raises(ValueError, match="at most 32"):
        build_speech_clauses("g", [f"S{i}." for i in range(33)], cue, seed=0)
    with pytest.raises(ValueError, match="4000"):
        build_speech_clauses("g", ["x" * 900] * 6, cue, seed=0)
    with pytest.raises(ValueError, match="seed"):
        build_speech_clauses("g", ["A."], cue, seed=2**32)


def test_split_then_build_stays_inside_alices_bounds():
    text = ("This is a sentence. " * 200).strip()
    clauses = build_speech_clauses("g", split_clauses(text), make_cue(), seed=1)
    assert len(clauses) <= MAX_CLAUSES
    assert sum(len(clause.text) for clause in clauses) <= MAX_RESPONSE_CHARS
