"""The whole loop over the example transcript, offline, from recorded live answers.

Every answer here came back from hosted Jev (``jev-1.13.0``) during the run written up
in ``docs/evidence/2026-09-22-kitchen-chat-live.md``, and was recorded with
``rt-agent replay --backend jev --record-fixtures``. Re-running that command re-records
them; the expectations below are the *recorded truth*, not a wish list, and where the
live model disagreed with the scenario's design the disagreement is spelled out in a
comment rather than papered over.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from rt_agent.contracts import MemoryRecord
from rt_agent.harness import ListeningAgent
from rt_agent.llm import ScriptedLLM
from rt_agent.memory import MemoryOutcome, SqliteMemoryStore
from rt_agent.policy import PolicyConfig
from rt_agent.systemone import MockSystemOne
from tests.harness.conftest import KITCHEN_FIXTURES, kitchen_utterances, make_agent

ALICE_SCHEMA = (
    Path(__file__).resolve().parents[1] / "fixtures" / "alice" / "speech-plan.schema.json"
)

#: The only two lines the robot may answer: both open with "Alice," and hand it the floor.
ADDRESSED_LINES = (12, 24)


@pytest.fixture
async def replayed(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> ListeningAgent:
    """One completed offline replay of the whole 28-line transcript."""
    agent = make_agent(tmp_path, kitchen_backend, scripted_llm)
    await agent.run(kitchen_utterances(agent.session_id))
    await agent.aclose()
    return agent


def turn_at(agent: ListeningAgent, number: int):
    """The recorded turn for 1-based line ``number`` of the transcript."""
    return agent.turns[number - 1]


def persisted_memories(agent: ListeningAgent) -> tuple[MemoryRecord, ...]:
    """Re-open the run's SQLite file. The agent closed it, which is the point."""
    store = SqliteMemoryStore(agent.config.memory_db_path(agent.session_id))
    try:
        return store.all()
    finally:
        store.close()


def state_for(agent: ListeningAgent, utterance_id: str) -> str:
    """The rendered state the backend was asked about one utterance.

    Found by searching, not by index: background memory writes make their own calls, so
    the n-th call is not the n-th utterance.
    """
    text = next(
        turn.utterance.text for turn in agent.turns if turn.utterance.utterance_id == utterance_id
    )
    for state, _questions in agent.backend.calls:
        _, _, tail = state.partition("Current utterance:\n")
        if tail and tail.partition("\n")[0].endswith(text):
            return state
    raise AssertionError(f"no bundle call was made for {utterance_id}")


# -- admission -------------------------------------------------------------------------


async def test_every_utterance_produces_exactly_one_decision(replayed: ListeningAgent) -> None:
    assert len(replayed.turns) == 28
    assert [turn.utterance.utterance_id for turn in replayed.turns] == [
        f"u{index:04d}" for index in range(1, 29)
    ]
    assert all(turn.decision.answers is not None for turn in replayed.turns)
    assert all(turn.backend_error is None for turn in replayed.turns)


async def test_only_the_two_addressed_lines_speak(replayed: ListeningAgent) -> None:
    spoke = [index for index, turn in enumerate(replayed.turns, start=1) if turn.decision.speak]
    assert spoke == list(ADDRESSED_LINES)
    for number in ADDRESSED_LINES:
        turn = turn_at(replayed, number)
        assert turn.spoke is True
        assert turn.reply
        assert turn.decision.addressee == "robot"


async def test_the_third_person_mention_does_not_take_the_turn(
    replayed: ListeningAgent,
) -> None:
    """Line 13 opens with the word "Alice" but is *about* her, said to S1.

    The two addressee signals disagree here, which is the interesting part. The
    admission ``noul`` is confident it is not being addressed (0.23), while the
    ``addressee`` choice names ``robot`` — at 0.51 probability and 0.33 confidence, its
    lowest of the whole transcript. Admission rests on the noul alone; a policy that
    branched on the choice would have answered a sentence spoken about it to someone
    else. Two earlier live runs of this same line chose ``other_human`` instead, so the
    choice is unstable here, and the noul is not.
    """
    turn = turn_at(replayed, 13)
    assert turn.decision.speak is False
    assert turn.decision.wait_reason == "not_addressed"
    answers = turn.decision.answers
    assert answers is not None
    assert answers.noul("addressed_to_robot") < 0.7
    assert answers.choice("addressee").confidence < 0.5


async def test_the_unintelligible_fragment_waits(replayed: ListeningAgent) -> None:
    turn = turn_at(replayed, 11)
    assert turn.decision.speak is False
    assert turn.decision.wait_reason == "not_intelligible"
    assert turn.decision.answers is not None
    assert turn.decision.answers.noul("intelligible_complete") < 0.6


async def test_the_floor_handoff_is_not_taken(replayed: ListeningAgent) -> None:
    # Line 18: S1 asks S2 about S2's own business. The robot must not answer for S2.
    turn = turn_at(replayed, 18)
    assert turn.decision.speak is False
    assert turn.decision.wait_reason == "not_addressed"


# -- the distressed passage --------------------------------------------------------------


async def test_the_distressed_passage_is_silent_with_a_visible_sad_face(
    replayed: ListeningAgent,
) -> None:
    """Lines 19-23: the robot says nothing and the face is visibly not neutral.

    The scenario expected the safety gate to fire on *both* 19 and 20. On the recorded
    run it fires from line 20 onward: line 19 measured P(stay quiet) = 0.40, one turn
    before the gate's 0.5. The distress is only in the current utterance at that point —
    the six-turn window behind it is still dinner logistics — and the signal accrues as
    the window fills, reaching 0.50, 0.53, 0.56, 0.53 on lines 20-23. The robot is silent
    at line 19 either way (nobody addressed it), and the face had already softened to
    ``sympathetic``. See the evidence doc for the whole trajectory.
    """
    for number in range(19, 24):
        turn = turn_at(replayed, number)
        assert turn.decision.speak is False, f"line {number} must stay quiet"
        assert turn.decision.emotion_preset in ("concerned", "sympathetic")

    assert turn_at(replayed, 19).decision.wait_reason == "not_addressed"
    for number in range(20, 24):
        assert turn_at(replayed, number).decision.wait_reason == "safety_quiet"

    # The safety hold overrides the chosen expression with ``concerned``, at an intensity
    # the policy floors at 0.5 so the face cannot shrug the moment off.
    held = turn_at(replayed, 20).decision
    assert held.emotion_preset == "concerned"
    assert held.intensity >= 0.5

    # Both presets sit at valence -0.3, past alice's 0.2 threshold: a real, visible frown.
    for number in (19, 20):
        cue = turn_at(replayed, number).emitted_cue
        assert cue is not None
        assert cue.visible_on_alice_face is True


async def test_the_face_timeline_follows_the_conversation(replayed: ListeningAgent) -> None:
    presets = [preset for _, preset in replayed.emotion_timeline]
    assert presets == ["attentive", "sympathetic", "concerned", "sympathetic"]


# -- memory ------------------------------------------------------------------------------


async def test_the_two_durable_facts_are_stored(replayed: ListeningAgent) -> None:
    stored = [
        outcome for outcome in replayed.memory_outcomes if outcome.outcome is MemoryOutcome.STORED
    ]
    assert {outcome.utterance_id for outcome in stored} == {"u0005", "u0006"}
    by_id = {outcome.utterance_id: outcome for outcome in stored}
    assert by_id["u0005"].kind == "preference"
    assert by_id["u0006"].kind == "event"
    for outcome in stored:
        assert outcome.record is not None
        assert outcome.faithfulness_p is not None
        assert outcome.faithfulness_p >= 0.7
        assert outcome.sensitive is False
    assert len(persisted_memories(replayed)) == 2


async def test_the_sensitive_health_line_stores_nothing(replayed: ListeningAgent) -> None:
    """Line 9 is the sensitive health mention, and nothing about it is written.

    The scenario expected it to demonstrate the *withheld* path. On this recording it
    does not get that far: P(worth remembering) measured 0.70 against the 0.75 gate, so
    the policy never names a memory kind and the writer is never scheduled. Three live
    runs of the same line measured 0.73, 0.75 and 0.70 — it straddles the threshold, and
    which side it lands on decides only which log line is written. The privacy outcome is
    the same either way, and
    :func:`test_the_withholding_path_fires_when_the_gate_is_lowered` exercises the
    withheld branch against these same recorded answers.
    """
    turn = turn_at(replayed, 9)
    assert turn.memory_scheduled is False
    assert turn.decision.remember is False
    assert turn.decision.memory_kind is None
    # The sensitivity itself was recognised; it is the durability gate that stopped it.
    assert turn.decision.sensitive is True
    assert all("blood pressure" not in record.text for record in persisted_memories(replayed))


async def test_the_withholding_path_fires_when_the_gate_is_lowered(
    tmp_path: Path, kitchen_backend: MockSystemOne, scripted_llm: ScriptedLLM
) -> None:
    """With the durability gate under line 9's recorded value, it is withheld, not stored.

    The threshold is derived from the fixture rather than hard-coded, so a re-record
    moves it with the measurement. Everything else is the same recorded live run.
    """
    fixture = json.loads((KITCHEN_FIXTURES / "u0009.json").read_text(encoding="utf-8"))
    worth = fixture["response"]["answers"]["worth_remembering"]["noul"]
    config = PolicyConfig(worth_remembering_min=worth - 0.01)

    agent = make_agent(tmp_path, kitchen_backend, scripted_llm, policy=config)
    await agent.run(kitchen_utterances(agent.session_id))
    await agent.aclose()

    withheld = [
        outcome
        for outcome in agent.memory_outcomes
        if outcome.outcome is MemoryOutcome.WITHHELD_SENSITIVE
    ]
    assert [outcome.utterance_id for outcome in withheld] == ["u0009"]
    # Withholding happens before any model call: the cheapest way not to leak something
    # is not to send it anywhere.
    assert withheld[0].candidate_text is None
    assert withheld[0].record is None
    assert withheld[0].sensitive is True
    assert withheld[0].kind == "health"
    stored = {record.memory_id for record in persisted_memories(agent)}
    assert len(stored) == 2
    assert all("blood pressure" not in record.text for record in persisted_memories(agent))
    assert agent.backend.calls, "the lowered gate still went through the same recordings"


async def test_no_other_line_attempts_a_memory(replayed: ListeningAgent) -> None:
    scheduled = {turn.utterance.utterance_id for turn in replayed.turns if turn.memory_scheduled}
    assert scheduled == {"u0005", "u0006"}


async def test_the_memory_worthiness_gate_is_close_on_line_9(
    replayed: ListeningAgent,
) -> None:
    """P(worth) on line 9 measured 0.70 against a 0.75 threshold: one bad round away.

    Recorded for the record, not as a target. Risk R3 in docs/ARCHITECTURE.md ("no
    threshold on a knife edge next to a measured value"), measured on a real line.
    """
    answers = turn_at(replayed, 9).decision.answers
    assert answers is not None
    assert answers.noul("worth_remembering") == pytest.approx(0.70, abs=0.001)
    assert answers.noul("sensitive_personal") > 0.9


# -- speaking ------------------------------------------------------------------------------


async def test_every_reply_exported_a_plan_alice_would_accept(
    replayed: ListeningAgent,
) -> None:
    schema = json.loads(ALICE_SCHEMA.read_text(encoding="utf-8"))
    plans = sorted(replayed.run_dir.glob("speech-plan-*.json"))
    assert [path.name for path in plans] == ["speech-plan-1.json", "speech-plan-2.json"]
    for path in plans:
        plan = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.validate(plan, schema)
        assert plan["voice"] == "azelma"
        assert plan["source_id"] == "rt-agent"
        assert plan["segments"]
        assert plan["segments"][0]["cues"][0]["progress"] == 0.0


async def test_the_clauses_carry_the_current_cue(replayed: ListeningAgent) -> None:
    lines = (replayed.run_dir / "clauses.jsonl").read_text(encoding="utf-8").splitlines()
    clauses = [json.loads(line) for line in lines]
    assert [clause["schema_version"] for clause in clauses] == ["speech-clause/v1"] * len(clauses)
    # Two clauses for the weather answer, one for the reminder.
    assert turn_at(replayed, 12).clause_count == 2
    assert turn_at(replayed, 24).clause_count == 1
    assert sum(1 for clause in clauses if clause["end_of_response"]) == 2
    # The reminder is spoken with the sympathetic face the distress left behind.
    reminder = clauses[-1]
    assert reminder["vector"][0] < -0.2


async def test_the_robot_hears_itself(replayed: ListeningAgent) -> None:
    """A ROBOT turn goes into the transcript and into the next decision window."""
    logged = replayed.transcript.read_session(replayed.session_id)
    robot_turns = [turn for turn in logged if turn.is_robot]
    assert len(robot_turns) == 2
    assert len(logged) == 30
    for turn, number in zip(robot_turns, ADDRESSED_LINES, strict=True):
        trigger = turn_at(replayed, number).utterance
        assert turn.t_start_s == pytest.approx(trigger.t_end_s)
        assert turn.t_end_s > turn.t_start_s
        assert turn.audio is None

    # The state the model saw for line 13 quotes the robot's own previous turn.
    state = state_for(replayed, "u0013")
    assert "ROBOT: I do not have a forecast" in state
    # An ongoing robot turn renders as "just now", never as a double-negative age.
    assert "[--" not in state


# -- the run's own paperwork ------------------------------------------------------------


async def test_summary_json_records_the_run(replayed: ListeningAgent) -> None:
    summary = json.loads((replayed.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "run-summary/v1"
    assert summary["utterances"] == 28
    assert summary["speak_count"] == 2
    assert summary["spoken_count"] == 2
    assert summary["wait_count"] == 26
    assert summary["wait_reasons"] == {
        "not_addressed": 21,
        "not_intelligible": 1,
        "safety_quiet": 4,
    }
    assert summary["memory_outcomes"] == {"stored": 2}
    assert summary["memories_stored"] == 2
    assert summary["emotion_changes"] == 4
    assert summary["backend_failures"] == 0
    assert summary["reply_failures"] == 0
    assert summary["backend_model_resolved"] == "jev-1.13.0"
    assert summary["policy_version"] == "policy/v1"
    assert summary["bundle_version"] == "decision-bundle/v1"
    assert summary["speech_plans"] == ["speech-plan-1.json", "speech-plan-2.json"]


async def test_decisions_jsonl_holds_one_complete_record_per_turn(
    replayed: ListeningAgent,
) -> None:
    lines = (replayed.run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 28
    records = [json.loads(line) for line in lines]
    assert [record["seq"] for record in records] == list(range(28))
    for record in records:
        assert record["schema_version"] == "decision-record/v1"
        assert record["decision"]["answers"]["model"] == "jev-1.13.0"
        assert record["bundle_ms"] >= 0.0
        assert record["total_ms"] >= record["policy_ms"]
    assert records[11]["llm_ms"] is not None


async def test_memories_jsonl_holds_every_outcome(replayed: ListeningAgent) -> None:
    lines = (replayed.run_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines()
    outcomes = sorted(json.loads(line)["outcome"] for line in lines)
    assert outcomes == ["stored", "stored"]


async def test_affect_jsonl_carries_the_simulated_alice_face(
    replayed: ListeningAgent,
) -> None:
    lines = (replayed.run_dir / "affect.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert records
    assert all(record["kind"] == "affect" for record in records)
    assert all(record["face"]["profile"] == "bench" for record in records)
    visible = [record for record in records if record["face"]["visible"]]
    # Every visible frame is from the distressed passage: nothing else moved the face.
    assert visible
    assert all(record["face"]["anchor"] == "frown_closed" for record in visible)
