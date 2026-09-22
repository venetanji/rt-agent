"""The ROS seam: exact message field sets, local guards, and a clear failure without rclpy."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from rt_agent.face.clauses import build_speech_clauses
from rt_agent.face.ros_bridge import (
    ACTION_RUN_SPEECH,
    AFFECT_CUE_FIELDS,
    AFFECT_CUE_MSG,
    MAX_AFFECT_VALID_NS,
    MAX_TRANSPORT_AGE_NS,
    SPEECH_CLAUSE_FIELDS,
    STREAM_HEADER_FIELDS,
    TOPIC_AFFECT_CUE,
    TOPIC_COMMITTED_CLAUSES,
    RosFaceBridge,
    RosUnavailableError,
    RunIdentity,
    RunSpeechClient,
    affect_cue_msg_dict,
    apply_msg_fields,
    speech_clause_msg_dict,
    stream_header_dict,
)
from tests.face.conftest import make_cue

IDENTITY = RunIdentity(run_id="run-1", epoch="epoch-from-feedback", generation_id="utt-0042")


@pytest.fixture
def clause():
    return build_speech_clauses("utt-0042", ["Ten past nine."], make_cue("warm", 0.5), 29)[0]


def test_importing_this_module_never_needs_ros():
    assert "rclpy" not in sys.modules
    assert importlib.import_module("rt_agent.face.ros_bridge") is not None


def test_topic_and_bound_constants_match_alice():
    assert TOPIC_COMMITTED_CLAUSES == "/alice/run/committed_clauses"
    assert TOPIC_AFFECT_CUE == "/alice/affect/cue"
    assert ACTION_RUN_SPEECH == "/alice/run_speech"
    assert MAX_TRANSPORT_AGE_NS == 250_000_000
    assert MAX_AFFECT_VALID_NS == 2_000_000_000


def test_speech_clause_msg_dict_has_exactly_the_msg_fields(clause):
    payload = speech_clause_msg_dict(clause, IDENTITY, source_monotonic_ns=1234)
    assert tuple(payload) == SPEECH_CLAUSE_FIELDS
    assert tuple(payload["header"]) == STREAM_HEADER_FIELDS
    assert tuple(payload["header"]["identity"]) == ("run_id", "epoch", "generation_id")

    assert payload["schema_version"] == "speech-clause/v1"
    assert payload["clause_id"] == clause.clause_id
    assert payload["clause_sequence"] == 0
    assert payload["text"] == "Ten past nine."
    assert payload["affect_vector"] == [0.5, 0.2, 0.1]
    assert payload["intensity"] == pytest.approx(0.5)
    assert payload["seed"] == 29
    assert payload["end_of_response"] is True
    assert payload["header"]["source_monotonic_ns"] == 1234
    assert payload["header"]["publisher_incarnation"] == "rt-agent"


def test_header_sequence_defaults_to_the_clause_sequence():
    cue = make_cue()
    clauses = build_speech_clauses("utt-0042", ["One.", "Two.", "Three."], cue, 0)
    payloads = [speech_clause_msg_dict(item, IDENTITY) for item in clauses]
    assert [item["header"]["sequence"] for item in payloads] == [0, 1, 2]
    assert [item["clause_sequence"] for item in payloads] == [0, 1, 2]


def test_source_monotonic_ns_is_stamped_at_call_time(clause):
    first = speech_clause_msg_dict(clause, IDENTITY)["header"]["source_monotonic_ns"]
    second = speech_clause_msg_dict(clause, IDENTITY)["header"]["source_monotonic_ns"]
    assert second >= first > 0


def test_a_generation_id_that_disagrees_with_the_run_is_refused(clause):
    other = RunIdentity(run_id="run-1", epoch="e", generation_id="some-other-generation")
    with pytest.raises(ValueError, match="generation_id"):
        speech_clause_msg_dict(clause, other)


def test_stream_header_rejects_locally_what_alice_would_fault_on():
    with pytest.raises(ValueError, match="sequence"):
        stream_header_dict(IDENTITY, -1, 10, "rt-agent")
    with pytest.raises(ValueError, match="monotonic"):
        stream_header_dict(IDENTITY, 0, -1, "rt-agent")
    with pytest.raises(ValueError, match="publisher_incarnation"):
        stream_header_dict(IDENTITY, 0, 10, "")


def test_affect_cue_msg_dict_has_exactly_the_proposed_msg_fields():
    cue = make_cue("concerned", 0.75, valid_for_ms=1500)
    payload = affect_cue_msg_dict(cue, IDENTITY, sequence=3, source_monotonic_ns=99)

    assert tuple(payload) == AFFECT_CUE_FIELDS
    assert payload["schema_version"] == "affect-cue/v1"
    assert payload["affect_vector"] == [-0.3, 0.35, -0.1]
    assert payload["intensity"] == pytest.approx(0.75)
    assert payload["valid_for_ns"] == 1_500_000_000
    assert payload["source_id"] == "rt-agent"
    assert payload["source_confidence"] == pytest.approx(0.9)
    assert payload["header"]["sequence"] == 3
    assert payload["header"]["source_monotonic_ns"] == 99


def test_affect_cue_lifetime_is_bounded_at_two_seconds():
    with pytest.raises(ValueError, match="2000000000 ns bound"):
        affect_cue_msg_dict(make_cue(valid_for_ms=2500), IDENTITY)
    assert affect_cue_msg_dict(make_cue(valid_for_ms=2000), IDENTITY)["valid_for_ns"] == (
        MAX_AFFECT_VALID_NS
    )


def test_the_proposed_msg_text_lists_every_field_it_builds():
    for field in AFFECT_CUE_FIELDS:
        assert field in AFFECT_CUE_MSG
    assert "StreamHeader header" in AFFECT_CUE_MSG
    assert "float64[3] affect_vector" in AFFECT_CUE_MSG


def test_apply_msg_fields_fills_nested_objects(clause):
    identity = types.SimpleNamespace(run_id="", epoch="", generation_id="")
    header = types.SimpleNamespace(
        identity=identity, sequence=0, source_monotonic_ns=0, publisher_incarnation=""
    )
    msg = types.SimpleNamespace(
        header=header,
        schema_version="",
        clause_id="",
        clause_sequence=0,
        text="",
        affect_vector=[],
        intensity=0.0,
        seed=0,
        end_of_response=False,
    )
    apply_msg_fields(msg, speech_clause_msg_dict(clause, IDENTITY, source_monotonic_ns=7))
    assert msg.header.identity.epoch == "epoch-from-feedback"
    assert msg.header.source_monotonic_ns == 7
    assert msg.text == "Ten past nine."
    assert msg.affect_vector == [0.5, 0.2, 0.1]


def test_apply_msg_fields_refuses_an_unknown_field():
    msg = types.SimpleNamespace(known=0)
    with pytest.raises(AttributeError, match="unknown"):
        apply_msg_fields(msg, {"unknown": 1})


def test_ros_face_bridge_fails_clearly_at_construction_without_rclpy():
    with pytest.raises(RosUnavailableError) as raised:
        RosFaceBridge(IDENTITY)
    message = str(raised.value)
    assert "ROS 2 is not available" in message
    assert "rclpy" in message
    assert "alice_interfaces" in message
    # The message must point at something that does work here.
    assert "UdpJsonFaceBridge" in message
    assert "export_speech_plan" in message


def test_run_speech_client_fails_clearly_at_construction_without_rclpy():
    with pytest.raises(RosUnavailableError, match="ROS 2 is not available"):
        RunSpeechClient(
            run_id="run-1",
            generation_id="utt-0042",
            requester_incarnation="rt-agent-1",
            selected_profile="visible-face",
            config_sha256="0" * 64,
            calibration_sha256="1" * 64,
            clock_domain_fingerprint="host-monotonic-zero/v2:deadbeef",
        )


def test_the_failure_is_a_runtime_error_so_a_harness_can_fall_back():
    assert issubclass(RosUnavailableError, RuntimeError)
