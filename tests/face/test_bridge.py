"""JSONL, UDP, composite and null bridges — record shapes and transport behaviour."""

from __future__ import annotations

import json
import socket

import pytest

from rt_agent.contracts import FaceBridge
from rt_agent.face.authored_face import AuthoredFaceModel
from rt_agent.face.bridge import (
    MAX_DATAGRAM_BYTES,
    CompositeFaceBridge,
    JsonlFaceBridge,
    NullFaceBridge,
    UdpJsonFaceBridge,
)
from rt_agent.face.clauses import build_speech_clauses
from tests.face.conftest import make_cue


@pytest.fixture
def clauses():
    return build_speech_clauses("utt-0042", ["Ten past nine.", "Nearly late."], make_cue(), 29)


def test_every_bridge_satisfies_the_face_bridge_protocol(tmp_path):
    bridges = [
        JsonlFaceBridge(tmp_path),
        UdpJsonFaceBridge("127.0.0.1", 9917),
        NullFaceBridge(),
        CompositeFaceBridge(NullFaceBridge()),
    ]
    for bridge in bridges:
        assert isinstance(bridge, FaceBridge)
        bridge.close()


def test_jsonl_writes_the_documented_affect_record(tmp_path):
    with JsonlFaceBridge(tmp_path) as bridge:
        bridge.emit_affect(make_cue("warm", 0.5))

    lines = (tmp_path / "affect.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"kind", "seq", "sent_at", "cue", "face"}
    assert record["kind"] == "affect"
    assert record["seq"] == 0
    assert record["cue"]["schema_version"] == "affect-cue/v1"
    assert record["cue"]["preset"] == "warm"
    assert record["cue"]["vector"] == [0.5, 0.2, 0.1]
    assert record["face"]["anchor"] == "smile_open"
    assert record["face"]["amplitude"] == pytest.approx(0.25)
    assert record["face"]["channels"]["right_mouth_corner"] == pytest.approx(-0.25)
    assert record["face"]["visible"] is True


def test_jsonl_clause_lines_are_plain_alice_clauses(tmp_path, clauses):
    with JsonlFaceBridge(tmp_path) as bridge:
        bridge.emit_clauses(clauses)

    lines = (tmp_path / "clauses.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    # No envelope: the line is exactly a speech-clause/v1 object, replayable by alice.
    assert set(first) == {
        "schema_version",
        "generation_id",
        "clause_id",
        "sequence",
        "text",
        "vector",
        "intensity",
        "seed",
        "end_of_response",
    }
    assert first["sequence"] == 0
    assert first["end_of_response"] is False
    assert json.loads(lines[1])["end_of_response"] is True


def test_jsonl_appends_across_calls_and_numbers_cues(tmp_path):
    bridge = JsonlFaceBridge(tmp_path)
    bridge.emit_affect(make_cue())
    bridge.emit_affect(make_cue("sad", 1.0))
    bridge.close()
    seqs = [
        json.loads(line)["seq"] for line in (tmp_path / "affect.jsonl").read_text().splitlines()
    ]
    assert seqs == [0, 1]


def test_jsonl_creates_the_directory_lazily_and_skips_empty_clause_runs(tmp_path):
    target = tmp_path / "deep" / "out"
    bridge = JsonlFaceBridge(target)
    assert not target.exists()
    bridge.emit_clauses([])
    assert not target.exists()
    bridge.emit_affect(make_cue())
    assert (target / "affect.jsonl").exists()
    assert not (target / "clauses.jsonl").exists()
    bridge.close()
    bridge.close()  # idempotent


def test_jsonl_face_can_be_switched_off(tmp_path):
    with JsonlFaceBridge(tmp_path, face=None) as bridge:
        bridge.emit_affect(make_cue())
    assert json.loads((tmp_path / "affect.jsonl").read_text())["face"] is None


def test_jsonl_accepts_a_prebuilt_face_model(tmp_path):
    with JsonlFaceBridge(tmp_path, face=AuthoredFaceModel("visible")) as bridge:
        bridge.emit_affect(make_cue("happy", 1.0))
    record = json.loads((tmp_path / "affect.jsonl").read_text())
    assert record["face"]["profile"] == "visible"
    assert record["face"]["amplitude"] == pytest.approx(1.0)


@pytest.fixture
def receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    yield sock
    sock.close()


def test_udp_round_trip_on_localhost(receiver, clauses):
    host, port = receiver.getsockname()
    with UdpJsonFaceBridge(host, port) as bridge:
        assert bridge.address == (host, port)
        bridge.emit_affect(make_cue("sad", 1.0))
        bridge.emit_clauses(clauses)
        assert bridge.sent == 3

    affect = json.loads(receiver.recv(MAX_DATAGRAM_BYTES))
    assert affect["kind"] == "affect"
    assert affect["seq"] == 0
    assert affect["cue"]["preset"] == "sad"
    assert affect["face"]["anchor"] == "frown_closed"

    received = [json.loads(receiver.recv(MAX_DATAGRAM_BYTES)) for _ in range(2)]
    assert [item["kind"] for item in received] == ["clause", "clause"]
    assert [item["seq"] for item in received] == [1, 2]
    assert [item["index"] for item in received] == [0, 1]
    assert all(item["count"] == 2 for item in received)
    assert received[0]["clause"]["schema_version"] == "speech-clause/v1"
    assert received[1]["clause"]["end_of_response"] is True
    assert set(received[0]) == {"kind", "seq", "sent_at", "clause", "index", "count"}


def test_udp_send_to_nobody_is_not_an_error():
    with UdpJsonFaceBridge("127.0.0.1", 9) as bridge:
        bridge.emit_affect(make_cue())
    assert bridge.sent == 1


def test_a_full_length_clause_still_fits_one_datagram(receiver):
    host, port = receiver.getsockname()
    with UdpJsonFaceBridge(host, port) as bridge:
        bridge.emit_clauses(build_speech_clauses("g", ["x" * 1000], make_cue(), 0))
        assert bridge.sent == 1
    assert len(receiver.recv(MAX_DATAGRAM_BYTES * 2)) <= MAX_DATAGRAM_BYTES


def test_udp_refuses_an_oversized_datagram(receiver, monkeypatch):
    monkeypatch.setattr("rt_agent.face.bridge.MAX_DATAGRAM_BYTES", 64)
    host, port = receiver.getsockname()
    with UdpJsonFaceBridge(host, port) as bridge, pytest.raises(ValueError, match="byte limit"):
        bridge.emit_affect(make_cue())


def test_udp_rejects_an_impossible_port():
    with pytest.raises(ValueError, match="port"):
        UdpJsonFaceBridge("127.0.0.1", 0)


def test_udp_does_not_close_a_socket_it_was_given(receiver):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    host, port = receiver.getsockname()
    bridge = UdpJsonFaceBridge(host, port, sock=sock)
    bridge.close()
    sock.sendto(b"{}", (host, port))  # still open
    sock.close()


def test_composite_fans_out_to_every_bridge(tmp_path, clauses):
    null = NullFaceBridge()
    jsonl = JsonlFaceBridge(tmp_path)
    composite = CompositeFaceBridge(jsonl, null)
    composite.emit_affect(make_cue())
    composite.emit_clauses(clauses)
    composite.close()

    assert (null.affect_count, null.clause_count, null.closed) == (1, 2, True)
    assert len((tmp_path / "clauses.jsonl").read_text().splitlines()) == 2


def test_composite_still_reaches_later_bridges_when_one_raises():
    class Broken:
        def emit_affect(self, cue):
            raise RuntimeError("preview died")

        def emit_clauses(self, clauses):
            raise RuntimeError("preview died")

        def close(self):
            raise RuntimeError("preview died")

    null = NullFaceBridge()
    composite = CompositeFaceBridge(Broken(), null)
    with pytest.raises(RuntimeError, match="preview died"):
        composite.emit_affect(make_cue())
    assert null.affect_count == 1


def test_null_bridge_counts_without_side_effects(clauses):
    null = NullFaceBridge()
    null.emit_affect(make_cue())
    null.emit_clauses(clauses)
    null.close()
    assert (null.affect_count, null.clause_count, null.closed) == (1, 2, True)
