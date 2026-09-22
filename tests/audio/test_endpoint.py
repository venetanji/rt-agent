"""The endpointer state machine, driven by synthetic probability sequences.

Pure logic: no model, no audio file, no event loop. Every constant asserted here is
a port of alice's qualified endpointer (``alice-workspace/src/alice/conversation/vad.py``),
so a change that breaks one of these tests is a change to turn-taking behaviour that
was tuned on a real robot.
"""

from __future__ import annotations

import numpy as np
import pytest

from rt_agent.audio.endpoint import AudioSegment, Endpointer

FRAME = 512
RATE = 16_000
SILENT = np.zeros(FRAME, dtype=np.float32)


def run(
    endpointer: Endpointer,
    probabilities: list[float],
    frame: np.ndarray = SILENT,
) -> list[AudioSegment]:
    """Push a whole probability sequence and collect every segment that closed."""
    segments = []
    for probability in probabilities:
        segment = endpointer.push(frame, probability)
        if segment is not None:
            segments.append(segment)
    return segments


def test_thresholds_match_the_ported_alice_values() -> None:
    endpointer = Endpointer()
    assert endpointer.start_probability == 0.5
    assert endpointer.release_probability == 0.35
    assert endpointer.endpoint_silence_samples == FRAME * 10 == int(0.32 * RATE)
    assert endpointer.preroll_samples == 3200 == int(0.2 * RATE)
    assert endpointer.max_segment_samples == RATE * 15
    assert endpointer.min_voiced_samples == 1536


def test_one_turn_closes_after_320ms_of_release() -> None:
    endpointer = Endpointer()
    # 10 quiet frames (preroll), 20 voiced frames, then silence.
    segments = run(endpointer, [0.0] * 10 + [0.9] * 20 + [0.0] * 15)
    assert len(segments) == 1
    segment = segments[0]
    # Preroll is capped at 200 ms even though 320 ms of silence preceded the trigger.
    assert segment.samples.size == 3200 + 30 * FRAME
    assert segment.sample_rate == RATE
    # The turn closed on the 10th released frame, not the 15th.
    assert segment.t_end_s == pytest.approx(40 * FRAME / RATE)
    assert segment.t_start_s == pytest.approx(segment.t_end_s - segment.duration_s)
    assert segment.t_start_s == pytest.approx(0.12)


def test_evidence_excludes_the_preroll_from_its_denominator() -> None:
    endpointer = Endpointer()
    segments = run(endpointer, [0.0] * 10 + [0.9] * 20 + [0.0] * 10)
    evidence = segments[0].evidence
    # 30 post-trigger frames: 20 at 0.9 and the 10 released frames at 0.0.
    assert evidence.vad_mean_prob == pytest.approx(20 * 0.9 / 30)
    assert evidence.voiced_fraction == pytest.approx(20 / 30)
    # ...while the waveform itself still carries the 200 ms of preroll.
    assert evidence.duration_s == pytest.approx((3200 + 30 * FRAME) / RATE)
    assert evidence.overlap_fraction is None


def test_a_dip_between_release_and_start_does_not_end_the_turn() -> None:
    endpointer = Endpointer()
    # 0.4 is below the 0.5 start threshold but above the 0.35 release threshold:
    # hysteresis, so a quiet syllable must not be counted as endpoint silence.
    segments = run(endpointer, [0.9] * 5 + [0.4] * 30 + [0.9] * 5 + [0.0] * 10)
    assert len(segments) == 1
    assert segments[0].samples.size == 50 * FRAME
    assert segments[0].evidence.voiced_fraction == pytest.approx(10 / 50)


def test_silence_counter_resets_on_a_voiced_frame() -> None:
    endpointer = Endpointer()
    segments = run(endpointer, [0.9] * 3 + [0.0] * 9 + [0.9] * 3 + [0.0] * 9)
    assert segments == []
    assert endpointer.speaking is True
    assert endpointer.silence_ms == 9 * FRAME * 1000 // RATE


def test_a_segment_with_too_little_voiced_audio_is_dropped() -> None:
    endpointer = Endpointer()
    # 2 voiced frames = 1024 samples < the 1536-sample (96 ms) floor.
    segments = run(endpointer, [0.9] * 2 + [0.0] * 10)
    assert segments == []
    assert endpointer.dropped_short == 1
    assert endpointer.last_discard_reason == "too_little_voiced_audio"


def test_an_overlong_utterance_is_discarded_and_rearms_after_320ms_quiet() -> None:
    endpointer = Endpointer()
    frames_to_overflow = RATE * 15 // FRAME + 1
    segments = run(endpointer, [0.9] * frames_to_overflow)
    assert segments == []
    assert endpointer.discarded_overlong == 1
    assert endpointer.last_discard_reason == "overlong_utterance"

    # Still loud: the detector refuses to re-arm, so no new segment can start.
    assert run(endpointer, [0.9] * 20 + [0.0] * 10) == []
    # 320 ms of continuous quiet re-arms it; the next turn is captured normally.
    segments = run(endpointer, [0.0] * 10 + [0.9] * 10 + [0.0] * 10)
    assert len(segments) == 1


def test_the_overlong_cap_is_not_a_truncation() -> None:
    endpointer = Endpointer()
    run(endpointer, [0.9] * (RATE * 15 // FRAME + 1))
    # Nothing is buffered after a discard: a 15 s monologue leaves no partial turn.
    assert endpointer.speaking is False
    assert endpointer.preroll.size == 0


def test_flush_closes_a_turn_that_the_stream_cut_off() -> None:
    endpointer = Endpointer()
    assert run(endpointer, [0.9] * 20) == []
    segment = endpointer.flush()
    assert segment is not None
    assert segment.samples.size == 20 * FRAME
    assert segment.t_end_s == pytest.approx(20 * FRAME / RATE)
    assert endpointer.flush() is None


def test_flush_respects_the_minimum_voiced_floor() -> None:
    endpointer = Endpointer()
    run(endpointer, [0.9] * 2)
    assert endpointer.flush() is None
    assert endpointer.dropped_short == 1


def test_two_turns_in_one_stream_get_independent_evidence() -> None:
    endpointer = Endpointer()
    sequence = [0.9] * 10 + [0.0] * 10 + [0.6] * 10 + [0.0] * 10
    segments = run(endpointer, sequence)
    assert len(segments) == 2
    assert segments[0].evidence.vad_mean_prob == pytest.approx(10 * 0.9 / 20)
    assert segments[1].evidence.vad_mean_prob == pytest.approx(10 * 0.6 / 20)
    assert segments[0].t_end_s < segments[1].t_start_s + 1e-9


def test_rms_and_clipping_are_measured_on_the_emitted_waveform() -> None:
    endpointer = Endpointer()
    loud = np.full(FRAME, 1.0, dtype=np.float32)
    for _ in range(20):
        endpointer.push(loud, 0.9)
    segment = endpointer.flush()
    assert segment is not None
    assert segment.evidence.rms == pytest.approx(1.0)
    assert segment.evidence.clipped_fraction == pytest.approx(1.0)


def test_reset_clears_the_stream_clock() -> None:
    endpointer = Endpointer()
    run(endpointer, [0.9] * 10 + [0.0] * 10)
    endpointer.reset()
    assert endpointer.position == 0
    assert endpointer.speaking is False
    segment = run(endpointer, [0.9] * 10 + [0.0] * 10)[0]
    assert segment.t_end_s == pytest.approx(20 * FRAME / RATE)


@pytest.mark.parametrize(
    ("frame", "probability"),
    [
        (np.zeros(256, dtype=np.float32), 0.5),
        (np.zeros(FRAME, dtype=np.float32), 1.5),
        (np.zeros(FRAME, dtype=np.float32), float("nan")),
        (np.full(FRAME, np.nan, dtype=np.float32), 0.5),
    ],
)
def test_malformed_input_is_rejected(frame: np.ndarray, probability: float) -> None:
    with pytest.raises(ValueError, match="invalid endpointer frame or probability"):
        Endpointer().push(frame, probability)


def test_threshold_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="release <= start"):
        Endpointer(start_probability=0.3, release_probability=0.6)
