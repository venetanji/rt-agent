"""Front-end composition, with fake models so the wiring is tested on its own."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile

from rt_agent.audio.asr import EmptyTranscriptError, FasterWhisperTranscriber
from rt_agent.audio.asr import _confidence_from_avg_logprob as confidence_from_avg_logprob
from rt_agent.audio.diarize import ScriptedDiarizer, SingleSpeakerDiarizer
from rt_agent.audio.endpoint import Endpointer
from rt_agent.audio.pipeline import AudioFrontEnd
from rt_agent.audio.replay import ReplaySource
from rt_agent.audio.sources import Pcm, WavFileSource
from rt_agent.contracts import AudioSource, Transcriber, Transcription, VoiceActivityDetector

RATE = 16_000


class EnergyVad:
    """A stand-in VAD: loud frames are speech. Deterministic and instant."""

    def __init__(self) -> None:
        self.frames = 0

    def probability(self, frame: Pcm) -> float:
        self.frames += 1
        return 1.0 if float(np.sqrt(np.mean(np.square(frame)))) > 0.05 else 0.0

    def reset(self) -> None:
        self.frames = 0


class CannedTranscriber:
    """Returns scripted text per segment; raises EmptyTranscriptError for ``None``."""

    def __init__(self, texts: list[str | None]) -> None:
        self.texts = texts
        self.calls = 0

    async def transcribe(self, pcm: Pcm, sample_rate: int) -> Transcription:
        del pcm, sample_rate
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        if text is None:
            raise EmptyTranscriptError("nothing said")
        return Transcription(text=text, language="en", confidence=0.77)


def speech_then_silence(tmp_path: Path) -> Path:
    """0.5 s tone, 0.5 s silence, 0.5 s tone, 0.5 s silence at 16 kHz."""

    def tone(seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * RATE), dtype=np.float32) / RATE
        return (0.4 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)

    quiet = np.zeros(int(0.5 * RATE), dtype=np.float32)
    samples = np.concatenate([quiet, tone(0.5), quiet, tone(0.5), quiet])
    path = tmp_path / "two-turns.wav"
    soundfile.write(str(path), samples, RATE, subtype="PCM_16")
    return path


def test_the_shipped_pieces_satisfy_their_protocols(tmp_path: Path) -> None:
    path = speech_then_silence(tmp_path)
    assert isinstance(WavFileSource(path), AudioSource)
    assert isinstance(EnergyVad(), VoiceActivityDetector)
    assert isinstance(FasterWhisperTranscriber(), Transcriber)


async def test_two_turns_become_two_utterances(tmp_path: Path) -> None:
    front_end = AudioFrontEnd(
        source=WavFileSource(speech_then_silence(tmp_path)),
        vad=EnergyVad(),
        endpointer=Endpointer(),
        transcriber=CannedTranscriber(["first turn", "second turn"]),
        diarizer=ScriptedDiarizer(["S1", "S2"]),
        session_id="compose",
    )
    utterances = [utterance async for utterance in front_end.utterances()]
    assert [u.utterance_id for u in utterances] == ["u0001", "u0002"]
    assert [u.speaker_label for u in utterances] == ["S1", "S2"]
    assert [u.text for u in utterances] == ["first turn", "second turn"]
    assert all(u.session_id == "compose" for u in utterances)
    assert all(u.is_final and u.language == "en" for u in utterances)
    assert utterances[0].t_end_s <= utterances[1].t_start_s + 1e-6
    assert front_end.segments_seen == 2
    assert front_end.utterances_emitted == 2


async def test_evidence_is_attached_to_every_utterance(tmp_path: Path) -> None:
    front_end = AudioFrontEnd(
        source=WavFileSource(speech_then_silence(tmp_path)),
        vad=EnergyVad(),
        endpointer=Endpointer(),
        transcriber=CannedTranscriber(["turn"]),
        diarizer=SingleSpeakerDiarizer(),
        session_id="evidence",
    )
    utterances = [utterance async for utterance in front_end.utterances()]
    evidence = utterances[0].audio
    assert evidence is not None
    assert evidence.vad_mean_prob > 0.5
    assert evidence.voiced_fraction > 0.5
    assert evidence.rms > 0.0
    assert evidence.clipped_fraction == 0.0
    assert evidence.overlap_fraction is None
    assert utterances[0].asr_confidence == pytest.approx(0.77)


async def test_a_segment_whisper_cannot_read_is_dropped_not_faked(tmp_path: Path) -> None:
    front_end = AudioFrontEnd(
        source=WavFileSource(speech_then_silence(tmp_path)),
        vad=EnergyVad(),
        endpointer=Endpointer(),
        transcriber=CannedTranscriber([None, "second turn"]),
        diarizer=SingleSpeakerDiarizer(),
        session_id="drop",
    )
    utterances = [utterance async for utterance in front_end.utterances()]
    assert [u.text for u in utterances] == ["second turn"]
    # The dropped segment must not consume an utterance id.
    assert [u.utterance_id for u in utterances] == ["u0001"]
    assert front_end.segments_seen == 2
    assert front_end.empty_transcripts == 1


async def test_reset_restarts_ids_and_clocks(tmp_path: Path) -> None:
    path = speech_then_silence(tmp_path)
    front_end = AudioFrontEnd(
        source=WavFileSource(path),
        vad=EnergyVad(),
        endpointer=Endpointer(),
        transcriber=CannedTranscriber(["turn"]),
        diarizer=SingleSpeakerDiarizer(),
        session_id="reset",
    )
    first = [utterance async for utterance in front_end.utterances()]
    front_end.source = WavFileSource(path)
    front_end.reset()
    second = [utterance async for utterance in front_end.utterances()]
    assert [u.utterance_id for u in first] == [u.utterance_id for u in second]
    assert first[0].t_start_s == pytest.approx(second[0].t_start_s)


async def test_the_replay_source_and_the_front_end_emit_the_same_type(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(
        '{"speaker": "S1", "text": "hello", "t_start_s": 0.0, "t_end_s": 1.0}\n',
        encoding="utf-8",
    )
    replayed = [utterance async for utterance in ReplaySource(path).utterances()]
    front_end = AudioFrontEnd(
        source=WavFileSource(speech_then_silence(tmp_path)),
        vad=EnergyVad(),
        endpointer=Endpointer(),
        transcriber=CannedTranscriber(["hello"]),
        diarizer=SingleSpeakerDiarizer(),
        session_id="replay",
    )
    live = [utterance async for utterance in front_end.utterances()]
    assert type(replayed[0]) is type(live[0])
    assert replayed[0].schema_version == live[0].schema_version == "utterance/v1"


@pytest.mark.parametrize(
    ("avg_logprob", "expected"),
    [(0.0, 1.0), (-0.1, math.exp(-0.1)), (-2.0, math.exp(-2.0)), (-40.0, 0.0)],
)
def test_confidence_is_the_geometric_mean_token_probability(
    avg_logprob: float, expected: float
) -> None:
    assert confidence_from_avg_logprob(avg_logprob) == pytest.approx(expected, abs=1e-9)


def test_confidence_survives_a_non_finite_logprob() -> None:
    assert confidence_from_avg_logprob(float("-inf")) == 0.0
    assert confidence_from_avg_logprob(float("nan")) == 0.0


async def test_the_transcriber_rejects_non_audio() -> None:
    with pytest.raises(ValueError, match="finite 1-D mono float32 PCM"):
        await FasterWhisperTranscriber().transcribe(np.zeros((2, 2), dtype=np.float32), RATE)
