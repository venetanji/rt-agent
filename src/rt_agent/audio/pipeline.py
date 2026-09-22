"""Compose source → VAD → endpointer → ASR → diarizer into a stream of utterances.

``AudioFrontEnd`` is the only place in this package where the pieces know about each
other, and it is deliberately thin: pull frames, score each one, hand whole segments
to Whisper, stamp the result with a session-local speaker label and a stable id.
Everything it emits is a final :class:`~rt_agent.contracts.Utterance` with the
acoustic evidence the decision bundle will quote.

Threading, following alice's shape: the VAD runs inline in the consuming coroutine
(~1 ms per 32 ms frame, so it cannot fall behind) while ASR runs in a thread
executor. The front end transcribes one segment at a time — a robot that starts
answering turn *n* while still decoding turn *n+1* would need barge-in handling,
which is out of PoC scope.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path

from rt_agent.audio.asr import EmptyTranscriptError, FasterWhisperTranscriber
from rt_agent.audio.diarize import SingleSpeakerDiarizer
from rt_agent.audio.endpoint import AudioSegment, Endpointer
from rt_agent.audio.sources import WavFileSource
from rt_agent.audio.vad import SileroVad
from rt_agent.contracts import AudioSource, Diarizer, Transcriber, Utterance, VoiceActivityDetector

__all__ = ["AudioFrontEnd", "transcribe_wav"]


class AudioFrontEnd:
    """Audio in, final utterances out.

    ``utterance_id`` runs ``u0001``, ``u0002``, ... within one front end instance;
    ``t_start_s`` / ``t_end_s`` are seconds from the first sample the endpointer saw,
    which is the "relative to session start" clock the contracts expect.
    """

    def __init__(
        self,
        source: AudioSource,
        vad: VoiceActivityDetector,
        endpointer: Endpointer,
        transcriber: Transcriber,
        diarizer: Diarizer,
        session_id: str,
    ) -> None:
        self.source = source
        self.vad = vad
        self.endpointer = endpointer
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.session_id = session_id
        #: Counters and timings, useful in logs and in the end-to-end test report.
        self.frames_seen = 0
        self.segments_seen = 0
        self.utterances_emitted = 0
        self.empty_transcripts = 0
        self.vad_seconds = 0.0
        self.asr_seconds = 0.0

    def reset(self) -> None:
        """Start a fresh session: VAD state, endpointer clock, speaker labels, ids."""
        self.vad.reset()
        self.endpointer.reset()
        self.diarizer.reset()
        self.frames_seen = 0
        self.segments_seen = 0
        self.utterances_emitted = 0
        self.empty_transcripts = 0
        self.vad_seconds = 0.0
        self.asr_seconds = 0.0

    async def utterances(self) -> AsyncIterator[Utterance]:
        """Drive the source to exhaustion, yielding one utterance per decoded segment."""
        async for frame in self.source.frames():
            self.frames_seen += 1
            started = time.perf_counter()
            probability = self.vad.probability(frame)
            self.vad_seconds += time.perf_counter() - started
            segment = self.endpointer.push(frame, probability)
            if segment is not None:
                utterance = await self._to_utterance(segment)
                if utterance is not None:
                    yield utterance
        tail = self.endpointer.flush()
        if tail is not None:
            utterance = await self._to_utterance(tail)
            if utterance is not None:
                yield utterance

    def __aiter__(self) -> AsyncIterator[Utterance]:
        """``async for utterance in front_end`` is the same as iterating :meth:`utterances`."""
        return self.utterances()

    async def _to_utterance(self, segment: AudioSegment) -> Utterance | None:
        self.segments_seen += 1
        started = time.perf_counter()
        try:
            transcription = await self.transcriber.transcribe(segment.samples, segment.sample_rate)
        except EmptyTranscriptError:
            # Silence, a cough, or a segment Whisper refuses to guess at. The audio
            # evidence stays in the counters; there is no utterance to decide about.
            self.empty_transcripts += 1
            return None
        finally:
            self.asr_seconds += time.perf_counter() - started
        label = self.diarizer.label(segment.samples, segment.sample_rate)
        self.utterances_emitted += 1
        return Utterance(
            utterance_id=f"u{self.utterances_emitted:04d}",
            session_id=self.session_id,
            speaker_label=label,
            text=transcription.text,
            t_start_s=segment.t_start_s,
            t_end_s=segment.t_end_s,
            asr_confidence=transcription.confidence,
            language=transcription.language,
            audio=segment.evidence,
            is_final=True,
        )

    async def aclose(self) -> None:
        """Release the source."""
        await self.source.aclose()


async def transcribe_wav(
    path: str | Path,
    *,
    session_id: str = "wav",
    model_size: str = "base.en",
    compute_type: str = "int8",
    device: str = "cpu",
    speaker_label: str = "S1",
    transcriber: Transcriber | None = None,
    vad: VoiceActivityDetector | None = None,
    diarizer: Diarizer | None = None,
    endpointer: Endpointer | None = None,
) -> list[Utterance]:
    """Run one sound file through the whole front end and collect the utterances.

    The convenience path for tests, fixture recording and "what did this recording
    actually say" questions. Pass ``transcriber`` / ``vad`` to reuse already-loaded
    models across calls; otherwise both are constructed here and thrown away.
    """
    source = WavFileSource(path)
    front_end = AudioFrontEnd(
        source=source,
        vad=vad if vad is not None else SileroVad(),
        endpointer=endpointer if endpointer is not None else Endpointer(),
        transcriber=transcriber
        if transcriber is not None
        else FasterWhisperTranscriber(
            model_size=model_size, compute_type=compute_type, device=device
        ),
        diarizer=diarizer if diarizer is not None else SingleSpeakerDiarizer(speaker_label),
        session_id=session_id,
    )
    return [utterance async for utterance in front_end.utterances()]
