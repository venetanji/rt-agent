# The audio front end

`rt_agent.audio` turns sound — or a recorded transcript — into
`Utterance` objects (`utterance/v1`), the only thing the rest of the harness
consumes. Everything downstream of this package is hardware-free by construction.

```
MicSource / WavFileSource ──frames(512 @ 16 kHz)──▶ SileroVad ──p(speech)──▶ Endpointer
                                                                                  │
                                                                           AudioSegment
                                                                                  │
                                            FasterWhisperTranscriber ◀────────────┤
                                                       Diarizer      ◀────────────┘
                                                            │
                                                     Utterance (final)

ReplaySource ──────────────────────────────────────────────────────────▶ Utterance (final)
```

`AudioFrontEnd` composes the top path; `ReplaySource` is the bottom one. They emit
the same type, so a harness written against `Utterance` cannot tell which it is
talking to — that is the point.

## Which path to use

| Situation | Source |
|---|---|
| Tests, CI, fixture recording, policy demos | `ReplaySource` (no models, no hardware) |
| A recording you want transcribed | `WavFileSource` + `AudioFrontEnd`, or `transcribe_wav()` |
| A robot with a microphone | `MicSource` + `AudioFrontEnd` |

This sandbox, and most containers, have no PortAudio and no capture device;
`MicSource` raises `MicrophoneUnavailableError` with a message that says which of
the two is missing, instead of an `OSError` from inside an import.

## Frames

Mono float32 PCM, 16 kHz, 512 samples (32 ms) per frame — not negotiable: Silero v5
only accepts 512-sample windows at 16 kHz, and it is the frame alice's ReSpeaker
bench captures with (`alice-workspace/src/alice/conversation/audio.py:191`).
`WavFileSource` accepts anything `libsndfile` reads, downmixes to mono and
linearly resamples to 16 kHz (with a box low-pass when downsampling; see
`resample_linear`). Sources expose a monotonic `sample_index` — the start offset of
the most recently yielded frame — and `elapsed_s`.

## VAD: Silero v5 on onnxruntime, without torch

`SileroVad` loads the `silero_vad.onnx` that the `silero-vad` wheel ships and runs
it on a single-threaded CPU `InferenceSession`. It never imports `silero_vad`
itself, because that package's `__init__` imports torch, and **the robot host should
not need torch to listen**. The model file is located with
`importlib.util.find_spec` (which does not execute the package), then the env
override `RT_AGENT_SILERO_ONNX`, then a cache, then an optional download of the
pinned upstream URL.

The v5 state handling is ported from alice
(`alice-workspace/src/alice/conversation/vad.py:128-166`): a `(2, 1, 128)` recurrent
state carried across frames, plus the previous window's last 64 samples prepended to
each inference.

Every instance records its weights' digest:

```python
vad = SileroVad()
vad.model_sha256   # '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
vad.is_qualified   # True — byte-identical to the model alice qualified
vad.model_id       # 'silero_vad.onnx@sha256:1a153a22…'
```

Pass `expected_sha256=QUALIFIED_SILERO_SHA256` to make a mismatch fatal. One
instance per session: the recurrent state is conversational context, so call
`reset()` at a session boundary and never share an instance between streams.

## Endpointing

`Endpointer` is a port of alice's `Endpoint` (`vad.py:33-127`) with its tuned
constants intact:

| Parameter | Value | Why |
|---|---|---|
| `start_probability` | 0.5 | speech starts |
| `release_probability` | 0.35 | hysteresis — a quiet syllable must not cut the turn |
| `endpoint_silence_s` | 0.32 (320 ms) | released audio that closes the turn |
| `preroll_s` | 0.2 (200 ms) | kept ahead of the trigger so the first phoneme survives |
| `max_segment_s` | 15.0 | hard cap |
| `min_voiced_s` | 0.096 (96 ms) | below this the segment is a click, not a turn |

Two behaviours are easy to get wrong and are worth restating:

- **The preroll is in the audio but not in the statistics.** `vad_mean_prob` and
  `voiced_fraction` are measured over post-trigger frames only; the waveform still
  carries the 200 ms of preroll. (alice `vad.py:34-39`.)
- **An overlong utterance is discarded, not truncated**, and the endpointer then
  refuses to start a new turn until it has seen 320 ms of continuous quiet. A
  15-second monologue is a television or a stuck microphone, not a turn.

A stream that ends mid-speech never reaches the 320 ms tail, so call
`Endpointer.flush()` at end-of-stream (`AudioFrontEnd` does this for you).

### `AudioSegment`

```python
@dataclass(frozen=True)
class AudioSegment:
    samples: NDArray[np.float32]   # mono float32, includes the preroll
    sample_rate: int               # 16000
    t_start_s: float               # seconds from the first sample the endpointer saw
    t_end_s: float                 # t_end_s - t_start_s == len(samples) / sample_rate
    evidence: AudioEvidence        # contracts.AudioEvidence
```

`evidence` carries `vad_mean_prob`, `voiced_fraction`, `duration_s`, `rms`,
`clipped_fraction` (`|x| >= 0.99`), and `overlap_fraction = None` — the PoC runs no
overlapped-speech model, and an unknown value stays `None` rather than becoming a
fabricated `0.0`. (alice does measure overlap with a pyannote-segmentation-3.0
sherpa-onnx export; wiring that in is the obvious next step.)

## ASR

`FasterWhisperTranscriber(model_size="base.en", compute_type="int8", device="cpu")`
runs CTranslate2 Whisper locally; weights come from Hugging Face on first use. The
model loads lazily and every decode runs in a thread executor, so the event loop
keeps servicing frames.

- `language="en"`, fixed: `base.en` cannot decode anything else and the question
  bundle is English.
- `beam_size=1` — greedy. On short endpointed turns beam 5 buys little accuracy and
  costs real latency.
- `vad_filter=False` — the audio is *already* endpointed. Letting Whisper
  re-segment it would contradict the evidence attached to the utterance.

**Confidence.** Whisper reports `avg_logprob` per segment: the mean natural log
probability of the decoded tokens. We take the duration-weighted mean across
segments, exponentiate once and clamp to `[0, 1]`, so `asr_confidence` is the
geometric-mean per-token probability. It is a monotone rescaling of the model's own
likelihood, **not a calibrated probability of being correct** — the policy must read
it as evidence, never as accuracy. Typical values: ~0.9 on clean read speech,
0.3–0.5 when the decoder is guessing.

A segment Whisper returns nothing for raises `EmptyTranscriptError` (the contract
forbids empty text); `AudioFrontEnd` catches it and drops the segment without
consuming an utterance id.

## Diarization

Labels are **session-local, anonymous, and advisory**. No real names, no voice
embeddings on disk, ever.

- `SingleSpeakerDiarizer(label="S1")` — one microphone, one person. The honest PoC
  default.
- `ScriptedDiarizer(labels)` — deterministic round-robin, for tests and for audio
  whose speaker order is already known.
- `EmbeddingDiarizer` — a placeholder that raises `NotImplementedError`. A real
  implementation would wrap [Diart](https://github.com/juanmc2005/diart) or
  [pyannote.audio](https://github.com/pyannote/pyannote-audio), keep clusters in
  memory for one session only, and still be treated as advisory: nothing downstream
  may gate a memory, an addressing decision or a safety behaviour on a label alone.

## Replay transcript format (JSONL)

One JSON object per line, UTF-8. Blank lines and `#` comment lines are skipped.

```jsonl
# a diarized transcript fixture
{"speaker": "S1", "text": "Alice, what time is it?", "t_start_s": 0.0, "t_end_s": 1.8}
{"speaker": "S2", "text": "She is not going to know that.", "t_start_s": 2.1, "t_end_s": 4.0, "audio": {"vad_mean_prob": 0.88, "voiced_fraction": 0.63, "rms": 0.05, "clipped_fraction": 0.0}}
{"speaker": "ROBOT", "text": "It is just after four.", "t_start_s": 4.2, "t_end_s": 5.6, "asr_confidence": 0.91, "utterance_id": "robot-1"}
```

| Key | Required | Notes |
|---|---|---|
| `speaker` | yes | anonymous label, `^[A-Z][A-Z0-9_-]{0,31}$` — `S1`, `S2`, `ROBOT` |
| `text` | yes | 1–1000 characters |
| `t_start_s`, `t_end_s` | yes | seconds from session start; `t_start_s` must not go backwards |
| `audio` | no | omit it entirely when there is no evidence |
| `asr_confidence` | no | `[0, 1]`, defaults to `null` |
| `language` | no | defaults to `"en"` |
| `utterance_id` | no | defaults to `u0001`, `u0002`, … in file order |
| `is_final` | no | defaults to `true` |

Inside `audio`, the four **measured** fields `vad_mean_prob`, `voiced_fraction`,
`rms` and `clipped_fraction` are required; `duration_s` defaults to
`t_end_s - t_start_s` and `overlap_fraction` to `null`. A partial `audio` object is
an error rather than something to fill in: a replay must not invent evidence the
decision policy is going to read. Unknown keys on either object are a hard error —
a typo in a fixture should fail the run, not quietly change a decision.

`session_id` comes from the constructor, not the file:

```python
ReplaySource("examples/kitchen.jsonl", session_id="demo", realtime=False, speed=1.0)
```

`realtime=True` paces the stream against the transcript's own clock: utterance *n*
is released `(t_start_s - first_t_start_s) / speed` seconds after the first one.

## Using it

```python
from rt_agent.audio import (
    AudioFrontEnd, Endpointer, FasterWhisperTranscriber,
    ReplaySource, SileroVad, SingleSpeakerDiarizer, WavFileSource, transcribe_wav,
)

# hardware-free
async for utterance in ReplaySource("examples/kitchen.jsonl", session_id="demo"):
    ...

# one call, one file
utterances = await transcribe_wav("recording.wav", session_id="s1")

# the full front end, models reused across sessions
front_end = AudioFrontEnd(
    source=WavFileSource("recording.wav"),
    vad=SileroVad(),
    endpointer=Endpointer(),
    transcriber=FasterWhisperTranscriber(),
    diarizer=SingleSpeakerDiarizer("S1"),
    session_id="s1",
)
async for utterance in front_end.utterances():
    ...
```

`AudioFrontEnd` keeps counters worth logging: `frames_seen`, `segments_seen`,
`utterances_emitted`, `empty_transcripts`, `vad_seconds`, `asr_seconds`.

## Tests

`tests/audio/` is offline and sub-second apart from one end-to-end test marked
`@pytest.mark.models`, which runs the committed LibriSpeech clip
(`tests/fixtures/audio/`, CC BY 4.0, see its `LICENSE.md`) through Silero and
`base.en`. Skip it with `RT_AGENT_SKIP_MODELS=1`.

Measured on this host (5.855 s clip, 183 frames): Silero session load 70 ms, VAD
39 ms total (0.0067 × realtime), `base.en` int8 decode 2.28 s for a 5.48 s segment
(0.42 × realtime), one segment, transcript
*"Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his
gospel."*
