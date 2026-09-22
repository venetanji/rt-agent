# rt-agent

`rt-agent` is a hardware-free **listening robot**. It consumes VAD-segmented, diarized
transcript events and, for each finalized utterance, asks a System One model — hosted Jev
at `api.typesafe.ai` or a local Kev server, same wire protocol — one frozen bundle of
typed questions: is this intelligible, is it addressed to me, does it invite an answer
now, should I stay quiet anyway, which expression should I wear, is any of this worth
remembering, is it sensitive. From those calibrated answers a thresholds-and-hysteresis
policy decides speak or wait, picks the robot's facial emotion, and decides whether to
write a memory. System One does not generate text, and it is never asked to: a separate
remote LLM writes what the robot says and the one sentence a memory is stored as, and
neither of them is on the admission path.

The point of the design is what happens when the answer is *no*, which is most of the
time. Every path that is not a confident, complete, well-formed yes ends in silence with
a named reason — a blown deadline, a missing answer, a malformed distribution and an
explicit "do not speak" all produce the same quiet. On the 28-line example conversation
the robot answers the two turns actually addressed to it, stays out of the other
twenty-six, holds a concerned face while two housemates work through a difficult moment,
and stores two durable facts while withholding a medical one. The face contract
(`affect-cue/v1`, `speech-clause/v1`, `speech-plan/v1`) is alice's own, so a decision made
here is playable on alice's bench without alice changing a line; ROS 2 is an optional
bridge, and nothing here has ever moved a servo.

---

## Quickstart

```bash
uv sync --all-extras

# Replay the example conversation against hosted Jev.
TYPESAFE_API_KEY=proxy-injected uv run rt-agent replay examples/kitchen_chat.jsonl \
    --backend jev --scripted-rules examples/scripted_llm.json

# The same run with no network at all, from recorded live answers.
uv run rt-agent replay examples/kitchen_chat.jsonl \
    --backend mock --fixtures tests/fixtures/systemone/kitchen_chat \
    --scripted-rules examples/scripted_llm.json

# Listen to a sound file: VAD, endpointing, Whisper, then the same loop.
TYPESAFE_API_KEY=proxy-injected uv run rt-agent listen \
    --wav tests/fixtures/audio/librispeech-1272-128104-0000.wav --backend jev

# What did it remember?
uv run rt-agent memories --db out/<session>/memories.sqlite3
```

Each run writes one directory, `out/<session>/`:

| File | What is in it |
| --- | --- |
| `decisions.jsonl` | one `decision-record/v1` per utterance: the whole `decision/v1`, the raw answers, and `bundle_ms` / `policy_ms` / `llm_ms` / `total_ms` |
| `affect.jsonl` | every cue the face was actually sent, with the `AuthoredFaceModel`'s prediction of what alice's face would do |
| `clauses.jsonl` | `speech-clause/v1` objects, byte-compatible with alice's own clause fixtures |
| `speech-plan-<n>.json` | one `speech-plan/v1` per reply, valid against alice's JSON Schema |
| `transcript.jsonl` | every turn heard, including the robot's own |
| `memories.jsonl` | every memory write attempt and its outcome |
| `memories.sqlite3` | the memories that survived the faithfulness gate |
| `summary.json` | counts by wait reason, emotion changes, memory outcomes, latency p50/p95/max, resolved model id |

## Commands

| Command | What it does |
| --- | --- |
| `rt-agent replay PATH` | a diarized transcript JSONL through the whole decision loop |
| `rt-agent listen --wav PATH` / `--mic` | sound through the audio front end, then the same loop |
| `rt-agent memories --db PATH` | read the memory store back (`--speaker`, `--search`, `--json`) |
| `rt-agent check-backend --backend jev\|kev` | `GET /v1/models` plus one tiny call: model id and latency, exit 1 on failure |

Shared options: `--backend jev|kev|mock`, `--llm scripted|openai`, `--scripted-rules FILE`,
`--face jsonl|udp|null|ros`, `--udp-host` / `--udp-port`, `--out DIR`, `--session`,
`--deadline-ms`, `--allow-sensitive`, and for `replay` also `--realtime/--no-realtime`,
`--speed`, `--fixtures DIR` and `--record-fixtures DIR`.

`--record-fixtures` saves every live call — the exact state sent, the raw response, the
request id and the measured latency — as a replayable fixture, including the background
`memory_faithful` calls. That is how `tests/fixtures/systemone/kitchen_chat/` was made,
and re-running the recording command is how it is refreshed.

## Environment

| Variable | Meaning |
| --- | --- |
| `TYPESAFE_API_KEY` | credential for hosted Jev. Required by `--backend jev`; `--backend kev` and `--backend mock` need nothing |
| `RT_AGENT_*` | every `AgentConfig` field, e.g. `RT_AGENT_BACKEND`, `RT_AGENT_OUT_DIR`, `RT_AGENT_DEADLINE_MS`, `RT_AGENT_FACE`, `RT_AGENT_VOICE` |
| `RT_AGENT_*` (thresholds) | every `PolicyConfig` field, e.g. `RT_AGENT_ADDRESSED_MIN`, `RT_AGENT_INVITES_MIN`, `RT_AGENT_SAFETY_QUIET_MAX`, `RT_AGENT_WORTH_REMEMBERING_MIN`, `RT_AGENT_ALLOW_SENSITIVE` |
| `RT_AGENT_LLM_BASE_URL`, `RT_AGENT_LLM_MODEL`, `RT_AGENT_LLM_API_KEY` | the OpenAI-compatible reply model used by `--llm openai` |
| `RT_AGENT_LIVE=1` | opt into the tests that make real network calls |
| `RT_AGENT_SKIP_MODELS=1` | skip the tests that download and run Silero and Whisper |

## Architecture

```
  microphone / WAV                      diarized transcript JSONL
        │                                          │
   SileroVad ─ Endpointer ─ Whisper ─ Diarizer     ReplaySource
        └───────────────┬──────────────────────────┘
                        ▼
                  utterance/v1 ──────────────▶ TranscriptLog (append-only, robot turns too)
                        │
          DecisionContext  ≤6 recent turns · ≤3 retrieved memories · robot state
                        │                              ▲
                  render_state ("state-render/v1")     │  MemoryRetriever
                        │                              │
      SystemOneBackend.ask(state, FROZEN decision-bundle/v1)     SqliteMemoryStore
        Jev (hosted) │ Kev (local) │ MockSystemOne (fixtures)             ▲
                        │  asyncio.wait_for(1500 ms)                      │
                  BundleAnswers ── or None ──▶ fail closed                │
                        │                                                 │
                    Policy v1 ──────────▶ decision/v1                     │
                        │                                                 │
        ┌───────────────┼──────────────────────┐                          │
        ▼               ▼                      ▼                          │
 EmotionController   speak?                remember?                      │
 (confidence gate,     │            MemoryWriter (background task)  ──────┘
  refractory, decay)   │              ChatLLM sentence → Jev memory_faithful ≥ 0.7
        │              ▼
  affect-cue/v1   ReplyGenerator → split_clauses → speech-clause/v1 (carrying the cue)
        │              │
        └──────────────┴────▶ FaceBridge: JSONL │ UDP JSON │ ROS 2 │ speech-plan/v1 export
                                                        │
                                                   alice's face
```

Further reading: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (the whole design and its
evidence), [`docs/alice-integration.md`](docs/alice-integration.md) (the three ways a
decision reaches alice), [`docs/audio-frontend.md`](docs/audio-frontend.md),
[`docs/memory.md`](docs/memory.md).

Evidence from live runs:
[`docs/evidence/2026-09-22-jev-bundle-smoke.md`](docs/evidence/2026-09-22-jev-bundle-smoke.md)
(six hand-written scenarios through the bundle) and
[`docs/evidence/2026-09-22-kitchen-chat-live.md`](docs/evidence/2026-09-22-kitchen-chat-live.md)
(the whole harness over the example conversation, per-utterance).

## What is proven, and what is not

**Proven, on real calls against hosted Jev**, recorded and replayable:

* one frozen ten-question bundle per utterance admits exactly the turns addressed to the
  robot and refuses the rest, including a third-person mention that opens with the
  robot's name and a question handed to another person;
* warm latency of 267 ms p50 / 577 ms p95 / 742 ms max, comfortably inside a 1500 ms
  deadline, with the harness itself costing under a millisecond per turn;
* the fail-closed paths: a timeout, a transport error, a rate limit, a missing fixture
  and a malformed answer all produce silence with a named reason, never a crash and
  never a reply;
* a memory is written only when a second System One call agrees the sentence is
  supported by the transcript — a plausible but unsupported sentence was caught at 0.64
  and discarded;
* an exported `speech-plan/v1` validates against alice's own JSON Schema.

**Not proven, and not claimed:**

* **No servo has moved.** ROS hardware admission rejects unconditionally; the
  `speech-plan/v1` export played by alice's host `alice-speak` is the only path that
  could reach hardware, and it has not been run on a robot from here. The
  `AuthoredFaceModel` in the logs is a *simulation* of alice's policy, not a measurement.
* **No real diarization.** The example transcript arrives pre-labelled and the live
  audio path defaults to `SingleSpeakerDiarizer`, which labels everything the same.
  Speaker labels are session-local, anonymous, and never a claim that two voices were
  told apart.
* **Kev is unqualified.** Nothing here measures it. `check-backend --backend kev` against
  a host with no Kev running fails cleanly, and that is the entire extent of the
  evidence.
* **Auth is untested.** This sandbox's proxy rewrites the `Authorization` header, so any
  placeholder is accepted. Credentials will be the first thing to configure on a real
  robot.
* **Sending live room transcripts to a hosted endpoint is an operator decision**, not a
  technical default. It is a new data-egress path (risk R1 in the architecture doc); the
  local Kev backend exists so that decision can go the other way.
* **This is not an emotion model.** The preset table is authored, and the expression is
  the robot's intended delivery, never a diagnosis of what a person feels.

## Tests

```bash
uv run pytest -q                      # the whole suite, network-free and hardware-free
uv run pytest -q tests/harness        # the agent loop, the CLI and the offline replay
uv run ruff check && uv run ruff format --check
uv run mypy src

RT_AGENT_SKIP_MODELS=1 uv run pytest -q          # skip the Silero/Whisper downloads
RT_AGENT_LIVE=1 TYPESAFE_API_KEY=... uv run pytest -q -m live    # real network calls
```

Markers: `live` (real network, opt in with `RT_AGENT_LIVE=1`), `models` (downloads and
runs real ASR/VAD models, opt out with `RT_AGENT_SKIP_MODELS=1`), `slow`.
