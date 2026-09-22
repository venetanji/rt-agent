# Integrating rt-agent with alice

How the decisions this package makes reach a robot face. Three paths exist; they differ
only in how much alice has to change, and none of them changes who is in charge — **alice
owns the face, rt-agent only proposes.**

| Path | alice changes needed | Can move a real servo today | rt-agent entry point |
|---|---|---|---|
| **A** — committed clauses during speech | none, the seam exists | no (ROS hardware admission rejects) | `rt_agent.face.ros_bridge.RunSpeechClient` |
| **B** — idle `AffectCue` while listening | **four**, proposed below | no | `rt_agent.face.ros_bridge.affect_cue_msg_dict` |
| **C** — `speech-plan/v1` export played by `alice-speak` | none | **yes**, the only path that can | `rt_agent.face.speech_plan.export_speech_plan` |

This document is the rt-agent-side companion to `docs/ARCHITECTURE.md` §7; where §7 states
the design decision, this states the API, the field tables and the commands. All alice
references are to `/home/user/alice-workspace` at commit `8ddcb6a`, branch
`claude/sharp-volta-wt499g`.

Everything rt-agent sends is an *authored* affect vector from a policy table, mapped from a
System One answer. Nothing here is a fitted emotion model. alice's own configs carry
`"trained_expression_model": false` and `affect_anchor_mappings: []`, and rt-agent must not
be described as doing better than that.

---

## 0. What rt-agent produces

Two contract objects, both defined in `src/rt_agent/contracts`:

* **`AffectCue`** (`affect-cue/v1`) — one expression target: `vector` (valence, arousal,
  dominance ∈ [-1,1]), `intensity` ∈ [0,1], `preset`, `source_id`, `source_confidence`,
  `valid_for_ms`, `issued_at`. It describes *how the robot intends to deliver*, never a
  diagnosis of a human's emotion.
* **`SpeechClauseOut`** (`speech-clause/v1`) — one spoken clause with the affect it should
  be delivered with. Field-for-field compatible with alice's own `SpeechClause`.

```python
from rt_agent.face import build_speech_clauses, split_clauses

texts   = split_clauses(reply_text)                      # <= 32 clauses, <= 1000 chars each
clauses = build_speech_clauses("utt-0042", texts, decision.affect, seed=29)
```

`split_clauses(text, max_len=1000, max_clauses=32)` splits on `.`/`!`/`?`/`…`, falls back to
commas and then word boundaries for an over-long sentence, and **merges** neighbours back
together rather than dropping text when there are more than 32 pieces.
`build_speech_clauses(generation_id, texts, cue, seed)` numbers them from 0, sets
`end_of_response` on the last, gives each clause seed `(seed + sequence) mod 2³²`, and
raises locally on anything alice's ledger would reject — over 32 clauses, over 4000
characters in total, a `generation_id` longer than the 128-character wire cap.

### Why one affect value per response

alice's granularity is **one affect value per clause**, applied at clause *onset*, and
"late emotion metadata cannot alter a committed/audible clause". rt-agent decides once per
utterance, so every clause of one reply carries the same vector. To vary affect inside a
reply, split into more clauses and call `build_speech_clauses` once per group.

---

## 1. Path A — committed clauses during speech

The seam alice already designed for exactly this. Topic
**`/alice/run/committed_clauses`**, type `alice_interfaces/msg/SpeechClause`, QoS RELIABLE
depth 128, subscribed by `SessionNode.external_clause`.

### Message fields

`alice_interfaces/msg/SpeechClause` — built by
`rt_agent.face.ros_bridge.speech_clause_msg_dict(clause, run_identity)`:

| Field | ROS type | Source in rt-agent | Bound |
|---|---|---|---|
| `header` | `StreamHeader` | built here, see below | — |
| `schema_version` | `string<=32` | `"speech-clause/v1"` | exact literal |
| `clause_id` | `string<=128` | `SpeechClauseOut.clause_id` | unique within the response |
| `clause_sequence` | `uint32` | `SpeechClauseOut.sequence` | **0–31**, exactly the next one |
| `text` | `string<=1000` | `SpeechClauseOut.text` | 1–1000 chars, ≤4000 per response |
| `affect_vector` | `float64[3]` | `AffectCue.vector` | each in [-1, 1] |
| `intensity` | `float64` | `AffectCue.intensity` | [0, 1] |
| `seed` | `uint32` | `SpeechClauseOut.seed` | 0…2³²-1 |
| `end_of_response` | `bool` | `SpeechClauseOut.end_of_response` | exactly one true, on the last |

`alice_interfaces/msg/StreamHeader` — built by `stream_header_dict(...)`:

| Field | ROS type | Value | Rule |
|---|---|---|---|
| `identity` | `RunIdentity` | `{run_id, epoch, generation_id}` | `epoch` comes from feedback, see §1.1 |
| `sequence` | `uint64` | 0, 1, 2, … | **exact** stream: starts at 0, increments by 1, never repeats or regresses |
| `source_monotonic_ns` | `uint64` | `time.monotonic_ns()` | **stamped at publish, never at decision** |
| `publisher_incarnation` | `string<=128` | the goal's `requester_incarnation` | must not change mid-run |

`alice_interfaces/msg/RunIdentity` is `{run_id: string<=128, epoch: string<=128,
generation_id: string<=128}`. `speech_clause_msg_dict` refuses to build a message whose
`run_identity.generation_id` disagrees with the clause's, because alice takes
`generation_id` from the *header*, not the body — a mismatch would be silently ignored.

### `RunSpeech` goal

Action `/alice/run_speech`, served only by `SessionNode`. `RunSpeechClient.goal_dict()`
returns exactly these fields:

| Field | Value rt-agent sends | Note |
|---|---|---|
| `schema_version` | `"run-speech/v1"` | |
| `identity` | `{run_id, "requested", generation_id}` | the epoch is a placeholder and is discarded |
| `source` | `COMMITTED_CLAUSES = 1` | `FIXTURE = 0` plays a file instead |
| `fixture_name` | `""` | must be empty for this source |
| `selected_profile` | e.g. `"visible-face"` | selects the authored-expression config |
| `seed` | u32 | |
| `hardware` | `false` | always: ROS hardware admission rejects unconditionally |
| `sad_hold_ms` | 0–2000 | post-speech closed-mouth sad pose, after a successful drain |
| `config_sha256` | `config_digest(config_root, profile)` | must match exactly |
| `calibration_sha256` | the face manifest hash | must match exactly |
| `clock_domain_fingerprint` | `clock_proof(epoch)` | must be computed **inside the same container and kernel namespace** |
| `requester_incarnation` | e.g. `"rt-agent-1"` | reused as `publisher_incarnation` |

A second concurrent goal is REJECTed: `SessionNode` takes a non-blocking admission lock and
there is exactly one run at a time.

### 1.1 The RunIdentity / epoch handshake — the gotcha

**You do not choose the epoch.** `SessionNode.execute` mints a fresh `uuid4` and throws
away whatever the goal asked for. The authoritative epoch appears in exactly one place: the
**first action feedback message**, published before the session starts accepting external
clauses. An external publisher must therefore subscribe to feedback *before* it can address
a single clause.

A clause addressed to the wrong epoch is not faulted, it is **silently dropped** — and the
run then fails later for missing clauses, which is a confusing way to lose 25 seconds. An
epoch can never be reused, and a restarted process gets a new incarnation and cannot resume
one.

```python
from rt_agent.face import RunSpeechClient

with RunSpeechClient(
    run_id="run-1",
    generation_id="utt-0042",
    requester_incarnation="rt-agent-1",
    selected_profile="visible-face",
    config_sha256=config_digest,
    calibration_sha256=calibration_digest,
    clock_domain_fingerprint=clock_proof,
) as client:
    handle_future = client.send_goal()
    identity = client.await_identity()       # blocks on the FIRST feedback message
    client.publish_clauses(clauses)          # stamps source_monotonic_ns per message
    result = client.result(handle_future.result())
```

`RunSpeechClient` needs `rclpy` and the generated `alice_interfaces` package, i.e. alice's
ROS container. Everywhere else it raises `RosUnavailableError` **at construction** — never
at import, and never silently later while a run is live. The message names the missing
module and points at the bridges that do work without ROS.

### 1.2 Guards that reject, and what each costs

Nearly all of these **fault the whole run**, because `SessionNode.external_clause` wraps its
body in a bare `except: self.fail(...)`.

| Guard | Where | Cost |
|---|---|---|
| `publisher_incarnation` ≠ goal's `requester_incarnation` | `session.py` `external_clause` | fault |
| wrong `run_id` / `epoch` / `generation_id` | `base.py` `current()` | **silent drop** |
| `sequence` not exactly the next one, duplicate, backwards, or a changed publisher incarnation | `SequenceGuard(exact=True)` | fault |
| `source_monotonic_ns` in the future, or older than **250 ms** at receipt | `transport.py` | fault |
| **relay re-check**: original timestamp older than 250 ms when the session republishes | `session.py` `relay_external` — "external clause original source expired at relay" | fault |
| out-of-order `clause_sequence`, duplicate `clause_id`, changed `generation_id`, >4000 chars, anything after `end_of_response` | `ClauseSequence.commit` | fault |
| `clause_sequence` > 31, vector not 3 finite values in [-1,1], `intensity` outside [0,1], empty or >1000-char text, `seed` out of range, unknown schema | pydantic in `speech_stream.py` | fault |
| internal queue full (32 clauses) | `session.py` `put_nowait` | fault |
| external phase not finished within 10 s of START | `session.py` | fault |
| total audio over 10 s including the 0.3 s tail; run over 25 s | `audio.py`, `base.py` watchdog | fault |

**The rule that matters most: stamp `source_monotonic_ns` at publish time, not at decision
time.** The 250 ms budget is checked twice — once on receipt and again at relay — so a
model that thinks for 300 ms after stamping a clause kills the run.
`speech_clause_msg_dict` defaults `source_monotonic_ns` to `time.monotonic_ns()` at call
time for exactly this reason; pass it explicitly only in tests.

`RunSpeechClient.publish_clauses` re-checks the local half of alice's ledger before
anything goes on the wire — contiguous sequences, unique `clause_id`s, one constant
`generation_id`, exactly one `end_of_response` and it last, ≤32 clauses, ≤4000 characters —
so a mistake costs a local `ValueError` instead of a faulted run.

### 1.3 What happens downstream, and how late

TTS → `PcmChunk` (the first packet carries the clause) → `PcmTimeline.commit_clause`
records the **onset** → `SpeechState.audible_vector` → `ExpressionBridge` → authored anchor
→ `compose_frame` → `FaceTarget` → Maestro.

Latency is structural: up to **400 ms** of accepted motion prefix after the audible clause
boundary, then the anchor transition (0.8 s on the visible profile, 1.6 s on the bench
profile), then per-channel rate caps (mouth corners move at 0.8/s). Fast emotion switching
looks smeared. alice's documentation forbids claiming instantaneous expression onset, and
so does this one.

A pure-affect update with no text is **not expressible on this topic**: `text` has
`min_length=1` and every clause is synthesised and spoken. That is what Path B is for.

---

## 2. Path B — the proposed idle `AffectCue`

**alice has no idle-expression path today.** Every participant rejects all data unless it
is inside an admitted `RunSpeech` run; `ExpressionNode` only acts while `SpeechState.phase
== PLAYING` with non-zero `speech_weight`; a run cannot succeed without audio draining; and
while nothing is proposing, `FaceRuntime` drives every channel to Home. Idle is *defined*
as neutral. There is no topic that moves a servo outside a run.

So a listening robot that shows anything at all needs a small addition. rt-agent already
builds the message for it; alice needs four changes.

### The proposed message

New `ros2_ws/src/alice_interfaces/msg/AffectCue.msg`, published on **`/alice/affect/cue`**
(RELIABLE, depth 128). The text below is also available at runtime as
`rt_agent.face.ros_bridge.AFFECT_CUE_MSG`:

```
StreamHeader header
string<=32 schema_version   # "affect-cue/v1"
float64[3] affect_vector    # each in [-1, 1]
float64 intensity           # [0, 1]
uint64 valid_for_ns         # bounded lifetime, <= 2e9
string<=64 source_id        # e.g. "rt-agent/jev-emotion-v1"
float64 source_confidence   # [0, 1]
```

| Field | Source in rt-agent | Bound enforced by rt-agent |
|---|---|---|
| `header` | `stream_header_dict(...)` | as Path A |
| `schema_version` | literal | `"affect-cue/v1"` |
| `affect_vector` | `AffectCue.vector` | 3 axes in [-1, 1] |
| `intensity` | `AffectCue.intensity` | [0, 1] |
| `valid_for_ns` | `AffectCue.valid_for_ms × 1e6` | ≤ 2 000 000 000 ns, refused locally above that |
| `source_id` | `AffectCue.source_id` | ≤64 characters |
| `source_confidence` | `AffectCue.source_confidence` | [0, 1] |

```python
from rt_agent.face import RunIdentity, affect_cue_msg_dict

payload = affect_cue_msg_dict(decision.affect, RunIdentity(
    run_id="run-1", epoch=authoritative_epoch, generation_id="idle-0007",
))
```

### The four alice changes

1. **Action.** Add `uint8 EXPRESSION_ONLY=2` to the `RunSpeech` goal `source` enum and
   extend `contracts.validate_run_speech_goal` to accept it with an empty `fixture_name`
   and a new `uint32 hold_ms` bounded by the 25 s run limit.
2. **Message.** Add `AffectCue.msg` as above and **route it through `session` as a relay**,
   exactly like `/alice/run/committed_clauses`, subscribing in `ExpressionNode` through the
   existing `admit_header(header, "session", "affect", sparse=True)`. The 250 ms freshness,
   identity, incarnation, ordering and roster guards then apply unchanged and no ninth
   participant is added.
3. **ExpressionNode.** In `EXPRESSION_ONLY` mode, replace the `SpeechState` trigger with a
   20 ms timer that synthesises `SpeechFrame(sample_index=k*rate/50, mouth_aperture=0,
   speech_weight=0, vector=cue.vector, intensity=cue.intensity)` and calls
   `bridge.advance(...)` on that synthetic sample clock — the bridge only needs a
   non-regressing sample index. `speech_weight=0` leaves `mouth_open` at the expression
   baseline, i.e. a closed, non-flapping mouth while listening, which is what is wanted.
   Relax the `PLAYING` early return **for this mode only**, and set `_control_source` from
   the cue's original source timestamp so the 250 ms control-progress watchdog still bites.
4. **Three relaxed success predicates.** `AudioNode.validate_end` requires no drain when no
   PCM was admitted; `MaestroNode.require_drain`/`validate_end` require
   `_control_source is not None` instead of a drained playback; `RecorderNode`'s audio
   corroboration relaxes correspondingly. Keep `observe_playback`'s drain check inert by not
   publishing DRAINED in this mode. Everything else — Home at start, Home at finish with PWM
   confirmation, per-channel caps, the 250 ms watchdogs, the 25 s bound, cancellation —
   stays untouched.

Listening idle is then a sequence of bounded expression-only runs (≤20 s each, re-admitted
while the robot listens), with rt-agent publishing an `AffectCue` whenever the decision
changes. A speech run preempts by cancelling the idle run; the admission lock already
guarantees one run at a time.

Until those land, `RosFaceBridge.emit_affect` is a **no-op unless `affect_enabled=True` is
passed explicitly**, and passing it against an `alice_interfaces` without `AffectCue` raises
`RosUnavailableError`. A harness must not be able to quietly believe it has an idle face
when it does not.

### Where the decision does *not* belong

* Never publish to `/alice/speech/state`, `/alice/expression/frame` or `/alice/face/target`.
  Each is guarded by `admit_header(header, "<expected producer>", …)`, which raises
  "publisher incarnation does not match prepared roster" for anyone outside the eight-node
  roster — and faults the run.
* Never write `cluster_labels` and expect them to act. They are documented as "never
  authoritative categorical labels"; consumers use `affect_schema_id` + `vector`.

---

## 3. Path C — `speech-plan/v1` export

ROS hardware admission is fail-closed and re-checked at the Maestro adapter factory, so the
**host pipeline is the only path that can move a real servo today**. It costs alice no code
at all: export a plan, play it with `alice-speak`.

```python
from rt_agent.face import export_speech_plan, write_speech_plan

plan = export_speech_plan(
    "utt-0042", "rt-agent/poc",
    clauses,                    # SpeechClauseOut objects, or plain strings
    decision.affect,            # one cue, or one cue per segment
    voice="azelma", seed=29,
    pause_after_s=0.2,          # float, or one value per segment
    blend_to_next=True,
)
write_speech_plan("out/plans/utt-0042.json", plan)
```

```bash
uv run --extra speech alice-speak --plan out/plans/utt-0042.json \
    --output artifacts/speech/utt-0042
```

The exporter enforces alice's own bounds before the file is ever written: 1–32 segments,
each 1–1000 characters and 4000 in total; 1–32 cues per segment; **the first cue of every
segment at `progress == 0` and progress strictly increasing**; `pause_after_s` in [0, 5];
`voice` from `{alba, marius, javert, jean, fantine, cosette, eponine, azelma}`; `seed` in
0…2³²-1. Two of those — the progress rule and the 4000-character total — live in alice's
pydantic model validators and are *not* expressed by the JSON Schema, so
`validate_speech_plan` re-implements them and the test suite checks both.

`blend_to_next=True` (the default) gives each segment two cues: its own affect at
`progress 0.0` and the next segment's at `progress 1.0`, so the face is already arriving at
the next clause's expression when that clause starts. This is the idiom of alice's own
`config/speech/alice-introduction.json`. `blend_to_next=False` emits one onset cue per
segment, matching what the streaming path would do.

**One asymmetry to remember:** intra-segment cue fractions matter *only* here. In the
streaming clause path alice uses clause-onset affect alone, because a final duration does
not exist yet. A plan and a clause stream built from the same decision will therefore not
look identical mid-segment.

`write_speech_plan` refuses to overwrite an existing file unless `overwrite=True`: an
utterance id identifies one plan, and silently replacing it destroys the evidence of what
was actually spoken.

The schema and alice's own demo plan are vendored at
`tests/fixtures/alice/speech-plan.schema.json` and
`tests/fixtures/alice/alice-introduction.json` (origin and commit in the README beside
them). The demo plan is a positive control: if it ever stops validating, the test harness
is broken, not the exporter.

---

## 4. Hardware-free bridges

For a run with no robot at all — the default.

```python
from rt_agent.face import CompositeFaceBridge, JsonlFaceBridge, UdpJsonFaceBridge

bridge = CompositeFaceBridge(
    JsonlFaceBridge("out/kitchen"),                 # durable log
    UdpJsonFaceBridge("127.0.0.1", 9917),           # live preview
)
bridge.emit_affect(decision.affect)
bridge.emit_clauses(clauses)
bridge.close()
```

All four bridges (`JsonlFaceBridge`, `UdpJsonFaceBridge`, `CompositeFaceBridge`,
`NullFaceBridge`) satisfy the `FaceBridge` protocol: `emit_affect(cue)`,
`emit_clauses(clauses)`, `close()`. `CompositeFaceBridge` always calls every bridge even if
an earlier one raised, then re-raises the first exception — a broken preview must not cost
the run its log.

### `out/affect.jsonl` — one object per cue

```json
{"kind": "affect", "seq": 0, "sent_at": "2026-09-22T12:00:00.123456+00:00",
 "cue": {"schema_version": "affect-cue/v1", "vector": [0.5, 0.2, 0.1], "intensity": 0.5,
         "preset": "warm", "source_id": "rt-agent", "source_confidence": 0.9,
         "valid_for_ms": 1500, "issued_at": "2026-09-22T12:00:00+00:00"},
 "face": {"anchor": "smile_open", "amplitude": 0.25,
          "channels": {"mouth_open": 0.25, "left_mouth_corner": 0.25,
                       "right_mouth_corner": -0.25},
          "profile": "bench", "visible": true}}
```

### `out/clauses.jsonl` — one object per clause, **no envelope**

```json
{"schema_version": "speech-clause/v1", "generation_id": "utt-0042",
 "clause_id": "utt-0042-00", "sequence": 0, "text": "Ten past nine.",
 "vector": [0.5, 0.2, 0.1], "intensity": 0.5, "seed": 29, "end_of_response": true}
```

The clause lines are deliberately bare so the file is byte-compatible with alice's own
clause fixtures and replays straight through the host CLI:

```bash
uv run --extra speech alice-speak --clauses out/kitchen/clauses.jsonl \
    --expression-mode authored --config-root config --output artifacts/speech/stream-run
```

### UDP datagrams — one JSON object each, discriminated by `kind`

```json
{"kind": "affect", "seq": 0, "sent_at": "...", "cue": {...}, "face": {...}}
{"kind": "clause", "seq": 1, "sent_at": "...", "clause": {...}, "index": 0, "count": 2}
```

`seq` counts every datagram of either kind; `index`/`count` locate a clause within its
response so a receiver can notice a lost one. Datagrams are refused above 8192 bytes; a
full 1000-character clause fits comfortably. Sending to a port nobody is listening on is
not an error, which is the right trade for a preview and the wrong one for alice's
committed-clause topic — that seam is RELIABLE and is served by Path A.

A minimal listener:

```bash
python3 -c "
import json, socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('127.0.0.1', 9917))
while True:
    record = json.loads(s.recv(8192))
    print(record['kind'], record.get('face', record.get('clause')))
"
```

---

## 5. The face simulator

`rt_agent.face.AuthoredFaceModel` predicts what alice's face would do. It is a **simulator,
not the authority** — alice decides, this only logs a prediction so a hardware-free run has
something to show.

It mirrors `_AuthoredAnchorPlanner._interpolated_target` exactly:

| | valence | anchor | `mouth_open` | `left_mouth_corner` | `right_mouth_corner` |
|---|---|---|---|---|---|
| positive | **> +0.2** | `smile_open` | +1 | +1 | **−1** |
| negative | **< −0.2** | `frown_closed` | −1 | −1 | **+1** |
| neutral | otherwise | `neutral` | 0 | 0 | 0 |

The right corner's inverted polarity is hardware, not a typo. `amplitude = intensity ×
anchor_scale`; channels are the anchor scaled by amplitude. The other eight channels stay
at Home.

| Profile | alice config | `anchor_scale` | `transition_s` |
|---|---|---|---|
| `bench` | `config/speech/authored-expression-v1.json` | 0.5 | 1.6 s |
| `visible` | `config/speech/authored-expression-visible-v1.json` | 1.0 | 0.8 s |

**The threshold is strict**, which is why the `surprised` preset in
`rt_agent/policy/presets.py` is authored at valence 0.25 and not 0.20: at exactly 0.2 the
face renders neutral and the emotion is invisible. `model.visibility(cue)` answers that
question directly. Only valence selects the anchor — arousal and dominance travel through
alice's whole stack and reach nothing but blink/gaze hazard rates, so two cues differing
only in arousal produce an identical face.

`advance(now)` interpolates linearly toward the target over `transition_s`. It does **not**
model the 400 ms accepted prefix, the per-channel rate and acceleration caps, blink and
gaze, the speech envelope's ownership of the jaw, or the `sad_hold_ms` post-speech pose.
Real onset is therefore always *slower* than what `advance` reports, never faster.

---

## 6. Known limits inherited from alice

* **ROS cannot move a real servo.** `require_ros_hardware_visibility()` rejects
  unconditionally and the Maestro re-checks before its adapter factory. Path C is the only
  route to hardware today.
* **Physical facial acceptance is unproven.** `"physical_motion_verified": false` is
  hard-coded in alice's own manifests; PWM readback is not mechanical arrival.
* **No fitted emotion model anywhere.** Authored policy over three anchors keyed on valence
  alone. Any claim of "recognised emotion" would be unsupported.
* **Response ceiling.** 10 s of audio including the 0.3 s tail, 25 s of wall time, 32
  clauses, 4000 characters.
* **A single malformed clause faults the entire run.** Validate locally first; rt-agent
  does.
* **Jev/Kev output is an unqualified candidate.** Keep `neutral` as the honest fallback —
  which is exactly what the policy's fail-closed path already does on a backend timeout.
