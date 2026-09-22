# Live kitchen-chat run: the whole harness against hosted Jev

Date: 2026-09-22. Endpoint: `POST https://api.typesafe.ai/v1/systemone`, model alias
`jev-latest`, resolved **`jev-1.13.0`** on every one of the 30 calls. Auth:
`TYPESAFE_API_KEY=proxy-injected` — this sandbox's HTTPS proxy rewrites the
`Authorization` header, so the placeholder is accepted and **auth is untested here**, as
it was in the bundle smoke test.

This is the first run of the *whole* loop rather than the bundle alone: 28 diarized
utterances of `examples/kitchen_chat.jsonl` through `ListeningAgent` — transcript log,
decision context, one frozen `decision-bundle/v1` call per utterance under a 1500 ms
deadline, policy, emotion controller, JSONL face bridge, background memory writer (LLM
sentence plus a second `memory_faithful` call), reply generation, clause split and
`speech-plan/v1` export.

* 28 bundle calls + 2 `memory_faithful` calls = **30 live calls**, 62,400 input tokens
  (~2,080 per call), 0 failures, 0 retries.
* Wall clock for the whole 96-second conversation: **8.4 s** (not realtime-paced).
* Text generation was the offline `ScriptedLLM` (`examples/scripted_llm.json`), so the
  only thing measured here is System One and the harness around it.

Reproduce, and re-record the fixtures the offline tests replay:

```bash
TYPESAFE_API_KEY=proxy-injected uv run rt-agent replay examples/kitchen_chat.jsonl \
    --backend jev --scripted-rules examples/scripted_llm.json \
    --out out --session kitchen-live \
    --record-fixtures tests/fixtures/systemone/kitchen_chat
```

Every call's rendered state, wire questions, raw response, request id and latency is in
`tests/fixtures/systemone/kitchen_chat/<utterance_id>.json`. Replaying them offline
(`--backend mock --fixtures tests/fixtures/systemone/kitchen_chat`) reproduces the
decisions below **byte for byte**, which is what `tests/harness/` asserts.

---

## Per-utterance results

`P(int)` = `intelligible_complete`, `P(addr)` = `addressed_to_robot`, `P(inv)` =
`invites_response_now`, `P(quiet)` = `robot_should_stay_quiet_safety`. `int` is the
`emotion_intensity` rubric score divided by 4. "worth / sens" is
`worth_remembering` / `sensitive_personal`, bold when it clears the 0.75 durability
gate. `emotion` is the preset the policy *chose*; what the face actually did, after the
confidence gate and the refractory period, is the next table.

| # | spk | text | P(int) | P(addr) | P(inv) | P(quiet) | emotion (conf) | int | decision | reason | worth / sens | bundle ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | ---: |
| 1 | S1 | Morning. Did you leave the kettle on a… | 0.96 | 0.24 | 0.72 | 0.15 | attentive (0.43) | 0.06 | wait | `not_addressed` | 0.06 / 0.05 | 728 |
| 2 | S2 | I did, sorry. I was halfway through th… | 0.96 | 0.09 | 0.19 | 0.17 | attentive (0.52) | 0.11 | wait | `not_addressed` | 0.06 / 0.04 | 246 |
| 3 | S1 | It's fine, it clicks off by itself. Is… | 0.95 | 0.09 | 0.49 | 0.18 | attentive (0.82) | 0.23 | wait | `not_addressed` | 0.06 / 0.03 | 239 |
| 4 | S2 | Half a carton in the door. I'll pick u… | 0.96 | 0.05 | 0.12 | 0.18 | attentive (0.92) | 0.21 | wait | `not_addressed` | 0.26 / 0.03 | 272 |
| 5 | S1 | Thanks. Get coffee as well if you're p… | 0.96 | 0.04 | 0.13 | 0.20 | attentive (0.84) | 0.18 | wait | `not_addressed` | **0.87 / 0.14** | 241 |
| 6 | S2 | Decaf after four. Noted. Oh, before I … | 0.95 | 0.11 | 0.14 | 0.19 | attentive (0.97) | 0.21 | wait | `not_addressed` | **0.83 / 0.06** | 742 |
| 7 | S1 | Thursday? I won't be here. I'm at the … | 0.93 | 0.05 | 0.27 | 0.20 | attentive (0.96) | 0.23 | wait | `not_addressed` | 0.64 / 0.79 | 237 |
| 8 | S2 | Right, your appointment. Is that the f… | 0.94 | 0.04 | 0.22 | 0.29 | attentive (0.72) | 0.24 | wait | `not_addressed` | 0.51 / 0.75 | 275 |
| 9 | S1 | Yeah. They want to check my blood pres… | 0.95 | 0.03 | 0.19 | 0.30 | attentive (0.64) | 0.30 | wait | `not_addressed` | 0.70 / 0.92 | 254 |
| 10 | S2 | Hope it's good news. | 0.96 | 0.04 | 0.22 | 0.33 | sympathetic (0.55) | 0.32 | wait | `not_addressed` | 0.10 / 0.27 | 291 |
| 11 | S1 | uh the, hm | **0.08** | 0.10 | 0.17 | 0.26 | attentive (0.92) | 0.27 | wait | `not_intelligible` | 0.08 / 0.14 | 291 |
| 12 | S1 | Alice, what's the weather doing on Thu… | 0.92 | 0.96 | 0.94 | 0.18 | attentive (0.88) | 0.29 | **SPEAK** | — | 0.12 / 0.11 | 257 |
| 13 | S2 | Alice is better at that than the thing… | 0.93 | 0.23 | 0.37 | 0.18 | attentive (0.69) | 0.27 | wait | `not_addressed` | 0.34 / 0.10 | 270 |
| 14 | S1 | That one answered the telly more than … | 0.92 | 0.17 | 0.37 | 0.17 | attentive (0.43) | 0.30 | wait | `not_addressed` | 0.09 / 0.09 | 297 |
| 15 | S2 | Anyway. What are we doing about dinner? | 0.95 | 0.15 | 0.40 | 0.11 | attentive (0.97) | 0.36 | wait | `not_addressed` | 0.06 / 0.03 | 284 |
| 16 | S1 | I could do the lentil thing. We've got… | 0.93 | 0.11 | 0.27 | 0.12 | attentive (0.98) | 0.33 | wait | `not_addressed` | 0.21 / 0.03 | 227 |
| 17 | S2 | Adding stock to the list. | 0.93 | 0.11 | 0.13 | 0.17 | attentive (0.98) | 0.30 | wait | `not_addressed` | 0.17 / 0.02 | 267 |
| 18 | S1 | Good. Oh, did you ever hear back about… | 0.93 | 0.36 | 0.49 | 0.18 | attentive (0.68) | 0.32 | wait | `not_addressed` | 0.26 / 0.45 | 260 |
| 19 | S2 | No. Nothing. It's been three weeks now… | 0.94 | 0.05 | 0.24 | 0.42 | sympathetic (0.92) | 0.35 | wait | `not_addressed` | 0.29 / 0.42 | 232 |
| 20 | S2 | Sorry. I'm, it's just getting to me a … | 0.92 | 0.08 | 0.38 | **0.51** | sympathetic (1.00) | 0.45 | wait | `safety_quiet` | 0.08 / 0.40 | 297 |
| 21 | S1 | Hey. Come here. Three weeks is nothing… | 0.91 | 0.07 | 0.35 | **0.55** | sympathetic (0.69) | 0.36 | wait | `safety_quiet` | 0.16 / 0.18 | 291 |
| 22 | S2 | I know. I know. | 0.91 | 0.04 | 0.17 | **0.56** | sympathetic (0.80) | 0.38 | wait | `safety_quiet` | 0.08 / 0.21 | 266 |
| 23 | S1 | Don't apologise for it. | 0.94 | 0.04 | 0.21 | **0.50** | sympathetic (0.69) | 0.36 | wait | `safety_quiet` | 0.06 / 0.11 | 250 |
| 24 | S2 | Alice, can you remind me on Friday mor… | 0.96 | 0.97 | 0.91 | 0.30 | sympathetic (0.63) | 0.38 | **SPEAK** | — | 0.61 / 0.40 | 284 |
| 25 | S1 | See, that's what she's for. | 0.92 | 0.11 | 0.14 | 0.35 | attentive (0.32) | 0.29 | wait | `not_addressed` | 0.19 / 0.09 | 267 |
| 26 | S2 | Right. I'm going to get out of these c… | 0.94 | 0.10 | 0.12 | 0.39 | sympathetic (0.53) | 0.34 | wait | `not_addressed` | 0.06 / 0.11 | 232 |
| 27 | S1 | Tea first? | 0.93 | 0.16 | 0.51 | 0.29 | warm (0.58) | 0.25 | wait | `not_addressed` | 0.05 / 0.03 | 241 |
| 28 | S2 | Go on then. | 0.92 | 0.09 | 0.17 | 0.39 | attentive (0.44) | 0.30 | wait | `not_addressed` | 0.08 / 0.05 | 284 |

### What the face actually did

The `EmotionController` sits between the chosen preset and the bridge: a proposal under
0.5 confidence never moves the face, a switch is held for 2 s after the previous one,
and a cue older than 1500 ms decays by half. `amp` is the simulated alice amplitude at
the `bench` profile (`intensity * 0.5`); `visible` is alice's own test, `|valence| > 0.2`.

| lines | emitted preset | amp | visible on alice's face | note |
| --- | --- | ---: | --- | --- |
| 1 | — | — | no | first proposal, `attentive` at 0.43, under the confidence gate |
| 2–18 | `attentive` | 0.03–0.18 | **no** | valence 0.1: the listening face is invisible on today's anchors |
| 19 | `sympathetic` | 0.18 | yes | the distress onset; the face softens one turn before the safety gate |
| 20–23 | `concerned` | 0.25 | yes | the safety hold overrides the preset and floors intensity at 0.5 |
| 24–26, 28 | `sympathetic` | 0.09–0.19 | yes | still visibly soft while answering line 24, then decaying |
| 27 | — | — | — | `warm` at 0.58 confidence, blocked by the 2 s refractory after line 26 |

Four effective expression changes, at t = 6.3 s (`attentive`), 72.6 s (`sympathetic`),
76.1 s (`concerned`) and 88.4 s (`sympathetic`).

### What the robot said

| line | reply | clauses | plan |
| ---: | --- | ---: | --- |
| 12 | "I do not have a forecast in front of me, so I would not want to guess. I can check it for you before Thursday morning." | 2 | `speech-plan-1.json` |
| 24 | "Yes, I will remind you on Friday morning to chase them about it." | 1 | `speech-plan-2.json` |

Both plans validate against alice's own `speech-plan/v1` JSON Schema. Both replies were
appended to the transcript as `ROBOT` turns and appear in the next decision's window —
the state for line 13 quotes the robot back to itself.

---

## Summary block

```json
{
  "utterances": 28,
  "speak_count": 2, "spoken_count": 2, "wait_count": 26,
  "wait_reasons": {"not_addressed": 21, "not_intelligible": 1, "safety_quiet": 4},
  "backend_failures": 0, "reply_failures": 0,
  "emotion_changes": 4,
  "emotion_counts": {"attentive": 19, "concerned": 4, "sympathetic": 4, "warm": 1},
  "memory_outcomes": {"stored": 2}, "memories_stored": 2,
  "bundle_ms_p50": 266.9, "bundle_ms_p95": 577.3, "bundle_ms_max": 741.8,
  "total_ms_p50": 267.5, "total_ms_p95": 577.9, "total_ms_max": 742.8,
  "backend_model_resolved": "jev-1.13.0"
}
```

**Latency.** Warm p50 **267 ms**, p95 **577 ms**, max **742 ms** — every call inside the
1500 ms deadline, with the two slowest being the first call of the run (728 ms, cold
connection) and line 6 (742 ms). The per-turn total is within 1 ms of the bundle call:
everything else the harness does — context assembly, retrieval, policy, face, logging —
costs under a millisecond, and the two speaking turns add only the scripted generator's
zero-latency reply. On a real remote reply model, that is the term that would dominate,
and it is *after* the decision, not inside it.

## Memory outcomes

| line | speaker | kind | P(worth) | P(sensitive) | sentence written | P(faithful) | outcome |
| ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| 5 | S1 | `preference` | 0.87 | 0.14 | "S1 only drinks decaf after four, because otherwise they are awake until two." | **0.93** | `stored` |
| 6 | S2 | `event` | 0.83 | 0.06 | "S2 says the building inspection is on Thursday morning between nine and eleven." | **0.96** | `stored` |
| 9 | S1 | — | 0.70 | 0.92 | *never written* | — | no memory attempt (see below) |

Two memories reached the store, both through a second live `memory_faithful` call. The
store holds nothing about the blood pressure or the medication.

---

## Discrepancies with the scenario's design, and one real defect

**1. The safety gate fires one turn later than the scenario expected (lines 20–23, not
19–20).** Line 19 is the distress onset and measured `P(stay quiet) = 0.42`, under the
0.5 gate; the following four turns measured 0.51, 0.55, 0.56, 0.50. This is defensible
and, on inspection, mechanical: at line 19 the distress exists only in the *current*
utterance — the six-turn window behind it is still dinner logistics — and the signal
accrues as the window fills with it. The robot is silent at line 19 regardless
(`not_addressed`), and the face had already gone `sympathetic` and visible. The test
asserts the recorded behaviour with this explanation in the docstring. The honest
reading is that `robot_should_stay_quiet_safety` is a *context* signal with roughly one
turn of lag, not an onset detector, and the emotion question is the faster of the two.

**2. Line 9's durability gate is a knife edge.** `worth_remembering` for "they want to
check my blood pressure again before they change the medication" measured **0.73, 0.75
and 0.70** across three live runs of the same line, against a 0.75 threshold. On the
recorded run it lands at 0.70, so the policy never names a memory kind and the *withheld*
branch is never reached — the line produces no memory at all instead of a logged
withholding. The privacy outcome is identical (nothing is stored, nothing is sent to the
reply model), but the log line differs, and a re-record can flip it. This is risk R3 in
`docs/ARCHITECTURE.md` — "no threshold on a knife edge next to a measured value" —
measured on a real line, and the threshold is on the wrong side of it.
`tests/harness/test_replay_end_to_end.py` records the measurement and exercises the
withheld branch separately, with the gate derived from the fixture rather than guessed.

**3. The `addressee` choice disagreed with the admission noul on line 13.** On this
recording the choice named `robot` at 0.51 probability and **0.33 confidence** — its
lowest of the transcript — for a sentence spoken *about* the robot to the other person,
while `addressed_to_robot` said 0.23. Two earlier runs of the same line chose
`other_human` at 0.28 and 0.42 confidence. Admission depends on the noul alone, so the
robot correctly said nothing; a policy that had branched on the choice would have
answered. The design rule "several labels may be true at once ⇒ one `noul` each, not one
`choice`" earned its keep here.

**4. The faithfulness gate caught a bad memory sentence, which was then fixed.** The
first recorded run wrote "S1 drinks only decaf coffee after four in the afternoon" and
the gate returned **0.64**, discarding it. The transcript says "I only drink decaf after
four" — "coffee" and "in the afternoon" are additions. The scripted rule was rewritten to
stay inside what was said and the next run scored 0.93. The gate did exactly the job it
exists for, on a sentence a human author thought was fine.

**5. A real defect in the state renderer, found and fixed.** The robot's own turn can end
*after* the next human turn — the harness estimates the length of a `ROBOT` turn because
it synthesises no audio — and `render_state` rendered that negative age as `[--3.6s]`,
which is not a time anyone can read. The same happens with genuinely overlapping
diarized speech. `_age_line` now clamps at zero, so an ongoing turn reads `[-0.0s]`, and
`render_faithfulness_state` in the memory writer got the same fix. No well-formed state
changed, so `state-render/v1` was not bumped; the fixtures were re-recorded afterwards
anyway, and every number on this page is from the run *after* the fix. `BUNDLE_VERSION`
was not touched.

## Local Kev: the clean failure

No Kev server runs on this host, and `check-backend` says so rather than hanging or
raising:

```
$ uv run rt-agent check-backend --backend kev --timeout-s 5
endpoint: http://127.0.0.1:8009  model requested: kev-latest
FAILED: SystemOneUnavailableError: model listing transport failure: [Errno 111] Connection refused
$ echo $?
1
```

The same command against hosted Jev:

```
$ TYPESAFE_API_KEY=proxy-injected uv run rt-agent check-backend --backend jev
endpoint: https://api.typesafe.ai  model requested: jev-latest
GET /v1/models:     731 ms  advertised: jev-latest, jev-preview
POST /v1/systemone:     324 ms  model resolved: jev-1.13.0  P(intelligible_complete)=0.97
```

Kev remains unqualified (risk R5): nothing here measures it, and nothing here should be
read as evidence about it.

---

## Interpretation

What worked is the part the design was least sure of: **the admission bundle picked out
exactly the two turns meant for the robot and refused the other twenty-six**, including
the two hardest negatives — the sentence that opens with the robot's name but is about it
(line 13, `P(addr)` 0.23) and the question one housemate hands to the other (line 18,
0.36). The highest `P(addressed_to_robot)` among the twenty-six refusals was 0.36 and the
lowest among the two admissions was 0.96, so the 0.7 threshold sits in an empty band
0.60 wide and is nowhere near a knife edge; the durability threshold, measured on line 9,
very much is. The face timeline reads the way a person would want it to — plain attention
through the domestic talk, softening at the moment the distress starts, held concerned
while the humans handle it, and still soft when the robot is finally spoken to again.

What looked wrong is smaller but real. The safety signal lags the distress by one turn
because it is a property of the window, not of the utterance, which argues for combining
it with the faster emotion answer rather than using it alone as a gate. The 1500 ms cue
lifetime is tuned for a face that is being driven continuously; in a conversation with
three-to-four-second gaps the cue expires *between every pair of turns*, so the
controller decays to neutral and re-proposes almost every time, and the refractory period
then blocks the one switch a viewer would actually have noticed (`warm` on "Tea first?",
line 27) while allowing a dozen re-issues nobody can see. The face is also left faintly
`sympathetic` at the end of a scene that has recovered, because neither closing line
mustered 0.5 confidence for anything warmer. And nineteen of the twenty-eight turns chose
`attentive`, whose valence of 0.1 is below alice's 0.2 anchor threshold: for most of this
conversation a correct expression decision produced no visible motion at all — which is
an argument about the preset table and the anchor threshold (open question O2/O3), not
about the model.

Latency is not the problem anyone expected it to be. One frozen bundle of ten questions
answered in 267 ms at the median and 742 ms at the worst, with the harness around it
costing under a millisecond per turn, leaves most of the 1500 ms budget unused — the
budget is there for the tail, and on this run the tail never arrived.
