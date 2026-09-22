# Live smoke test: `decision-bundle/v1` against hosted Jev

Date: 2026-09-22. Endpoint: `POST https://api.typesafe.ai/v1/systemone`, model alias
`jev-latest`, resolved **`jev-1.13.0`** on every call. Auth: `TYPESAFE_API_KEY=proxy-injected`
(this sandbox's HTTPS proxy rewrites the `Authorization` header, so the placeholder is
accepted — auth is therefore *untested* here and will be the first thing to configure on
the real robot).

Six hand-written `DecisionContext` scenarios, one bundle call each: the whole frozen
bundle (10 questions: 6 `noul`, 3 `choice`, 1 `score`) is answered in a single request.
Each call's rendered state, wire questions, raw response, request id and latency are
stored in `tests/fixtures/systemone/<scenario>.json` and replayed by `MockSystemOne`, so
the default test suite is network-free.

Reproduce:

```bash
TYPESAFE_API_KEY=... uv run python scripts/record_systemone_fixtures.py
RT_AGENT_LIVE=1 TYPESAFE_API_KEY=... uv run pytest -m live
```

## Results

Probabilities are as returned; `intensity` is the 0–4 rubric score divided by 4.
"Decision" is what `Policy` (policy/v1 defaults) makes of the answers.

| Scenario | P(intel) | P(addr) | P(invite) | P(quiet) | addressee (conf) | emotion (conf) | intensity | P(worth) | P(sens) | kind | Decision | Face | ms |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| two humans planning a weekend, robot not addressed | 0.95 | 0.04 | 0.16 | 0.18 | other_human (0.91) | attentive (0.74) | 0.05 | 0.37 | 0.06 | plan | WAIT `not_addressed` | attentive | 245 |
| "Alice, what time is it?" | 0.97 | 0.97 | 0.96 | 0.07 | robot (1.00) | attentive (0.91) | 0.33 | 0.04 | 0.03 | other | **SPEAK** | attentive | 318 |
| daughter's birthday next Tuesday | 0.97 | 0.89 | 0.42 | 0.09 | robot (1.00) | happy (0.61) | 0.36 | **0.96** | 0.10 | event | WAIT `no_invitation` | happy | 278 |
| cardiology appointment on Thursday | 0.96 | 0.97 | 0.87 | 0.10 | robot (1.00) | attentive (0.90) | 0.33 | **0.87** | **0.91** | health | **SPEAK** | attentive | 269 |
| distressed: father in intensive care | 0.94 | 0.10 | 0.50 | **0.91** | other_human (0.45) | sympathetic (0.89) | 0.62 | 0.67 | 0.98 | health | WAIT `safety_quiet` | **concerned** | 258 |
| recognition garbage: "uh the the when i-- with the" | **0.05** | 0.11 | 0.12 | 0.24 | unclear (0.44) | attentive (0.82) | 0.06 | 0.03 | 0.03 | other | WAIT `not_intelligible` | attentive | 263 |

Memory outcomes under the default `allow_sensitive=False`:

| Scenario | remember | kind | sensitive | note |
| --- | --- | --- | --- | --- |
| daughter's birthday | yes | `event` | no | ordinary durable fact, stored |
| cardiology appointment | **no** | `health` | yes | worth 0.87 but sensitive 0.91 → **withheld**, kind kept for the log |
| distressed person | no | — | yes | worth 0.67 is below the 0.75 gate |
| the other three | no | — | no | nothing durable |

Usage: 2,065–2,115 input tokens per call (the 10 questions are ~1.6k of that and are
re-sent every time), 350–355 output tokens (free). At $42/Btok that is about **$0.088 per
1,000 decisions**.

Latency: **245–318 ms warm**, every call inside the 1,500 ms hard deadline and inside the
500 ms p95 target the alice plan already froze. The first call of a cold process costs
~700 ms, so the harness must warm the client at startup.

## Reading the numbers

* **The three admission `noul`s behave as the plan predicted.** `intelligible_complete` is
  the sharpest single signal (0.05 on garbage, 0.94–0.97 on everything else), and
  `addressed_to_robot` separates cleanly (0.04 / 0.10 for the two non-addressed rooms,
  0.89–0.97 for the three addressed ones). No threshold sits on a knife edge: the closest
  approach to a boundary is `addressed_to_robot` 0.89 against a 0.70 gate.
* **`invites_response_now` is the gate that actually discriminates among addressed turns**
  (0.96 for a question, 0.87 for a request, 0.42 for a volunteered fact). The birthday
  case therefore ends in silence — correct for a conservative admission policy, but worth
  revisiting if the robot should acknowledge things it is told.
* **The safety gate works and is decisive** (0.91 on the distress case, 0.07–0.24
  everywhere else), and it is what makes that case a WAIT: `addressed_to_robot` alone
  would also have blocked it, but the gate is what forces the concerned face.
* **`sensitive_personal` is a clean binary here** (0.91/0.98 vs 0.03–0.10), which is what
  the withhold path depends on.
* **`emotion` is confidently boring.** Four of six rooms get `attentive`, which is honest —
  nothing in them calls for a face. The two that should move do:
  `happy` (0.61) for the birthday and `sympathetic` (0.89) for the distress. Intensity
  tracks it: 0.05–0.06 on flat rooms, 0.62 on the distress case.
* **Known soft spots.** `addressee` confidence on the distress case is 0.45 and on the
  garbage case 0.44 — the low-confidence band, exactly as the scouting report warned.
  Nothing in policy v1 depends on `addressee` beyond logging, so this is recorded rather
  than papered over. `memory_kind` on the weekend room is `plan` at 0.42 confidence, but
  it is never read because `worth_remembering` (0.37) does not pass.

## Wording iterations

Three rounds were recorded; the final wording is what ships in
`src/rt_agent/systemone/bundle.py`.

| Round | Change | Effect |
| --- | --- | --- |
| 1 | Initial wording. | Five of six scenarios correct. `addressed_to_robot` returned **0.68** for the birthday case even though the speaker literally begins with "Alice," — below the 0.70 gate. |
| 2 | Two changes. (a) Scenario fix: the birthday room had a *competing* addressee (another person had just asked a question), so 0.68 was defensible; the recent turns were changed so the speaker is not answering someone else. (b) Wording fix: the `addressed_to_robot` false-criterion said "mentioning the robot's name while talking about it to someone else does not count", which was suppressing genuine vocatives. It now reads "Talking *about* the robot in the third person to someone else is not addressing it; starting a sentence with the robot's name and then telling it something is." | `addressed_to_robot` moved **0.68 → 0.90** on that scenario; every other value stayed inside the ±0.02 noise band. |
| 3 | The `emotion_intensity` rubric had "strong" as a bare one-word level, violating the rule that score levels must stand alone. Changed to "strong and hard to miss". | No behavioural change beyond noise (intensity scores moved by ≤0.03); kept because a self-contained level is what makes the rubric portable. |

Everything else in the bundle is unchanged from the first draft. The bundle is versioned
as a whole (`decision-bundle/v1`): because an answer shifts when its *siblings* change,
any future edit means bumping the version and re-recording these fixtures.
`tests/systemone/test_mock.py` fails loudly if the shipped bundle or state renderer
drifts away from what the fixtures were recorded with.

## Caveats

* Six hand-written cases are an encouraging signal, **not a qualification**. The frozen
  targets from the alice plan — p95 ≤ 500 ms, false-SPEAK ≤ 2%, valid-call recall ≥ 95%,
  per-language recall ≥ 90% on 120 development + 120 confirmation cases — have not been run.
* English only. Cantonese, which is in scope for alice, was not probed here at all.
* Answers are stable to about ±0.02 between identical calls and shift more when the
  question set changes; treat every number above as a measurement, not a constant.
* Hosted Jev is a third-party data-egress path for live room transcripts. This smoke test
  used synthetic transcripts only. `SystemOneClient.kev()` points the same code at a local
  Kev server for the privacy-preserving deployment.
