# alice fixtures — copied, not authored

These files are **verbatim copies** from the alice workspace. They are the authority for
what rt-agent's `speech-plan/v1` exporter must produce; nothing here is edited, and a test
that fails against them means rt-agent is wrong, not the fixture.

| File | Origin in alice-workspace |
|---|---|
| `speech-plan.schema.json` | `config/speech/speech-plan.schema.json` |
| `alice-introduction.json` | `config/speech/alice-introduction.json` |

* Source repository: `/home/user/alice-workspace`
* Branch: `claude/sharp-volta-wt499g` (`== feature/streaming-affect-motion`)
* Commit: **`8ddcb6a`** — "docs: preserve speech experiments and expressive-motion handoff"
* Copied: 2026-09-22

`speech-plan.schema.json` is the JSON Schema exported from alice's `SpeechPlan` pydantic
model (`src/alice/contracts/speech.py`). `alice-introduction.json` is alice's own authored
demo plan and is used here as a **positive control**: if it ever fails to validate against
the schema, the test harness itself is broken.

Two rules the schema does *not* express, which alice enforces in `SpeechSegment` and
`SpeechPlan` model validators and which `rt_agent.face.speech_plan.validate_speech_plan`
therefore re-implements:

1. the first cue of every segment must be at `progress == 0` and cue progress must be
   **strictly increasing**;
2. the total text of all segments may not exceed **4000 characters**.

To refresh these copies, re-copy them from the same paths and update the commit above.
