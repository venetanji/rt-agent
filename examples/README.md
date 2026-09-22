# examples

Synthetic material for hardware-free runs. Nothing here is a recording, nothing here
describes a real person, and no real names, addresses, dates of birth, phone numbers or
medical records appear in any of it. The two speakers are session-local anonymous labels
`S1` and `S2`, which is the only speaker identity rt-agent ever stores.

## `kitchen_chat.jsonl`

A 28-line diarized transcript of two housemates in a kitchen/living room over about 96 seconds, with a robot (Alice) listening. It is the default input for the replay harness:

```bash
uv run --no-sync rt-agent replay examples/kitchen_chat.jsonl --out out/kitchen
```

### Format

One JSON object per line. `speaker`, `text`, `t_start_s` and `t_end_s` are required;
`audio` is optional and carries the acoustic evidence the decision bundle gets to see.

| Field | Type | Meaning |
|---|---|---|
| `speaker` | `"S1"` \| `"S2"` | session-local anonymous label; the robot's own turns are `ROBOT` and are appended by the harness, never present in the file |
| `text` | string, 1–1000 chars | the ASR transcript of the turn, warts included |
| `t_start_s` | float ≥ 0 | turn start, seconds from session start |
| `t_end_s` | float ≥ `t_start_s` | turn end |
| `audio.vad_mean_prob` | 0–1 | mean VAD speech probability over the segment |
| `audio.voiced_fraction` | 0–1 | fraction of frames above the VAD threshold |
| `audio.duration_s` | float | `t_end_s - t_start_s` |
| `audio.rms` | float | segment RMS amplitude |
| `audio.clipped_fraction` | 0–1 | fraction of samples at full scale |

Timestamps are strictly monotonic and non-overlapping: every `t_start_s` is at least the
previous `t_end_s`, so a replay source can sleep the gaps and reproduce the pacing.

### Scenario

S1 and S2 share a flat. They start on domestic logistics, drift into S1's medical
appointment, S1 addresses Alice for the first time, S2 mentions Alice to S1 without
addressing her, they settle dinner, and then S1 asks about a job interview and S2 —
who has heard nothing for three weeks — gets upset. S1 comforts them, S2 asks Alice for a
reminder, and the scene closes on tea.

The point of the scenario is that **most turns are not for the robot**. A listening robot
that answers the oat milk question has failed; one that stays quiet while S2 is upset, and
answers the two questions actually addressed to it, has not.

### What each line is for

Line numbers are 1-based, as in the file.

| Line | Speaker | What it exercises |
|---|---|---|
| 1–4 | S1, S2 | ordinary human-to-human turns; **must not** trigger a reply |
| 5 | S1 | **memory-worthy preference** — decaf only after four (`memory_kind: preference`) |
| 6 | S2 | **memory-worthy plan/event** — building inspection Thursday, 9–11 (`memory_kind: plan`) |
| 7–8 | S1, S2 | a clinic appointment is raised; health-adjacent but not yet the sensitive detail |
| 9 | S1 | **sensitive health mention** — blood pressure and a medication change (`sensitive_personal` should fire; stored only when `allow_sensitive=True`) |
| 10 | S2 | sympathetic reply between humans |
| 11 | S1 | **unintelligible fragment** — `"uh the, hm"`, low VAD and low RMS; `intelligible_complete` should fail and the robot should wait |
| 12 | S1 | **direct address 1** — "Alice, what's the weather doing on Thursday morning?" |
| 13 | S2 | **third-person mention that is not an address** — the turn begins with the word "Alice" but is *about* her, spoken to S1; the addressee classifier must not take it |
| 14 | S1 | follow-on between humans |
| 15 | S2 | topic shift, still human-to-human |
| 16–17 | S1, S2 | dinner logistics; weakly memory-worthy at most |
| 18 | S1 | **floor hand-off between the humans** — S1 asks S2 a question about S2's own business and yields the floor; the robot must not take the turn that was just handed to someone else |
| 19 | S2 | **distressed moment** — the longest, loudest turn in the file (note the non-zero `clipped_fraction`); `robot_should_stay_quiet_safety` should force WAIT and a concerned face |
| 20 | S2 | distress continues, quieter |
| 21–23 | S1, S2 | the humans handle it themselves; the robot stays out |
| 24 | S2 | **direct address 2** — "Alice, can you remind me on Friday morning to chase them about it?" |
| 25–28 | S1, S2 | wind-down; line 25 mentions the robot as "she", again not an address |

Expected behaviour in one sentence: **two replies (lines 12 and 24), two memories (lines 5
and 6, plus line 9 only if sensitive storage is enabled), and silence everywhere else.**
