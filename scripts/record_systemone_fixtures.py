#!/usr/bin/env python3
"""Record live System One answers for the hand-written scenarios into test fixtures.

Every fixture holds the exact request that was sent (rendered state + wire questions),
the raw response, the measured latency and the decision context it came from, so the
network-free tests replay a real call rather than an invented one.

    TYPESAFE_API_KEY=... uv run python scripts/record_systemone_fixtures.py

Re-recording is required whenever the question bundle or the state renderer changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from rt_agent.contracts import AudioEvidence, DecisionContext, RobotState, Utterance
from rt_agent.systemone import QUESTION_BUNDLE_V1, SystemOneClient
from rt_agent.systemone.client import REQUEST_ID_HEADER
from rt_agent.systemone.mock import FIXTURE_SCHEMA_VERSION
from rt_agent.systemone.state import render_state

SESSION = "rec-2026-09-22"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "systemone"


def audio(
    vad: float = 0.93,
    voiced: float = 0.72,
    duration: float = 2.0,
    rms: float = 0.06,
    clipped: float = 0.0,
    overlap: float | None = 0.0,
) -> AudioEvidence:
    """Plausible Silero/pyannote evidence for a clean close-range utterance."""
    return AudioEvidence(
        vad_mean_prob=vad,
        voiced_fraction=voiced,
        duration_s=duration,
        rms=rms,
        clipped_fraction=clipped,
        overlap_fraction=overlap,
    )


def turn(
    index: int,
    speaker: str,
    text: str,
    start: float,
    end: float,
    evidence: AudioEvidence | None = None,
) -> Utterance:
    """One diarized turn of the recorded scenarios."""
    return Utterance(
        utterance_id=f"u{index}",
        session_id=SESSION,
        speaker_label=speaker,
        text=text,
        t_start_s=start,
        t_end_s=end,
        audio=evidence,
    )


def scenarios() -> dict[str, DecisionContext]:
    """Six hand-written rooms, chosen to exercise every branch of policy v1."""
    return {
        # 1. Two people planning a weekend; the robot is furniture here.
        "two_humans_weekend": DecisionContext(
            session_id=SESSION,
            recent=(
                turn(1, "S1", "Are you around on Saturday at all?", 10.0, 12.1),
                turn(2, "S2", "I think so, I only have the dentist in the morning.", 12.4, 15.2),
                turn(3, "S1", "We could finally do that hike then.", 15.6, 17.8),
            ),
            current=turn(
                4,
                "S2",
                "If the weather holds we could drive up to the lake instead, it is nicer this time of year.",
                18.1,
                22.0,
                audio(duration=3.9),
            ),
            robot_state=RobotState(
                speaking=False, last_spoke_age_s=None, current_emotion="neutral"
            ),
        ),
        # 2. A plain, direct question to the robot with the floor open.
        "direct_question_time": DecisionContext(
            session_id=SESSION,
            recent=(
                turn(1, "S1", "I think we are running late.", 30.0, 31.6),
                turn(2, "S2", "Probably, I stopped watching the clock.", 31.9, 34.0),
            ),
            current=turn(3, "S1", "Alice, what time is it?", 34.4, 35.8, audio(duration=1.4)),
            robot_state=RobotState(
                speaking=False, last_spoke_age_s=None, current_emotion="attentive"
            ),
        ),
        # 3. A durable, ordinary fact volunteered to the robot: remember, do not hide.
        "daughter_birthday": DecisionContext(
            session_id=SESSION,
            recent=(
                turn(1, "S1", "The house is going to be busy next week.", 40.0, 42.3),
                turn(2, "S1", "Actually, I should tell you now so you do not forget.", 42.6, 45.1),
            ),
            current=turn(
                3,
                "S1",
                "Alice, my daughter Nora turns seven next Tuesday, so we are having a party here.",
                44.2,
                48.6,
                audio(duration=4.4),
            ),
            robot_state=RobotState(
                speaking=False, last_spoke_age_s=20.0, current_emotion="attentive"
            ),
        ),
        # 4. A durable fact that is medical: remember-worthy but sensitive.
        "medical_appointment": DecisionContext(
            session_id=SESSION,
            recent=(
                turn(1, "S2", "I have to move a few things around this week.", 60.0, 62.4),
                turn(2, "S1", "Anything I can help with?", 62.7, 64.0),
            ),
            current=turn(
                3,
                "S2",
                "Alice, remind me that I have a cardiology appointment at the hospital on Thursday afternoon.",
                64.3,
                69.1,
                audio(duration=4.8),
            ),
            robot_state=RobotState(
                speaking=False, last_spoke_age_s=42.0, current_emotion="attentive"
            ),
        ),
        # 5. Real distress: the safety gate should hold the floor and the face should soften.
        "distressed_person": DecisionContext(
            session_id=SESSION,
            recent=(
                turn(1, "S2", "Hey, hey, sit down. What happened?", 80.0, 82.2),
                turn(2, "S1", "They called from the hospital an hour ago.", 82.5, 85.0),
            ),
            current=turn(
                3,
                "S1",
                "I am sorry, I cannot stop crying, my father is in intensive care and they do not know if he will make it.",
                85.3,
                91.4,
                audio(vad=0.88, voiced=0.64, duration=6.1, rms=0.04, overlap=0.02),
            ),
            robot_state=RobotState(
                speaking=False, last_spoke_age_s=120.0, current_emotion="attentive"
            ),
        ),
        # 6. Recognition garbage: the sharpest single signal in the bundle.
        "unintelligible_fragment": DecisionContext(
            session_id=SESSION,
            recent=(turn(1, "S1", "Did you get the box from the hallway?", 100.0, 102.1),),
            current=turn(
                2,
                "S2",
                "uh the the when i-- with the",
                102.4,
                103.6,
                audio(vad=0.61, voiced=0.38, duration=1.2, rms=0.02, overlap=0.11),
            ),
            robot_state=RobotState(
                speaking=False, last_spoke_age_s=None, current_emotion="neutral"
            ),
        ),
    }


def record(out_dir: Path, model: str, timeout_s: float) -> int:
    """Call the live model once per scenario and write one fixture each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    client = SystemOneClient.jev(model=model, timeout_s=timeout_s)
    advertised = client.check_model()
    print(f"models advertised: {', '.join(advertised)}")

    questions = QUESTION_BUNDLE_V1.to_wire()
    failures = 0
    for name, ctx in scenarios().items():
        state = render_state(ctx)
        body = {"state": state, "model": model, "questions": questions}
        started = time.perf_counter()
        response = client._sync_client().post("/v1/systemone", json=body)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            print(f"{name}: HTTP {response.status_code} {response.text[:200]}", file=sys.stderr)
            failures += 1
            continue
        payload = response.json()
        document = {
            "fixture_version": FIXTURE_SCHEMA_VERSION,
            "scenario": name,
            "utterance_id": ctx.current.utterance_id,
            "recorded_at": datetime.now(UTC).isoformat(),
            "endpoint": f"{client.base_url}/v1/systemone",
            "model_requested": model,
            "bundle_version": QUESTION_BUNDLE_V1.version,
            "context": ctx.model_dump(mode="json"),
            "state": state,
            "questions": questions,
            "response": payload,
            "latency_ms": round(latency_ms, 1),
            "request_id": response.headers.get(REQUEST_ID_HEADER),
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        answers = payload["answers"]
        print(
            f"{name}: {latency_ms:7.0f} ms  "
            f"intel={answers['intelligible_complete']['noul']:.2f} "
            f"addr={answers['addressed_to_robot']['noul']:.2f} "
            f"invite={answers['invites_response_now']['noul']:.2f} "
            f"quiet={answers['robot_should_stay_quiet_safety']['noul']:.2f} "
            f"emotion={answers['emotion']['choice']}@{answers['emotion']['confidence']:.2f} "
            f"worth={answers['worth_remembering']['noul']:.2f} "
            f"sens={answers['sensitive_personal']['noul']:.2f}"
        )
    client.close()
    return failures


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    return record(args.out, args.model, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
