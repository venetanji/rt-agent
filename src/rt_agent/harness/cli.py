"""``rt-agent`` — the command line around the listening harness.

Four commands, each a thin shell over something that already exists:

``replay``          a diarized transcript JSONL through the whole decision loop;
``listen``          a WAV file or a microphone through the audio front end and then the
                    same loop;
``memories``        what the run remembered, read back out of the SQLite store;
``check-backend``   is a System One endpoint there, which model does it advertise, and
                    how long does one small call take.

Nothing here holds logic of its own: the CLI parses arguments into an
:class:`~rt_agent.harness.config.AgentConfig`, builds a
:class:`~rt_agent.harness.agent.ListeningAgent` from it, and prints what came back.
``rich`` renders the tables when it is importable and plain text is printed when it is
not, so the output is never the reason a run fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from rt_agent.audio import ReplaySource
from rt_agent.harness.agent import ListeningAgent, RunSummary
from rt_agent.harness.config import AgentConfig, BackendName, FaceName, LlmName
from rt_agent.harness.record import FixtureRecorder
from rt_agent.memory import SqliteMemoryStore
from rt_agent.systemone import (
    JEV_MODEL,
    KEV_MODEL,
    SystemOneClient,
    SystemOneError,
)
from rt_agent.systemone.bundle import Q_INTELLIGIBLE_COMPLETE, QUESTION_BUNDLE_V1

__all__ = ["app", "main"]

app = typer.Typer(
    name="rt-agent",
    help="A listening robot: one typed System One bundle per utterance decides "
    "speak/wait, the face and the memory.",
    no_args_is_help=True,
    add_completion=False,
)


# --------------------------------------------------------------------------------------
# Output helpers — rich when it is there, plain text when it is not
# --------------------------------------------------------------------------------------


#: Width used when the output is redirected. rich would otherwise fall back to 80
#: columns and ellipsise the turn table down to its separators.
_REDIRECTED_WIDTH = 150


def _console() -> Any:
    try:
        from rich.console import Console
    except ImportError:  # pragma: no cover - rich is a hard dependency, but never assume
        return None
    console = Console()
    if console.is_terminal:
        return console
    return Console(width=_REDIRECTED_WIDTH)


def _print(text: str) -> None:
    typer.echo(text)


def _table(title: str, columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Render a table with rich if it is importable, otherwise as aligned plain text."""
    console = _console()
    if console is not None:
        from rich.table import Table

        table = Table(title=title)
        for column in columns:
            # Values are already truncated by the caller; wrapping them again would
            # turn a 28-line table into three screens of confetti.
            table.add_column(column, no_wrap=True, overflow="ellipsis")
        for row in rows:
            table.add_row(*row)
        console.print(table)
        return
    _print(title)
    widths = [
        max([len(column), *(len(row[index]) for row in rows)])
        for index, column in enumerate(columns)
    ]
    _print("  ".join(name.ljust(widths[index]) for index, name in enumerate(columns)))
    _print("  ".join("-" * width for width in widths))
    for row in rows:
        _print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_summary(summary: RunSummary, run_dir: Path) -> None:
    rows = [
        ("session", summary.session_id),
        ("backend", f"{summary.backend} -> {summary.backend_model_resolved or 'n/a'}"),
        ("llm / face", f"{summary.llm} / {summary.face}"),
        ("utterances", str(summary.utterances)),
        ("spoke", f"{summary.spoken_count} of {summary.speak_count} admitted"),
        ("waited", str(summary.wait_count)),
        (
            "wait reasons",
            ", ".join(f"{name}={count}" for name, count in summary.wait_reasons.items()) or "-",
        ),
        ("emotion changes", str(summary.emotion_changes)),
        (
            "emotions",
            ", ".join(f"{name}={count}" for name, count in summary.emotion_counts.items()) or "-",
        ),
        (
            "memories",
            ", ".join(f"{name}={count}" for name, count in summary.memory_outcomes.items()) or "-",
        ),
        ("backend failures", str(summary.backend_failures)),
        (
            "bundle ms",
            f"p50 {summary.bundle_ms_p50:.0f}  p95 {summary.bundle_ms_p95:.0f}  "
            f"max {summary.bundle_ms_max:.0f}",
        ),
        (
            "turn ms",
            f"p50 {summary.total_ms_p50:.0f}  p95 {summary.total_ms_p95:.0f}  "
            f"max {summary.total_ms_max:.0f}",
        ),
        ("output", str(run_dir)),
    ]
    _table("run summary", ("field", "value"), [(name, value) for name, value in rows])


def _print_turns(agent: ListeningAgent) -> None:
    rows: list[tuple[str, ...]] = []
    for turn in agent.turns:
        answers = turn.decision.answers
        probability = (
            f"{answers.noul_or(Q_INTELLIGIBLE_COMPLETE, 0.0):.2f}" if answers is not None else "-"
        )
        rows.append(
            (
                turn.utterance.utterance_id,
                turn.utterance.speaker_label,
                turn.utterance.text[:44],
                probability,
                "SPEAK" if turn.decision.speak else "wait",
                turn.decision.wait_reason or "-",
                turn.decision.emotion_preset,
                "yes" if turn.memory_scheduled else "-",
                f"{turn.bundle_ms:.0f}",
            )
        )
    # "preset" is what the policy chose for this turn. What the face actually did —
    # after the confidence gate, the refractory period and decay — is in affect.jsonl.
    _table(
        "turns",
        ("id", "spk", "text", "P(intel)", "decision", "reason", "preset", "mem", "ms"),
        rows,
    )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------------------------
# Shared options
# --------------------------------------------------------------------------------------


def _build_config(
    *,
    backend: BackendName,
    llm: LlmName,
    scripted_rules: Path | None,
    face: FaceName,
    udp_host: str,
    udp_port: int,
    out: Path,
    session: str | None,
    realtime: bool,
    speed: float,
    fixtures: Path | None,
    deadline_ms: int,
    allow_sensitive: bool,
) -> AgentConfig:
    overrides: dict[str, Any] = {
        "backend": backend,
        "llm": llm,
        "face": face,
        "udp_host": udp_host,
        "udp_port": udp_port,
        "out_dir": out,
        "realtime": realtime,
        "speed": speed,
        "deadline_ms": deadline_ms,
    }
    if scripted_rules is not None:
        overrides["scripted_rules"] = scripted_rules
    if fixtures is not None:
        overrides["fixtures_dir"] = fixtures
    if session:
        overrides["session_id"] = session
    config = AgentConfig(**overrides)
    if allow_sensitive:
        config = config.model_copy(
            update={"policy": config.policy.model_copy(update={"allow_sensitive": True})}
        )
    return config.validated()


# --------------------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------------------


@app.command()
def replay(
    path: Annotated[Path, typer.Argument(help="Diarized transcript JSONL to replay.")],
    backend: Annotated[str, typer.Option(help="System One backend: jev, kev or mock.")] = "jev",
    llm: Annotated[str, typer.Option(help="Text generator: scripted or openai.")] = "scripted",
    scripted_rules: Annotated[
        Path | None, typer.Option(help="Rules file for --llm scripted.")
    ] = None,
    face: Annotated[str, typer.Option(help="Face bridge: jsonl, udp, null or ros.")] = "jsonl",
    udp_host: Annotated[str, typer.Option(help="Destination for --face udp.")] = "127.0.0.1",
    udp_port: Annotated[int, typer.Option(help="Destination port for --face udp.")] = 9917,
    out: Annotated[Path, typer.Option(help="Root output directory.")] = Path("out"),
    session: Annotated[str | None, typer.Option(help="Session id; default is generated.")] = None,
    realtime: Annotated[
        bool, typer.Option("--realtime/--no-realtime", help="Pace to the transcript's clock.")
    ] = False,
    speed: Annotated[float, typer.Option(help="Realtime speed multiplier.")] = 1.0,
    fixtures: Annotated[
        Path | None, typer.Option(help="Recorded calls for --backend mock.")
    ] = None,
    record_fixtures: Annotated[
        Path | None, typer.Option(help="Write every live call to this directory as a fixture.")
    ] = None,
    deadline_ms: Annotated[int, typer.Option(help="Hard budget for one bundle call.")] = 1500,
    allow_sensitive: Annotated[
        bool, typer.Option(help="Store memories the model marks sensitive.")
    ] = False,
    turns: Annotated[bool, typer.Option(help="Print the per-utterance table.")] = True,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Log every turn.")] = False,
) -> None:
    """Replay a diarized transcript through the whole decision loop."""
    _configure_logging(verbose)
    if not path.exists():
        raise typer.BadParameter(f"{path} does not exist")
    config = _build_config(
        backend=_backend_name(backend),
        llm=_llm_name(llm),
        scripted_rules=scripted_rules,
        face=_face_name(face),
        udp_host=udp_host,
        udp_port=udp_port,
        out=out,
        session=session,
        realtime=realtime,
        speed=speed,
        fixtures=fixtures,
        deadline_ms=deadline_ms,
        allow_sensitive=allow_sensitive,
    )
    recorder = FixtureRecorder(record_fixtures) if record_fixtures is not None else None
    summary = asyncio.run(_run_replay(path, config, recorder, turns))
    if summary.backend_failures:
        raise typer.Exit(code=1)


async def _run_replay(
    path: Path, config: AgentConfig, recorder: FixtureRecorder | None, show_turns: bool
) -> RunSummary:
    session_id = config.resolved_session_id()
    source = ReplaySource(path, session_id=session_id, realtime=config.realtime, speed=config.speed)
    agent = ListeningAgent.from_config(config, session_id=session_id, recorder=recorder)
    try:
        summary = await agent.run(source.utterances())
        if show_turns:
            _print_turns(agent)
        _print_summary(summary, agent.run_dir)
        return summary
    finally:
        await agent.aclose()


# --------------------------------------------------------------------------------------
# listen
# --------------------------------------------------------------------------------------


@app.command()
def listen(
    wav: Annotated[Path | None, typer.Option(help="WAV file to listen to.")] = None,
    mic: Annotated[bool, typer.Option(help="Listen to the default microphone.")] = False,
    backend: Annotated[str, typer.Option(help="System One backend: jev, kev or mock.")] = "jev",
    llm: Annotated[str, typer.Option(help="Text generator: scripted or openai.")] = "scripted",
    scripted_rules: Annotated[
        Path | None, typer.Option(help="Rules file for --llm scripted.")
    ] = None,
    face: Annotated[str, typer.Option(help="Face bridge: jsonl, udp, null or ros.")] = "jsonl",
    udp_host: Annotated[str, typer.Option(help="Destination for --face udp.")] = "127.0.0.1",
    udp_port: Annotated[int, typer.Option(help="Destination port for --face udp.")] = 9917,
    out: Annotated[Path, typer.Option(help="Root output directory.")] = Path("out"),
    session: Annotated[str | None, typer.Option(help="Session id; default is generated.")] = None,
    speaker: Annotated[
        str, typer.Option(help="Label the default single-speaker diarizer assigns.")
    ] = "S1",
    model_size: Annotated[str, typer.Option(help="faster-whisper model size.")] = "base.en",
    fixtures: Annotated[
        Path | None, typer.Option(help="Recorded calls for --backend mock.")
    ] = None,
    deadline_ms: Annotated[int, typer.Option(help="Hard budget for one bundle call.")] = 1500,
    allow_sensitive: Annotated[
        bool, typer.Option(help="Store memories the model marks sensitive.")
    ] = False,
    turns: Annotated[bool, typer.Option(help="Print the per-utterance table.")] = True,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="Log every turn.")] = False,
) -> None:
    """Listen to a WAV file or a microphone: VAD, endpointing, ASR, then the same loop.

    Diarization defaults to ``SingleSpeakerDiarizer``, which labels every segment the
    same: honest for one speaker, and never a claim that two voices were told apart.
    """
    _configure_logging(verbose)
    if (wav is None) == (not mic):
        raise typer.BadParameter("pass exactly one of --wav PATH or --mic")
    if wav is not None and not wav.exists():
        raise typer.BadParameter(f"{wav} does not exist")
    config = _build_config(
        backend=_backend_name(backend),
        llm=_llm_name(llm),
        scripted_rules=scripted_rules,
        face=_face_name(face),
        udp_host=udp_host,
        udp_port=udp_port,
        out=out,
        session=session,
        realtime=False,
        speed=1.0,
        fixtures=fixtures,
        deadline_ms=deadline_ms,
        allow_sensitive=allow_sensitive,
    )
    summary = asyncio.run(_run_listen(wav, config, speaker, model_size, turns))
    if summary.backend_failures:
        raise typer.Exit(code=1)


async def _run_listen(
    wav: Path | None, config: AgentConfig, speaker: str, model_size: str, show_turns: bool
) -> RunSummary:
    # Imported here, not at module scope: the audio extras are optional and the other
    # three commands must work on a host that has none of them.
    try:
        from rt_agent.audio import (
            AudioFrontEnd,
            Endpointer,
            FasterWhisperTranscriber,
            MicSource,
            SileroVad,
            SingleSpeakerDiarizer,
            WavFileSource,
        )
    except ImportError as error:  # pragma: no cover - depends on the host's extras
        raise typer.BadParameter(
            f"the audio front end needs the 'audio' extra: uv sync --extra audio ({error})"
        ) from error

    session_id = config.resolved_session_id()
    source = WavFileSource(wav) if wav is not None else MicSource()
    front_end = AudioFrontEnd(
        source=source,
        vad=SileroVad(),
        endpointer=Endpointer(),
        transcriber=FasterWhisperTranscriber(model_size=model_size),
        diarizer=SingleSpeakerDiarizer(speaker),
        session_id=session_id,
    )
    agent = ListeningAgent.from_config(config, session_id=session_id)
    try:
        summary = await agent.run(front_end.utterances())
        if show_turns:
            _print_turns(agent)
        _print_summary(summary, agent.run_dir)
        return summary
    finally:
        await front_end.aclose()
        await agent.aclose()


# --------------------------------------------------------------------------------------
# memories
# --------------------------------------------------------------------------------------


@app.command()
def memories(
    db: Annotated[Path, typer.Option(help="SQLite memory store to read.")],
    speaker: Annotated[str | None, typer.Option(help="Only memories about this label.")] = None,
    search: Annotated[str | None, typer.Option(help="Rank by keyword and recency.")] = None,
    session: Annotated[str | None, typer.Option(help="Only this session's memories.")] = None,
    limit: Annotated[int, typer.Option(help="How many rows to print.")] = 50,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table.")] = False,
) -> None:
    """Read back what the robot remembered."""
    if not db.exists():
        raise typer.BadParameter(f"{db} does not exist")
    store = SqliteMemoryStore(db)
    try:
        if search:
            records = store.search(session, search, limit=limit)
            if speaker is not None:
                records = tuple(item for item in records if item.speaker_label == speaker)
        else:
            records = store.list(session_id=session, speaker_label=speaker, limit=limit)
    finally:
        store.close()

    if as_json:
        _print(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if not records:
        _print("no memories stored")
        return
    _table(
        f"memories ({len(records)})",
        ("id", "speaker", "kind", "text", "worth", "faithful", "sensitive", "created"),
        [
            (
                record.memory_id,
                record.speaker_label,
                record.kind,
                record.text[:60],
                f"{record.worth_p:.2f}",
                f"{record.faithfulness_p:.2f}",
                "yes" if record.sensitive else "-",
                record.created_at.strftime("%Y-%m-%d %H:%M"),
            )
            for record in records
        ],
    )


# --------------------------------------------------------------------------------------
# check-backend
# --------------------------------------------------------------------------------------


@app.command(name="check-backend")
def check_backend(
    backend: Annotated[str, typer.Option(help="Which endpoint to probe: jev or kev.")] = "jev",
    base_url: Annotated[str | None, typer.Option(help="Override the endpoint URL.")] = None,
    model: Annotated[str | None, typer.Option(help="Override the model id.")] = None,
    timeout_s: Annotated[float, typer.Option(help="Transport timeout for the probe.")] = 20.0,
) -> None:
    """Probe a System One endpoint: list the models, then make one tiny bundle call.

    Exit code 1 on any failure, with the typed reason printed — this is what tells a
    "no local Kev is running" apart from "the model is not advertised".
    """
    name = backend.strip().lower()
    if name not in ("jev", "kev"):
        raise typer.BadParameter("--backend must be jev or kev")
    options: dict[str, Any] = {"timeout_s": timeout_s}
    if base_url is not None:
        options["base_url"] = base_url
    if model is not None:
        options["model"] = model
        if name == "kev":
            options["expected_model"] = model
    client = SystemOneClient.kev(**options) if name == "kev" else SystemOneClient.jev(**options)
    expected = model or (KEV_MODEL if name == "kev" else JEV_MODEL)
    _print(f"endpoint: {client.base_url}  model requested: {expected}")
    try:
        started = time.perf_counter()
        advertised = client.check_model()
        models_ms = (time.perf_counter() - started) * 1000.0
        _print(f"GET /v1/models: {models_ms:7.0f} ms  advertised: {', '.join(advertised)}")

        question = QUESTION_BUNDLE_V1.get(Q_INTELLIGIBLE_COMPLETE)
        state = (
            "Robot: Alice (social robot, listens in a shared room)\n"
            "Current utterance:\nS1: Alice, are you there?"
        )
        started = time.perf_counter()
        answers = client.ask(state, {Q_INTELLIGIBLE_COMPLETE: question.to_wire()})
        call_ms = (time.perf_counter() - started) * 1000.0
        _print(
            f"POST /v1/systemone: {call_ms:7.0f} ms  model resolved: {answers.model}  "
            f"P({Q_INTELLIGIBLE_COMPLETE})={answers.noul(Q_INTELLIGIBLE_COMPLETE):.2f}"
        )
    except SystemOneError as error:
        _print(f"FAILED: {type(error).__name__}: {error}")
        raise typer.Exit(code=1) from error
    finally:
        client.close()


# --------------------------------------------------------------------------------------
# argument coercion — typer 0.12 does not narrow a Literal, so it is done here
# --------------------------------------------------------------------------------------


def _backend_name(value: str) -> BackendName:
    name = value.strip().lower()
    if name not in ("jev", "kev", "mock"):
        raise typer.BadParameter("--backend must be jev, kev or mock")
    return name  # type: ignore[return-value]


def _llm_name(value: str) -> LlmName:
    name = value.strip().lower()
    if name not in ("scripted", "openai"):
        raise typer.BadParameter("--llm must be scripted or openai")
    return name  # type: ignore[return-value]


def _face_name(value: str) -> FaceName:
    name = value.strip().lower()
    if name not in ("jsonl", "udp", "null", "ros"):
        raise typer.BadParameter("--face must be jsonl, udp, null or ros")
    return name  # type: ignore[return-value]


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
