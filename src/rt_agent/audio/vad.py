"""Silero v5 voice activity detection through onnxruntime only — never torch.

Ported from alice's qualified VAD (``alice-workspace/src/alice/conversation/vad.py``
class ``Silero``, lines 128-166): same model file, same 512-sample windows, same
(2, 1, 128) recurrent state plus 64-sample context, same single-threaded ORT session,
same "hash the weights and keep the digest" provenance rule.

The one deliberate difference is *why* this file exists: the robot host should be
able to run the VAD without a torch install, so we never import ``silero_vad``
(whose ``__init__`` imports torch) — we only locate the ``.onnx`` file that package
ships, using ``importlib.util.find_spec``, which does not execute it.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from rt_agent.audio.sources import Pcm

__all__ = [
    "QUALIFIED_SILERO_SHA256",
    "SILERO_CONTEXT_SAMPLES",
    "SILERO_DOWNLOAD_URL",
    "SILERO_FRAME_SAMPLES",
    "SILERO_SAMPLE_RATE",
    "SileroVad",
    "download_silero_model",
    "find_silero_model",
]

#: Silero v5 accepts exactly 512-sample windows at 16 kHz.
SILERO_FRAME_SAMPLES = 512
SILERO_SAMPLE_RATE = 16_000

#: v5 prepends the previous window's last 64 samples to every inference.
SILERO_CONTEXT_SAMPLES = 64

#: SHA-256 of the ``silero_vad.onnx`` shipped by silero-vad 6.2.2 — the exact file
#: alice qualified (``alice-workspace/src/alice/conversation/vad.py:14``).
QUALIFIED_SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

#: Fallback when the package data is absent; pinned to the upstream release tag.
SILERO_DOWNLOAD_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"

#: Override the model location without touching code.
SILERO_PATH_ENV = "RT_AGENT_SILERO_ONNX"

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rt-agent"


def find_silero_model(*, allow_download: bool = False) -> Path:
    """Locate ``silero_vad.onnx``: env override, installed package data, then cache.

    ``importlib.util.find_spec`` gives us the installed ``silero_vad`` package
    directory *without importing it*, which matters because importing it would pull
    in torch — the whole point of this module is that the robot host does not need
    torch to listen.
    """
    override = os.environ.get(SILERO_PATH_ENV)
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"{SILERO_PATH_ENV} points at a missing file: {path}")
        return path
    spec = find_spec("silero_vad")
    if spec is not None and spec.origin is not None:
        candidate = Path(spec.origin).parent / "data" / "silero_vad.onnx"
        if candidate.is_file():
            return candidate
    cached = _CACHE_DIR / "silero_vad.onnx"
    if cached.is_file():
        return cached
    if allow_download:
        return download_silero_model(cached)
    raise FileNotFoundError(
        "silero_vad.onnx not found. Install rt-agent[audio] (the silero-vad wheel "
        f"ships the model), set {SILERO_PATH_ENV}, or construct SileroVad with "
        "allow_download=True."
    )


def download_silero_model(destination: Path) -> Path:
    """Fetch the pinned ONNX weights to ``destination`` and return the path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".onnx.partial")
    with urllib.request.urlopen(SILERO_DOWNLOAD_URL, timeout=60) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)
    return destination


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in one shot (the model is ~2 MB)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SileroVad:
    """Frame-level speech probability from Silero v5, stateful across a session.

    Satisfies :class:`rt_agent.contracts.VoiceActivityDetector`. One session, one
    instance: the recurrent state carries conversational context, so call
    :meth:`reset` at a session boundary and never share an instance between streams.

    The model digest is recorded in :attr:`model_sha256` for provenance — every
    decision the harness logs can be traced back to the exact weights that produced
    the audio evidence. Pass ``expected_sha256`` to make a mismatch fatal, as alice
    does for its qualified checkpoint.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        expected_sha256: str | None = None,
        allow_download: bool = False,
        threads: int = 1,
    ) -> None:
        path = (
            Path(model_path)
            if model_path is not None
            else find_silero_model(allow_download=allow_download)
        )
        if not path.is_file():
            raise FileNotFoundError(f"Silero ONNX model not found: {path}")
        self.model_path = path
        self.model_sha256 = sha256_file(path)
        if expected_sha256 is not None and self.model_sha256 != expected_sha256:
            raise ValueError(
                f"Silero model hash {self.model_sha256} differs from the qualified "
                f"{expected_sha256} ({path})"
            )
        #: True when these are byte-for-byte the weights alice qualified.
        self.is_qualified = self.model_sha256 == QUALIFIED_SILERO_SHA256
        self.sample_rate = SILERO_SAMPLE_RATE
        self.frame_samples = SILERO_FRAME_SAMPLES

        try:
            import onnxruntime
        except ImportError as error:  # pragma: no cover - depends on the host
            raise RuntimeError("SileroVad needs onnxruntime: install rt-agent[audio]") from error
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = threads
        self.session: Any = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=options
        )
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, SILERO_CONTEXT_SAMPLES), dtype=np.float32)
        # Warm the graph so the first real frame is not also the first allocation.
        self.probability(np.zeros(SILERO_FRAME_SAMPLES, dtype=np.float32))
        self.reset()

    @property
    def model_id(self) -> str:
        """``silero_vad.onnx@sha256:<digest>`` — safe to log, identifies the weights."""
        return f"{self.model_path.name}@sha256:{self.model_sha256}"

    def reset(self) -> None:
        """Drop the recurrent state and the 64-sample context at a session boundary."""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, SILERO_CONTEXT_SAMPLES), dtype=np.float32)

    def probability(self, frame: Pcm) -> float:
        """Probability that this 512-sample 16 kHz frame contains speech."""
        array = np.asarray(frame, dtype=np.float32)
        if array.shape != (SILERO_FRAME_SAMPLES,) or not np.isfinite(array).all():
            raise ValueError(
                f"SileroVad needs a finite {SILERO_FRAME_SAMPLES}-sample mono frame, "
                f"got shape {array.shape}"
            )
        batch = array.reshape(1, SILERO_FRAME_SAMPLES)
        output, self.state = self.session.run(
            None,
            {
                "input": np.concatenate((self.context, batch), axis=1),
                "state": self.state,
                "sr": np.array(SILERO_SAMPLE_RATE, dtype=np.int64),
            },
        )
        self.context = batch[:, -SILERO_CONTEXT_SAMPLES:].copy()
        return float(output[0, 0])
