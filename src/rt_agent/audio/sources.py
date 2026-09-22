"""Audio sources: a WAV file replayed as frames, and a live microphone.

Every source normalises to the one format the rest of the front end speaks — mono
float32 PCM at 16 kHz in fixed 512-sample frames (32 ms). That frame size is not a
free parameter: Silero v5 only accepts 512-sample windows at 16 kHz, and it is the
frame alice's bench captures with (``pw-cat`` ``readexactly(2048)`` = 512 mono
samples, ``alice-workspace/src/alice/conversation/audio.py:191``).

Nothing here imports ``soundfile`` or ``sounddevice`` at module scope, so
``rt_agent.audio`` stays importable on a host that has neither.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_FRAME_SAMPLES",
    "TARGET_SAMPLE_RATE",
    "MicSource",
    "MicrophoneUnavailableError",
    "Pcm",
    "WavFileSource",
    "resample_linear",
    "to_mono",
]

#: Mono float32 PCM — the only waveform type crossing a seam in this package.
Pcm = NDArray[np.float32]

#: The one sample rate the VAD, the endpointer and Whisper all agree on.
TARGET_SAMPLE_RATE = 16_000

#: 512 samples = 32 ms at 16 kHz. Required by Silero v5, matches alice's capture.
DEFAULT_FRAME_SAMPLES = 512


def to_mono(data: NDArray[Any]) -> Pcm:
    """Collapse an ``(n,)`` or ``(n, channels)`` array to mono float32 by averaging."""
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return np.asarray(array.mean(axis=1), dtype=np.float32)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {array.shape}")


def resample_linear(data: Pcm, source_rate: int, target_rate: int) -> Pcm:
    """Resample mono float32 PCM by linear interpolation.

    Deliberately simple: no polyphase filter, no SciPy. When downsampling we first
    apply a box (moving-average) low pass of the decimation width, which removes the
    worst of the aliasing for the two things that consume this audio — a VAD looking
    at 32 ms energy/spectral envelopes and Whisper's 80-bin log-mel front end. It is
    not good enough for archival resampling; use ``soxr`` if that ever matters.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or data.size == 0:
        return np.asarray(data, dtype=np.float32)
    signal = np.asarray(data, dtype=np.float32)
    if target_rate < source_rate:
        width = round(source_rate / target_rate)
        if width > 1:
            kernel = np.full(width, 1.0 / width, dtype=np.float32)
            signal = np.convolve(signal, kernel, mode="same").astype(np.float32)
    count = math.floor(signal.size * target_rate / source_rate)
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    positions = np.arange(count, dtype=np.float64) * (source_rate / target_rate)
    resampled = np.interp(positions, np.arange(signal.size, dtype=np.float64), signal)
    return np.asarray(resampled, dtype=np.float32)


def _frame_view(samples: Pcm, frame_samples: int, *, pad_tail: bool) -> list[Pcm]:
    """Split a waveform into fixed-size frames, optionally zero-padding the last one."""
    total = samples.size
    whole = total // frame_samples
    frames = [samples[i * frame_samples : (i + 1) * frame_samples] for i in range(whole)]
    remainder = total - whole * frame_samples
    if remainder and pad_tail:
        tail = np.zeros(frame_samples, dtype=np.float32)
        tail[:remainder] = samples[whole * frame_samples :]
        frames.append(tail)
    return frames


class WavFileSource:
    """Replay a sound file as 16 kHz mono float32 frames, as fast as the consumer asks.

    Any format ``libsndfile`` reads works, not just WAV; multi-channel input is
    downmixed and a differing sample rate is linearly resampled (see
    :func:`resample_linear`). The file is decoded eagerly in ``__init__`` so a bad
    path or an unreadable codec fails at construction rather than mid-stream.
    """

    def __init__(
        self,
        path: str | Path,
        frame_samples: int = DEFAULT_FRAME_SAMPLES,
        *,
        pad_final_frame: bool = True,
    ) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        self.path = Path(path)
        self.frame_samples = frame_samples
        self.sample_rate = TARGET_SAMPLE_RATE
        self.pad_final_frame = pad_final_frame
        raw, source_rate, channels = _read_sound_file(self.path)
        self.source_sample_rate = source_rate
        self.source_channels = channels
        self.samples: Pcm = resample_linear(to_mono(raw), source_rate, TARGET_SAMPLE_RATE)
        #: Start offset of the most recently yielded frame, in samples. Monotonic.
        self.sample_index = 0
        self.frame_index = 0
        self._closed = False

    @property
    def duration_s(self) -> float:
        """Length of the decoded (resampled) waveform in seconds."""
        return self.samples.size / self.sample_rate

    @property
    def elapsed_s(self) -> float:
        """Stream position of the most recently yielded frame, in seconds."""
        return self.sample_index / self.sample_rate

    async def frames(self) -> AsyncIterator[Pcm]:
        """Yield every frame of the file in order, then stop."""
        for index, frame in enumerate(
            _frame_view(self.samples, self.frame_samples, pad_tail=self.pad_final_frame)
        ):
            if self._closed:
                return
            self.frame_index = index
            self.sample_index = index * self.frame_samples
            yield np.asarray(frame, dtype=np.float32)
            # Stay cooperative: a tight file replay must not starve the ASR task.
            await asyncio.sleep(0)

    def __aiter__(self) -> AsyncIterator[Pcm]:
        """``async for frame in source`` is the same as iterating :meth:`frames`."""
        return self.frames()

    async def aclose(self) -> None:
        """Stop the iterator at the next frame boundary. The file is already closed."""
        self._closed = True


def _read_sound_file(path: Path) -> tuple[NDArray[Any], int, int]:
    """Decode a sound file to ``(samples, sample_rate, channels)``; lazy ``soundfile``."""
    try:
        import soundfile
    except (ImportError, OSError) as error:  # pragma: no cover - depends on the host
        raise RuntimeError(
            "reading audio files needs the 'soundfile' package: install rt-agent[audio]"
        ) from error
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")
    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    array = np.asarray(data, dtype=np.float32)
    return array, int(sample_rate), int(array.shape[1])


class MicrophoneUnavailableError(RuntimeError):
    """No usable capture device — missing PortAudio, missing package, or no input."""


class MicSource:
    """Live capture from a local input device through ``sounddevice``/PortAudio.

    ``sounddevice`` is imported lazily inside :meth:`frames` so that importing this
    module (or running the whole replay path) never needs PortAudio. Hosts without a
    sound card — CI, containers, this sandbox — get a
    :class:`MicrophoneUnavailableError` that says exactly what is missing instead of
    an ``OSError: PortAudio library not found`` from deep inside the import.

    The bounded queue mirrors alice's capture guard
    (``alice-workspace/src/alice/conversation/audio.py:112``): a consumer that falls
    behind is a fault, not something to paper over with unbounded buffering.
    """

    def __init__(
        self,
        device: int | str | None = None,
        *,
        frame_samples: int = DEFAULT_FRAME_SAMPLES,
        queue_frames: int = 8,
    ) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        if queue_frames < 1:
            raise ValueError("queue_frames must be at least 1")
        self.device = device
        self.frame_samples = frame_samples
        self.sample_rate = TARGET_SAMPLE_RATE
        self.queue_frames = queue_frames
        self.sample_index = 0
        self.frame_index = 0
        self.dropped_frames = 0
        self._closed = False

    @property
    def elapsed_s(self) -> float:
        """Stream position of the most recently yielded frame, in seconds."""
        return self.sample_index / self.sample_rate

    async def frames(self) -> AsyncIterator[Pcm]:
        """Yield captured frames until :meth:`aclose` is called or the stream faults."""
        sounddevice = _import_sounddevice()
        self._check_input_device(sounddevice)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Pcm] = asyncio.Queue(maxsize=self.queue_frames)
        fault: list[BaseException] = []

        def on_audio(indata: NDArray[Any], _frames: int, _time: Any, status: Any) -> None:
            if status:  # overflow/underflow reported by PortAudio
                self.dropped_frames += 1
            frame = to_mono(np.array(indata, dtype=np.float32, copy=True))
            loop.call_soon_threadsafe(_offer, frame)

        def _offer(frame: Pcm) -> None:
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull:
                fault.append(RuntimeError("microphone capture queue overflow"))

        stream = sounddevice.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=on_audio,
        )
        try:
            stream.start()
        except Exception as error:  # pragma: no cover - needs a real device
            raise MicrophoneUnavailableError(f"cannot start capture device: {error}") from error
        try:
            index = 0
            while not self._closed:
                frame = await queue.get()
                if fault:
                    raise fault[0]
                self.frame_index = index
                self.sample_index = index * self.frame_samples
                index += 1
                yield frame
        finally:
            stream.stop()
            stream.close()

    def __aiter__(self) -> AsyncIterator[Pcm]:
        """``async for frame in source`` is the same as iterating :meth:`frames`."""
        return self.frames()

    async def aclose(self) -> None:
        """Ask the capture loop to stop after the next frame."""
        self._closed = True

    def _check_input_device(self, sounddevice: Any) -> None:
        try:
            sounddevice.query_devices(self.device, "input")
        except Exception as error:
            raise MicrophoneUnavailableError(
                f"no usable audio input device (device={self.device!r}): {error}. "
                "Use WavFileSource or ReplaySource on a host without a microphone."
            ) from error


def _import_sounddevice() -> Any:
    """Import ``sounddevice``, turning both failure modes into one clear error."""
    try:
        import sounddevice
    except ImportError as error:
        raise MicrophoneUnavailableError(
            "microphone capture needs the 'sounddevice' package: install rt-agent[audio]"
        ) from error
    except OSError as error:
        raise MicrophoneUnavailableError(
            "microphone capture needs the PortAudio shared library, which is not "
            f"installed on this host ({error}). Install libportaudio2, or use "
            "WavFileSource / ReplaySource instead."
        ) from error
    return sounddevice
