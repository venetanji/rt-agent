"""WAV framing, downmixing and resampling — the boring parts that break silently."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile

from rt_agent.audio.sources import (
    DEFAULT_FRAME_SAMPLES,
    TARGET_SAMPLE_RATE,
    MicrophoneUnavailableError,
    MicSource,
    WavFileSource,
    resample_linear,
    to_mono,
)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write float32 PCM (mono or ``(n, channels)``) to ``path``."""
    soundfile.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


def sine(seconds: float, sample_rate: int, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


async def collect(source: WavFileSource) -> list[np.ndarray]:
    return [frame async for frame in source.frames()]


async def test_a_16khz_mono_file_frames_at_512_samples(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "mono.wav", sine(1.0, 16_000), 16_000)
    source = WavFileSource(path)
    assert source.sample_rate == TARGET_SAMPLE_RATE
    assert source.frame_samples == DEFAULT_FRAME_SAMPLES
    assert source.source_sample_rate == 16_000
    assert source.source_channels == 1
    frames = await collect(source)
    # 16000 samples / 512 = 31.25 frames; the tail is zero-padded into a 32nd frame.
    assert len(frames) == 32
    assert all(frame.shape == (512,) for frame in frames)
    assert all(frame.dtype == np.float32 for frame in frames)
    assert frames[-1][16000 - 31 * 512 :].tolist() == [0.0] * (32 * 512 - 16000)


async def test_the_sample_index_is_monotonic_and_frame_aligned(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "mono.wav", sine(0.5, 16_000), 16_000)
    source = WavFileSource(path)
    seen: list[int] = []
    async for _frame in source.frames():
        seen.append(source.sample_index)
    assert seen == sorted(seen)
    assert seen == [index * 512 for index in range(len(seen))]
    assert source.elapsed_s == pytest.approx(seen[-1] / TARGET_SAMPLE_RATE)


async def test_an_unpadded_source_drops_the_partial_tail(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "mono.wav", sine(1.0, 16_000), 16_000)
    frames = await collect(WavFileSource(path, pad_final_frame=False))
    assert len(frames) == 31


async def test_stereo_is_downmixed_to_mono(tmp_path: Path) -> None:
    left = sine(0.2, 16_000, hz=200.0)
    right = sine(0.2, 16_000, hz=400.0)
    path = write_wav(tmp_path / "stereo.wav", np.stack([left, right], axis=1), 16_000)
    source = WavFileSource(path)
    assert source.source_channels == 2
    expected = ((left + right) / 2).astype(np.float32)
    assert np.allclose(source.samples[: expected.size], expected, atol=2e-4)


async def test_a_48khz_file_is_resampled_to_16khz(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "hi.wav", sine(1.0, 48_000, hz=220.0), 48_000)
    source = WavFileSource(path)
    assert source.source_sample_rate == 48_000
    assert source.sample_rate == 16_000
    assert source.samples.size == 16_000
    assert source.duration_s == pytest.approx(1.0)
    # The tone survives the resampler: the dominant FFT bin is still 220 Hz.
    spectrum = np.abs(np.fft.rfft(source.samples))
    peak_hz = float(np.fft.rfftfreq(source.samples.size, 1 / 16_000)[int(np.argmax(spectrum))])
    assert peak_hz == pytest.approx(220.0, abs=2.0)


async def test_an_8khz_file_is_upsampled_to_16khz(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "lo.wav", sine(0.5, 8_000, hz=300.0), 8_000)
    source = WavFileSource(path)
    assert source.samples.size == 8_000
    spectrum = np.abs(np.fft.rfft(source.samples))
    peak_hz = float(np.fft.rfftfreq(source.samples.size, 1 / 16_000)[int(np.argmax(spectrum))])
    assert peak_hz == pytest.approx(300.0, abs=3.0)


def test_resample_linear_is_identity_at_the_same_rate() -> None:
    data = sine(0.1, 16_000)
    assert resample_linear(data, 16_000, 16_000) is not None
    assert np.array_equal(resample_linear(data, 16_000, 16_000), data)


def test_resample_linear_handles_empty_and_rejects_bad_rates() -> None:
    empty = np.zeros(0, dtype=np.float32)
    assert resample_linear(empty, 44_100, 16_000).size == 0
    with pytest.raises(ValueError, match="positive"):
        resample_linear(empty, 0, 16_000)


def test_to_mono_rejects_3d_audio() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        to_mono(np.zeros((2, 2, 2), dtype=np.float32))


async def test_aclose_stops_the_iterator(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "mono.wav", sine(1.0, 16_000), 16_000)
    source = WavFileSource(path)
    frames = 0
    async for _frame in source.frames():
        frames += 1
        if frames == 3:
            await source.aclose()
    assert frames == 3


def test_a_missing_file_fails_at_construction(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WavFileSource(tmp_path / "nope.wav")


def test_frame_size_is_validated(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "mono.wav", sine(0.1, 16_000), 16_000)
    with pytest.raises(ValueError, match="frame_samples must be positive"):
        WavFileSource(path, frame_samples=0)


async def test_a_missing_microphone_raises_a_readable_error() -> None:
    """This sandbox has no PortAudio and no device; either way the message must say so."""
    source = MicSource(device="rt-agent-no-such-device")
    with pytest.raises(MicrophoneUnavailableError):
        async for _frame in source.frames():  # pragma: no cover - never yields here
            break
