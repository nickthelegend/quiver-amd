"""Play the recovered vibration back as sound.

`true_audio` keeps real pitch and duration: what the housing is actually doing.
Most machine fundamentals (10-60 Hz) sit below what laptop speakers can
reproduce, so `audible_audio` also lifts the pitch by an integer factor and
loops the sped-up clip back to the original length.
"""

from __future__ import annotations

import io

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

SR = 44100


def _clean(x: np.ndarray, fps: float) -> np.ndarray:
    x = sps.detrend(x)
    hi = min(0.45 * fps, fps / 2 - 1)
    if hi > 2:
        sos = sps.butter(4, [1.0, hi], btype="bandpass", fs=fps, output="sos")
        x = sps.sosfiltfilt(sos, x)
    peak = np.max(np.abs(x)) or 1.0
    return x / peak


def _fade(x: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(x) // 2)
    if n > 0:
        ramp = np.linspace(0, 1, n)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def true_audio(sig: np.ndarray, fps: float) -> np.ndarray:
    x = _clean(sig, fps)
    n = int(round(len(x) * SR / fps))
    return _fade(sps.resample(x, n), SR // 50)


def pitch_factor(fundamental_hz: float | None) -> int:
    if not fundamental_hz:
        return 8
    return int(np.clip(round(260 / fundamental_hz), 2, 32))


def audible_audio(sig: np.ndarray, fps: float, factor: int) -> np.ndarray:
    """Play the samples `factor` times faster (pitch x factor), then loop to the original length."""
    x = _clean(sig, fps)
    n = int(round(len(x) * SR / (fps * factor)))
    once = _fade(sps.resample(x, max(n, 32)), SR // 200)
    reps = int(np.ceil(len(x) / fps * SR / len(once)))
    return np.tile(once, reps)[: int(len(x) / fps * SR)]


def to_wav_bytes(x: np.ndarray, gain: float = 0.8) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, SR, (np.clip(x * gain, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()
