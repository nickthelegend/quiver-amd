"""Turn per-tile displacement into one vibration signal and its spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .motion import MotionResult

MIN_HZ = 1.0  # below this is drift, breathing of the tripod, auto-exposure
MIN_MATCH = 0.25  # trackability below this (texture < MIN_TEXTURE) = untrackable tile


@dataclass
class Peak:
    hz: float
    amp_px: float
    snr_db: float


@dataclass
class Spectrum:
    fps: float
    duration_s: float
    resolution_hz: float
    freqs: np.ndarray
    amp: np.ndarray  # amplitude spectrum, pixels
    noise_floor: float
    peaks: list[Peak]
    signal: np.ndarray  # combined displacement signal, pixels
    hot_tiles: list[int]  # tile indices that carry the vibration
    tile_energy: np.ndarray  # per-tile band energy, for the heatmap


def _principal_axis(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Project 2-D motion onto the direction it mostly moves along."""
    m = np.vstack([dx - dx.mean(), dy - dy.mean()])
    cov = m @ m.T
    w, v = np.linalg.eigh(cov)
    axis = v[:, np.argmax(w)]
    return axis @ m


def _amp_spectrum(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    x = sps.detrend(x)
    win = np.hanning(len(x))
    spec = np.fft.rfft(x * win)
    # Scale so a pure sine of amplitude A shows a peak of height A.
    amp = 2.0 * np.abs(spec) / win.sum()
    freqs = np.fft.rfftfreq(len(x), 1.0 / fps)
    return freqs, amp


def analyze(motion: MotionResult, top_k: int = 6, max_peaks: int = 8) -> Spectrum:
    n_t = len(motion.tiles)
    sigs = np.array([_principal_axis(motion.dx[i], motion.dy[i]) for i in range(n_t)])

    freqs, _ = _amp_spectrum(sigs[0], motion.fps)
    band = freqs >= MIN_HZ
    energy = np.zeros(n_t)
    for i in range(n_t):
        _, a = _amp_spectrum(sigs[i], motion.fps)
        # Textureless tiles return random shifts with a weak correlation peak; drop
        # them and down-weight marginal ones so noise can't win.
        r = motion.response[i]
        energy[i] = float(np.sum(a[band] ** 2)) * r * r if r >= MIN_MATCH else 0.0

    # Combine only tiles that carry most of the vibration: ones at the edge of the
    # machine (half machine, half background) would drag the amplitude down.
    order = list(np.argsort(energy)[::-1])
    hot = [i for i in order[:top_k] if energy[i] >= 0.5 * energy[order[0]]] or order[:1]
    best = sigs[hot[0]]
    combined = np.zeros_like(best)
    for i in hot:
        s = sigs[i]
        # The principal axis has an arbitrary sign; align every tile with the best one.
        if np.dot(s, best) < 0:
            s = -s
        combined += s
    combined /= len(hot)

    freqs, amp = _amp_spectrum(combined, motion.fps)
    floor = float(np.median(amp[band])) + 1e-12
    idx, _ = sps.find_peaks(amp, height=floor * 4, distance=max(1, int(0.5 / (freqs[1] - freqs[0]))))
    idx = [i for i in idx if freqs[i] >= MIN_HZ]
    idx = sorted(idx, key=lambda i: amp[i], reverse=True)[:max_peaks]
    peaks = [Peak(float(_refine_freq(freqs, amp, i)), float(amp[i]), float(20 * np.log10(amp[i] / floor))) for i in idx]

    duration = motion.n_frames / motion.fps
    return Spectrum(motion.fps, duration, float(freqs[1] - freqs[0]), freqs, amp, floor, peaks, combined, hot, energy)


def _refine_freq(freqs: np.ndarray, amp: np.ndarray, i: int) -> float:
    """Parabolic interpolation of the peak for sub-bin frequency accuracy."""
    if i <= 0 or i >= len(amp) - 1:
        return freqs[i]
    a, b, c = np.log(amp[i - 1] + 1e-12), np.log(amp[i] + 1e-12), np.log(amp[i + 1] + 1e-12)
    denom = a - 2 * b + c
    off = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return freqs[i] + off * (freqs[1] - freqs[0])


def amplitude_at(spec: Spectrum, hz: float, tol: float = 0.03) -> float:
    """Largest amplitude within ±tol (relative) of hz, or 0 if hz is past Nyquist."""
    if hz <= 0 or hz >= spec.fps / 2:
        return 0.0
    lo, hi = hz * (1 - tol) - spec.resolution_hz, hz * (1 + tol) + spec.resolution_hz
    sel = (spec.freqs >= lo) & (spec.freqs <= hi)
    return float(spec.amp[sel].max()) if sel.any() else 0.0
