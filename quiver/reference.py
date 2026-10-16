"""Ground truth from a real accelerometer: a phone running phyphox, taped to the machine.

phyphox ("Acceleration without g" or "Accelerometer" experiment) exports CSV
with a time column and x/y/z acceleration columns. We take the axis with the
most energy above 1 Hz, compute its amplitude spectrum, and report the top
peaks so they can be compared with what Quiver measured from video.

Accelerometers report acceleration; video measures displacement. For one
frequency the two differ by (2*pi*f)^2, so peaks line up in frequency, and the
spectra are each normalised to their own maximum for the overlay.
"""

from __future__ import annotations

import csv
import io

import numpy as np
from scipy import signal as sps


def parse_phyphox(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, acc[n, axes]). Accepts comma, semicolon or tab separated exports."""
    text = data.decode("utf-8-sig", errors="replace")
    dialect = csv.Sniffer().sniff(text.splitlines()[0] + "\n", delimiters=",;\t")
    rows = list(csv.reader(io.StringIO(text), dialect))
    header, body = rows[0], rows[1:]
    num = []
    for r in body:
        try:
            num.append([float(x.replace(",", ".")) if dialect.delimiter != "," else float(x) for x in r[: len(header)]])
        except ValueError:
            continue
    a = np.array(num)
    if a.ndim != 2 or a.shape[0] < 64 or a.shape[1] < 2:
        raise ValueError("expected a phyphox CSV: a time column then acceleration columns")
    tcol = next((i for i, h in enumerate(header) if "time" in h.lower()), 0)
    axes = [i for i, h in enumerate(header) if i != tcol and "abs" not in h.lower()][:3] or [i for i in range(a.shape[1]) if i != tcol]
    return a[:, tcol], a[:, axes]


def spectrum(t: np.ndarray, acc: np.ndarray, max_hz: float | None = None) -> dict:
    t = t - t[0]
    fs = (len(t) - 1) / (t[-1] - t[0])
    grid = np.arange(0, t[-1], 1 / fs)
    best = None
    for k in range(acc.shape[1]):
        x = sps.detrend(np.interp(grid, t, acc[:, k]))
        win = np.hanning(len(x))
        amp = 2 * np.abs(np.fft.rfft(x * win)) / win.sum()
        f = np.fft.rfftfreq(len(x), 1 / fs)
        e = float(np.sum(amp[f >= 1] ** 2))
        if best is None or e > best[0]:
            best = (e, f, amp, k)
    _, f, amp, axis = best
    keep = (f >= 1) & (f <= (max_hz or fs / 2))
    f, amp = f[keep], amp[keep]
    floor = float(np.median(amp)) + 1e-12
    idx, _ = sps.find_peaks(amp, height=floor * 6, distance=max(1, int(0.5 / (f[1] - f[0]))))
    idx = sorted(idx, key=lambda i: amp[i], reverse=True)[:6]
    return {
        "sample_hz": round(fs, 1),
        "axis": int(axis),
        "hz": np.round(f, 3).tolist(),
        "amp_norm": np.round(amp / (amp.max() or 1), 4).tolist(),
        "peaks": [{"hz": round(float(f[i]), 2), "rel": round(float(amp[i] / amp.max()), 3)} for i in idx],
    }


def compare(ref: dict, video_peaks: list[dict]) -> dict:
    """Match the accelerometer's top peak against the nearest video peak."""
    if not ref["peaks"] or not video_peaks:
        return {"match": False}
    r = ref["peaks"][0]["hz"]
    v = min(video_peaks, key=lambda p: abs(p["hz"] - r))["hz"]
    return {"match": abs(v - r) <= max(0.3, 0.02 * r), "accelerometer_hz": r, "video_hz": v,
            "error_hz": round(abs(v - r), 3), "error_pct": round(100 * abs(v - r) / r, 2)}
