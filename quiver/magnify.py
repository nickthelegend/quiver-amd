"""Linear Eulerian motion magnification (Wu et al., SIGGRAPH 2012).

Blur each frame down a Gaussian pyramid, band-pass every pixel over time
around the vibration frequency, amplify that band and add it back. Motion far
below a pixel becomes a visible wobble.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy import signal as sps


def _down(img: np.ndarray, levels: int) -> np.ndarray:
    for _ in range(levels):
        img = cv2.pyrDown(img)
    return img


def magnify(path: str, out_path: str, fps: float, band_hz: tuple[float, float], alpha: float = 40.0,
            levels: int = 3, max_frames: int = 960, width: int = 480, playback_fps: float = 30.0) -> str:
    """Write a side-by-side (original | magnified) H.264 mp4 and return its path."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = width / w
        frames.append(cv2.resize(f, (width, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
    cap.release()
    if len(frames) < 16:
        raise ValueError("not enough frames to magnify")

    small = np.stack([_down(f, levels) for f in frames])  # (T, h, w, 3)
    lo, hi = band_hz
    hi = min(hi, fps / 2 * 0.95)
    lo = max(0.5, min(lo, hi * 0.9))
    sos = sps.butter(2, [lo, hi], btype="bandpass", fs=fps, output="sos")
    band = sps.sosfiltfilt(sos, small, axis=0).astype(np.float32) * alpha

    H, W = frames[0].shape[:2]
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (W * 2, H))
    for f, b in zip(frames, band):
        up = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
        mag = np.clip(f + up, 0, 1)
        both = np.hstack([f, mag])
        both = (both * 255).astype(np.uint8)
        cv2.putText(both, "original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(both, f"magnified x{alpha:.0f}  {lo:.1f}-{hi:.1f} Hz", (W + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2)
        vw.write(both)
    vw.release()

    # Browsers can't play mp4v; re-encode to H.264 when ffmpeg is present.
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], check=True)
    else:
        shutil.copy(raw, out_path)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_path
