"""Motion magnification of the vibrating region.

We already measured how the machine moves (its displacement over time, to a
few hundredths of a pixel). To make that visible we band-pass the measured
motion around the running speed, multiply it, and warp only the vibrating
region by the amplified amount (Lagrangian magnification, Liu et al. 2005).
The background stays still and sensor noise is not amplified, unlike
pixel-wise Eulerian magnification.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy import signal as sps


def _mask(rects: list[tuple[int, int, int, int]], src_w: int, size: tuple[int, int]) -> np.ndarray:
    W, H = size
    s = W / src_w
    m = np.zeros((H, W), np.float32)
    for x, y, w, h in rects:
        cv2.rectangle(m, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), 1.0, -1)
    k = max(3, int(0.04 * W)) | 1
    return cv2.GaussianBlur(m, (k, k), 0)[..., None]


def magnify(path: str, out_path: str, fps: float, band_hz: tuple[float, float], xy: np.ndarray,
            rects: list[tuple[int, int, int, int]], src_w: int, alpha: float = 30.0,
            max_frames: int = 960, width: int = 480, playback_fps: float = 30.0) -> str:
    """Write a side-by-side (original | magnified) H.264 mp4 and return its path.

    xy: (2, T) measured displacement in pixels of the analysis frame (width src_w).
    rects: vibrating regions (x, y, w, h) in analysis-frame pixels.
    """
    lo, hi = band_hz
    hi = min(hi, fps / 2 * 0.95)
    lo = max(0.5, min(lo, hi * 0.9))
    sos = sps.butter(2, [lo, hi], btype="bandpass", fs=fps, output="sos")
    motion = sps.sosfiltfilt(sos, xy, axis=1) * (alpha - 1)

    cap = cv2.VideoCapture(path)
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw.mp4"
    vw = None
    mask = None
    scale = 1.0
    i = 0
    while i < min(max_frames, motion.shape[1]):
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        H = int(h * width / w) // 2 * 2
        f = cv2.resize(f, (width, H), interpolation=cv2.INTER_AREA)
        if vw is None:
            vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (width * 2, H))
            mask = _mask(rects, src_w, (width, H))
            scale = width / src_w
        sx, sy = motion[0, i] * scale, motion[1, i] * scale
        T = np.float32([[1, 0, sx], [0, 1, sy]])
        moved = cv2.warpAffine(f, T, (width, H), borderMode=cv2.BORDER_REFLECT)
        mag = (f * (1 - mask) + moved * mask).astype(np.uint8)
        both = np.hstack([f, mag])
        cv2.putText(both, "original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(both, f"motion x{alpha:.0f}  {lo:.1f}-{hi:.1f} Hz", (width + 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2)
        vw.write(both)
        i += 1
    cap.release()
    if vw is None:
        raise ValueError("no frames to magnify")
    vw.release()

    # Browsers can't play mp4v; re-encode to H.264 when ffmpeg is present.
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], check=True)
    else:
        shutil.copy(raw, out_path)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_path
