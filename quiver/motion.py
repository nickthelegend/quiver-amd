"""Sub-pixel motion extraction from video.

Each frame is split into a grid of tiles. For every tile we estimate its shift
against a reference image with a gradient (Lucas-Kanade) solve: for motions
well under a pixel, the brightness change at each pixel is the image gradient
times the shift, so one 2x2 least-squares system per tile per frame gives the
displacement to a few hundredths of a pixel.

The reference is the temporal mean of the clip (the machine's average
position), which averages away sensor noise. Camera shake moves every
background tile together, so the median shift across trackable tiles is
removed from each tile before analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

BLUR_SIGMA = 1.0  # suppress sensor noise before taking gradients
MIN_TEXTURE = 0.5  # structure-tensor min eigenvalue per pixel (after blur) below this = untrackable


@dataclass
class Tile:
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int
    texture: float  # min eigenvalue of the structure tensor per pixel (Shi-Tomasi score)


@dataclass
class MotionResult:
    fps: float
    n_frames: int
    frame_size: tuple[int, int]  # (width, height)
    tiles: list[Tile]
    dx: np.ndarray  # (n_tiles, n_frames) horizontal displacement, pixels
    dy: np.ndarray  # (n_tiles, n_frames) vertical displacement, pixels
    response: np.ndarray  # (n_tiles,) trackability 0..1 (texture, saturating)
    shake: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))  # removed global motion
    first_frame: np.ndarray | None = None


def read_video(path: str, max_frames: int = 1200, max_width: int = 480, blur: float = 0.0) -> tuple[np.ndarray, float, np.ndarray]:
    """Read grayscale float32 frames (T, H, W), downscaled so the long side is at most max_width.

    Frames are blurred as they are read (to keep memory flat); the first frame is also
    returned unblurred for display.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = min(1.0, max_width / max(w, h))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if not frames:
            first = g.copy()
        frames.append(cv2.GaussianBlur(g, (0, 0), blur) if blur else g)
    cap.release()
    if len(frames) < 16:
        raise ValueError(f"video too short: {len(frames)} frames")
    return np.stack(frames), float(fps), first


def _tile_shifts(F: np.ndarray, R: np.ndarray, win: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Lucas-Kanade shift of every frame in F (T, h, w) relative to reference R (h, w)."""
    gy, gx = np.gradient(R)
    wgx, wgy = gx * win, gy * win
    a11, a12, a22 = float((wgx * gx).sum()), float((wgx * gy).sum()), float((wgy * gy).sum())
    det = a11 * a22 - a12 * a12
    min_eig = 0.5 * (a11 + a22 - np.sqrt((a11 - a22) ** 2 + 4 * a12 * a12)) / win.sum()
    if det <= 1e-9:
        n = F.shape[0]
        return np.zeros(n), np.zeros(n), 0.0
    It = F - R  # (T, h, w)
    b1 = -(It * wgx).sum(axis=(1, 2))
    b2 = -(It * wgy).sum(axis=(1, 2))
    dx = (a22 * b1 - a12 * b2) / det
    dy = (a11 * b2 - a12 * b1) / det
    return dx, dy, float(min_eig)


def make_tiles(shape: tuple[int, int], grid: tuple[int, int]) -> list[Tile]:
    rows, cols = grid
    h, w = shape
    th, tw = h // rows, w // cols
    return [Tile(r, c, c * tw, r * th, tw, th, 0.0) for r in range(rows) for c in range(cols)]


def extract_motion(path: str, grid: tuple[int, int] = (6, 8), fps_override: float | None = None) -> MotionResult:
    frames, fps, first = read_video(path, blur=BLUR_SIGMA)
    if fps_override:
        fps = fps_override
    ref = frames.mean(axis=0)
    n_f, h, w = frames.shape
    tiles = make_tiles((h, w), grid)
    n_t = len(tiles)

    dx = np.zeros((n_t, n_f))
    dy = np.zeros((n_t, n_f))
    wins: dict[tuple[int, int], np.ndarray] = {}
    for i, t in enumerate(tiles):
        key = (t.w, t.h)
        if key not in wins:
            wins[key] = np.outer(np.hanning(t.h), np.hanning(t.w)).astype(np.float32)
        sl = (slice(t.y, t.y + t.h), slice(t.x, t.x + t.w))
        dx[i], dy[i], t.texture = _tile_shifts(frames[:, sl[0], sl[1]], ref[sl], wins[key])

    texture = np.array([t.texture for t in tiles])
    response = np.clip(texture / (4 * MIN_TEXTURE), 0, 1)

    # Camera shake: the median motion of trackable tiles. The machine usually
    # covers a minority of them, so the median follows the static scene.
    good = texture >= MIN_TEXTURE
    if good.sum() >= 4:
        shake = np.vstack([np.median(dx[good], axis=0), np.median(dy[good], axis=0)])
        dx -= shake[0]
        dy -= shake[1]
    else:
        shake = np.zeros((2, n_f))

    return MotionResult(fps, n_f, (w, h), tiles, dx, dy, response, shake, first)
