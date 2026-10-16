"""Render a test clip with a known vibration, so the pipeline can be checked against ground truth.

A textured plate (the "machine") shakes at f Hz with a sub-pixel amplitude,
plus optional 2x and 3x components, on a static textured background, with
sensor noise and a little camera shake.

    python tools/make_synthetic.py out.mp4 --hz 19.5 --amp 0.3 --h2 0.1 --fps 240
"""

import argparse
import subprocess

import cv2
import numpy as np


def render(path: str, hz: float, amp: float, h2: float, h3: float, fps: float, seconds: float,
           noise: float, shake: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    W, H, S = 640, 480, 4  # render at 4x and downsample, so sub-pixel motion is real
    bg = cv2.GaussianBlur(rng.uniform(30, 220, (H * S, W * S)).astype(np.float32), (0, 0), 3)
    plate = cv2.GaussianBlur(rng.uniform(0, 255, (160 * S, 220 * S)).astype(np.float32), (0, 0), 3)
    px, py = 300 * S, 160 * S
    canvas = np.zeros_like(bg)
    canvas[py:py + plate.shape[0], px:px + plate.shape[1]] = plate
    mask = np.zeros_like(bg)
    mask[py:py + plate.shape[0], px:px + plate.shape[1]] = 1.0
    n = int(fps * seconds)
    # High-bitrate H.264, like a phone's slow-motion recording. OpenCV's default
    # mp4v encoder is low quality and smears exactly the sub-pixel motion we test.
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12",
                           "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / fps
        d = amp * (np.sin(2 * np.pi * hz * t) + h2 / amp * np.sin(2 * np.pi * 2 * hz * t + 0.7)
                   + h3 / amp * np.sin(2 * np.pi * 3 * hz * t + 1.3)) if amp else 0.0
        cam = shake * np.array([np.sin(2 * np.pi * 1.3 * t), np.cos(2 * np.pi * 0.9 * t)])
        # Bilinear warp keeps the plate's shift sub-pixel instead of snapping to the grid.
        T = np.float32([[1, 0, d * S], [0, 1, 0]])
        moved = cv2.warpAffine(canvas, T, (W * S, H * S))
        m = cv2.warpAffine(mask, T, (W * S, H * S))
        frame = bg * (1 - m) + moved
        M = np.float32([[1, 0, cam[0] * S], [0, 1, cam[1] * S]])
        frame = cv2.warpAffine(frame, M, (W * S, H * S), borderMode=cv2.BORDER_REFLECT)
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        small += rng.normal(0, noise, small.shape)
        ff.stdin.write(np.clip(small, 0, 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--hz", type=float, default=19.5)
    ap.add_argument("--amp", type=float, default=0.3, help="1x amplitude in pixels")
    ap.add_argument("--h2", type=float, default=0.0, help="2x amplitude in pixels")
    ap.add_argument("--h3", type=float, default=0.0, help="3x amplitude in pixels")
    ap.add_argument("--fps", type=float, default=240)
    ap.add_argument("--seconds", type=float, default=4)
    ap.add_argument("--noise", type=float, default=2.0)
    ap.add_argument("--shake", type=float, default=0.1)
    a = ap.parse_args()
    render(a.out, a.hz, a.amp, a.h2, a.h3, a.fps, a.seconds, a.noise, a.shake)
