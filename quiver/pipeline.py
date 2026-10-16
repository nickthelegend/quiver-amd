"""End-to-end analysis of one clip: motion -> spectrum -> rules -> sound -> magnified video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import sonify
from .diagnose import diagnose
from .magnify import magnify
from .motion import extract_motion
from .spectrum import analyze


def heatmap_png(motion, spec, out_path: Path) -> None:
    """First frame with the tiles that carry the vibration highlighted."""
    base = cv2.cvtColor(np.clip(motion.first_frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    e = spec.tile_energy / (spec.tile_energy.max() or 1)
    overlay = base.copy()
    for t, v in zip(motion.tiles, e):
        color = cv2.applyColorMap(np.uint8([[int(255 * v)]]), cv2.COLORMAP_INFERNO)[0, 0].tolist()
        cv2.rectangle(overlay, (t.x, t.y), (t.x + t.w, t.y + t.h), color, -1)
    out = cv2.addWeighted(overlay, 0.45, base, 0.55, 0)
    for i in spec.hot_tiles:
        t = motion.tiles[i]
        cv2.rectangle(out, (t.x, t.y), (t.x + t.w, t.y + t.h), (80, 220, 255), 2)
    cv2.imwrite(str(out_path), out)


def run(video: str, out_dir: str, rpm: float | None = None, mm_per_px: float | None = None,
        baseline_1x_px: float | None = None, fps_override: float | None = None,
        make_video: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    motion = extract_motion(video, fps_override=fps_override)
    spec = analyze(motion)
    diag = diagnose(motion, spec, rpm=rpm, mm_per_px=mm_per_px, baseline_1x_px=baseline_1x_px)

    f1 = diag.running_hz
    (out / "true.wav").write_bytes(sonify.to_wav_bytes(sonify.true_audio(spec.signal, spec.fps)))
    factor = sonify.pitch_factor(f1)
    (out / "audible.wav").write_bytes(sonify.to_wav_bytes(sonify.audible_audio(spec.signal, spec.fps, factor)))
    heatmap_png(motion, spec, out / "heatmap.png")
    cv2.imwrite(str(out / "frame.jpg"), np.clip(motion.first_frame, 0, 255).astype(np.uint8))

    if make_video and f1:
        magnify(video, str(out / "magnified.mp4"), spec.fps, (0.8 * f1, 1.2 * f1))

    keep = spec.freqs <= spec.fps / 2
    result = {
        "video": Path(video).name,
        "diagnosis": diag.to_dict(),
        "spectrum": {"hz": np.round(spec.freqs[keep], 3).tolist(), "amp_px": np.round(spec.amp[keep], 5).tolist(),
                     "noise_floor_px": spec.noise_floor},
        "signal": {"fps": spec.fps, "px": np.round(spec.signal[: int(spec.fps * 2)], 4).tolist()},
        "audio": {"true": "true.wav", "audible": "audible.wav", "pitch_factor": factor},
        "media": {"heatmap": "heatmap.png", "frame": "frame.jpg",
                  "magnified": "magnified.mp4" if (out / "magnified.mp4").exists() else None},
    }
    (out / "result.json").write_text(json.dumps(result))
    return result
