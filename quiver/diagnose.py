"""Vibration rules and evidence-quality checks.

The rules are the standard first-pass reading of a rotating-machine spectrum:
  1x dominant, little else          -> mass imbalance
  strong 2x (>= half of 1x)          -> misalignment (or bent shaft)
  a run of 3+ harmonics              -> mechanical looseness
  peaks just under 0.5x              -> sub-synchronous (rub / whirl)
  strong peaks unrelated to 1x       -> another source (bearing, belt, structure)

The quality checks decide whether the evidence is good enough to diagnose at
all, and if not, what the next recording should be. That decision is what
makes Quiver an agent rather than a report generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .motion import MIN_TEXTURE, MotionResult
from .spectrum import Spectrum, amplitude_at

MIN_SNR_DB = 12.0
MIN_AMP_PX = 0.02  # below this, image motion is indistinguishable from sensor noise
MIN_ORDER_FRAC = 0.1  # a harmonic must be at least this fraction of 1x to count


@dataclass
class Finding:
    fault: str
    confidence: float  # 0..1
    evidence: list[str]


@dataclass
class Action:
    kind: str  # "diagnose" | "recapture" | "escalate"
    reason: str
    instructions: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    running_hz: float | None
    running_source: str  # "nameplate rpm" | "dominant peak" | "half of dominant peak" | "none"
    orders: dict[str, float]  # "1x": amplitude in px, ...
    findings: list[Finding]
    quality: dict
    action: Action
    severity: dict | None
    peaks: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _quality(motion: MotionResult, spec: Spectrum) -> dict:
    top = spec.peaks[0] if spec.peaks else None
    shake_rms = float(np.sqrt(np.mean(motion.shake ** 2))) if motion.shake.size else 0.0
    hot_texture = float(np.mean([motion.tiles[i].texture for i in spec.hot_tiles]))
    hot_response = float(np.mean([motion.response[i] for i in spec.hot_tiles]))
    return {
        "fps": spec.fps,
        "nyquist_hz": spec.fps / 2,
        "duration_s": round(spec.duration_s, 2),
        "resolution_hz": round(spec.resolution_hz, 3),
        "top_peak_snr_db": round(top.snr_db, 1) if top else 0.0,
        "top_peak_amp_px": round(top.amp_px, 4) if top else 0.0,
        "camera_shake_px_rms": round(shake_rms, 4),
        "hot_tile_texture": round(hot_texture, 2),
        "hot_tile_match": round(hot_response, 3),
    }


def _pick_running_speed(spec: Spectrum, rpm: float | None) -> tuple[float | None, str]:
    if rpm:
        return rpm / 60.0, "nameplate rpm"
    if not spec.peaks:
        return None, "none"
    top = spec.peaks[0]
    # If there is a real peak at half the dominant one, the dominant one is 2x.
    half = amplitude_at(spec, top.hz / 2)
    if top.hz / 2 >= 1.0 and half > 0.3 * top.amp_px and half > spec.noise_floor * 6:
        return top.hz / 2, "half of dominant peak"
    return top.hz, "dominant peak"


def _decide(q: dict, spec: Spectrum, f1: float | None, rpm: float | None) -> Action:
    steps: list[str] = []
    reasons: list[str] = []
    if q["top_peak_amp_px"] < MIN_AMP_PX:
        reasons.append(f"no vibration above {MIN_AMP_PX} px was found (largest peak {q['top_peak_amp_px']} px); "
                       "either the machine is very smooth or the camera is too far away")
        steps += ["If the machine is running, move the phone closer or zoom in on the housing",
                  "Make sure the clip is filmed in slow motion"]
    elif q["top_peak_snr_db"] < MIN_SNR_DB:
        reasons.append(f"vibration peak is only {q['top_peak_snr_db']} dB above noise (need {MIN_SNR_DB})")
        steps += ["Move the phone closer or zoom so the machine fills more of the frame",
                  "Add light; slow-motion footage gets noisy in dim rooms"]
        if q["hot_tile_texture"] < 2 * MIN_TEXTURE:
            steps.append("Stick a strip of high-contrast tape on the machine housing to give the camera an edge to track")
    if q["camera_shake_px_rms"] > 0.5:
        reasons.append(f"the camera itself moved {q['camera_shake_px_rms']} px RMS")
        steps.append("Brace the phone on a tripod or a solid surface; do not hold it")
    if q["duration_s"] < 2.0:
        reasons.append(f"clip is {q['duration_s']} s, too short to separate nearby frequencies")
        steps.append("Record at least 4 seconds")
    if f1 is not None:
        if 2 * f1 >= q["nyquist_hz"]:
            reasons.append(f"at {q['fps']:.0f} fps the camera can't see 2x running speed ({2 * f1:.1f} Hz), so misalignment can't be ruled out")
            steps.append(f"Re-record in slow motion at {int(np.ceil(6 * f1 / 30) * 30)} fps or more")
        if not rpm and q["fps"] <= 60:
            reasons.append("frame rate is low enough that the dominant peak may be aliased")
            steps.append("Re-record the same view in 240 fps slow motion so Quiver can confirm the true frequency")
    if steps:
        return Action("recapture", "; ".join(reasons), steps)
    return Action("diagnose", "evidence is clean enough to diagnose")


def _severity(spec: Spectrum, f1: float, mm_per_px: float | None, baseline_1x_px: float | None) -> dict | None:
    a1 = amplitude_at(spec, f1)
    out: dict = {}
    if baseline_1x_px:
        out["vs_baseline"] = round(a1 / baseline_1x_px, 2)
    if mm_per_px:
        # Peak displacement -> RMS velocity at 1x, the quantity ISO 10816/20816 zones use.
        v_rms = 2 * np.pi * f1 * (a1 * mm_per_px) / np.sqrt(2)
        out["velocity_mm_s_rms"] = round(float(v_rms), 2)
        out["iso_zone"] = "A (new)" if v_rms < 1.4 else "B (acceptable)" if v_rms < 2.8 else "C (restricted)" if v_rms < 4.5 else "D (damage likely)"
    return out or None


def diagnose(motion: MotionResult, spec: Spectrum, rpm: float | None = None,
             mm_per_px: float | None = None, baseline_1x_px: float | None = None) -> Diagnosis:
    q = _quality(motion, spec)
    f1, source = _pick_running_speed(spec, rpm)
    action = _decide(q, spec, f1, rpm)
    peaks = [{"hz": round(p.hz, 2), "amp_px": round(p.amp_px, 4), "snr_db": round(p.snr_db, 1)} for p in spec.peaks]

    if f1 is None:
        return Diagnosis(None, source, {}, [], q, action, None, peaks)

    orders = {f"{k}x": round(amplitude_at(spec, k * f1), 4) for k in (0.5, 1, 2, 3, 4, 5)}
    a1, a2 = orders["1x"], orders["2x"]
    floor6 = max(spec.noise_floor * 6, MIN_ORDER_FRAC * a1)
    harmonics = [k for k in (2, 3, 4, 5) if orders[f"{k}x"] > floor6]
    high = [k for k in harmonics if k >= 3]
    findings: list[Finding] = []

    if a1 > floor6 and a2 < 0.5 * a1 and not high:
        findings.append(Finding("mass imbalance", min(0.95, 0.6 + 0.1 * (a1 / max(a2, spec.noise_floor))),
                                [f"1x ({f1:.2f} Hz = {f1 * 60:.0f} rpm) dominates at {a1:.3f} px",
                                 f"2x is only {a2 / a1:.0%} of 1x"]))
    if a1 > 0 and a2 >= 0.5 * a1 and a2 > floor6:
        findings.append(Finding("misalignment", min(0.9, 0.5 + 0.2 * a2 / a1),
                                [f"2x ({2 * f1:.2f} Hz) is {a2 / a1:.0%} of 1x",
                                 "confirm with a side view: misalignment also moves the shaft axially"]))
    if len(harmonics) >= 2 and high:
        findings.append(Finding("mechanical looseness", min(0.9, 0.4 + 0.15 * len(harmonics)),
                                [f"harmonics at {', '.join(f'{k}x' for k in harmonics)} stand above the noise",
                                 "check mounting bolts, base plate and bearing housing"]))
    sub = orders["0.5x"]
    if sub > floor6 and sub > 0.3 * a1:
        findings.append(Finding("sub-synchronous vibration (rub or whirl)", 0.5,
                                [f"peak near 0.5x ({0.5 * f1:.2f} Hz) at {sub / max(a1, 1e-9):.0%} of 1x"]))
    others = [p for p in spec.peaks if min(abs(p.hz / f1 - k) for k in (0.5, 1, 2, 3, 4, 5)) > 0.06
              and p.snr_db > 20 and p.amp_px > 0.15 * a1 and p.hz > 3]
    if others:
        findings.append(Finding("non-synchronous source", 0.4,
                                [f"{p.hz:.2f} Hz is not a multiple of running speed" for p in others[:3]] +
                                ["could be a bearing defect, belt, or a neighbouring machine"]))
    if not findings:
        findings.append(Finding("no significant fault", 0.6, [f"no order above {MIN_SNR_DB} dB except 1x at {a1:.3f} px"]))

    sev = _severity(spec, f1, mm_per_px, baseline_1x_px)
    if sev and (sev.get("iso_zone", "").startswith("D") or sev.get("vs_baseline", 0) >= 4) and action.kind == "diagnose":
        action = Action("escalate", "vibration is in the damage range; ask a person before the machine keeps running",
                        ["Request supervisor approval to stop the machine", "Open a priority work order"])

    findings.sort(key=lambda f: f.confidence, reverse=True)
    return Diagnosis(f1, source, orders, findings, q, action, sev, peaks)


def resolve_alias(obs_a: float, fps_a: float, obs_b: float, fps_b: float, f_max: float = 500.0, tol: float = 0.25) -> list[float]:
    """True frequencies consistent with peaks seen at two different frame rates.

    A tone at f seen at sample rate fs shows up at |f - k*fs| for the k that
    folds it into [0, fs/2]. Two recordings at different rates disagree on
    every alias except the true frequency.
    """
    def candidates(obs: float, fs: float) -> list[float]:
        out = []
        k = 0
        while k * fs - obs <= f_max:
            for f in (k * fs + obs, k * fs - obs):
                if 0 < f <= f_max:
                    out.append(f)
            k += 1
        return out

    ca, cb = candidates(obs_a, fps_a), candidates(obs_b, fps_b)
    matches = sorted({round((x + y) / 2, 2) for x in ca for y in cb if abs(x - y) <= tol})
    return matches
