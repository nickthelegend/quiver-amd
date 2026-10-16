"""Ground-truth tests: render clips with a known vibration and check Quiver recovers it."""

import subprocess
import sys
from pathlib import Path

import pytest

from quiver.diagnose import diagnose, resolve_alias
from quiver.motion import extract_motion
from quiver.spectrum import analyze

ROOT = Path(__file__).resolve().parents[1]
SCALE = 480 / 640  # analysis downscales the 640-wide test clip to 480


def clip(tmp_path_factory, name, **kw) -> str:
    out = tmp_path_factory.mktemp("clips") / f"{name}.mp4"
    args = [sys.executable, str(ROOT / "tools/make_synthetic.py"), str(out)]
    for k, v in kw.items():
        args += [f"--{k}", str(v)]
    subprocess.run(args, check=True)
    return str(out)


def run(path, **kw):
    m = extract_motion(path)
    s = analyze(m)
    return diagnose(m, s, **kw)


@pytest.fixture(scope="module")
def imbalance(tmp_path_factory):
    return clip(tmp_path_factory, "imbalance", hz=19.5, amp=0.3, h2=0.04)


def test_imbalance_frequency_and_amplitude(imbalance):
    d = run(imbalance)
    assert d.running_hz == pytest.approx(19.5, abs=0.1)
    assert d.orders["1x"] == pytest.approx(0.3 * SCALE, rel=0.25)
    assert d.findings[0].fault == "mass imbalance"
    assert d.action.kind == "diagnose"


def test_misalignment(tmp_path_factory):
    d = run(clip(tmp_path_factory, "misalign", hz=24.0, amp=0.25, h2=0.2))
    faults = [f.fault for f in d.findings]
    assert "misalignment" in faults
    assert d.running_hz == pytest.approx(24.0, abs=0.15)


def test_looseness(tmp_path_factory):
    d = run(clip(tmp_path_factory, "loose", hz=15.0, amp=0.25, h2=0.1, h3=0.1))
    assert any(f.fault in ("mechanical looseness", "misalignment") for f in d.findings)


def test_still_machine_asks_for_recapture(tmp_path_factory):
    d = run(clip(tmp_path_factory, "still", hz=19.5, amp=0.0))
    assert d.action.kind == "recapture"


def test_low_fps_requests_slow_motion(tmp_path_factory):
    # 30 fps: Nyquist is 15 Hz, so a 19.5 Hz machine aliases to 10.5 Hz.
    d = run(clip(tmp_path_factory, "lowfps", hz=19.5, amp=0.3, fps=30, seconds=8))
    assert d.action.kind == "recapture"
    assert any("slow motion" in s or "fps" in s for s in d.action.instructions)


def test_nameplate_rpm_overrides_peak(imbalance):
    d = run(imbalance, rpm=1170)
    assert d.running_source == "nameplate rpm"
    assert d.running_hz == pytest.approx(19.5)


def test_resolve_alias_finds_true_frequency():
    # 19.5 Hz seen at 30 fps -> 10.5 Hz; at 60 fps -> 19.5 Hz (below Nyquist 30).
    assert 19.5 in resolve_alias(10.5, 30, 19.5, 60, f_max=100)
    # 47 Hz at 30 fps -> 13 Hz; at 240 fps -> 47 Hz.
    assert 47.0 in resolve_alias(13.0, 30, 47.0, 240, f_max=100)
