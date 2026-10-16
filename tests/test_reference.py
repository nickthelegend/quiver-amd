"""Accelerometer ground truth: phyphox CSV exports in both locales."""

import numpy as np

from quiver.reference import compare, parse_phyphox, spectrum


def fake_export(hz=19.5, fs=100.0, seconds=10, sep=",", decimal="."):
    rng = np.random.default_rng(0)
    t = np.arange(0, seconds, 1 / fs) + rng.normal(0, 1e-4, int(seconds * fs))
    x = 0.02 * rng.normal(size=t.size)
    y = 0.8 * np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(2 * np.pi * 2 * hz * t) + 0.02 * rng.normal(size=t.size)
    z = 0.02 * rng.normal(size=t.size)
    head = sep.join(['"Time (s)"', '"Linear Acceleration x (m/s^2)"', '"Linear Acceleration y (m/s^2)"',
                     '"Linear Acceleration z (m/s^2)"', '"Absolute acceleration (m/s^2)"'])
    lines = [head]
    for row in zip(t, x, y, z, np.sqrt(x * x + y * y + z * z)):
        lines.append(sep.join(f"{v:.6f}".replace(".", decimal) for v in row))
    return "\n".join(lines).encode()


def test_parses_and_finds_running_speed():
    t, acc = parse_phyphox(fake_export())
    ref = spectrum(t, acc)
    assert ref["axis"] == 1
    assert abs(ref["peaks"][0]["hz"] - 19.5) < 0.15
    assert any(abs(p["hz"] - 39.0) < 0.2 for p in ref["peaks"])


def test_european_export_semicolon_comma_decimal():
    t, acc = parse_phyphox(fake_export(sep=";", decimal=","))
    assert abs(spectrum(t, acc)["peaks"][0]["hz"] - 19.5) < 0.15


def test_compare_with_video_peaks():
    t, acc = parse_phyphox(fake_export())
    c = compare(spectrum(t, acc), [{"hz": 19.5, "amp_px": 0.24}, {"hz": 39.0, "amp_px": 0.04}])
    assert c["match"] and c["error_hz"] < 0.15
