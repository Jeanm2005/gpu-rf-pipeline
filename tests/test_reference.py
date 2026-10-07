"""Checks on the CPU reference itself (no GPU needed).

The GPU tests trust tools/reference.py, so it is checked here two ways: against slow
brute-force loops written straight from the definitions in its docstring, and against the
ground truth stored in the fixture sidecars.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "tools"))
import reference  # noqa: E402

# float64 against float64: only rounding-order differences are allowed.
RTOL = 1e-9
ATOL = 1e-9


def random_complex(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


def meta(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


# --- Brute-force definitions ---------------------------------------------------------------

def fir_brute(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    y = np.zeros(len(x), dtype=np.complex128)
    for n in range(len(x)):
        for k in range(len(taps)):
            if n - k >= 0:
                y[n] += taps[k] * x[n - k]
    return y


def xcorr_brute(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.zeros(len(a) + len(b) - 1, dtype=np.complex128)
    for i, k in enumerate(range(-(len(a) - 1), len(b))):
        for n in range(len(a)):
            if 0 <= n + k < len(b):
                r[i] += b[n + k] * np.conj(a[n])
    return r


def dft_brute(frame: np.ndarray) -> np.ndarray:
    n = len(frame)
    k = np.arange(n)
    return np.array([np.sum(frame * np.exp(-2j * np.pi * m * k / n)) for m in range(n)])


# --- File I/O ------------------------------------------------------------------------------

def test_cf32_round_trip(tmp_path: Path) -> None:
    x = random_complex(1, 100)
    reference.write_cf32(tmp_path / "x.cf32", x)
    # Stored as float32, so the round trip is exact only to float32 precision.
    np.testing.assert_allclose(reference.read_cf32(tmp_path / "x.cf32"), x, rtol=1e-6, atol=1e-6)


def test_read_cf32_rejects_odd_float_count(tmp_path: Path) -> None:
    np.zeros(3, dtype="<f4").tofile(tmp_path / "bad.cf32")
    with pytest.raises(ValueError):
        reference.read_cf32(tmp_path / "bad.cf32")


# --- Spectrum ------------------------------------------------------------------------------

def test_spectrum_matches_brute_force_dft() -> None:
    x = random_complex(2, 40)  # 2.5 frames of 16: the last frame is zero-padded
    frames = reference.spectrum(x, 16)
    assert frames.shape == (3, 16)
    padded = np.concatenate([x, np.zeros(8)])
    for f in range(3):
        np.testing.assert_allclose(frames[f], dft_brute(padded[16 * f: 16 * (f + 1)]),
                                   rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name,expected_frames", [("tone", 64), ("tone_odd", 10), ("short", 1)])
def test_spectrum_frame_count(name: str, expected_frames: int) -> None:
    x = reference.read_cf32(FIXTURES / f"{name}.cf32")
    assert reference.spectrum(x, 1024).shape == (expected_frames, 1024)


def test_spectrum_peaks_on_the_fixture_tone_bins() -> None:
    m = meta("tone")
    mag = reference.magnitude(reference.spectrum(reference.read_cf32(FIXTURES / "tone.cf32"), 1024))
    expected = sorted(t["bin"] % 1024 for t in m["tones"])
    for frame in mag:
        assert sorted(np.argsort(frame)[-len(expected):]) == expected
    # Unnormalized FFT: a tone of amplitude A on an exact bin has magnitude A * nfft. The 2 %
    # allowance covers the noise (std 0.05, about 1.6 per bin against a peak of 512 or more).
    for tone in m["tones"]:
        np.testing.assert_allclose(mag[:, tone["bin"] % 1024], tone["amplitude"] * 1024,
                                   rtol=0.02, atol=0)


def test_spectrum_of_zeros_is_zero() -> None:
    frames = reference.spectrum(reference.read_cf32(FIXTURES / "zeros.cf32"), 1024)
    assert not np.any(frames)


# --- FIR -----------------------------------------------------------------------------------

@pytest.mark.parametrize("num_samples,num_taps", [(50, 7), (50, 1), (5, 12), (1, 1)])
def test_fir_matches_brute_force(num_samples: int, num_taps: int) -> None:
    x = random_complex(3, num_samples)
    taps = np.random.default_rng(4).standard_normal(num_taps)
    y = reference.fir(x, taps)
    assert len(y) == num_samples
    np.testing.assert_allclose(y, fir_brute(x, taps), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name", ["tone", "tone_odd", "short"])
def test_fir_identity_returns_the_input(name: str) -> None:
    x = reference.read_cf32(FIXTURES / f"{name}.cf32")
    y = reference.fir(x, reference.read_f32(FIXTURES / "identity.f32"))
    np.testing.assert_allclose(y, x, rtol=0, atol=0)


@pytest.mark.parametrize("taps_name", ["lowpass", "lowpass_long"])
def test_fir_lowpass_keeps_passband_and_removes_stopband(taps_name: str) -> None:
    x = reference.read_cf32(FIXTURES / "tone.cf32")
    taps = reference.read_f32(FIXTURES / f"{taps_name}.f32")
    y = reference.fir(x, taps)
    # Frame 2 starts at sample 2048, after the start-up transient of both filters (<= 1024).
    mag = reference.magnitude(reference.spectrum(y, 1024))[2]
    # Passband tone (bin 40, amplitude 1.0) keeps its level: within 2 % of 1024.
    np.testing.assert_allclose(mag[40], 1024.0, rtol=0.02, atol=0)
    # Stopband tone (bin -300, 512 before filtering) drops by more than 40 dB (a factor of 100).
    assert mag[-300 % 1024] < 512.0 / 100.0


@pytest.mark.parametrize("taps_name", ["lowpass", "lowpass_long", "identity"])
def test_fir_of_zeros_is_zero(taps_name: str) -> None:
    y = reference.fir(reference.read_cf32(FIXTURES / "zeros.cf32"),
                      reference.read_f32(FIXTURES / f"{taps_name}.f32"))
    assert len(y) == 4096 and not np.any(y)


def test_fir_input_shorter_than_filter() -> None:
    x = reference.read_cf32(FIXTURES / "short.cf32")
    taps = reference.read_f32(FIXTURES / "lowpass.f32")
    np.testing.assert_allclose(reference.fir(x, taps), fir_brute(x, taps), rtol=RTOL, atol=ATOL)


# --- Cross-correlation ---------------------------------------------------------------------

@pytest.mark.parametrize("num_a,num_b", [(20, 20), (9, 31), (31, 9), (1, 5), (1, 1)])
def test_xcorr_matches_brute_force(num_a: int, num_b: int) -> None:
    a = random_complex(5, num_a)
    b = random_complex(6, num_b)
    r = reference.xcorr(a, b)
    assert len(r) == num_a + num_b - 1 == len(reference.xcorr_lags(num_a, num_b))
    np.testing.assert_allclose(r, xcorr_brute(a, b), rtol=RTOL, atol=ATOL)


def test_xcorr_recovers_the_fixture_delay() -> None:
    a = reference.read_cf32(FIXTURES / "ch0.cf32")
    b = reference.read_cf32(FIXTURES / "ch1.cf32")
    r = reference.xcorr(a, b)
    lags = reference.xcorr_lags(len(a), len(b))
    assert lags[np.argmax(np.abs(r))] == meta("ch1")["true_delay_samples"]


def test_xcorr_of_zeros_is_zero() -> None:
    z = reference.read_cf32(FIXTURES / "zeros.cf32")
    assert not np.any(reference.xcorr(z, z))


# --- CLI -----------------------------------------------------------------------------------

def test_cli_writes_the_same_result_as_the_function(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "fir.cf32"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "reference.py"), "fir",
         "--in", str(FIXTURES / "tone_odd.cf32"), "--taps", str(FIXTURES / "lowpass.f32"),
         "--out", str(out)],
        check=True, capture_output=True)
    expected = reference.fir(reference.read_cf32(FIXTURES / "tone_odd.cf32"),
                             reference.read_f32(FIXTURES / "lowpass.f32"))
    # The file holds float32, so allow float32 rounding of values of order 1.
    np.testing.assert_allclose(reference.read_cf32(out), expected, rtol=1e-6, atol=1e-6)
