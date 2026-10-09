"""GPU results against the CPU reference.

Each test runs the rfgpu binary on a fixture and compares its output file to the function of
the same name in tools/reference.py. The reference is float64 and the GPU is float32, so the
tolerances cover float32 rounding only, and they are stated next to each operation.

The whole module is skipped when build/rfgpu does not exist or cannot reach a GPU. Set
RFGPU_BIN to test a binary somewhere else.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
RFGPU = Path(os.environ.get("RFGPU_BIN", ROOT / "build" / "rfgpu"))
sys.path.insert(0, str(ROOT / "tools"))
import reference  # noqa: E402


def gpu_info() -> dict:
    """Output of `rfgpu info`, or a module-level skip if the binary or the GPU is missing."""
    if not RFGPU.exists():
        pytest.skip(f"{RFGPU} not built", allow_module_level=True)
    try:
        proc = subprocess.run([str(RFGPU), "info"], capture_output=True, text=True, timeout=60)
    except OSError as e:
        pytest.skip(f"cannot run {RFGPU}: {e}", allow_module_level=True)
    if proc.returncode != 0:
        pytest.skip(f"rfgpu cannot use a GPU: {proc.stderr.strip()}", allow_module_level=True)
    return json.loads(proc.stdout)


INFO = gpu_info()
FIR_IMPLS: list[str] = INFO["impls"]["fir"]
XCORR_IMPLS: list[str] = INFO["impls"]["xcorr"]


def rfgpu(*args: object) -> subprocess.CompletedProcess[str]:
    """Run rfgpu and fail the test with its stderr if it exits with an error."""
    proc = subprocess.run([str(RFGPU), *map(str, args)], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, f"rfgpu {' '.join(map(str, args))}\n{proc.stderr}"
    return proc


def cf32(name: str) -> Path:
    return FIXTURES / f"{name}.cf32"


def taps_file(name: str) -> Path:
    return FIXTURES / f"{name}.f32"


# --- Spectrum (cuFFT) ----------------------------------------------------------------------
# Unnormalized FFT, so values reach A * nfft (about 1000 here). Measured cuFFT error on the
# fixtures is about 1.5e-4 absolute on every bin; atol leaves roughly a factor of 10 above it.
SPECTRUM_RTOL = 1e-5
SPECTRUM_ATOL = 2e-3


@pytest.mark.parametrize("nfft", [1024, 256])
@pytest.mark.parametrize("name", ["tone", "tone_odd", "short"])
def test_spectrum_matches_reference(tmp_path: Path, name: str, nfft: int) -> None:
    # tone_odd ends in a partial frame; short is smaller than one frame.
    out = tmp_path / "spectrum.cf32"
    rfgpu("spectrum", "--in", cf32(name), "--out", out, "--nfft", nfft)
    expected = reference.spectrum(reference.read_cf32(cf32(name)), nfft).ravel()
    actual = reference.read_cf32(out)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=SPECTRUM_RTOL, atol=SPECTRUM_ATOL)


def test_spectrum_of_zeros_is_exactly_zero(tmp_path: Path) -> None:
    out = tmp_path / "spectrum.cf32"
    rfgpu("spectrum", "--in", cf32("zeros"), "--out", out, "--nfft", 1024)
    actual = reference.read_cf32(out)
    assert actual.shape == (4096,)
    assert not np.any(actual)  # also rules out NaN


# --- FIR -----------------------------------------------------------------------------------
# Output values are of order 1. Summing up to 1025 float32 products in order gives an error
# below 1e-6 on the fixtures (estimated on the CPU in float32); atol leaves a factor of 10.
FIR_RTOL = 1e-5
FIR_ATOL = 1e-5


@pytest.mark.parametrize("impl", FIR_IMPLS)
@pytest.mark.parametrize("name,taps", [
    ("tone", "lowpass"),
    ("tone", "lowpass_long"),      # filter longer than one tile and one 1024-thread block
    ("tone_odd", "lowpass"),       # length not a multiple of any block size
    ("tone_odd", "lowpass_long"),
    ("short", "lowpass"),          # input shorter than the filter
    ("short", "lowpass_long"),
])
def test_fir_matches_reference(tmp_path: Path, impl: str, name: str, taps: str) -> None:
    out = tmp_path / "fir.cf32"
    rfgpu("fir", "--in", cf32(name), "--taps", taps_file(taps), "--out", out, "--impl", impl)
    expected = reference.fir(reference.read_cf32(cf32(name)), reference.read_f32(taps_file(taps)))
    actual = reference.read_cf32(out)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=FIR_RTOL, atol=FIR_ATOL)


@pytest.mark.parametrize("impl", FIR_IMPLS)
@pytest.mark.parametrize("name", ["tone", "tone_odd", "short"])
def test_fir_identity_returns_the_input_exactly(tmp_path: Path, impl: str, name: str) -> None:
    # One tap of 1.0: every output is 1.0f * x[n], which float32 computes without rounding.
    out = tmp_path / "fir.cf32"
    rfgpu("fir", "--in", cf32(name), "--taps", taps_file("identity"), "--out", out,
          "--impl", impl)
    np.testing.assert_allclose(reference.read_cf32(out), reference.read_cf32(cf32(name)),
                               rtol=0, atol=0)


@pytest.mark.parametrize("impl", FIR_IMPLS)
@pytest.mark.parametrize("taps", ["lowpass", "lowpass_long", "identity"])
def test_fir_of_zeros_is_exactly_zero(tmp_path: Path, impl: str, taps: str) -> None:
    out = tmp_path / "fir.cf32"
    rfgpu("fir", "--in", cf32("zeros"), "--taps", taps_file(taps), "--out", out, "--impl", impl)
    actual = reference.read_cf32(out)
    assert actual.shape == (4096,)
    assert not np.any(actual)  # also rules out NaN


# --- Cross-correlation ---------------------------------------------------------------------
# The ch0 / ch1 peak is about 1.65e4 and the other lags are of order 100. Measured error
# against the float64 reference on that pair: 1.1e-2 on the peak and up to 1.5e-3 elsewhere for
# the time-domain kernels (one running float32 sum per lag), 2.6e-3 and 1.0e-3 for the cuFFT
# version. rtol covers the peak (0.165 allowed) and atol the rest, with a factor of 3 or more
# to spare.
XCORR_RTOL = 1e-5
XCORR_ATOL = 5e-3


@pytest.mark.parametrize("impl", XCORR_IMPLS)
@pytest.mark.parametrize("a,b", [
    ("ch0", "ch1"),
    ("ch1", "ch0"),
    ("tone_odd", "short"),   # unequal lengths, a longer than b
    ("short", "tone_odd"),   # unequal lengths, b longer than a
    ("short", "short"),
])
def test_xcorr_matches_reference(tmp_path: Path, impl: str, a: str, b: str) -> None:
    out = tmp_path / "xcorr.cf32"
    rfgpu("xcorr", "--a", cf32(a), "--b", cf32(b), "--out", out, "--impl", impl)
    expected = reference.xcorr(reference.read_cf32(cf32(a)), reference.read_cf32(cf32(b)))
    actual = reference.read_cf32(out)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=XCORR_RTOL, atol=XCORR_ATOL)


@pytest.mark.parametrize("impl", XCORR_IMPLS)
def test_xcorr_recovers_the_known_delay(tmp_path: Path, impl: str) -> None:
    out = tmp_path / "xcorr.cf32"
    rfgpu("xcorr", "--a", cf32("ch0"), "--b", cf32("ch1"), "--out", out, "--impl", impl)
    r = reference.read_cf32(out)
    meta = json.loads((FIXTURES / "ch1.json").read_text())
    lags = reference.xcorr_lags(meta["num_samples"], meta["num_samples"])
    assert len(r) == len(lags)
    assert lags[np.argmax(np.abs(r))] == meta["true_delay_samples"]


@pytest.mark.parametrize("impl", XCORR_IMPLS)
def test_xcorr_of_zeros_is_exactly_zero(tmp_path: Path, impl: str) -> None:
    out = tmp_path / "xcorr.cf32"
    rfgpu("xcorr", "--a", cf32("zeros"), "--b", cf32("zeros"), "--out", out, "--impl", impl)
    actual = reference.read_cf32(out)
    assert actual.shape == (2 * 4096 - 1,)
    assert not np.any(actual)  # also rules out NaN


# --- CLI and timing harness ----------------------------------------------------------------

def test_bench_reports_each_phase_separately(tmp_path: Path) -> None:
    proc = rfgpu("spectrum", "--in", cf32("tone"), "--out", tmp_path / "s.cf32",
                 "--bench", 10, "--warmup", 2)
    report = json.loads(proc.stdout)
    assert report["op"] == "spectrum" and report["runs"] == 10 and report["warmup"] == 2
    assert report["gpu"] == INFO["gpu"]
    for phase in ("h2d_ms", "kernel_ms", "d2h_ms"):
        assert report[phase]["mean"] > 0
        assert report[phase]["std"] >= 0


def test_bench_does_not_change_the_output(tmp_path: Path) -> None:
    rfgpu("spectrum", "--in", cf32("tone_odd"), "--out", tmp_path / "once.cf32")
    rfgpu("spectrum", "--in", cf32("tone_odd"), "--out", tmp_path / "bench.cf32", "--bench", 10)
    assert (tmp_path / "once.cf32").read_bytes() == (tmp_path / "bench.cf32").read_bytes()


def run_expecting_failure(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(RFGPU), *map(str, args)], capture_output=True, text=True,
                          timeout=60)


def test_unknown_impl_is_a_usage_error(tmp_path: Path) -> None:
    proc = run_expecting_failure("fir", "--in", cf32("tone"), "--taps", taps_file("lowpass"),
                                 "--out", tmp_path / "y.cf32", "--impl", "no_such_impl")
    assert proc.returncode == 2
    assert "no_such_impl" in proc.stderr
    assert not (tmp_path / "y.cf32").exists()


def test_missing_input_file_is_a_runtime_error(tmp_path: Path) -> None:
    proc = run_expecting_failure("spectrum", "--in", tmp_path / "missing.cf32",
                                 "--out", tmp_path / "s.cf32")
    assert proc.returncode == 1
    assert "missing.cf32" in proc.stderr


def test_truncated_cf32_is_rejected(tmp_path: Path) -> None:
    np.zeros(3, dtype="<f4").tofile(tmp_path / "bad.cf32")  # 12 bytes: one and a half samples
    proc = run_expecting_failure("spectrum", "--in", tmp_path / "bad.cf32",
                                 "--out", tmp_path / "s.cf32")
    assert proc.returncode == 1
    assert "bad.cf32" in proc.stderr
