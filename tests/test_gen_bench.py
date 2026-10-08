"""Checks on the benchmark-input generator, at small sizes (no GPU needed).

The benchmark inputs are too large to commit, so they are regenerated on every machine that
profiles. These tests make sure a regenerated set is the same data with the same ground truth.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gen_bench  # noqa: E402

FIR_SAMPLES = 5000
XCORR_SAMPLES = 3000
NAMES = {"bench_tone": FIR_SAMPLES, "bench_ch0": XCORR_SAMPLES, "bench_ch1": XCORR_SAMPLES}


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bench")
    gen_bench.generate(out, FIR_SAMPLES, XCORR_SAMPLES)
    return out


def load_cf32(path: Path) -> np.ndarray:
    iq = np.fromfile(path, dtype="<f4")
    return iq[0::2] + 1j * iq[1::2]


def digest(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir())}


def test_generation_is_deterministic(bench: Path, tmp_path: Path) -> None:
    gen_bench.generate(tmp_path, FIR_SAMPLES, XCORR_SAMPLES)
    assert digest(tmp_path) == digest(bench)


@pytest.mark.parametrize("name", sorted(NAMES))
def test_cf32_matches_sidecar(bench: Path, name: str) -> None:
    meta = json.loads((bench / f"{name}.json").read_text())
    assert meta["num_samples"] == NAMES[name]
    assert meta["generator"] == gen_bench.GENERATOR
    assert meta["seed"] is not None
    assert (bench / f"{name}.cf32").stat().st_size == NAMES[name] * 8
    assert np.all(np.isfinite(load_cf32(bench / f"{name}.cf32")))


def test_two_channel_delay_is_recoverable(bench: Path) -> None:
    meta = json.loads((bench / "bench_ch1.json").read_text())
    a = load_cf32(bench / "bench_ch0.cf32")
    b = load_cf32(bench / "bench_ch1.cf32")
    # r[k] = sum_n b[n + k] * conj(a[n]); np.correlate conjugates its second argument.
    r = np.correlate(b, a, mode="full")
    lags = np.arange(-(len(a) - 1), len(b))
    assert lags[np.argmax(np.abs(r))] == meta["true_delay_samples"]
