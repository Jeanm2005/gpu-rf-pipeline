"""Checks on the synthetic fixtures themselves (no GPU needed).

If the ground truth is wrong, every GPU-vs-reference comparison built on it is meaningless,
so the generator gets its own tests.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gen_fixtures  # noqa: E402

CF32 = ["tone", "tone_odd", "short", "zeros", "ch0", "ch1"]
TAPS = ["lowpass", "lowpass_long", "identity"]


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("fixtures")
    gen_fixtures.generate(out)
    return out


def load_cf32(path: Path) -> np.ndarray:
    iq = np.fromfile(path, dtype="<f4")
    return iq[0::2] + 1j * iq[1::2]


def digest(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.iterdir())}


def test_generation_is_deterministic(fixtures: Path, tmp_path: Path) -> None:
    gen_fixtures.generate(tmp_path)
    assert digest(tmp_path) == digest(fixtures)


@pytest.mark.parametrize("name", CF32)
def test_cf32_matches_sidecar(fixtures: Path, name: str) -> None:
    meta = json.loads((fixtures / f"{name}.json").read_text())
    for key in ("sample_rate", "num_samples", "generator", "seed"):
        assert key in meta
    size = (fixtures / f"{name}.cf32").stat().st_size
    assert size == meta["num_samples"] * 8  # 2 float32 per complex sample
    assert size < 5e6
    assert np.all(np.isfinite(load_cf32(fixtures / f"{name}.cf32")))


@pytest.mark.parametrize("name", TAPS)
def test_taps_match_sidecar(fixtures: Path, name: str) -> None:
    meta = json.loads((fixtures / f"{name}.json").read_text())
    taps = np.fromfile(fixtures / f"{name}.f32", dtype="<f4")
    assert len(taps) == meta["num_taps"]


def test_tone_peaks_are_on_the_stated_bins(fixtures: Path) -> None:
    meta = json.loads((fixtures / "tone.json").read_text())
    nfft = meta["ref_nfft"]
    spectrum = np.abs(np.fft.fft(load_cf32(fixtures / "tone.cf32")[:nfft]))
    expected = sorted(t["bin"] % nfft for t in meta["tones"])
    assert sorted(np.argsort(spectrum)[-len(expected):]) == expected


def test_zeros_is_all_zero(fixtures: Path) -> None:
    assert not np.any(np.fromfile(fixtures / "zeros.cf32", dtype="<f4"))


def test_two_channel_delay_is_recoverable(fixtures: Path) -> None:
    meta = json.loads((fixtures / "ch1.json").read_text())
    a = load_cf32(fixtures / "ch0.cf32")
    b = load_cf32(fixtures / "ch1.cf32")
    # r[k] = sum_n b[n + k] * conj(a[n]); np.correlate conjugates its second argument.
    r = np.correlate(b, a, mode="full")
    lags = np.arange(-(len(a) - 1), len(b))
    assert lags[np.argmax(np.abs(r))] == meta["true_delay_samples"]
