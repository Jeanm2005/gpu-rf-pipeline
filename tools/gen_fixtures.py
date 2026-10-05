#!/usr/bin/env python3
"""Generate deterministic synthetic IQ fixtures with known ground truth.

Outputs, per fixture:
  <name>.cf32  interleaved little-endian float32 [I0, Q0, I1, Q1, ...] (SigMF cf32_le)
  <name>.json  sidecar metadata: sample_rate, num_samples, generator, seed, ground truth
Filter taps:
  <name>.f32   raw little-endian float32 array (real taps), plus a <name>.json sidecar

Every fixture has its own fixed seed, so adding or reordering fixtures never changes the
bytes of an existing one.

Usage:
    python tools/gen_fixtures.py --out tests/fixtures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

GENERATOR = "gpu-rf-pipeline/tools/gen_fixtures.py"
SAMPLE_RATE = 1_000_000.0  # Hz
REF_NFFT = 1024  # tones are placed on exact bins of this FFT size

# Two-channel capture: channel B is channel A delayed by this many samples.
TRUE_DELAY_SAMPLES = 37


def complex_noise(rng: np.random.Generator, n: int, std: float) -> np.ndarray:
    """Circular complex Gaussian noise with total standard deviation `std`.

    The power is split evenly between I and Q, so each component has std / sqrt(2).
    """
    scale = std / np.sqrt(2.0)
    return scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def tones_plus_noise(
    rng: np.random.Generator,
    n: int,
    tones: list[dict[str, float]],
    noise_std: float,
) -> np.ndarray:
    """Sum of complex exponentials plus complex Gaussian noise.

    Each tone is {"bin": k, "amplitude": a}: frequency k * SAMPLE_RATE / REF_NFFT. A negative
    bin is a negative frequency, which a complex IQ signal can represent and a real one cannot.
    """
    t = np.arange(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.complex128)
    for tone in tones:
        x += tone["amplitude"] * np.exp(2j * np.pi * tone["bin"] / REF_NFFT * t)
    return x + complex_noise(rng, n, noise_std)


def delay(x: np.ndarray, d: int) -> np.ndarray:
    """Delay by d >= 0 samples, zero-filled (not circular): y[n] = x[n - d], y[n < d] = 0."""
    y = np.zeros_like(x)
    y[d:] = x[: len(x) - d]
    return y


def write_cf32(out: Path, name: str, x: np.ndarray, meta: dict[str, Any]) -> None:
    iq = np.empty(2 * len(x), dtype="<f4")
    iq[0::2] = x.real
    iq[1::2] = x.imag
    iq.tofile(out / f"{name}.cf32")
    full = {
        "datatype": "cf32_le",
        "sample_rate": SAMPLE_RATE,
        "num_samples": len(x),
        "generator": GENERATOR,
        **meta,
    }
    (out / f"{name}.json").write_text(json.dumps(full, indent=2) + "\n")


def write_taps(out: Path, name: str, taps: np.ndarray, meta: dict[str, Any]) -> None:
    taps.astype("<f4").tofile(out / f"{name}.f32")
    full = {"datatype": "rf32_le", "num_taps": len(taps), "generator": GENERATOR, **meta}
    (out / f"{name}.json").write_text(json.dumps(full, indent=2) + "\n")


def tone_meta(tones: list[dict[str, float]], noise_std: float) -> dict[str, Any]:
    return {
        "ref_nfft": REF_NFFT,
        "tones": [
            {**t, "freq_hz": t["bin"] * SAMPLE_RATE / REF_NFFT} for t in tones
        ],
        "noise_std": noise_std,
    }


def generate(out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # --- Tones plus noise: the main fixture for spectrum and FIR tests -----------------------
    # One tone inside the lowpass passband (bin 40), one in the stopband at a negative
    # frequency (bin -300). After lowpass filtering the second one should be gone.
    tones = [{"bin": 40, "amplitude": 1.0}, {"bin": -300, "amplitude": 0.5}]
    seed = 1001
    x = tones_plus_noise(np.random.default_rng(seed), 65536, tones, noise_std=0.05)
    write_cf32(out, "tone", x, {"seed": seed, "description": "two tones plus noise",
                                **tone_meta(tones, 0.05)})
    written.append("tone")

    # --- Edge case: length that is not a multiple of any block or FFT size (prime) -----------
    seed = 1002
    x = tones_plus_noise(np.random.default_rng(seed), 10007, tones, noise_std=0.05)
    write_cf32(out, "tone_odd", x, {"seed": seed, "description": "prime length, 10007 samples",
                                    **tone_meta(tones, 0.05)})
    written.append("tone_odd")

    # --- Edge case: very short input, shorter than every filter and every thread block -------
    seed = 1003
    x = tones_plus_noise(np.random.default_rng(seed), 7, tones, noise_std=0.05)
    write_cf32(out, "short", x, {"seed": seed, "description": "7 samples",
                                 **tone_meta(tones, 0.05)})
    written.append("short")

    # --- Edge case: all-zero input (output must be exactly zero, no NaN) ---------------------
    write_cf32(out, "zeros", np.zeros(4096, dtype=np.complex128),
               {"seed": None, "description": "all zeros"})
    written.append("zeros")

    # --- Two-channel capture with a known delay ---------------------------------------------
    # The source is white noise, not a tone: a tone's correlation is periodic, so its peak is
    # ambiguous, while white noise correlates with itself at exactly one lag. Each channel
    # gets independent receiver noise, as two real antennas would.
    seed = 2001
    rng = np.random.default_rng(seed)
    n = 16384
    source = complex_noise(rng, n, std=1.0)
    ch0 = source + complex_noise(rng, n, std=0.3)
    ch1 = delay(source, TRUE_DELAY_SAMPLES) + complex_noise(rng, n, std=0.3)
    pair = {
        "seed": seed,
        "description": "ch1 is ch0's source delayed by true_delay_samples, independent noise",
        "true_delay_samples": TRUE_DELAY_SAMPLES,
        "delay_convention": "ch1[n] = source[n - true_delay_samples], zero-filled",
        "source_std": 1.0,
        "noise_std": 0.3,
    }
    write_cf32(out, "ch0", ch0, {**pair, "channel": 0, "paired_with": "ch1"})
    write_cf32(out, "ch1", ch1, {**pair, "channel": 1, "paired_with": "ch0"})
    written += ["ch0", "ch1"]

    # --- Filter taps -------------------------------------------------------------------------
    # 63-tap lowpass, cutoff at 0.2 of Nyquist: passes bin 40, rejects bin -300.
    cutoff = 0.2
    write_taps(out, "lowpass", signal.firwin(63, cutoff),
               {"design": "scipy.signal.firwin", "cutoff_fraction_of_nyquist": cutoff})
    # Edge case: a filter longer than one shared-memory tile (and than a 1024-thread block).
    write_taps(out, "lowpass_long", signal.firwin(1025, cutoff),
               {"design": "scipy.signal.firwin", "cutoff_fraction_of_nyquist": cutoff})
    # Single tap of 1.0: output must equal input. Useful when debugging indexing.
    write_taps(out, "identity", np.array([1.0]), {"design": "identity"})
    written += ["lowpass", "lowpass_long", "identity"]

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"),
                        help="output directory (default: tests/fixtures)")
    args = parser.parse_args()
    names = generate(args.out)
    total = sum(f.stat().st_size for f in args.out.iterdir() if f.is_file())
    print(f"wrote {len(names)} fixtures to {args.out} ({total / 1e6:.2f} MB total)")


if __name__ == "__main__":
    main()
