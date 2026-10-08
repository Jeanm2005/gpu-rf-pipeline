#!/usr/bin/env python3
"""Generate large benchmark inputs for profiling (written to out/, never committed).

The test fixtures are a few hundred kilobytes, which fits in the GPU's caches and finishes
before a profiler sees steady-state behaviour. These inputs use the same signal models and
file formats as tools/gen_fixtures.py, only longer:

  bench_tone.cf32            two tones plus noise, for `rfgpu fir` and `rfgpu spectrum`
  bench_ch0.cf32, bench_ch1  two-channel capture with a known delay, for `rfgpu xcorr`

The two operations get different lengths because their costs grow differently: FIR does
num_samples * num_taps multiply-adds, naive cross-correlation does about num_samples ** 2.
Filter taps are not generated here; use tests/fixtures/lowpass.f32 (63 taps) and
tests/fixtures/lowpass_long.f32 (1025 taps).

Usage:
    python tools/gen_bench.py --out out/bench
    python tools/gen_bench.py --out out/bench --fir-samples 4194304 --xcorr-samples 65536
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import gen_fixtures
from gen_fixtures import TRUE_DELAY_SAMPLES, complex_noise, delay, tone_meta, tones_plus_noise

GENERATOR = "gpu-rf-pipeline/tools/gen_bench.py"

# Defaults: 2**24 samples is 128 MB of cf32, far larger than any cache on the GPU. 2**18
# samples per channel is 2 MB each, but the naive kernel does about 7e10 multiply-adds on it.
DEFAULT_FIR_SAMPLES = 1 << 24
DEFAULT_XCORR_SAMPLES = 1 << 18

# Seeds are separate from the fixture seeds (1001.., 2001) so the two sets stay independent.
SEED_TONE = 9001
SEED_PAIR = 9002


def generate(out: Path, fir_samples: int, xcorr_samples: int) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)

    # Same tones as the `tone` fixture: bin 40 is in the lowpass passband, bin -300 is not.
    tones = [{"bin": 40, "amplitude": 1.0}, {"bin": -300, "amplitude": 0.5}]
    x = tones_plus_noise(np.random.default_rng(SEED_TONE), fir_samples, tones, noise_std=0.05)
    gen_fixtures.write_cf32(out, "bench_tone", x,
                            {"generator": GENERATOR, "seed": SEED_TONE,
                             "description": "two tones plus noise, benchmark size",
                             **tone_meta(tones, 0.05)})
    del x

    # Same model as the `ch0` / `ch1` fixtures: a white-noise source, delayed in channel 1,
    # with independent receiver noise on each channel.
    rng = np.random.default_rng(SEED_PAIR)
    source = complex_noise(rng, xcorr_samples, std=1.0)
    ch0 = source + complex_noise(rng, xcorr_samples, std=0.3)
    ch1 = delay(source, TRUE_DELAY_SAMPLES) + complex_noise(rng, xcorr_samples, std=0.3)
    pair = {
        "generator": GENERATOR,
        "seed": SEED_PAIR,
        "description": "ch1 is ch0's source delayed by true_delay_samples, benchmark size",
        "true_delay_samples": TRUE_DELAY_SAMPLES,
        "delay_convention": "ch1[n] = source[n - true_delay_samples], zero-filled",
        "source_std": 1.0,
        "noise_std": 0.3,
    }
    gen_fixtures.write_cf32(out, "bench_ch0", ch0,
                            {**pair, "channel": 0, "paired_with": "bench_ch1"})
    gen_fixtures.write_cf32(out, "bench_ch1", ch1,
                            {**pair, "channel": 1, "paired_with": "bench_ch0"})
    return ["bench_tone", "bench_ch0", "bench_ch1"]


def sample_count(text: str) -> int:
    value = int(text)
    if value <= TRUE_DELAY_SAMPLES:
        raise argparse.ArgumentTypeError(f"needs more than {TRUE_DELAY_SAMPLES} samples")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("out/bench"),
                        help="output directory (default: out/bench)")
    parser.add_argument("--fir-samples", type=sample_count, default=DEFAULT_FIR_SAMPLES,
                        help=f"length of bench_tone (default: {DEFAULT_FIR_SAMPLES})")
    parser.add_argument("--xcorr-samples", type=sample_count, default=DEFAULT_XCORR_SAMPLES,
                        help=f"length of each channel (default: {DEFAULT_XCORR_SAMPLES})")
    args = parser.parse_args()
    names = generate(args.out, args.fir_samples, args.xcorr_samples)
    for name in names:
        size = (args.out / f"{name}.cf32").stat().st_size
        print(f"{args.out / name}.cf32  {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
