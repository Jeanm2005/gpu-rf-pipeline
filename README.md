# gpu-rf-pipeline

GPU signal processing for recorded radio data, written in CUDA C++ with a Python test harness.

The pipeline takes complex IQ (in-phase/quadrature) captures, the raw output of a
software-defined radio (SDR), and computes three core operations on an NVIDIA GPU:

| Operation | GPU implementation | Status |
|---|---|---|
| FFT (fast Fourier transform) magnitude spectrum | cuFFT (library baseline) | ✅ |
| FIR (finite impulse response) filter | Custom CUDA kernel(s) | ✅ naive, tiled, tiled with constant-memory taps |
| Cross-correlation | Custom CUDA kernel (time-domain) + cuFFT version for comparison | ✅ naive, tiled, cuFFT-based |

Every GPU result is verified against a NumPy/SciPy CPU reference, and every optimization is
timed with CUDA events, profiled with NVIDIA Nsight Compute, and written up in
[`docs/PROFILING_LOG.md`](docs/PROFILING_LOG.md).

## Why

Radio direction-finding and signal-intelligence systems are moving from dedicated hardware to
software-defined radios paired with embedded GPUs such as NVIDIA Jetson. The core workload
(spectral computation, filtering, and correlation on streaming samples) is memory-bound and
latency-sensitive, so it is a good place to learn how GPU memory hierarchy, coalescing, and
occupancy actually affect performance. The long-term goal is direction finding: estimating where
a signal comes from by correlating captures from multiple antennas.

## Quick start

```bash
# Requirements: Linux or WSL2, NVIDIA GPU + driver, CUDA Toolkit, CMake >= 3.24,
# C++17 compiler, Python >= 3.10
pip install numpy scipy pytest

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

python tools/gen_fixtures.py --out tests/fixtures
pytest -q tests/
```

If the default compiler is too new for your CUDA Toolkit's `nvcc` (for example GCC 15 with
CUDA 12.8), pass an older host compiler to CMake. The exact setup used on the development
machine is in the "Toolchain notes" section of [`docs/ROADMAP.md`](docs/ROADMAP.md).

Example:

```bash
./build/rfgpu xcorr --a tests/fixtures/ch0.cf32 --b tests/fixtures/ch1.cf32 \
                    --out out/xcorr.cf32 --impl fft
```

`./build/rfgpu --help` lists every implementation variant. To reproduce the measurements:

```bash
python tools/gen_bench.py --out out/bench          # large inputs, not committed
./build/rfgpu fir --in out/bench/bench_tone.cf32 --taps tests/fixtures/lowpass_long.f32 \
                  --out out/fir.cf32 --impl tiled_const --bench 20
mkdir -p docs/profiles
ncu --set full -o docs/profiles/fir ./build/rfgpu fir --in out/bench/bench_tone.cf32 \
    --taps tests/fixtures/lowpass_long.f32 --out out/fir.cf32 --impl tiled_const
```

## How it is tested

- `tools/gen_fixtures.py` generates synthetic captures with known ground truth: tones plus noise,
  and a 2-channel capture where one channel is a delayed copy of the other.
- `tools/reference.py` computes the same operations on the CPU with NumPy/SciPy.
- `tests/test_pipeline.py` runs the GPU tool on each fixture and checks the output against the
  reference with explicit tolerances. It covers edge cases (lengths not divisible by the block
  size, short inputs, long filters, zero input) and checks that cross-correlation recovers the
  known delay.

Data format: `.cf32` files are interleaved little-endian float32 `[I, Q, I, Q, ...]` (SigMF
`cf32_le` convention), each with a `.json` sidecar holding sample rate, seed, and ground truth.

## Results

> Filled in from real profiler output only. See `docs/PROFILING_LOG.md` for the full entries.

Kernel time is the mean of 20 runs timed with CUDA events after 3 warm-up runs, with transfers
excluded. Memory bandwidth is Nsight Compute's "Max Bandwidth": the utilization of the busiest
memory unit, as a percentage of its peak.

FIR, 2^24 samples (128 MB):

| Variant | Taps | Kernel time | Memory bandwidth (% of peak) | Speedup vs. naive |
|---|---|---|---|---|
| naive | 63 | 1.559 ms | 98.83 | 1.0× |
| shared-memory tiled | 63 | 1.130 ms | 98.29 | 1.38× |
| tiled, taps in constant memory | 63 | 0.655 ms | 94.51 | 2.38× |
| naive | 1025 | 23.74 ms | 93.81 | 1.0× |
| shared-memory tiled | 1025 | 15.10 ms | 98.81 | 1.57× |
| tiled, taps in constant memory | 1025 | 8.22 ms | 98.90 | 2.89× |

Cross-correlation, 2^18 samples per channel (524287 lags):

| Variant | Kernel time | Memory bandwidth (% of peak) | Speedup vs. naive |
|---|---|---|---|
| naive | 150.6 ms | 95.88 | 1.0× |
| shared-memory tiled | 68.7 ms | 92.42 | 2.19× |
| cuFFT-based | 0.140 ms | 57.01 (conjugate-multiply kernel) | about 1070× |

The cuFFT-based version overtakes the time-domain kernels between 2^10 and 2^12 samples per
channel. Host-to-device plus device-to-host transfers take about 25 ms for the FIR input and
about 1.1 ms for the cross-correlation input, which is more than the fastest kernel in both
cases.

GPU: NVIDIA GeForce RTX 5080 Laptop GPU · CUDA: 12.8 · Driver: 572.76 (WSL2)

## What I built, what I profiled, what I changed

> Draft written by an AI agent from the profiling log. To be rewritten in the author's own
> words.

**Built.** A command-line tool, `rfgpu`, with three operations on recorded IQ samples: a framed
FFT spectrum (cuFFT), an FIR filter, and cross-correlation. The FIR filter has three kernel
variants and cross-correlation has three, all selectable with `--impl`, so every older variant
stays available as a baseline. A pytest harness runs each variant on synthetic fixtures with
known ground truth and compares it to a float64 NumPy/SciPy reference.

**Profiled.** Kernel and transfer times come from CUDA events (20 runs after warm-up, setup
excluded). Per-kernel metrics come from Nsight Compute. Nsight Systems recorded no GPU-side
data under WSL2 on this machine, so there are no timeline captures.

**Changed.**

- The naive FIR and cross-correlation kernels were not limited by DRAM or by arithmetic. Their
  warps spent 67 to 75 % of the cycles between instructions waiting on the queue for global
  memory loads, with occupancy already near 100 %.
- Staging the samples each block needs in shared memory gave 1.4× to 1.6× for FIR and 2.2×
  for cross-correlation.
- After that, the FIR tap loop still made one global load per multiply-add. Moving the taps to
  constant memory gave a further 1.7× to 1.8×.
- For cross-correlation the larger gain was algorithmic: the FFT version does `O(N log N)`
  work instead of `O(N^2)`, and is about 1000× faster at 2^18 samples per channel. It is also
  more accurate there, because the time-domain kernels keep one long float32 running sum per
  lag.
- With the kernels this fast, the transfers dominate end-to-end time. That is what the
  streaming item on the roadmap (pinned memory, overlapping transfers with compute) addresses.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Planned after the MVP (minimum viable product):

1. **Streaming:** pinned host memory, a ring buffer, and CUDA streams that overlap data
   transfers with compute, measured against real-time sample rates.
2. **Channelizer:** a polyphase filter bank that splits a wideband capture into sub-channels.
3. **Direction finding:** GCC-PHAT (generalized cross-correlation with phase transform) for TDOA
   (time difference of arrival) between antennas, then DOA (direction of arrival) estimation for
   a simulated antenna array.
4. **Real recordings:** public SigMF datasets in addition to synthetic data.
5. **Jetson portability:** unified memory, and the power and clock limits of embedded GPUs.

## Repository layout

```
src/        C++17 / CUDA sources and the rfgpu CLI
tools/      Python fixture generator and CPU reference
tests/      pytest harness and fixtures
docs/       profiling log, roadmap, Nsight Compute reports (docs/profiles)
```

## Working with AI agents

Agent instructions live in [`AGENTS.md`](AGENTS.md). Custom kernels are written by hand by the
author; agents help with host code, tests, tooling, and reviews.
