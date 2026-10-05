# gpu-rf-pipeline

GPU signal processing for recorded radio data, written in CUDA C++ with a Python test harness.

The pipeline takes complex IQ (in-phase/quadrature) captures, the raw output of a
software-defined radio (SDR), and computes three core operations on an NVIDIA GPU:

| Operation | GPU implementation | Status |
|---|---|---|
| FFT (fast Fourier transform) magnitude spectrum | cuFFT (library baseline) | ⬜ |
| FIR (finite impulse response) filter | Custom CUDA kernel(s) | ⬜ |
| Cross-correlation | Custom CUDA kernel (time-domain) + cuFFT version for comparison | ⬜ |

Every GPU result is verified against a NumPy/SciPy CPU reference, and every optimization is
profiled with NVIDIA Nsight Systems and Nsight Compute and written up in
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

Example:

```bash
./build/rfgpu xcorr --a tests/fixtures/ch0.cf32 --b tests/fixtures/ch1.cf32 \
                    --out out/xcorr.cf32 --impl tiled
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

> Filled in from real profiler output only. See `docs/PROFILING_LOG.md` for details and
> screenshots.

| Kernel | Variant | Kernel time | Memory bandwidth (% of peak) | Speedup vs. naive |
|---|---|---|---|---|
| FIR | naive | — | — | 1.0× |
| FIR | shared-memory tiled | — | — | — |
| Cross-correlation | naive | — | — | 1.0× |
| Cross-correlation | tiled | — | — | — |
| Cross-correlation | cuFFT-based | — | — | — |

GPU: — · CUDA: — · Driver: —

## What I built, what I profiled, what I changed

> Written after the profiling iterations are complete.

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
docs/       profiling log, roadmap, screenshots
```

## Working with AI agents

Agent instructions live in [`AGENTS.md`](AGENTS.md). Custom kernels are written by hand by the
author; agents help with host code, tests, tooling, and reviews.

Folder structure (the files in the tree that these docs mention don't exist yet; you'll create them as you build):

gpu-rf-pipeline/
├── AGENTS.md
├── README.md
├── CMakeLists.txt
├── .github/
│   └── copilot-instructions.md      # copy of AGENTS.md
├── src/                             # C++17 / CUDA
│   ├── main.cpp                     # rfgpu CLI entry point
│   ├── io.hpp / io.cpp              # .cf32 + metadata read/write
│   ├── fir.cu                       # custom FIR kernel(s)
│   ├── xcorr.cu                     # custom cross-correlation kernel(s)
│   └── spectrum.cu                  # cuFFT spectrum
├── tools/                           # Python
│   ├── gen_fixtures.py              # synthetic IQ generator with ground truth
│   └── reference.py                 # NumPy/SciPy CPU reference
├── tests/
│   ├── fixtures/                    # generated .cf32 + .json
│   └── test_pipeline.py             # pytest harness
└── docs/
    ├── PROFILING_LOG.md
    ├── ROADMAP.md
    └── screenshots/