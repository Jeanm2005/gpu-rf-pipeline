# AGENTS.md

Instructions for AI coding agents (Codex, GitHub Copilot, Claude Code, and others) working in
this repository. Read this file before making any change.

## Project in one paragraph

`gpu-rf-pipeline` is a GPU signal-processing pipeline for recorded radio data. It reads complex
IQ (in-phase/quadrature) samples and computes an FFT (fast Fourier transform) spectrum, FIR
(finite impulse response) filtering, and cross-correlation on an NVIDIA GPU using CUDA
(Compute Unified Device Architecture). Every GPU result is checked against a NumPy/SciPy CPU
reference by a fixture-driven pytest harness, and every optimization is measured with NVIDIA
Nsight profilers and recorded in `docs/PROFILING_LOG.md`.

## The most important rule: the owner writes the kernels

This is a learning and portfolio project. The repository owner (Jean) writes all custom CUDA
kernels in `src/fir.cu` and `src/xcorr.cu` himself, because he must be able to explain every
line and every optimization in interviews.

- **Do NOT write, complete, or rewrite the body of a custom CUDA kernel** (any `__global__` or
  `__device__` function in `fir.cu` or `xcorr.cu`) unless the owner explicitly asks for it in
  the current request.
- You MAY: explain CUDA concepts, sketch pseudocode, give function signatures and skeletons with
  `// TODO` bodies, review his kernel code, point out bugs and race conditions, and suggest the
  next optimization along with the profiler metric that motivates it.
- You MAY write freely: host-side C++ (CLI, file I/O, memory management, timing harness), CMake,
  Python tools, tests, and documentation.
- Code autocomplete (Copilot) inside kernel bodies should stay minimal: no multi-line kernel
  completions.

## Build, run, test

Requirements: Linux or WSL2 (Windows Subsystem for Linux), NVIDIA GPU and driver, CUDA Toolkit
(nvcc, cuFFT), CMake 3.24 or newer, a C++17 compiler, Python 3.10 or newer with
`numpy scipy pytest`.

```bash
# Build (Release for any timing; Debug for debugging)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Generate fixtures (deterministic: fixed random seeds)
python tools/gen_fixtures.py --out tests/fixtures

# Run all tests (builds must be current first)
pytest -q tests/

# Run the CLI
./build/rfgpu spectrum --in tests/fixtures/tone.cf32 --out out/spectrum.cf32 --nfft 1024
./build/rfgpu fir      --in tests/fixtures/tone.cf32 --taps tests/fixtures/lowpass.f32 --out out/fir.cf32 --impl tiled
./build/rfgpu xcorr    --a tests/fixtures/ch0.cf32 --b tests/fixtures/ch1.cf32 --out out/xcorr.cf32 --impl naive

# Profile
nsys profile -o docs/profiles/run ./build/rfgpu fir ...    # Nsight Systems: timeline, transfers vs. kernels
ncu --set full -o docs/profiles/fir ./build/rfgpu fir ...  # Nsight Compute: per-kernel metrics
```

On WSL2, Nsight Compute may not be able to read GPU performance counters. If `ncu` fails with a
permissions or counters error, say so and use CUDA events for timing plus Nsight Systems. Never
invent profiler numbers.

## Data format

- Samples: `.cf32` = interleaved little-endian float32 complex values `[I0, Q0, I1, Q1, ...]`
  (SigMF `cf32_le` convention).
- Each `.cf32` has a sidecar `.json` with at least: `sample_rate`, `num_samples`, `generator`,
  `seed`, and ground truth when known (for example `true_delay_samples` for 2-channel captures).
- Filter taps: `.f32` = raw little-endian float32 array.

## Code conventions

- C++17 and CUDA; `.cu` for files with device code, `.cpp`/`.hpp` for host-only code.
- Every CUDA API and cuFFT call is wrapped in the error-check macros (`CUDA_CHECK`,
  `CUFFT_CHECK`). No unchecked calls.
- Each custom operation keeps every implementation variant selectable from the CLI
  (`--impl naive|tiled|...`). Never delete an older variant; it is the baseline for the
  profiling log.
- Kernel launch configuration lives next to the launch site with a comment explaining the
  choice.
- Python: type hints, NumPy/SciPy only for references, no GPU libraries in the reference path.

## Testing rules

- Correctness before speed: no optimization work while any test fails.
- Every GPU operation is compared to `tools/reference.py` with explicit tolerances (`rtol`,
  `atol`) stated in the test, not hidden in helpers.
- Required edge cases: input length not a multiple of the block size, very short inputs, a
  filter longer than one tile, all-zero input, and recovery of the known delay in the 2-channel
  fixture.
- Fixtures are deterministic (fixed seeds) and small enough to commit (keep each under ~5 MB).

## Measurement rules

- Never claim a speedup without numbers from Nsight or CUDA events.
- Warm up before timing, average at least 10 runs, exclude one-time setup (context creation,
  cuFFT plan creation).
- Report host-to-device and device-to-host transfer time separately from kernel time.
- Every optimization iteration gets an entry in `docs/PROFILING_LOG.md` (template below),
  including negative results.

```markdown
## Iteration N — <kernel> — <short name of change>
- GPU / driver / CUDA version:
- Input size and parameters:
- Kernel time (mean ± std):          Transfer time:
- Achieved memory bandwidth (% of peak):     Occupancy:
- Top bottleneck reported by the profiler:
- What I changed and why:
- Result (speedup vs. previous iteration and vs. naive):
- Screenshot: docs/screenshots/...
```

## Scope

The current milestone is the MVP (minimum viable product) described in README.md. Anything
beyond it (streaming with CUDA streams, polyphase channelizer, GCC-PHAT direction finding, real
SigMF recordings, Jetson porting) goes into `docs/ROADMAP.md`, not into code.

## Git

- Small commits, one working step each, imperative messages ("Add tiled FIR kernel").
- Don't commit build outputs, profiler reports larger than ~10 MB, or files under `out/`.
- Don't push or force-push unless asked.

## Subprojects

- `compiler/` — SigFlow compiler (`sigflowc`). Read `compiler/AGENTS.md` before changing
  anything there; it adds stricter owner-written rules for the core compiler passes.
