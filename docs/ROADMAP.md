# Roadmap

The project has three stages, each built on a finished and measured previous one:

1. **GPU pipeline (`rfgpu`).** Spectrum, FIR filtering, and cross-correlation on recorded IQ
   data, verified against a CPU reference and optimized kernel by kernel.
2. **SigFlow compiler (`sigflowc`).** A DSL for these pipelines, compiled through a graph IR to
   scheduled, memory-planned, fused GPU programs.
3. **Extensions.** Streaming, channelization, direction finding, real recordings, and embedded
   targets.

Measurements behind every completed optimization are in [`PROFILING_LOG.md`](PROFILING_LOG.md).

## Stage 1: GPU pipeline — complete

| Milestone | Delivered |
|---|---|
| Foundations | Deterministic synthetic fixtures with ground truth (`tools/gen_fixtures.py`) |
| CPU reference | `tools/reference.py`: spectrum, magnitude, FIR, and cross-correlation in NumPy/SciPy, checked against brute-force definitions |
| Host scaffold | `rfgpu` CLI, `.cf32` / `.f32` I/O, error-checked CUDA and cuFFT calls, CUDA-event timing harness (`--bench`), CI |
| Spectrum | Batched cuFFT over fixed-size frames, last frame zero-padded |
| FIR | `naive` → `tiled` (shared-memory tile with halo) → `tiled_const` (taps in constant memory): 2.38× to 2.89× over naive |
| Cross-correlation | `naive` → `tiled` (2.19×) → `fft` (cuFFT, about 1070× at 2^18 samples per channel) |
| Profiling | Nsight Compute reports for every variant in `docs/profiles/`, figures in `docs/figures/` |

Open items:

- [ ] Further FIR iterations, each motivated by a profiler metric: a block-size sweep, and a
      structure-of-arrays layout for the I and Q components.

## Stage 2: SigFlow compiler — complete

| Phase | Delivered |
|---|---|
| A: front end and script backend | Lexer, recursive-descent parser, semantic analysis (names, types, sample rates, cycles), graph IR, Kahn scheduler with source-order tie-breaking, dead-node elimination, a backend that emits a script of `rfgpu` commands |
| B: memory planning and viewer | Liveness analysis and buffer reuse (−40 % GPU memory on `two_channel.sf`, measured), TypeScript DAG viewer |
| C: fusion and CUDA backend | Horizontal fusion of FIRs that share taps, a CUDA backend that keeps every value on the GPU, batched FIR kernel, fused vs. unfused measurements |

Kernel selection (`--fir-impl`, `--xcorr-impl`) lets both backends use any kernel variant,
including batched versions of every FIR variant for fusion groups.

## Stage 3: Extensions

In priority order; the order can change.

1. **Streaming.** A ring buffer of pinned host memory and CUDA streams that overlap transfers
   with compute, processing continuous chunks the way live radio data arrives, measured against
   real-time throughput (samples per second). Then streaming execution in `sigflowc`.
2. **Channelizer.** A polyphase filter bank that splits a wideband capture into sub-channels.
3. **Direction finding.** GCC-PHAT (generalized cross-correlation with phase transform) for TDOA
   (time difference of arrival) across channels, then DOA (direction of arrival) estimation for
   a simulated antenna array, with matching SigFlow operations.
4. **Real recordings.** Public SigMF datasets in addition to synthetic data.
5. **Jetson portability.** Unified memory, and the power and clock limits of embedded GPUs.
6. **Parser fuzzing.** A libFuzzer target for the SigFlow lexer and parser.

## Deferred ideas

- Vertical fusion of `fft |> magnitude`: compute the magnitude in a cuFFT store callback, so
  the complex spectrum never reaches GPU memory. Needs cuFFT's LTO callbacks (or the legacy
  callbacks with static cuFFT and relocatable device code).
- Fractional-sample delays in the two-channel fixture (it currently uses an integer delay).
- A `--max-lag` option for cross-correlation, so real-time use does not compute every lag.
- GPU tests in CI, which would need a self-hosted runner with an NVIDIA GPU.
