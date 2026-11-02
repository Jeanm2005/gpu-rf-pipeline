# Roadmap

The project has three stages, each built on a finished and measured previous one:

1. **GPU pipeline (`rfgpu`).** Spectrum, FIR filtering, and cross-correlation on recorded IQ
   data, verified against a CPU reference and optimized kernel by kernel.
2. **SigFlow compiler (`sigflowc`).** A DSL for these pipelines, compiled through a graph IR to
   scheduled, memory-planned, fused GPU programs.
3. **Extensions.** Streaming, channelization, direction finding, real recordings, embedded
   targets, and fuzzing.

All three are complete. Measurements behind every optimization are in
[`PROFILING_LOG.md`](PROFILING_LOG.md).

## Stage 1: GPU pipeline — complete

| Milestone | Delivered |
|---|---|
| Foundations | Deterministic synthetic fixtures with ground truth (`tools/gen_fixtures.py`) |
| CPU reference | `tools/reference.py`: every operation in NumPy/SciPy, checked against brute-force definitions |
| Host scaffold | `rfgpu` CLI, `.cf32` / `.f32` I/O, error-checked CUDA and cuFFT calls, CUDA-event timing harness (`--bench`), CI |
| Spectrum | Batched cuFFT over fixed-size frames, last frame zero-padded |
| FIR | `naive` → `tiled` (shared-memory tile with halo) → `tiled_const` (taps in constant memory) → `register` (4 or 8 outputs per thread from an I/Q-split tile): 2.7× (63 taps) and 5.0× (1025 taps) over naive; block-size sweep with `--block` |
| Cross-correlation | `naive` → `tiled` (2.19×) → `fft` (cuFFT, about 1070× at 2^18 samples per channel) |
| Profiling | Nsight Compute reports for every variant in `docs/profiles/`, figures in `docs/figures/` |

## Stage 2: SigFlow compiler — complete

| Phase | Delivered |
|---|---|
| A: front end and script backend | Lexer, recursive-descent parser, semantic analysis (names, types, sample rates, cycles), graph IR, Kahn scheduler with source-order tie-breaking, dead-node elimination, a backend that emits a script of `rfgpu` commands |
| B: memory planning and viewer | Liveness analysis and buffer reuse (−40 % GPU memory on `two_channel.sf`, measured), TypeScript DAG viewer |
| C: fusion and CUDA backend | Horizontal fusion of FIRs that share taps, a CUDA backend that keeps every value on the GPU, batched FIR kernels, fused vs. unfused measurements |

Kernel selection (`--fir-impl`, `--xcorr-impl`) lets both backends use any kernel variant,
including batched versions of every FIR variant for fusion groups.

## Stage 3: Extensions — complete

| Extension | Delivered |
|---|---|
| Streaming | `rfgpu stream`: a ring of pinned buffers and CUDA streams overlapping upload, kernel, and download of successive chunks, bit-identical to the one-shot FIR: up to 1061 MS/s end to end, 2.5× the one-shot path, with copy, registered, and zero-copy memory modes |
| Channelizer | `rfgpu channelize` and SigFlow `channelize`: a critically sampled polyphase filter bank (branch-sum kernel and one batched cuFFT call) |
| Direction finding | `rfgpu gcc_phat` and SigFlow `gcc_phat` (GCC-PHAT, generalized cross-correlation with phase transform, on the cuFFT path) for time differences of arrival, and `rfgpu doa` for the direction of arrival at a uniform linear array, checked on a simulated four-receiver array |
| Real recordings | Every `rfgpu` input accepts a SigMF recording (`.sigmf-meta` / `.sigmf-data`) in `cf32_le`, `ci16_le`, or `ci8`, with its sample rate taken from the metadata |
| Jetson portability | Device code built for Orin (8.7) and Thor (11.0) in CI, zero-copy streaming over mapped pinned memory (`--memory zero-copy`), and [`JETSON.md`](JETSON.md) on power modes, clocks, and measurement |
| Parser fuzzing | A libFuzzer target over the whole compiler under ASan and UBSan, run in CI |

## Deferred ideas

- Streaming execution in `sigflowc`: generated programs that process captures in chunks.
  Element-wise and FIR stages chunk the way `rfgpu stream` does, but cross-correlation and
  GCC-PHAT need the whole signal (or an overlap-save formulation with a maximum lag), so the
  language first needs a notion of bounded lag.
- Measurements on a Jetson board and on public SigMF recordings. The code paths exist and are
  tested; the numbers in this repository come from one discrete GPU and synthetic data.
- Vertical fusion of `fft |> magnitude`: compute the magnitude in a cuFFT store callback, so
  the complex spectrum never reaches GPU memory. Needs cuFFT's LTO callbacks (or the legacy
  callbacks with static cuFFT and relocatable device code).
- Fractional-sample delays in the two-channel and array fixtures (they use integer delays).
- A `--max-lag` option for cross-correlation, so real-time use does not compute every lag.
- GPU tests in CI, which would need a self-hosted runner with an NVIDIA GPU.
- Unified memory (`cudaMallocManaged`) for the SigFlow runtime's buffer slots on Jetson.
