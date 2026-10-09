# Roadmap

Every task from the current state of the repository to the end of the project, in the order it
should be done. Three stages:

1. **MVP** (minimum viable product): the `rfgpu` pipeline described in `README.md`.
2. **SigFlow compiler** (`compiler/`): phases A, B, C from `compiler/README.md`.
3. **Extensions:** streaming, channelizer, direction finding, real recordings, Jetson.

Nothing in a later stage gets code until the stage before it is finished and measured.

Who writes each task, following `AGENTS.md` and `compiler/AGENTS.md`:

- **[J]** Jean writes it by hand (custom kernels, core compiler passes, profiling write-ups).
  Agents explain, sketch skeletons, review, and propose tests.
- **[A]** An agent may write it (host code, CMake, Python, tests, docs).
- **[J+A]** Jean decides or runs it on his machine; an agent prepares or assists.

## Stage 1: MVP

### M0. Foundations

- [x] **[A]** Project docs, repository scaffolding, `.gitignore`.
- [x] **[A]** `tools/gen_fixtures.py`: deterministic fixtures with ground truth, plus
      `tests/test_fixtures.py`.

### M1. CPU reference

The reference defines what every GPU operation must compute, so it comes before any GPU code.

- [x] **[A]** `tools/reference.py`: `spectrum`, `magnitude`, `fir`, `xcorr` in NumPy/SciPy, with
      `.cf32` / `.f32` read and write helpers and a CLI that mirrors `rfgpu`.
- [x] **[A]** `tests/test_reference.py`: the reference checked against brute-force definitions
      and the fixture ground truth (tone bins, stopband rejection, known delay).
- [x] **[J]** Read the three output conventions in the `tools/reference.py` docstring and
      confirm or change them. The kernels have to match them exactly. Confirmed unchanged by
      Jean on 2026-10-08.

### M2. Toolchain and host scaffold

- [x] **[J+A]** Install the CUDA Toolkit in WSL2: CUDA 12.8.1 in `~/cuda-12.8` (`nvcc`, cuFFT,
      `nsys`, `ncu`), built with `g++-14`. See "Toolchain notes" below.
- [x] **[A]** `CMakeLists.txt`: C++17 and CUDA, `rfgpu` target, cuFFT linked, Release and Debug.
- [x] **[A]** `src/check.hpp`: `CUDA_CHECK` and `CUFFT_CHECK` macros (file, line, error string).
- [x] **[A]** `src/io.hpp` / `src/io.cpp`: read and write `.cf32` and `.f32`, with clear errors
      for a missing file or an odd float count.
- [x] **[A]** `src/main.cpp`: `rfgpu` CLI with `spectrum`, `fir`, `xcorr` subcommands and
      `--impl` dispatch through a table, so adding a variant is one line. Also `rfgpu info`
      (GPU, CUDA versions, implementation list as JSON).
- [x] **[A]** Timing harness: CUDA events, warm-up, at least 10 runs, mean and standard
      deviation, host-to-device and device-to-host time reported separately from kernel time
      (`--bench N`, machine-readable output).
- [x] **[A]** `.github/copilot-instructions.md` as a copy of `AGENTS.md`.

#### Toolchain notes (measured on this machine, 2026-10-07)

The machine is Ubuntu 26.04 (GCC 15, glibc 2.43) with Windows driver 572.76, and that
combination has no CUDA Toolkit that works unmodified:

- Driver 572.76 supports CUDA up to 12.8. A CUDA 13 runtime fails at start-up with "CUDA driver
  version is insufficient for CUDA runtime version", and kernels compiled by a CUDA 13 `nvcc`
  are rejected ("the provided PTX was compiled with an unsupported toolchain").
- CUDA 12.8's `nvcc` cannot parse GCC 15's standard library headers, and its
  `crt/math_functions.h` conflicts with glibc 2.43 (`rsqrt`, `rsqrtf`, `sinpi`, `sinpif`,
  `cospi`, `cospif` declared without `noexcept`).

What is installed (option chosen: keep the driver):

- `g++-14` from apt, and CUDA Toolkit 12.8.1 from the runfile, installed without root into
  `~/cuda-12.8`.
- The runfile's installer needs `libxml2.so.2`, which Ubuntu 26.04 no longer ships, so with
  `--silent` it exits without installing anything. It was run with `libxml2.so.2` and
  `libicu74` taken from the Ubuntu 24.04 packages on `LD_LIBRARY_PATH`.
- `~/cuda-12.8/include/crt/math_functions.h` is patched: `noexcept(true)` added to those six
  declarations (original kept as `math_functions.h.bak`).
- Configure with:
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14`
  with `~/cuda-12.8/bin` on `PATH`.

The alternative is to update the Windows NVIDIA driver to one that supports CUDA 13 and
install CUDA Toolkit 13.x, whose `nvcc` accepts GCC 15 without these workarounds.

### M3. Spectrum (cuFFT)

- [x] **[A]** `src/spectrum.cu`: batched cuFFT C2C plan over `nfft`-sized frames (Jean chose
      to have the agent write it; it has no custom kernel).
- [x] **[A]** `tests/test_pipeline.py`: harness that runs `rfgpu` on a fixture and compares to
      `tools/reference.py` with `rtol` / `atol` stated in each test. Skips cleanly when the
      binary or the GPU is missing.
- [x] **[A]** Spectrum tests: `tone`, `tone_odd` (partial last frame), `short` (shorter than one
      frame), `zeros`.

### M4. FIR, naive

- [x] **[A]** `src/fir.cu` host side: buffer allocation, transfers, launch site with a
      launch-configuration comment, and a kernel signature with a `// TODO` body.
- [x] **[J]** Naive FIR kernel: one thread per output sample, taps read from global memory.
- [ ] **[J]** Read `fir_naive_kernel` until every line can be explained without notes.
- [x] **[A]** FIR tests: `identity`, `lowpass`, `lowpass_long` (filter longer than one tile),
      `tone_odd`, `short` (input shorter than the filter), `zeros`.

### M5. Cross-correlation, naive

- [x] **[A]** `src/xcorr.cu` host side and kernel skeleton.
- [x] **[J]** Naive time-domain kernel: one thread per lag.
- [ ] **[J]** Read `xcorr_naive_kernel` until every line can be explained without notes.
- [x] **[A]** Cross-correlation tests: full output against the reference, recovery of
      `true_delay_samples` from `ch0` / `ch1`, unequal input lengths, `short`, `zeros`.

All tests must pass before M6 starts (correctness before speed).

### M6. Profiling baseline

- [x] **[A]** `tools/gen_bench.py`: large benchmark inputs written to `out/bench/` (not
      committed), with `tests/test_gen_bench.py`. The fixtures are too small to show memory
      behaviour. Defaults: `bench_tone` with 2^24 samples (128 MB) for FIR, and `bench_ch0` /
      `bench_ch1` with 2^18 samples each for cross-correlation.
- [x] **[J+A]** Get `ncu` reading performance counters under WSL2. First try (2026-10-07):
      `ncu` starts but fails with `ERR_NVGPUCTRPERM` (no permission to access GPU performance
      counters). The setting is on the Windows side: NVIDIA Control Panel, Desktop menu, Enable
      Developer Settings, then Developer, Manage GPU Performance Counters, allow access for
      all users. If it still fails after that, record it and use CUDA events plus Nsight
      Systems. Second try (2026-10-08): same `ERR_NVGPUCTRPERM`. `nsys profile` runs and
      records the host-side CUDA API calls (`cuda_api_sum`), but its report has no GPU-side
      data: `cuda_gpu_kern_sum` and `cuda_gpu_mem_time_sum` are skipped with "does not contain
      CUDA kernel data". Later on 2026-10-08, after the Windows setting was changed, `ncu
      --set full` works (the output directory must exist first: `mkdir -p docs/profiles`).
      `nsys` still records no GPU-side data, so kernel and transfer times come from CUDA
      events (`--bench`) and everything else from `ncu`.
- [x] **[J]** `docs/PROFILING_LOG.md`: iteration 0 for naive FIR and naive cross-correlation
      (kernel time, transfer time, memory bandwidth, occupancy, top bottleneck). Jean handed
      the logging of iterations to the agent on 2026-10-08; the analysis in each entry is
      still his to check.

### M7. FIR optimization

- [x] **[A]** `--impl tiled` host side in `src/fir.cu`: wrapper, launch site with dynamic
      shared memory for the tile, a clear error when the halo does not fit in one block's
      shared memory, and `fir_tiled_kernel` with a `// TODO` body. The FIR tests pick the
      variant up from `rfgpu info` and fail for it until the kernel is written.
- [x] **[J]** `--impl tiled`: shared-memory tiles with a halo of `num_taps - 1` samples. Must
      handle a filter longer than one tile. Written by an agent at Jean's explicit request
      (2026-10-08).
- [x] **[J]** Profile it and write the log entry. Negative results are logged too. Done by an
      agent (iteration 1).
- [ ] **[J]** Read `fir_tiled_kernel` until every line can be explained without notes.
- [x] **[A]** `--impl tiled_const` host side: taps copied to a `__constant__` array with
      `cudaMemcpyToSymbol` (at most 8192 taps), same launch configuration as `tiled`, and
      `fir_tiled_const_kernel` with a `// TODO` body. Motivated by iteration 1: the tap loop
      still does one global load per multiply-add.
- [x] **[J]** `fir_tiled_const_kernel` body, then profile it and log iteration 2. Written,
      profiled, and logged by an agent at Jean's explicit request (2026-10-08).
- [ ] **[J]** Read `fir_tiled_const_kernel` until every line can be explained without notes.
- [ ] **[J+A]** Further iterations, each motivated by a profiler metric that an agent may point
      out (candidates: taps in constant memory, block-size sweep, structure-of-arrays layout for
      I and Q).

### M8. Cross-correlation optimization

The three kernels and log entries below were written by an agent at Jean's explicit request
(2026-10-08).

- [x] **[J]** `--impl tiled`, profiled, with a log entry (cross-correlation iteration 1).
- [x] **[J+A]** `--impl fft`: cuFFT-based correlation (forward FFTs, conjugate multiply, inverse
      FFT) for comparison. `FftPlan` moved to `src/fft_plan.hpp` so `spectrum.cu` and
      `xcorr.cu` share it.
- [x] **[J]** Log entry comparing naive, tiled, and FFT-based, with the input size where the
      FFT version starts to win (cross-correlation iteration 2: between 2^10 and 2^12 samples
      per channel).
- [ ] **[J]** Read `xcorr_tiled_kernel` and `xcorr_conj_mul_kernel` until every line can be
      explained without notes.

### M9. MVP write-up

- [x] **[J+A]** `README.md` results table filled in from the profiling log (real numbers only),
      GPU / CUDA / driver line, status column updated.
- [ ] **[J]** "What I built, what I profiled, what I changed" section. An agent wrote a draft
      from the profiling log (2026-10-08), marked as a draft in the README; Jean rewrites it
      in his own words.
- [ ] **[J]** Screenshots in `docs/screenshots/`. None yet: the profiles were read from the
      `ncu` command line. Open the reports in `docs/profiles/` with the Nsight Compute GUI
      (`ncu-ui`) to capture them.
- [x] **[A]** Clean-clone check: the Quick start commands build and pass on a fresh checkout
      (2026-10-08: fresh clone, 116 tests pass, regenerated fixtures are byte-identical to the
      committed ones). On this machine the configure step needs the `g++-14` flags from the
      toolchain notes; plain `cmake -S . -B build` fails because CUDA 12.8 cannot use GCC 15.

## Stage 2: SigFlow compiler (`compiler/`)

Phase A only needs a working `rfgpu`, so it can start after M5 if Jean wants to alternate
between kernel work and compiler work. Phase C needs the MVP profiling numbers as its baseline.

### C-A. Phase A: front end, IR, scheduling, script backend

- [ ] **[A]** `compiler/CMakeLists.txt`, GoogleTest, the golden-test runner, and
      `--update-golden`.
- [ ] **[J]** Design the token, AST, and IR data structures. **[A]** fills in header boilerplate
      once they are designed.
- [ ] **[A]** `src/main.cpp` (`--emit=tokens|ast|ir|schedule|json`, `-o`) and the diagnostics
      printer (file, line, column, source line with a caret).
- [ ] **[J]** `src/lexer.cpp`. **[A]** lexer unit tests and goldens (unterminated string, stray
      character, empty program).
- [ ] **[J]** `src/parser.cpp`, recursive descent. Grammar changes are recorded in
      `compiler/README.md`. **[A]** one positive and one negative golden per grammar rule.
- [ ] **[J]** `src/sema.cpp`: name resolution, argument, type, and sample-rate checks.
      **[A]** negative goldens (unknown name, wrong argument count, unknown keyword argument,
      type mismatch, rate mismatch).
- [ ] **[J+A]** IR builder (AST to DAG) and the text and JSON dumps.
- [ ] **[J]** `src/schedule.cpp`: Kahn's algorithm, cycle detection, source-order tie-breaking.
      **[A]** unit tests on hand-built graphs.
- [ ] **[J+A]** Dead-node elimination with `--no-dce`.
- [ ] **[A]** `rfgpu` gains a magnitude output (`frames<f32, N>`), which `two_channel.sf` needs
      and the MVP CLI does not have.
- [ ] **[A]** `src/backend_script.cpp`: emits a script of `rfgpu` commands.
- [ ] **[A]** `examples/*.sf` and `tests/e2e`: compile, run on the fixtures, compare to
      `tools/reference.py`.

### C-B. Phase B: liveness and viewer

- [ ] **[J]** `src/liveness.cpp`: last use of each buffer, and buffer reuse. **[A]** unit tests.
- [ ] **[J+A]** Measure peak GPU memory with and without reuse.
- [ ] **[A]** `viewer/` scaffolding (TypeScript build, JSON loading).
- [ ] **[J]** Graph layout and rendering of the IR before and after each pass.

### C-C. Phase C: fusion and CUDA backend

- [ ] **[J]** `src/fuse.cpp` with `--no-fuse`. **[A]** unit tests on hand-built graphs.
- [ ] **[J+A]** CUDA backend: generated C++ that calls the existing kernels. The fused kernel
      templates are Jean's.
- [ ] **[J]** Measurements for `two_channel.sf`: kernel launches, peak GPU memory, end-to-end
      time, fused against `--no-fuse`, logged in `docs/PROFILING_LOG.md`.
- [ ] **[J+A]** `compiler/README.md` results table and stage status column.

## Stage 3: Extensions

Out of scope until stages 1 and 2 are finished and measured. Order can change.

1. **Streaming.** A ring buffer of pinned host memory and CUDA streams that overlap transfers
   with compute, processing continuous chunks the way live radio data arrives. Measured against
   real-time throughput (samples per second). Then streaming execution in `sigflowc`.
2. **Channelizer.** A polyphase filter bank that splits a wideband capture into sub-channels.
3. **Direction finding.** GCC-PHAT (generalized cross-correlation with phase transform) for TDOA
   (time difference of arrival) across channels, then DOA (direction of arrival) estimation for
   a simulated antenna array. Then the matching SigFlow operations.
4. **Real recordings.** Public SigMF datasets in addition to synthetic data.
5. **Jetson portability.** Notes on running on Jetson: unified memory, power and clock limits.
6. **Parser fuzzer.** libFuzzer target for the SigFlow lexer and parser.

## Deferred ideas

- Fractional-sample delays in the two-channel fixture (the MVP fixture uses an integer delay).
- A `--max-lag` option for cross-correlation, so a real-time user does not compute every lag.
