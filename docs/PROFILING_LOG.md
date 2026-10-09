# Profiling log

One entry per optimization iteration, including negative results. Numbers come from Nsight
Systems, Nsight Compute, or CUDA events only. Timing rules are in `AGENTS.md` ("Measurement
rules"): warm up, average at least 10 runs, exclude one-time setup, and report transfer time
separately from kernel time.

Iteration 0 was measured and written up by an agent at Jean's request (2026-10-08). Kernel and
transfer times are CUDA events from `rfgpu --bench 20` (3 warm-up runs). Everything else is
from Nsight Compute 2025.1.1 (`ncu --set full`), which profiles a single launch with its own
overhead, so its "Duration" is longer than the CUDA-event time and is listed separately.
Nsight Systems records no GPU-side data on this machine (see M6 in [`ROADMAP.md`](ROADMAP.md)),
so there are no timeline screenshots. Inputs come from `python tools/gen_bench.py --out out/bench`.

## Iteration 0 — FIR — naive baseline
- GPU / driver / CUDA version: NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0,
  60 SMs) / 572.76 / 12.8
- Input size and parameters: `bench_tone.cf32`, 2^24 samples (128 MB), `--impl naive`, block
  size 256, grid 65536. Two filters: `lowpass.f32` (63 taps) and `lowpass_long.f32` (1025 taps).

| | 63 taps | 1025 taps |
|---|---|---|
| Kernel time (CUDA events, mean ± std) | 1.553 ± 0.009 ms (second round: 1.429 ± 0.004) | 24.43 ± 0.97 ms (second round: 24.41 ± 1.02) |
| Host-to-device transfer | 11.05 ± 0.70 ms | 11.14 ± 0.77 ms |
| Device-to-host transfer | 13.00 ± 1.64 ms | 14.28 ± 1.73 ms |
| `ncu` Duration (SM clock during profiling) | 3.50 ms (970 MHz) | 61.46 ms (935 MHz) |
| Max Bandwidth (% of peak; the busiest memory unit) | 98.83 | 93.81 |
| L1/TEX Cache Throughput | 99.05 % | 93.86 % |
| L2 Cache Throughput | 8.41 % | 1.74 % |
| DRAM Throughput | 15.77 % (71.47 GB/s) | 0.91 % (4.13 GB/s) |
| L1/TEX hit rate / L2 hit rate | 98.48 % / 13.56 % | 99.75 % / 68.06 % |
| Occupancy, theoretical / achieved | 100 % / 89.76 % | 100 % / 97.30 % |
| Registers per thread | 40 | 40 |
| Issue Slots Busy / SM Busy | 27.79 % / 35.08 % | 23.18 % / 31.39 % |
| Eligible warps per scheduler (of active) | 1.74 of 10.76 | 1.73 of 11.68 |
| Warp cycles per issued instruction | 38.71 | 50.38 |

- Top bottleneck reported by the profiler: the L1/TEX cache path, not DRAM and not arithmetic.
  L1/TEX throughput is 94 to 99 % of peak while DRAM is under 16 % and the kernel reaches 8 %
  of the device's fp32 peak (63 taps). The dominant warp stall is "waiting for the L1
  instruction queue for local and global memory operations to be not full": 27.6 of 38.7
  cycles between issued instructions (71.2 %) with 63 taps, 37.7 of 50.4 (74.9 %) with 1025.
  `ncu` also flags uncoalesced global accesses: 51 % of sectors are "excessive", and on
  average only 14.1 of the 32 bytes per sector are used by global loads and 16.0 of 32 by
  global stores.
- What I changed and why: nothing; this is the baseline. Every thread reads each tap and each
  sample from global memory, so the kernel issues `2 * num_taps` loads per output sample
  through L1 and neighbouring threads re-read almost the same samples.
- Result (speedup vs. previous iteration and vs. naive): 1.0× (baseline). With 63 taps the two
  transfers (about 24 ms) take about 15 times longer than the kernel (about 1.5 ms); with 1025
  taps they are about equal to it.
- Screenshot: none. Reports: `docs/profiles/fir_naive_63.ncu-rep`,
  `docs/profiles/fir_naive_1025.ncu-rep`.

## Iteration 0 — cross-correlation — naive baseline
- GPU / driver / CUDA version: NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0,
  60 SMs) / 572.76 / 12.8
- Input size and parameters: `bench_ch0.cf32` and `bench_ch1.cf32`, 2^18 samples each (2 MB
  each), 524287 lags, `--impl naive`, block size 256, grid 2048.
- Kernel time (mean ± std): 150.5 ± 3.1 ms (second round: 151.3 ± 3.2 ms).
  Transfer time: host-to-device 0.71 ± 0.07 ms, device-to-host 1.17 ± 0.14 ms.
  `ncu` Duration: 317.41 ms at an SM clock of 940 MHz.
- Achieved memory bandwidth (% of peak): Max Bandwidth 95.88 %, set by L1/TEX Cache Throughput
  (96.85 %). L2 Cache Throughput 5.11 %, DRAM Throughput 0.00 % (14.62 MB/s). L1/TEX hit rate
  99.18 %, L2 hit rate reported as 100.39 %.
  Occupancy: theoretical 100 %, achieved 98.19 %, 40 registers per thread.
- Top bottleneck reported by the profiler: the same L1/TEX path as FIR. Both inputs (4 MB
  together) stay in cache, so DRAM is idle, but L1/TEX is at 97 % of peak. Issue Slots Busy is
  33.29 %, with 2.28 of 11.78 active warps eligible per scheduler. The dominant stall is
  again the L1 instruction queue for local and global memory operations: 23.5 of 35.4 cycles
  between issued instructions (66.6 %). `ncu` flags 49 % of sectors as excessive
  (uncoalesced global accesses).
- What I changed and why: nothing; this is the baseline. Every lag re-reads both inputs from
  global memory, about `num_a * num_b` sample pairs in total.
- Result (speedup vs. previous iteration and vs. naive): 1.0× (baseline). Transfers are about
  1 % of the kernel time.
- Screenshot: none. Report: `docs/profiles/xcorr_naive.ncu-rep`.

## Iteration 1 — FIR — shared-memory tiles (`--impl tiled`)

Kernel, measurements, and this entry were written by an agent at Jean's request (2026-10-08).

- GPU / driver / CUDA version: NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0,
  60 SMs) / 572.76 / 12.8
- Input size and parameters: `bench_tone.cf32`, 2^24 samples (128 MB), `--impl tiled`, block
  size 256, grid 65536, `lowpass.f32` (63 taps) and `lowpass_long.f32` (1025 taps). Naive was
  re-timed in the same session. `--bench 20`, 3 warm-up runs.

| | 63 taps | 1025 taps |
|---|---|---|
| Kernel time, naive, same session (mean ± std) | 1.565 ± 0.044 ms | 24.71 ± 0.71 ms |
| Kernel time, tiled (mean ± std) | 1.099 ± 0.023 ms | 15.50 ± 0.97 ms |
| Host-to-device transfer (tiled) | 12.75 ± 0.90 ms | 13.13 ± 1.76 ms |
| Device-to-host transfer (tiled) | 13.99 ± 1.56 ms | 15.16 ± 2.67 ms |
| `ncu` Duration (SM clock during profiling) | 2.58 ms (914 MHz) | 39.04 ms (939 MHz) |
| Max Bandwidth (% of peak; the busiest memory unit) | 98.29 | 98.81 |
| L1/TEX Cache Throughput | 98.55 % | 98.89 % |
| L2 Cache Throughput | 11.19 % | 1.78 % |
| DRAM Throughput | 21.30 % (96.52 GB/s) | 1.42 % (6.45 GB/s) |
| L1/TEX hit rate / L2 hit rate | 82.16 % / 18.53 % | 95.18 % / 71.00 % |
| Dynamic shared memory per block | 2.54 KB | 10.24 KB |
| Occupancy, theoretical / achieved | 100 % / 95.63 % | 100 % / 99.42 % |
| Registers per thread | 39 | 39 |
| Issue Slots Busy / SM Busy | 36.69 % / 36.69 % | 29.94 % / 33.23 % |
| Eligible warps per scheduler (of active) | 1.49 of 11.46 | 1.52 of 11.93 |
| Warp cycles per issued instruction | 31.23 | 39.85 |
| Excessive (uncoalesced) sectors | 20 % | 4 % |

- Top bottleneck reported by the profiler: still the L1/TEX unit at 98 to 99 % of peak, which
  in `ncu` also carries shared-memory traffic. The dominant stall changed from the L1
  local/global queue to "waiting for the MIO (memory input/output) instruction queue": 13.5 of
  31.2 cycles between issued instructions (43.3 %) with 63 taps, 20.6 of 39.8 (51.6 %) with
  1025. The tap loop still does one global load (the tap) and one shared load (the sample)
  per multiply-add.
- What I changed and why: iteration 0 showed warps stalled on global loads, with every thread
  re-reading samples its neighbours also read. Each block now copies the `num_taps - 1 + 256`
  samples it needs into shared memory once (a cooperative, coalesced load that also covers a
  halo longer than the block), and the tap loop reads samples from the tile. Samples outside
  the input are stored as zeros, so the loop has no bounds test. Taps are unchanged (global
  memory). Shared memory does not limit occupancy at either filter length (block limit from
  shared memory 18 and 9, from registers and warps 6).
- Result (speedup vs. previous iteration and vs. naive): 1.42× with 63 taps and 1.59× with
  1025 taps (naive is the previous iteration). Output is bit-identical to naive on the
  benchmark input for both filters. Transfers are unchanged and still about 25 times the
  kernel time with 63 taps.
- Screenshot: none. Reports: `docs/profiles/fir_tiled_63.ncu-rep`,
  `docs/profiles/fir_tiled_1025.ncu-rep`.

## Iteration 2 — FIR — taps in constant memory (`--impl tiled_const`)

Kernel, measurements, and this entry were written by an agent at Jean's request (2026-10-08).

- GPU / driver / CUDA version: NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0,
  60 SMs) / 572.76 / 12.8
- Input size and parameters: `bench_tone.cf32`, 2^24 samples (128 MB), `--impl tiled_const`,
  block size 256, grid 65536, `lowpass.f32` (63 taps) and `lowpass_long.f32` (1025 taps).
  Naive and tiled were re-timed in the same session. `--bench 20`, 3 warm-up runs, two rounds
  (the second round is in parentheses).

| | 63 taps | 1025 taps |
|---|---|---|
| Kernel time, naive (mean ± std) | 1.559 ± 0.023 ms (1.445 ± 0.034) | 23.74 ± 1.08 ms (23.74 ± 1.01) |
| Kernel time, tiled | 1.130 ± 0.041 ms (1.041 ± 0.031) | 15.10 ± 0.55 ms (15.20 ± 0.69) |
| Kernel time, tiled_const | 0.655 ± 0.008 ms (0.660 ± 0.017) | 8.22 ± 0.20 ms (8.32 ± 0.27) |
| Host-to-device transfer (tiled_const) | 12.44 ± 1.08 ms | 12.94 ± 1.83 ms |
| Device-to-host transfer (tiled_const) | 12.99 ± 0.71 ms | 15.16 ± 1.81 ms |
| `ncu` Duration (SM clock during profiling) | 1.45 ms (893 MHz) | 19.72 ms (934 MHz) |
| Max Bandwidth (% of peak; the busiest memory unit) | 94.51 | 98.90 |
| L1/TEX Cache Throughput | 95.93 % | 99.12 % |
| L2 Cache Throughput | 18.24 % | 3.06 % |
| DRAM Throughput | 38.11 % (172.69 GB/s) | 2.81 % (12.71 GB/s) |
| L1/TEX hit rate / L2 hit rate | 52.89 % / 13.81 % | 49.65 % / 67.23 % |
| Dynamic shared memory per block | 2.54 KB | 10.24 KB |
| Occupancy, theoretical / achieved | 100 % / 94.64 % | 100 % / 99.08 % |
| Registers per thread | 27 | 27 |
| Issue Slots Busy / SM Busy | 62.52 % / 62.52 % | 54.76 % / 54.76 % |
| Eligible warps per scheduler (of active) | 2.24 of 11.26 | 1.98 of 11.89 |
| Warp cycles per issued instruction | 18.02 | 21.71 |
| Excessive (uncoalesced) sectors | 53 % (10.6 million) | 50 % (25.2 million) |

- Top bottleneck reported by the profiler: L1/TEX is still the busiest unit (96 to 99 % of
  peak) and the MIO queue is still the largest stall, but smaller: 6.1 of 18.0 cycles between
  issued instructions (33.6 %) with 63 taps, 9.7 of 21.7 (44.7 %) with 1025. Issue Slots Busy
  rose from 30 to 37 % to 55 to 63 %, so the schedulers issue instructions about twice as
  often. What remains on the global path is the tile load and the output store, which `ncu`
  flags as using only about half of each 32-byte sector. The count of excessive sectors is
  the same as in iteration 1 (10.6 and 25.2 million); the percentage is higher only because
  the tap loads no longer count as global sectors.
- What I changed and why: iteration 1 left one global load (the tap) per multiply-add, and the
  MIO queue was the top stall. The taps now live in a `__constant__` array filled with
  `cudaMemcpyToSymbol`; every thread of a warp reads the same tap in the same loop iteration,
  which the constant cache serves without a global-memory request. The kernel is otherwise
  identical to `tiled`. The host limits this variant to 8192 taps.
- Result (speedup vs. previous iteration and vs. naive): vs. tiled 1.73× (63 taps) and 1.84×
  (1025 taps); vs. naive 2.38× and 2.89× (first round). Output is bit-identical to `tiled` on
  the benchmark input for both filters. With 63 taps the kernel (0.66 ms) is now about 2.6 %
  of the transfer time (about 25 ms), so further kernel work on this filter length changes
  end-to-end time very little.
- Screenshot: none. Reports: `docs/profiles/fir_tiled_const_63.ncu-rep`,
  `docs/profiles/fir_tiled_const_1025.ncu-rep`.

## Iteration 1 — cross-correlation — shared-memory tiles (`--impl tiled`)

Kernel, measurements, and this entry were written by an agent at Jean's request (2026-10-08).

- GPU / driver / CUDA version: NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0,
  60 SMs) / 572.76 / 12.8
- Input size and parameters: `bench_ch0.cf32` and `bench_ch1.cf32`, 2^18 samples each, 524287
  lags, `--impl tiled`, block size 256 (also the chunk length), grid 2048, 6.14 KB of dynamic
  shared memory per block. `--bench 20`, 3 warm-up runs; naive re-timed in the same session.
- Kernel time (mean ± std): 68.7 ± 2.4 ms (naive, same session: 150.6 ± 4.5 ms).
  Transfer time: host-to-device 0.74 ± 0.10 ms, device-to-host 1.23 ± 0.30 ms.
  `ncu` Duration: 127.57 ms at an SM clock of 933 MHz (naive: 317.41 ms at 940 MHz).
- Achieved memory bandwidth (% of peak): Max Bandwidth 92.42 %, L1/TEX Cache Throughput
  93.58 %, L2 Cache Throughput 2.91 %, DRAM Throughput 0.01 %. L1/TEX hit rate 63.78 %, L2 hit
  rate 99.77 %.
  Occupancy: theoretical 100 %, achieved 98.93 %, 40 registers per thread.
- Top bottleneck reported by the profiler: the MIO (memory input/output) instruction queue,
  8.4 of 18.2 cycles between issued instructions (46.2 %). In iteration 0 it was the L1 queue
  for global loads, 23.5 of 35.4 cycles (66.6 %). Issue Slots Busy rose from 33.29 % to
  65.09 % and eligible warps per scheduler from 2.28 to 3.50. `ncu` flags 54 % of the
  remaining global sectors as excessive (the tile loads and the output store).
- What I changed and why: iteration 0 showed every lag re-reading both inputs from global
  memory, with warps stalled on those loads. A block of 256 lags now walks input `a` in chunks
  of 256 samples; per chunk it stages those 256 samples of `a` and the 511 samples of `b` that
  the block's lags pair with them, and each thread accumulates the chunk from shared memory.
  Indices outside either input are stored as zeros, so the inner loop has no bounds test, and
  the chunk range is the same for every thread of a block, so all threads reach the same
  barriers. Each thread still adds its products in the same order as naive.
- Result (speedup vs. previous iteration and vs. naive): 2.19× (naive is the previous
  iteration). Output is bit-identical to naive at every size in the sweep below.
- Screenshot: none. Report: `docs/profiles/xcorr_tiled.ncu-rep`.

## Iteration 2 — cross-correlation — cuFFT-based (`--impl fft`), and the three variants compared

Kernel, measurements, and this entry were written by an agent at Jean's request (2026-10-08).

- GPU / driver / CUDA version: NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0,
  60 SMs) / 572.76 / 12.8
- Input size and parameters: `python tools/gen_bench.py --xcorr-samples N` for N from 2^8 to
  2^22 samples per channel (the 2^18 pair is identical to `bench_ch0` / `bench_ch1`).
  `--bench 20`, 3 warm-up runs. The FFT length is the next power of two that holds all
  `2N - 1` lags, which is `2N` here.

Kernel time in ms (CUDA events, mean ± std). For `fft` it covers two forward FFTs, the
conjugate-multiply kernel, and the inverse FFT; plan creation is outside the timed region.

| Samples per channel | naive | tiled | fft |
|---|---|---|---|
| 2^8 | 0.020 ± 0.003 | 0.018 ± 0.016 | 0.047 ± 0.028 |
| 2^10 | 0.061 ± 0.040 | 0.026 ± 0.008 | 0.035 ± 0.015 |
| 2^12 | 0.164 ± 0.014 | 0.069 ± 0.008 | 0.034 ± 0.004 |
| 2^14 | 0.733 ± 0.021 | 0.355 ± 0.044 | 0.100 ± 0.027 |
| 2^16 | 9.29 ± 0.06 | 3.99 ± 0.21 | 0.108 ± 0.021 |
| 2^18 | 150.6 ± 4.5 | 68.7 ± 2.4 | 0.140 ± 0.022 |
| 2^20 | not run | not run | 0.464 ± 0.035 |
| 2^22 | not run | not run | 2.42 ± 0.22 |

- Kernel time (mean ± std) at 2^18: 0.140 ± 0.022 ms.
  Transfer time: host-to-device 0.51 ± 0.14 ms, device-to-host 0.63 ± 0.12 ms.
- Achieved memory bandwidth (% of peak) and occupancy, from `ncu` at 2^18. The run has seven
  kernels of 33 to 40 µs each: six cuFFT kernels (two per transform) and
  `xcorr_conj_mul_kernel`.
  - `xcorr_conj_mul_kernel`: Duration 32.51 µs, Max Bandwidth 57.01 % (DRAM Throughput
    57.01 %, L1/TEX 23.10 %, L2 28.91 %), occupancy 100 % theoretical and 72.50 % achieved,
    16 registers per thread, Issue Slots Busy 9.74 %.
  - cuFFT kernels: Max Bandwidth 23 to 27 %, achieved occupancy 33 to 64 % (theoretical 50 %
    or 100 %), 32.77 KB of shared memory per block.
- Top bottleneck reported by the profiler: for `xcorr_conj_mul_kernel`, warps waiting on a
  scoreboard dependency on an L1TEX operation, 81.9 of 90.1 cycles between issued
  instructions (91.0 %). The kernel is two loads, a few multiplies, and a store per element,
  so it waits on data arriving from DRAM; that is the expected profile for a streaming kernel
  over an 8 MB working set. `ncu` flags 50 % of its sectors as excessive.
- What I changed and why: the time-domain kernels do about `N^2` multiply-adds, which no
  memory optimization changes. The correlation theorem gives the same result in
  `O(N log N)`: forward FFT of both inputs, `B * conj(A)` per bin, inverse FFT. Input `a` is
  uploaded rotated left by `num_a - 1` samples, which shifts the circular result so that
  output index `i` already holds lag `i - (num_a - 1)` and no reordering pass is needed. The
  `1 / fft_length` normalization is folded into the conjugate-multiply kernel.
- Result (speedup vs. previous iteration and vs. naive): at 2^18, 490× faster than tiled and
  1070× faster than naive. The FFT version starts to win between 2^10 and 2^12 samples per
  channel: at 2^10 tiled is ahead (0.026 against 0.035 ms, within one standard deviation of
  each other) and at 2^8 both time-domain kernels beat it; from 2^12 up `fft` is fastest. All
  times below about 0.1 ms carry large relative noise. At 2^18 the transfers (1.1 ms) are
  about 8 times the `fft` kernel time.
  Accuracy against the float64 reference at 2^18 (peak value 2.6e5): maximum absolute error
  0.064 for `fft` and 0.72 for naive and tiled, whose single running float32 sum loses more
  precision as the inputs grow. All three recover the true delay of 37 samples at every size.
- Screenshot: none. Report: `docs/profiles/xcorr_fft.ncu-rep`.

## Entry template

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
