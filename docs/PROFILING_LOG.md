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
