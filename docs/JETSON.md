# Running on Jetson

Notes for building and measuring rfgpu and SigFlow programs on NVIDIA Jetson boards (Orin,
compute capability 8.7; Thor, 11.0). Everything here is checked by building for those
architectures; no measurement in this repository was taken on a Jetson.

## What changes on an embedded GPU

| | RTX 5080 Laptop GPU (the reference machine) | Jetson Orin / Thor |
|---|---|---|
| Memory | 16 GB of GDDR7 on the card, reached over PCIe | One LPDDR5 / LPDDR5X DRAM shared by CPU and GPU |
| Host-to-device copy | Crosses PCIe; for the 63-tap FIR the copies take about 30 times as long as the kernel | A copy inside the same DRAM: it costs bandwidth and buys nothing |
| Bandwidth | 896 GB/s | 68 to 205 GB/s (Orin Nano to AGX Orin), 273 GB/s (Thor) |
| Power | Up to 150 W, clocks limited by the laptop's power cap | 7 to 60 W (Orin), 40 to 130 W (Thor), set by the power mode |

Two consequences for this code:

- **Avoid copies, not just overlap them.** On the discrete GPU, `rfgpu stream` hides copies
  behind kernels. On a Jetson the copies themselves are waste, so `rfgpu stream --memory
  zero-copy` lets the kernel read and write mapped pinned host memory directly. On Orin and Thor
  that memory is cached and I/O-coherent, so the kernel reads it at DRAM speed. On the
  discrete GPU the same mode crosses PCIe on every access, and the profiling log shows what
  that costs (Stage 3, streaming).
- **The bottlenecks move.** The FIR and cross-correlation kernels are bound by the L1/TEX unit
  and by instruction issue on the RTX GPU, not by DRAM, so they should scale with SM count and
  clock rather than with the much lower bandwidth. `magnitude` is DRAM-bound and will slow down
  in proportion to bandwidth. Re-profile before assuming either.

## Building

The code builds unchanged; only the architecture differs. On the board, with the CUDA toolkit
from JetPack (12.6 or newer for Orin, 13 for Thor):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87   # Orin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110  # Thor
cmake --build build -j
```

The default `native` architecture also works when building on the board itself. CI compiles the
device code for 8.7 and 11.0 as well as 12.0, so an architecture-specific break shows up there.

`src/io.cpp` assumes a little-endian host; the Jetson's Arm cores are little-endian.

## Measuring

Clocks on a Jetson follow its power mode and thermal state, so fix them before timing:

```bash
sudo nvpmodel -q                  # current power mode
sudo nvpmodel -m 0                # maximum performance mode (MAXN on Orin AGX)
sudo jetson_clocks                # pin CPU, GPU, and memory clocks at the mode's maximum
tegrastats --interval 500         # GPU load, clocks, power rails, temperatures while running
```

Record the power mode next to every result, as the profiling log records the driver and CUDA
version. The measurement rules do not change: warm up, average at least 10 runs, and report
transfers separately from kernels. For streaming, compare the two memory modes on the same
input:

```bash
python tools/gen_bench.py --out out/bench
./build/rfgpu stream --in out/bench/bench_tone.cf32 --taps tests/fixtures/lowpass.f32 \
    --out out/y.cf32 --memory copy --bench 20
./build/rfgpu stream --in out/bench/bench_tone.cf32 --taps tests/fixtures/lowpass.f32 \
    --out out/y.cf32 --memory zero-copy --bench 20
```

Nsight Compute and Nsight Systems run on the board (`ncu` and `nsys` from JetPack); as on the
desktop, collecting GPU performance counters may need `sudo`.

## Unified memory

`cudaMallocManaged` is the other way to share one buffer between CPU and GPU. On Jetson it
avoids copies like mapped pinned memory, but with page-migration bookkeeping that the explicit
ring in `rfgpu stream` does not need; the stream's mapped pinned buffers are the simpler choice
for data that arrives in fixed chunks. Managed memory fits the SigFlow CUDA backend better,
where values are reused across operations: allocating the runtime's buffer slots with
`cudaMallocManaged` instead of `cudaMalloc` would remove the upload and download phases on a
Jetson. That is not implemented.
