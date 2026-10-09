// FIR filter: custom CUDA kernels and their host-side wrappers.
//
// Definition, matching tools/reference.py:
//   y[n] = sum_{k=0}^{num_taps-1} taps[k] * x[n - k],   with x[m] = 0 for m < 0
// Real taps, complex samples, output length equal to input length.

#include "check.hpp"
#include "device_buffer.hpp"
#include "ops.hpp"

#include <stdexcept>
#include <string>

// --- naive ---------------------------------------------------------------------------------

// One thread per output sample; samples and taps are read straight from global memory.
// The last block usually has threads whose index is past num_samples.
__global__ void fir_naive_kernel(const cf32* x, const float* taps, cf32* y, int num_samples,
                                 int num_taps)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;  // output sample this thread computes
    if (n >= num_samples) {
        return;
    }

    // x[n - k] only exists for k <= n. Stopping the loop there is the same as treating the
    // samples before the start of the input as zero, without reading out of bounds.
    const int last_tap = (n < num_taps - 1) ? n : num_taps - 1;

    // Accumulate in registers and write y[n] once, so the loop does no global-memory writes.
    float re = 0.0f;
    float im = 0.0f;
    for (int k = 0; k <= last_tap; ++k) {
        const float tap = taps[k];
        const cf32 sample = x[n - k];
        re += tap * sample.re;
        im += tap * sample.im;
    }
    y[n].re = re;
    y[n].im = im;
}

static std::vector<cf32> fir_naive(const std::vector<cf32>& x, const std::vector<float>& taps,
                                   Bench& bench)
{
    const int num_samples = static_cast<int>(x.size());
    const int num_taps = static_cast<int>(taps.size());
    std::vector<cf32> y(x.size());

    // One-time setup, outside the timed region.
    DeviceBuffer<cf32> d_x(x.size());
    DeviceBuffer<float> d_taps(taps.size());
    DeviceBuffer<cf32> d_y(y.size());

    // Launch configuration: one thread per output sample in a 1-D grid. 256 threads per block
    // is a neutral starting point (a multiple of the 32-thread warp, and several blocks fit
    // on each streaming multiprocessor). It is a placeholder until the profiler's occupancy
    // numbers say otherwise.
    const int block = 256;
    const int grid = (num_samples + block - 1) / block;

    bench.run(
        [&] {
            d_x.upload(x.data(), x.size());
            d_taps.upload(taps.data(), taps.size());
        },
        [&] {
            fir_naive_kernel<<<grid, block>>>(d_x.get(), d_taps.get(), d_y.get(), num_samples,
                                              num_taps);
            CUDA_CHECK(cudaGetLastError());
        },
        [&] { d_y.download(y.data(), y.size()); });
    return y;
}

// --- tiled ----------------------------------------------------------------------------------

// One block computes blockDim.x consecutive output samples. It first copies every input sample
// those outputs need into shared memory (the "tile"), then each thread runs its tap loop
// against the tile instead of global memory. Taps are still read from global memory.
//
// Tile layout, with halo = num_taps - 1 and block_start = blockIdx.x * blockDim.x:
//   tile[i] = x[block_start - halo + i]   for i in [0, halo + blockDim.x)
// and zero where that index is before the start or past the end of the input. The halo is the
// history the first output of the block reaches back to. It can be longer than blockDim.x
// (1025 taps against 256 threads), so each thread may have to load more than one tile entry.
//
// The host passes (halo + blockDim.x) * sizeof(cf32) bytes of dynamic shared memory.
__global__ void fir_tiled_kernel(const cf32* x, const float* taps, cf32* y, int num_samples,
                                 int num_taps)
{
    extern __shared__ cf32 tile[];

    const int halo = num_taps - 1;                    // history the first output reaches back to
    const int block_start = blockIdx.x * blockDim.x;  // first output sample of this block
    const int tile_len = halo + blockDim.x;
    const int tile_start = block_start - halo;        // input index held in tile[0]; can be < 0

    // Cooperative load: thread t fills tile[t], tile[t + blockDim.x], ... so the threads of a
    // warp read consecutive input samples (coalesced) and the loop also covers a halo longer
    // than the block. Samples before the start or past the end of the input are zero.
    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        const int m = tile_start + i;
        if (m >= 0 && m < num_samples) {
            tile[i] = x[m];
        } else {
            tile[i].re = 0.0f;
            tile[i].im = 0.0f;
        }
    }
    // Every thread reaches this barrier, including those whose output index is past
    // num_samples: they loaded part of the tile that other threads read below.
    __syncthreads();

    const int n = block_start + threadIdx.x;  // output sample this thread computes
    if (n >= num_samples) {
        return;
    }

    // x[n - k] is tile[halo + threadIdx.x - k]. The zero-filled entries stand in for the
    // samples before the start of the input, so the loop needs no bounds test.
    const int newest = halo + threadIdx.x;
    float re = 0.0f;
    float im = 0.0f;
    for (int k = 0; k < num_taps; ++k) {
        const float tap = taps[k];
        const cf32 sample = tile[newest - k];
        re += tap * sample.re;
        im += tap * sample.im;
    }
    y[n].re = re;
    y[n].im = im;
}

static std::vector<cf32> fir_tiled(const std::vector<cf32>& x, const std::vector<float>& taps,
                                   Bench& bench)
{
    const int num_samples = static_cast<int>(x.size());
    const int num_taps = static_cast<int>(taps.size());
    std::vector<cf32> y(x.size());

    // Launch configuration: one thread per output sample, 256 threads per block as in the
    // naive variant so the two differ only in where the samples are read from. The tile holds
    // the block's own samples plus a halo of num_taps - 1 samples of history.
    const int block = 256;
    const int grid = (num_samples + block - 1) / block;
    const std::size_t tile_bytes =
        (static_cast<std::size_t>(num_taps) - 1 + static_cast<std::size_t>(block)) * sizeof(cf32);

    // The whole halo has to fit in one block's shared memory (48 KB by default, about 5800
    // taps at this block size). Longer filters need a variant that walks the taps in pieces.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (tile_bytes > prop.sharedMemPerBlock) {
        throw std::runtime_error("fir tiled: " + std::to_string(num_taps) + " taps need "
                                 + std::to_string(tile_bytes) + " bytes of shared memory per "
                                 "block, more than the " + std::to_string(prop.sharedMemPerBlock)
                                 + " available; use --impl naive");
    }

    // One-time setup, outside the timed region.
    DeviceBuffer<cf32> d_x(x.size());
    DeviceBuffer<float> d_taps(taps.size());
    DeviceBuffer<cf32> d_y(y.size());

    bench.run(
        [&] {
            d_x.upload(x.data(), x.size());
            d_taps.upload(taps.data(), taps.size());
        },
        [&] {
            fir_tiled_kernel<<<grid, block, tile_bytes>>>(d_x.get(), d_taps.get(), d_y.get(),
                                                          num_samples, num_taps);
            CUDA_CHECK(cudaGetLastError());
        },
        [&] { d_y.download(y.data(), y.size()); });
    return y;
}

// --- implementation table ------------------------------------------------------------------

const std::vector<Impl<FirFn>>& fir_impls()
{
    static const std::vector<Impl<FirFn>> impls = {
        {"naive", "one thread per output sample, global memory only", fir_naive},
        {"tiled", "samples staged in a shared-memory tile with a halo", fir_tiled},
    };
    return impls;
}
