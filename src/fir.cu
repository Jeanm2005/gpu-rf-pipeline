// FIR filter: custom CUDA kernels (written by Jean) and their host-side wrappers.
//
// Definition, matching tools/reference.py:
//   y[n] = sum_{k=0}^{num_taps-1} taps[k] * x[n - k],   with x[m] = 0 for m < 0
// Real taps, complex samples, output length equal to input length.

#include "check.hpp"
#include "device_buffer.hpp"
#include "ops.hpp"

// --- naive ---------------------------------------------------------------------------------

// One thread per output sample; samples and taps are read straight from global memory.
// The last block usually has threads whose index is past num_samples.
__global__ void fir_naive_kernel(const cf32* x, const float* taps, cf32* y, int num_samples,
                                 int num_taps)
{
    // TODO(Jean): kernel body.
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

// --- implementation table ------------------------------------------------------------------

const std::vector<Impl<FirFn>>& fir_impls()
{
    static const std::vector<Impl<FirFn>> impls = {
        {"naive", "one thread per output sample, global memory only", fir_naive},
    };
    return impls;
}
