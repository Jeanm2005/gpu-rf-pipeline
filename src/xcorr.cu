// Cross-correlation: custom CUDA kernels and their host-side wrappers.
//
// Definition, matching tools/reference.py:
//   r[k] = sum_n b[n + k] * conj(a[n])   for every lag k from -(num_a - 1) to num_b - 1,
// with samples outside either input taken as zero. The output has num_a + num_b - 1 values,
// and output index i holds lag k = i - (num_a - 1). If b is a delayed by d samples, |r| peaks
// at lag +d.

#include "check.hpp"
#include "device_buffer.hpp"
#include "ops.hpp"

// --- naive ---------------------------------------------------------------------------------

// One thread per lag (per output index); both inputs are read straight from global memory.
// The last block usually has threads whose index is past num_a + num_b - 1.
__global__ void xcorr_naive_kernel(const cf32* a, const cf32* b, cf32* r, int num_a, int num_b)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index this thread computes
    if (i >= num_a + num_b - 1) {
        return;
    }
    const int lag = i - (num_a - 1);

    // The sum needs a[n] and b[n + lag] to both exist: 0 <= n < num_a and 0 <= n + lag < num_b.
    // Clamping the loop range to that overlap is the same as treating samples outside either
    // input as zero, and keeps a bounds test out of the loop.
    const int first = (lag < 0) ? -lag : 0;
    const int end = (num_b - lag < num_a) ? num_b - lag : num_a;

    // Accumulate in registers and write r[i] once, so the loop does no global-memory writes.
    float re = 0.0f;
    float im = 0.0f;
    for (int n = first; n < end; ++n) {
        const cf32 sa = a[n];
        const cf32 sb = b[n + lag];
        // sb * conj(sa) = (sb.re + j sb.im) * (sa.re - j sa.im)
        re += sb.re * sa.re + sb.im * sa.im;
        im += sb.im * sa.re - sb.re * sa.im;
    }
    r[i].re = re;
    r[i].im = im;
}

static std::vector<cf32> xcorr_naive(const std::vector<cf32>& a, const std::vector<cf32>& b,
                                     Bench& bench)
{
    const int num_a = static_cast<int>(a.size());
    const int num_b = static_cast<int>(b.size());
    const int num_lags = num_a + num_b - 1;
    std::vector<cf32> r(static_cast<std::size_t>(num_lags));

    // One-time setup, outside the timed region.
    DeviceBuffer<cf32> d_a(a.size());
    DeviceBuffer<cf32> d_b(b.size());
    DeviceBuffer<cf32> d_r(r.size());

    // Launch configuration: one thread per lag in a 1-D grid. 256 threads per block is a
    // neutral starting point (a multiple of the 32-thread warp, and several blocks fit on
    // each streaming multiprocessor). It is a placeholder until the profiler's occupancy
    // numbers say otherwise.
    const int block = 256;
    const int grid = (num_lags + block - 1) / block;

    bench.run(
        [&] {
            d_a.upload(a.data(), a.size());
            d_b.upload(b.data(), b.size());
        },
        [&] {
            xcorr_naive_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_r.get(), num_a, num_b);
            CUDA_CHECK(cudaGetLastError());
        },
        [&] { d_r.download(r.data(), r.size()); });
    return r;
}

// --- implementation table ------------------------------------------------------------------

const std::vector<Impl<XcorrFn>>& xcorr_impls()
{
    static const std::vector<Impl<XcorrFn>> impls = {
        {"naive", "one thread per lag, global memory only", xcorr_naive},
    };
    return impls;
}
