// Fused kernels: one launch doing the work of several operations, for the fusion groups
// sigflowc forms (compiler/src/fuse.cpp). Each batched kernel does the same arithmetic as the
// single-operation kernel it batches, so results match it exactly. The tiled batched kernels
// live in fir.cu, next to the constant-memory taps they share with the single-channel ones.

#include "check.hpp"
#include "device_ops.hpp"
#include "fir_batch.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

int make_fir_batch(const cf32* const* x, cf32* const* y, const int* num_samples, int count,
                   FirBatch& batch)
{
    if (count < 1 || count > kMaxFirBatch) {
        throw std::runtime_error("batched fir: " + std::to_string(count)
                                 + " channels, expected 1 to " + std::to_string(kMaxFirBatch));
    }
    batch = FirBatch{};
    int longest = 0;
    for (int c = 0; c < count; ++c) {
        batch.x[c] = x[c];
        batch.y[c] = y[c];
        batch.num_samples[c] = num_samples[c];
        longest = num_samples[c] > longest ? num_samples[c] : longest;
    }
    return longest;
}

// fir_naive_kernel over several channels that share one set of taps: blockIdx.y picks the
// channel, and within it each thread computes one output sample exactly as the naive kernel
// does (taps and samples straight from global memory). The grid is as wide as the longest
// channel, so threads past a shorter channel's end return at once.
__global__ void fir_naive_batched_kernel(FirBatch batch, const float* taps, int num_taps)
{
    const int channel = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;  // output sample this thread computes
    if (n >= batch.num_samples[channel]) {
        return;
    }
    const cf32* x = batch.x[channel];

    // x[n - k] only exists for k <= n; stopping the loop there treats earlier samples as zero.
    const int last_tap = (n < num_taps - 1) ? n : num_taps - 1;
    float re = 0.0f;
    float im = 0.0f;
    for (int k = 0; k <= last_tap; ++k) {
        const float tap = taps[k];
        const cf32 sample = x[n - k];
        re += tap * sample.re;
        im += tap * sample.im;
    }
    batch.y[channel][n].re = re;
    batch.y[channel][n].im = im;
}

void fir_naive_batched_device(const cf32* const* x, cf32* const* y, const int* num_samples,
                              int count, const float* taps, int num_taps)
{
    FirBatch batch;
    const int longest = make_fir_batch(x, y, num_samples, count, batch);
    if (longest == 0) {
        return;
    }
    // Launch configuration: the naive kernel's 256 threads per block along x, one grid row per
    // channel along y, so each channel gets the same blocks it would get on its own and the
    // fused launch differs from the separate ones only in being one launch.
    const int block = 256;
    const dim3 grid((longest + block - 1) / block, count);
    fir_naive_batched_kernel<<<grid, block>>>(batch, taps, num_taps);
    CUDA_CHECK(cudaGetLastError());
}

bool parse_fir_impl(const char* name, FirImpl& impl)
{
    if (std::strcmp(name, "naive") == 0) {
        impl = FirImpl::Naive;
    } else if (std::strcmp(name, "tiled") == 0) {
        impl = FirImpl::Tiled;
    } else if (std::strcmp(name, "tiled_const") == 0) {
        impl = FirImpl::TiledConst;
    } else {
        return false;
    }
    return true;
}

void fir_batched_device(FirImpl impl, const cf32* const* x, cf32* const* y,
                        const int* num_samples, int count, const float* taps, int num_taps)
{
    switch (impl) {
    case FirImpl::Naive:
        fir_naive_batched_device(x, y, num_samples, count, taps, num_taps);
        return;
    case FirImpl::Tiled:
        fir_tiled_batched_device(x, y, num_samples, count, taps, num_taps);
        return;
    case FirImpl::TiledConst:
        fir_tiled_const_batched_device(x, y, num_samples, count, taps, num_taps);
        return;
    }
}
