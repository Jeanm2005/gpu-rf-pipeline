// Cross-correlation: custom CUDA kernels and their host-side wrappers.
//
// Definition, matching tools/reference.py:
//   r[k] = sum_n b[n + k] * conj(a[n])   for every lag k from -(num_a - 1) to num_b - 1,
// with samples outside either input taken as zero. The output has num_a + num_b - 1 values,
// and output index i holds lag k = i - (num_a - 1). If b is a delayed by d samples, |r| peaks
// at lag +d.

#include "check.hpp"
#include "device_buffer.hpp"
#include "fft_plan.hpp"
#include "ops.hpp"

#include <stdexcept>
#include <string>

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

// --- tiled ----------------------------------------------------------------------------------

// One block computes blockDim.x consecutive lags, starting at lag0. It walks input a in chunks
// of blockDim.x samples. For the chunk starting at n0 it stages in shared memory
//   tile_a[j] = a[n0 + j]           for j in [0, blockDim.x)
//   tile_b[j] = b[n0 + lag0 + j]    for j in [0, 2 * blockDim.x - 1)
// with zeros where an index is outside its input, so thread t (lag0 + t) finds the partner of
// tile_a[j] at tile_b[j + t]. Each thread then accumulates the chunk from shared memory.
//
// The host passes (3 * blockDim.x - 1) * sizeof(cf32) bytes of dynamic shared memory.
__global__ void xcorr_tiled_kernel(const cf32* a, const cf32* b, cf32* r, int num_a, int num_b)
{
    extern __shared__ cf32 tile[];
    cf32* tile_a = tile;
    cf32* tile_b = tile + blockDim.x;

    const int chunk = blockDim.x;
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;  // output index this thread computes
    const bool active = i < num_a + num_b - 1;    // the last block has threads past the end
    const int lag0 = blockIdx.x * blockDim.x - (num_a - 1);  // lag of the block's thread 0

    // Range of n in which at least one lag of this block overlaps both inputs. It depends only
    // on the block, so every thread runs the same number of loop iterations and reaches the
    // same barriers.
    const int last_lag = (lag0 + chunk - 1 < num_b - 1) ? lag0 + chunk - 1 : num_b - 1;
    const int n_first = (last_lag < 0) ? -last_lag : 0;
    const int n_end = (num_b - lag0 < num_a) ? num_b - lag0 : num_a;

    float re = 0.0f;
    float im = 0.0f;
    for (int n0 = n_first; n0 < n_end; n0 += chunk) {
        // Cooperative, coalesced load: one sample of a and up to two of b per thread.
        const int na = n0 + tid;
        if (na < num_a) {
            tile_a[tid] = a[na];
        } else {
            tile_a[tid].re = 0.0f;
            tile_a[tid].im = 0.0f;
        }
        for (int j = tid; j < 2 * chunk - 1; j += chunk) {
            const int m = n0 + lag0 + j;
            if (m >= 0 && m < num_b) {
                tile_b[j] = b[m];
            } else {
                tile_b[j].re = 0.0f;
                tile_b[j].im = 0.0f;
            }
        }
        __syncthreads();  // the tile is complete before anyone reads it

        if (active) {
            for (int j = 0; j < chunk; ++j) {
                const cf32 sa = tile_a[j];
                const cf32 sb = tile_b[j + tid];
                // sb * conj(sa) = (sb.re + j sb.im) * (sa.re - j sa.im)
                re += sb.re * sa.re + sb.im * sa.im;
                im += sb.im * sa.re - sb.re * sa.im;
            }
        }
        __syncthreads();  // everyone is done reading before the next chunk overwrites the tile
    }

    if (active) {
        r[i].re = re;
        r[i].im = im;
    }
}

static std::vector<cf32> xcorr_tiled(const std::vector<cf32>& a, const std::vector<cf32>& b,
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

    // Launch configuration: one thread per lag, 256 threads per block as in the naive variant
    // so the two differ only in where the samples are read from. The block size is also the
    // chunk length, so the tile is 256 samples of a plus 511 of b: about 6 KB per block.
    const int block = 256;
    const int grid = (num_lags + block - 1) / block;
    const std::size_t tile_bytes = (3 * static_cast<std::size_t>(block) - 1) * sizeof(cf32);

    bench.run(
        [&] {
            d_a.upload(a.data(), a.size());
            d_b.upload(b.data(), b.size());
        },
        [&] {
            xcorr_tiled_kernel<<<grid, block, tile_bytes>>>(d_a.get(), d_b.get(), d_r.get(),
                                                            num_a, num_b);
            CUDA_CHECK(cudaGetLastError());
        },
        [&] { d_r.download(r.data(), r.size()); });
    return r;
}

// --- fft ------------------------------------------------------------------------------------

// Correlation theorem: with A = FFT(a) and B = FFT(b) over the same length,
// IFFT(B * conj(A))[k] = sum_m b[m + k] * conj(a[m]) with indices taken modulo that length.
// This kernel is the element-wise step: fb[i] = fb[i] * conj(fa[i]) * scale. cuFFT's inverse
// transform is unnormalized, so the host passes scale = 1 / fft_length.
__global__ void xcorr_conj_mul_kernel(const cf32* fa, cf32* fb, int n, float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const cf32 va = fa[i];
    const cf32 vb = fb[i];
    // vb * conj(va) = (vb.re + j vb.im) * (va.re - j va.im)
    fb[i].re = (vb.re * va.re + vb.im * va.im) * scale;
    fb[i].im = (vb.im * va.re - vb.re * va.im) * scale;
}

static std::vector<cf32> xcorr_fft(const std::vector<cf32>& a, const std::vector<cf32>& b,
                                   Bench& bench)
{
    const std::size_t num_a = a.size();
    const std::size_t num_lags = a.size() + b.size() - 1;
    std::vector<cf32> r(num_lags);

    // FFT length: the next power of two that holds every lag, so the circular correlation
    // does not wrap one lag onto another.
    std::size_t fft_len = 1;
    while (fft_len < num_lags) {
        fft_len *= 2;
    }
    if (fft_len > (std::size_t{1} << 30)) {
        throw std::runtime_error("xcorr fft: " + std::to_string(num_lags)
                                 + " lags need an FFT longer than the supported 2^30 points");
    }
    const int n = static_cast<int>(fft_len);

    // One-time setup, outside the timed region. The forward transforms are out of place so
    // that the zero padding written here survives every benchmark pass.
    DeviceBuffer<cf32> d_a(fft_len);
    DeviceBuffer<cf32> d_b(fft_len);
    DeviceBuffer<cf32> d_fa(fft_len);
    DeviceBuffer<cf32> d_fb(fft_len);
    d_a.zero();
    d_b.zero();
    FftPlan plan(n, 1);

    // Launch configuration for the conjugate multiply: one thread per frequency bin, 256
    // threads per block. Each thread does two loads and one store with no loop, so the block
    // size only has to be a multiple of the warp.
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const float scale = 1.0f / static_cast<float>(fft_len);

    auto fft = [&](DeviceBuffer<cf32>& in, DeviceBuffer<cf32>& out, int direction) {
        CUFFT_CHECK(cufftExecC2C(plan.get(), reinterpret_cast<cufftComplex*>(in.get()),
                                 reinterpret_cast<cufftComplex*>(out.get()), direction));
    };

    bench.run(
        [&] {
            // a is stored rotated left by num_a - 1: a[num_a - 1] at index 0 and a[0 .. num_a - 2]
            // at the end of the buffer. That shifts the circular result so that index i holds
            // lag i - (num_a - 1), which is the output order, with no reordering afterwards.
            d_a.upload(a.data() + (num_a - 1), 1);
            d_a.upload_at(fft_len - (num_a - 1), a.data(), num_a - 1);
            d_b.upload(b.data(), b.size());
        },
        [&] {
            fft(d_a, d_fa, CUFFT_FORWARD);
            fft(d_b, d_fb, CUFFT_FORWARD);
            xcorr_conj_mul_kernel<<<grid, block>>>(d_fa.get(), d_fb.get(), n, scale);
            CUDA_CHECK(cudaGetLastError());
            fft(d_fb, d_fb, CUFFT_INVERSE);
        },
        [&] { d_fb.download(r.data(), r.size()); });
    return r;
}

// --- implementation table ------------------------------------------------------------------

const std::vector<Impl<XcorrFn>>& xcorr_impls()
{
    static const std::vector<Impl<XcorrFn>> impls = {
        {"naive", "one thread per lag, global memory only", xcorr_naive},
        {"tiled", "both inputs staged in shared-memory tiles, chunk by chunk", xcorr_tiled},
        {"fft", "cuFFT: forward FFTs, conjugate multiply, inverse FFT", xcorr_fft},
    };
    return impls;
}
