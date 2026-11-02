// Polyphase channelizer: a critically sampled analysis filter bank with M channels.
//
// Definition, matching tools/reference.py:
//   z[t][k] = sum_l h[l] * x[t*M - l] * exp(+2j*pi*k*l / M),   x[m] = 0 for m < 0
// for t < ceil(N / M) frames and k < M channels. Writing l = p*M + m splits it into M
// polyphase branches followed by an M-point inverse DFT:
//   u[t][m] = sum_p h[p*M + m] * x[t*M - p*M - m]
//   z[t][k] = sum_m u[t][m] * exp(+2j*pi*k*m / M)        (cuFFT's unnormalized inverse)
// so the GPU work is one custom kernel for the branch sums and one batched cuFFT call. Each
// output frame needs ceil(L / M) multiply-adds per branch instead of the L * M of a separate
// filter-and-mix per channel.

#include "check.hpp"
#include "device_buffer.hpp"
#include "device_ops.hpp"
#include "fft_plan.hpp"
#include "ops.hpp"

#include <stdexcept>
#include <string>

// One thread per (frame t, branch m): idx = t * channels + m, the layout cuFFT's batch expects.
// Taps l = m, m + M, m + 2M, ... reach back to x[t*M - l]; the loop stops at the start of the
// input, which treats earlier samples as zero.
__global__ void channelize_branch_kernel(const cf32* x, const float* taps, cf32* u,
                                         int channels, int num_taps, int frames)
{
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(frames) * channels) {
        return;
    }
    const int t = static_cast<int>(idx / channels);
    const int m = static_cast<int>(idx % channels);
    const long long newest = static_cast<long long>(t) * channels;  // index of x[t*M]
    float re = 0.0f;
    float im = 0.0f;
    for (int l = m; l < num_taps; l += channels) {
        const long long n = newest - l;
        if (n < 0) {
            break;
        }
        const float tap = taps[l];
        const cf32 sample = x[n];
        re += tap * sample.re;
        im += tap * sample.im;
    }
    u[idx].re = re;
    u[idx].im = im;
}

int channelize_frames(int num_samples, int channels)
{
    return (num_samples + channels - 1) / channels;
}

void channelize_device(cufftHandle plan, const cf32* x, const float* taps, cf32* out,
                       int num_samples, int channels, int num_taps)
{
    const int frames = channelize_frames(num_samples, channels);
    if (frames == 0) {
        return;
    }
    // Launch configuration: one thread per output value, 256 threads per block as elsewhere.
    const long long total = static_cast<long long>(frames) * channels;
    const int block = 256;
    const long long grid = (total + block - 1) / block;
    channelize_branch_kernel<<<static_cast<unsigned>(grid), block>>>(x, taps, out, channels,
                                                                     num_taps, frames);
    CUDA_CHECK(cudaGetLastError());
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(out),
                             reinterpret_cast<cufftComplex*>(out), CUFFT_INVERSE));
}

std::vector<cf32> channelize_cuda(const std::vector<cf32>& x, const std::vector<float>& taps,
                                  int channels, Bench& bench)
{
    if (channels < 1) {
        throw std::runtime_error("channelize: need at least one channel");
    }
    const int num_samples = static_cast<int>(x.size());
    const int num_taps = static_cast<int>(taps.size());
    const int frames = channelize_frames(num_samples, channels);
    std::vector<cf32> z(static_cast<std::size_t>(frames) * static_cast<std::size_t>(channels));
    if (frames == 0) {
        return z;
    }

    // One-time setup, outside the timed region: buffers and the batched inverse FFT plan.
    DeviceBuffer<cf32> d_x(x.size());
    DeviceBuffer<float> d_taps(taps.size());
    DeviceBuffer<cf32> d_z(z.size());
    FftPlan plan(channels, frames);

    bench.run(
        [&] {
            d_x.upload(x.data(), x.size());
            d_taps.upload(taps.data(), taps.size());
        },
        [&] {
            channelize_device(plan.get(), d_x.get(), d_taps.get(), d_z.get(), num_samples,
                              channels, num_taps);
        },
        [&] { d_z.download(z.data(), z.size()); });
    return z;
}
