// Magnitude: custom CUDA kernel and its host-side wrapper.
//
// Definition, matching tools/reference.py:
//   mag[n] = |x[n]| = sqrt(x[n].re^2 + x[n].im^2)
// One real output per complex input sample. Magnitude does not depend on framing, so this runs
// over the whole buffer regardless of how spectrum() divided it into frames upstream.

#include "check.hpp"
#include "device_buffer.hpp"
#include "device_ops.hpp"
#include "ops.hpp"

#include <vector>

// One thread per sample: a load, a sqrt, and a store, with no neighbouring threads to share
// data with, so there is nothing to tile or stage in shared memory.
__global__ void magnitude_kernel(const cf32* x, float* mag, int num_samples)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= num_samples) {
        return;
    }
    const cf32 sample = x[n];
    mag[n] = sqrtf(sample.re * sample.re + sample.im * sample.im);
}

void magnitude_device(const cf32* x, float* mag, int num_samples)
{
    if (num_samples == 0) {
        return;
    }
    // Launch configuration: one thread per sample in a 1-D grid, 256 threads per block (a
    // multiple of the warp size, as in the other kernels).
    const int block = 256;
    const int grid = (num_samples + block - 1) / block;
    magnitude_kernel<<<grid, block>>>(x, mag, num_samples);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<float> magnitude_cuda(const std::vector<cf32>& x, Bench& bench)
{
    const int num_samples = static_cast<int>(x.size());
    std::vector<float> mag(x.size());

    // One-time setup, outside the timed region.
    DeviceBuffer<cf32> d_x(x.size());
    DeviceBuffer<float> d_mag(mag.size());

    bench.run(
        [&] { d_x.upload(x.data(), x.size()); },
        [&] { magnitude_device(d_x.get(), d_mag.get(), num_samples); },
        [&] { d_mag.download(mag.data(), mag.size()); });
    return mag;
}
