// FIR filter: custom CUDA kernels and their host-side wrappers.
//
// Definition, matching tools/reference.py:
//   y[n] = sum_{k=0}^{num_taps-1} taps[k] * x[n - k],   with x[m] = 0 for m < 0
// Real taps, complex samples, output length equal to input length.

#include "check.hpp"
#include "device_buffer.hpp"
#include "device_ops.hpp"
#include "fir_batch.hpp"
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

// Launch configuration: one thread per output sample in a 1-D grid. 256 threads per block
// (kDefaultFirBlock) is the default; iteration 3 in docs/PROFILING_LOG.md sweeps the others.
static void launch_fir_naive(const cf32* x, const float* taps, cf32* y, int num_samples,
                             int num_taps, int block)
{
    if (num_samples == 0) {
        return;
    }
    const int grid = (num_samples + block - 1) / block;
    fir_naive_kernel<<<grid, block>>>(x, taps, y, num_samples, num_taps);
    CUDA_CHECK(cudaGetLastError());
}

void fir_naive_device(const cf32* x, const float* taps, cf32* y, int num_samples, int num_taps)
{
    launch_fir_naive(x, taps, y, num_samples, num_taps, kDefaultFirBlock);
}

static std::vector<cf32> fir_naive(const std::vector<cf32>& x, const std::vector<float>& taps,
                                   int block, Bench& bench)
{
    const int num_samples = static_cast<int>(x.size());
    const int num_taps = static_cast<int>(taps.size());
    std::vector<cf32> y(x.size());

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
            launch_fir_naive(d_x.get(), d_taps.get(), d_y.get(), num_samples, num_taps, block);
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

// Bytes of dynamic shared memory for one tile: the block's own samples plus the halo. The whole
// halo has to fit in one block's shared memory (48 KB by default, about 5800 taps with 256
// threads per block). Longer filters need a variant that walks the taps in pieces.
static std::size_t fir_tile_bytes(int num_taps, int block, const char* impl)
{
    const std::size_t tile_bytes =
        (static_cast<std::size_t>(num_taps) - 1 + static_cast<std::size_t>(block)) * sizeof(cf32);
    // cudaDeviceGetAttribute rather than cudaGetDeviceProperties: it is cheap enough to call on
    // every launch from the device-pointer entry points, inside timed regions.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int shared_per_block = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
                                      device));
    if (tile_bytes > static_cast<std::size_t>(shared_per_block)) {
        throw std::runtime_error(std::string("fir ") + impl + ": " + std::to_string(num_taps)
                                 + " taps need " + std::to_string(tile_bytes)
                                 + " bytes of shared memory per block, more than the "
                                 + std::to_string(shared_per_block)
                                 + " available; use --impl naive");
    }
    return tile_bytes;
}

static std::vector<cf32> fir_tiled(const std::vector<cf32>& x, const std::vector<float>& taps,
                                   int block, Bench& bench)
{
    const int num_samples = static_cast<int>(x.size());
    const int num_taps = static_cast<int>(taps.size());
    std::vector<cf32> y(x.size());

    // Launch configuration: one thread per output sample, the same block size as the naive
    // variant so the two differ only in where the samples are read from. The tile holds the
    // block's own samples plus a halo of num_taps - 1 samples of history.
    const int grid = (num_samples + block - 1) / block;
    const std::size_t tile_bytes = fir_tile_bytes(num_taps, block, "tiled");

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

// --- tiled_const ----------------------------------------------------------------------------

// Taps in constant memory. Every thread of a warp reads the same taps[k] in the same loop
// iteration, which is the access pattern the constant cache is built for, and it takes the
// tap loads off the global-memory path that iteration 1 still stalls on. Constant memory is
// 64 KB per module; 8192 taps use half of it and are more than a tile can hold anyway.
constexpr int kMaxConstTaps = 8192;
__constant__ float c_taps[kMaxConstTaps];

// Same tile as fir_tiled_kernel (see the layout there); the only difference is that taps come
// from c_taps instead of a global-memory pointer.
__global__ void fir_tiled_const_kernel(const cf32* x, cf32* y, int num_samples, int num_taps)
{
    extern __shared__ cf32 tile[];

    const int halo = num_taps - 1;                    // history the first output reaches back to
    const int block_start = blockIdx.x * blockDim.x;  // first output sample of this block
    const int tile_len = halo + blockDim.x;
    const int tile_start = block_start - halo;        // input index held in tile[0]; can be < 0

    // Cooperative load, as in fir_tiled_kernel: coalesced, covers a halo longer than the
    // block, and stores zeros for samples outside the input.
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

    // x[n - k] is tile[halo + threadIdx.x - k]; the tap now comes from constant memory.
    const int newest = halo + threadIdx.x;
    float re = 0.0f;
    float im = 0.0f;
    for (int k = 0; k < num_taps; ++k) {
        const float tap = c_taps[k];
        const cf32 sample = tile[newest - k];
        re += tap * sample.re;
        im += tap * sample.im;
    }
    y[n].re = re;
    y[n].im = im;
}

static std::vector<cf32> fir_tiled_const(const std::vector<cf32>& x,
                                         const std::vector<float>& taps, int block,
                                         Bench& bench)
{
    const int num_samples = static_cast<int>(x.size());
    const int num_taps = static_cast<int>(taps.size());
    std::vector<cf32> y(x.size());

    if (num_taps > kMaxConstTaps) {
        throw std::runtime_error("fir tiled_const: " + std::to_string(num_taps)
                                 + " taps, more than the " + std::to_string(kMaxConstTaps)
                                 + " that fit in constant memory; use --impl naive");
    }

    // Launch configuration: identical to the tiled variant (one tile of dynamic shared memory
    // per block), so the two differ only in where the taps are read from.
    const int grid = (num_samples + block - 1) / block;
    const std::size_t tile_bytes = fir_tile_bytes(num_taps, block, "tiled_const");

    // One-time setup, outside the timed region.
    DeviceBuffer<cf32> d_x(x.size());
    DeviceBuffer<cf32> d_y(y.size());

    bench.run(
        [&] {
            d_x.upload(x.data(), x.size());
            // The taps go to the constant-memory symbol instead of a device buffer.
            CUDA_CHECK(cudaMemcpyToSymbol(c_taps, taps.data(), taps.size() * sizeof(float)));
        },
        [&] {
            fir_tiled_const_kernel<<<grid, block, tile_bytes>>>(d_x.get(), d_y.get(),
                                                                num_samples, num_taps);
            CUDA_CHECK(cudaGetLastError());
        },
        [&] { d_y.download(y.data(), y.size()); });
    return y;
}

// --- register -------------------------------------------------------------------------------

// tiled_const is bound by shared-memory traffic: Nsight Compute shows its warps stalled on the
// MIO queue, and every multiply-add pair loads one 8-byte sample from the tile. Here each thread
// computes R consecutive outputs and keeps the samples they share in registers, so it loads one
// new sample per tap for all of its outputs:
//
//   y[n + j] = sum_k taps[k] * x[n + j - k],  j = 0 .. R - 1
//
// For tap k the thread needs x[n - k + j] for every j; moving to tap k + 1 shifts that window by
// one sample, so w[j] takes the old w[j - 1] and only x[n - k - 1] is new.
//
// Consecutive threads now read samples R apart. With the tile as an array of cf32 that stride
// puts R threads of a warp on each shared-memory bank. The tile is therefore stored as two float
// arrays, I and Q (structure of arrays), with one padding float after every 32: thread t reads
// float index base + R t, and the padding moves each group of 32 / R threads to the next bank,
// so the 32 reads of a warp fall in 32 different banks (for R = 4 and R = 8).
//
// R is a template parameter, chosen per filter by register_outputs(): 8 for long filters, where
// the kernel is bound by instruction issue and more outputs per thread spread the loads and
// index arithmetic over more multiply-adds; 4 for short ones, where loading the tile dominates
// and the smaller tile keeps twice as many blocks resident (PROFILING_LOG.md, iteration 4).
constexpr int kRegLongFilterTaps = 128;
constexpr std::size_t kDefaultSharedBytes = 48 * 1024;  // per block, without opting in

static int register_outputs(int num_taps)
{
    return num_taps >= kRegLongFilterTaps ? 8 : 4;
}

__device__ __forceinline__ int padded_index(int i)
{
    return i + (i >> 5);
}

// Floats in one padded array of `len` samples.
__host__ __device__ __forceinline__ int padded_len(int len)
{
    return len + (len >> 5) + 1;
}

// One block of the register variant: outputs [blockIdx.x * blockDim.x * R, ...) of x. Shared by
// the single-channel and the batched kernel.
template <int R>
__device__ void fir_register_block(const cf32* x, cf32* y, int num_samples, int num_taps)
{
    extern __shared__ float soa[];

    const int halo = num_taps - 1;
    const int outputs = blockDim.x * R;   // outputs per block
    const int block_start = blockIdx.x * outputs;
    const int tile_len = halo + outputs;
    const int tile_start = block_start - halo;
    float* tile_re = soa;
    float* tile_im = soa + padded_len(tile_len);

    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        const int m = tile_start + i;
        const bool inside = m >= 0 && m < num_samples;
        const cf32 sample = inside ? x[m] : cf32{0.0f, 0.0f};
        tile_re[padded_index(i)] = sample.re;
        tile_im[padded_index(i)] = sample.im;
    }
    __syncthreads();

    const int first = block_start + threadIdx.x * R;  // this thread's first output
    if (first >= num_samples) {
        return;
    }

    // x[first + j - k] is tile[newest + j - k]. Window for tap 0: w[j] = x[first + j].
    const int newest = halo + threadIdx.x * R;
    float w_re[R];
    float w_im[R];
    float acc_re[R];
    float acc_im[R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        w_re[j] = tile_re[padded_index(newest + j)];
        w_im[j] = tile_im[padded_index(newest + j)];
        acc_re[j] = 0.0f;
        acc_im[j] = 0.0f;
    }
    for (int k = 0; k < num_taps; ++k) {
        const float tap = c_taps[k];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            acc_re[j] += tap * w_re[j];
            acc_im[j] += tap * w_im[j];
        }
        // Shift the window to tap k + 1. Its new sample, tile[newest - k - 1], exists for every
        // tap but the last (newest >= halo = num_taps - 1).
#pragma unroll
        for (int j = R - 1; j > 0; --j) {
            w_re[j] = w_re[j - 1];
            w_im[j] = w_im[j - 1];
        }
        if (k + 1 < num_taps) {
            w_re[0] = tile_re[padded_index(newest - k - 1)];
            w_im[0] = tile_im[padded_index(newest - k - 1)];
        }
    }
#pragma unroll
    for (int j = 0; j < R; ++j) {
        if (first + j < num_samples) {
            y[first + j].re = acc_re[j];
            y[first + j].im = acc_im[j];
        }
    }
}

template <int R>
__global__ void fir_register_kernel(const cf32* x, cf32* y, int num_samples, int num_taps)
{
    fir_register_block<R>(x, y, num_samples, num_taps);
}

// Bytes of dynamic shared memory for fir_register_kernel: two padded float arrays of
// halo + block * R samples. With 8 outputs per thread, 1024-thread blocks need more than the
// default 48 KB per block, so the limit is the device's opt-in maximum (99 KB here), and
// launch_fir_register raises the kernel's limit when it has to.
static std::size_t fir_register_bytes(int num_taps, int block)
{
    const int tile_len = num_taps - 1 + block * register_outputs(num_taps);
    const std::size_t bytes = 2 * static_cast<std::size_t>(padded_len(tile_len)) * sizeof(float);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int shared_per_block = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      device));
    if (bytes > static_cast<std::size_t>(shared_per_block)) {
        throw std::runtime_error("fir register: " + std::to_string(num_taps) + " taps need "
                                 + std::to_string(bytes)
                                 + " bytes of shared memory per block, more than the "
                                 + std::to_string(shared_per_block)
                                 + " available; use --impl naive");
    }
    return bytes;
}

// Launch configuration: `block` threads per block, each block computing block * R outputs.
// Expects the taps in constant memory already.
static void launch_fir_register(const cf32* x, cf32* y, int num_samples, int num_taps, int block,
                                cudaStream_t stream)
{
    const int outputs = register_outputs(num_taps);
    const int per_block = block * outputs;
    const int grid = (num_samples + per_block - 1) / per_block;
    const std::size_t bytes = fir_register_bytes(num_taps, block);
    const int shared = static_cast<int>(bytes);
    if (outputs == 8) {
        if (bytes > kDefaultSharedBytes) {
            CUDA_CHECK(cudaFuncSetAttribute(fir_register_kernel<8>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
        }
        fir_register_kernel<8><<<grid, block, bytes, stream>>>(x, y, num_samples, num_taps);
    } else {
        if (bytes > kDefaultSharedBytes) {
            CUDA_CHECK(cudaFuncSetAttribute(fir_register_kernel<4>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
        }
        fir_register_kernel<4><<<grid, block, bytes, stream>>>(x, y, num_samples, num_taps);
    }
    CUDA_CHECK(cudaGetLastError());
}

static std::vector<cf32> fir_register(const std::vector<cf32>& x,
                                      const std::vector<float>& taps, int block, Bench& bench)
{
    const int num_samples = static_cast<int>(x.size());
    const int num_taps = static_cast<int>(taps.size());
    std::vector<cf32> y(x.size());

    if (num_taps > kMaxConstTaps) {
        throw std::runtime_error("fir register: " + std::to_string(num_taps)
                                 + " taps, more than the " + std::to_string(kMaxConstTaps)
                                 + " that fit in constant memory; use --impl naive");
    }
    fir_register_bytes(num_taps, block);  // fails before any allocation if the tile is too big

    DeviceBuffer<cf32> d_x(x.size());
    DeviceBuffer<cf32> d_y(y.size());

    bench.run(
        [&] {
            d_x.upload(x.data(), x.size());
            CUDA_CHECK(cudaMemcpyToSymbol(c_taps, taps.data(), taps.size() * sizeof(float)));
        },
        [&] { launch_fir_register(d_x.get(), d_y.get(), num_samples, num_taps, block, 0); },
        [&] { d_y.download(y.data(), y.size()); });
    return y;
}

// --- device-pointer entry points -----------------------------------------------------------

// Launch configuration for every launch below: kDefaultFirBlock (256) threads per block, as in
// the host wrappers' default, with the tile of the block's samples plus a halo of num_taps - 1
// in dynamic shared memory for the tiled variants.
constexpr int kTiledBlock = kDefaultFirBlock;

static bool uses_const_taps(FirImpl impl)
{
    return impl == FirImpl::TiledConst || impl == FirImpl::Register;
}

void fir_load_const_taps(const float* taps, int num_taps, cudaStream_t stream)
{
    if (num_taps > kMaxConstTaps) {
        throw std::runtime_error("fir: " + std::to_string(num_taps) + " taps, more than the "
                                 + std::to_string(kMaxConstTaps)
                                 + " that fit in constant memory; use naive");
    }
    // Device-to-device into the constant bank, in stream order, so FIRs with different taps can
    // follow each other in one pipeline.
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_taps, taps, static_cast<std::size_t>(num_taps)
                                                         * sizeof(float),
                                       0, cudaMemcpyDeviceToDevice, stream));
}

void fir_launch(FirImpl impl, const cf32* x, const float* taps, cf32* y, int num_samples,
                int num_taps, cudaStream_t stream)
{
    if (num_samples == 0) {
        return;
    }
    switch (impl) {
    case FirImpl::Naive: {
        const int grid = (num_samples + kTiledBlock - 1) / kTiledBlock;
        fir_naive_kernel<<<grid, kTiledBlock, 0, stream>>>(x, taps, y, num_samples, num_taps);
        break;
    }
    case FirImpl::Tiled: {
        const int grid = (num_samples + kTiledBlock - 1) / kTiledBlock;
        const std::size_t tile_bytes = fir_tile_bytes(num_taps, kTiledBlock, "tiled");
        fir_tiled_kernel<<<grid, kTiledBlock, tile_bytes, stream>>>(x, taps, y, num_samples,
                                                                    num_taps);
        break;
    }
    case FirImpl::TiledConst: {
        const int grid = (num_samples + kTiledBlock - 1) / kTiledBlock;
        const std::size_t tile_bytes = fir_tile_bytes(num_taps, kTiledBlock, "tiled_const");
        fir_tiled_const_kernel<<<grid, kTiledBlock, tile_bytes, stream>>>(x, y, num_samples,
                                                                          num_taps);
        break;
    }
    case FirImpl::Register:
        launch_fir_register(x, y, num_samples, num_taps, kTiledBlock, stream);
        return;
    }
    CUDA_CHECK(cudaGetLastError());
}

void fir_device(FirImpl impl, const cf32* x, const float* taps, cf32* y, int num_samples,
                int num_taps)
{
    if (num_samples == 0) {
        return;
    }
    if (uses_const_taps(impl)) {
        fir_load_const_taps(taps, num_taps, 0);
    }
    fir_launch(impl, x, taps, y, num_samples, num_taps, 0);
}

// fir_tiled_kernel over several channels that share one set of taps: blockIdx.y picks the
// channel, and within it the block builds the same tile and computes the same outputs as the
// single-channel kernel. A block entirely past its channel's end returns before the barrier,
// which is safe because every thread of that block returns.
__global__ void fir_tiled_batched_kernel(FirBatch batch, const float* taps, int num_taps)
{
    extern __shared__ cf32 tile[];

    const int channel = blockIdx.y;
    const int num_samples = batch.num_samples[channel];
    const int block_start = blockIdx.x * blockDim.x;
    if (block_start >= num_samples) {
        return;
    }
    const cf32* x = batch.x[channel];
    const int halo = num_taps - 1;
    const int tile_len = halo + blockDim.x;
    const int tile_start = block_start - halo;

    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        const int m = tile_start + i;
        if (m >= 0 && m < num_samples) {
            tile[i] = x[m];
        } else {
            tile[i].re = 0.0f;
            tile[i].im = 0.0f;
        }
    }
    __syncthreads();

    const int n = block_start + threadIdx.x;
    if (n >= num_samples) {
        return;
    }
    const int newest = halo + threadIdx.x;
    float re = 0.0f;
    float im = 0.0f;
    for (int k = 0; k < num_taps; ++k) {
        const float tap = taps[k];
        const cf32 sample = tile[newest - k];
        re += tap * sample.re;
        im += tap * sample.im;
    }
    batch.y[channel][n].re = re;
    batch.y[channel][n].im = im;
}

// The same with the taps in constant memory, as fir_tiled_const_kernel.
__global__ void fir_tiled_const_batched_kernel(FirBatch batch, int num_taps)
{
    extern __shared__ cf32 tile[];

    const int channel = blockIdx.y;
    const int num_samples = batch.num_samples[channel];
    const int block_start = blockIdx.x * blockDim.x;
    if (block_start >= num_samples) {
        return;
    }
    const cf32* x = batch.x[channel];
    const int halo = num_taps - 1;
    const int tile_len = halo + blockDim.x;
    const int tile_start = block_start - halo;

    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        const int m = tile_start + i;
        if (m >= 0 && m < num_samples) {
            tile[i] = x[m];
        } else {
            tile[i].re = 0.0f;
            tile[i].im = 0.0f;
        }
    }
    __syncthreads();

    const int n = block_start + threadIdx.x;
    if (n >= num_samples) {
        return;
    }
    const int newest = halo + threadIdx.x;
    float re = 0.0f;
    float im = 0.0f;
    for (int k = 0; k < num_taps; ++k) {
        const float tap = c_taps[k];
        const cf32 sample = tile[newest - k];
        re += tap * sample.re;
        im += tap * sample.im;
    }
    batch.y[channel][n].re = re;
    batch.y[channel][n].im = im;
}

void fir_tiled_batched_device(const cf32* const* x, cf32* const* y, const int* num_samples,
                              int count, const float* taps, int num_taps)
{
    FirBatch batch;
    const int longest = make_fir_batch(x, y, num_samples, count, batch);
    if (longest == 0) {
        return;
    }
    // One grid row per channel along y; along x, the single-channel launch for the longest one.
    const std::size_t tile_bytes = fir_tile_bytes(num_taps, kTiledBlock, "tiled");
    const dim3 grid((longest + kTiledBlock - 1) / kTiledBlock, count);
    fir_tiled_batched_kernel<<<grid, kTiledBlock, tile_bytes>>>(batch, taps, num_taps);
    CUDA_CHECK(cudaGetLastError());
}

void fir_tiled_const_batched_device(const cf32* const* x, cf32* const* y,
                                    const int* num_samples, int count, const float* taps,
                                    int num_taps)
{
    FirBatch batch;
    const int longest = make_fir_batch(x, y, num_samples, count, batch);
    if (longest == 0) {
        return;
    }
    const std::size_t tile_bytes = fir_tile_bytes(num_taps, kTiledBlock, "tiled_const");
    fir_load_const_taps(taps, num_taps, 0);
    const dim3 grid((longest + kTiledBlock - 1) / kTiledBlock, count);
    fir_tiled_const_batched_kernel<<<grid, kTiledBlock, tile_bytes>>>(batch, num_taps);
    CUDA_CHECK(cudaGetLastError());
}

// fir_register_kernel over several channels that share one set of taps, as the batched tiled
// kernels: blockIdx.y picks the channel, and blocks past its end return before the barrier.
template <int R>
__global__ void fir_register_batched_kernel(FirBatch batch, int num_taps)
{
    const int channel = blockIdx.y;
    const int num_samples = batch.num_samples[channel];
    if (blockIdx.x * blockDim.x * R >= num_samples) {
        return;
    }
    fir_register_block<R>(batch.x[channel], batch.y[channel], num_samples, num_taps);
}

void fir_register_batched_device(const cf32* const* x, cf32* const* y, const int* num_samples,
                                 int count, const float* taps, int num_taps)
{
    FirBatch batch;
    const int longest = make_fir_batch(x, y, num_samples, count, batch);
    if (longest == 0) {
        return;
    }
    const std::size_t bytes = fir_register_bytes(num_taps, kTiledBlock);
    fir_load_const_taps(taps, num_taps, 0);
    const int outputs = register_outputs(num_taps);
    const int per_block = kTiledBlock * outputs;
    const dim3 grid((longest + per_block - 1) / per_block, count);
    const int shared = static_cast<int>(bytes);
    if (outputs == 8) {
        if (bytes > kDefaultSharedBytes) {
            CUDA_CHECK(cudaFuncSetAttribute(fir_register_batched_kernel<8>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
        }
        fir_register_batched_kernel<8><<<grid, kTiledBlock, bytes>>>(batch, num_taps);
    } else {
        if (bytes > kDefaultSharedBytes) {
            CUDA_CHECK(cudaFuncSetAttribute(fir_register_batched_kernel<4>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
        }
        fir_register_batched_kernel<4><<<grid, kTiledBlock, bytes>>>(batch, num_taps);
    }
    CUDA_CHECK(cudaGetLastError());
}

// --- implementation table ------------------------------------------------------------------

const std::vector<Impl<FirFn>>& fir_impls()
{
    static const std::vector<Impl<FirFn>> impls = {
        {"naive", "one thread per output sample, global memory only", fir_naive},
        {"tiled", "samples staged in a shared-memory tile with a halo", fir_tiled},
        {"tiled_const", "tiled, with the taps in constant memory", fir_tiled_const},
        {"register", "tiled_const, 4 or 8 outputs per thread from an I/Q-split tile", fir_register},
    };
    return impls;
}
