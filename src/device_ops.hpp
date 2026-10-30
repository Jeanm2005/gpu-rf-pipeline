#pragma once

// Device-pointer entry points for the GPU operations, for callers that keep their data on the
// GPU between operations (the code sigflowc generates with --emit=cuda). Each one only enqueues
// work on the default stream: no allocation, no host transfers, no synchronization. Launch
// configurations live here, next to their launch sites, and the host wrappers in ops.hpp call
// these same functions where they exist, so both paths run identical launches.

#include "device_buffer.hpp"
#include "fft_plan.hpp"
#include "types.hpp"

#include <cstddef>
#include <cufft.h>

// --- FIR (fir.cu, fused.cu) ----------------------------------------------------------------

// The FIR variants, as rfgpu's --impl names them: naive, tiled, tiled_const.
enum class FirImpl { Naive, Tiled, TiledConst };

// Parses an --impl name; returns false for an unknown one.
bool parse_fir_impl(const char* name, FirImpl& impl);

// y[n] = sum_k taps[k] * x[n - k] for n < num_samples, with `taps` in GPU memory. TiledConst
// first copies the taps into constant memory (at most 8192), which is part of the enqueued
// work. Tiled and TiledConst throw if the halo does not fit in one block's shared memory.
void fir_device(FirImpl impl, const cf32* x, const float* taps, cf32* y, int num_samples,
                int num_taps);

// The naive variant on its own (used by rfgpu's host wrapper).
void fir_naive_device(const cf32* x, const float* taps, cf32* y, int num_samples, int num_taps);

// The same FIR on `count` channels that share one set of taps, in a single launch: channel c
// filters x[c] (num_samples[c] samples) into y[c]. At most kMaxFirBatch channels, which matches
// the largest group sigflowc's fusion pass forms (kMaxFusedFirs). Results are identical to
// fir_device with the same variant on each channel.
constexpr int kMaxFirBatch = 8;
void fir_batched_device(FirImpl impl, const cf32* const* x, cf32* const* y,
                        const int* num_samples, int count, const float* taps, int num_taps);

// Kept for programs generated before variants were selectable: the naive batched launch.
void fir_naive_batched_device(const cf32* const* x, cf32* const* y, const int* num_samples,
                              int count, const float* taps, int num_taps);

// --- magnitude (magnitude.cu) --------------------------------------------------------------

// mag[n] = |x[n]| for n < num_samples.
void magnitude_device(const cf32* x, float* mag, int num_samples);

// --- cross-correlation (xcorr.cu) ----------------------------------------------------------

// num_a + num_b - 1 lags of b against a into r, one thread per lag.
void xcorr_naive_device(const cf32* a, const cf32* b, cf32* r, int num_a, int num_b);
void xcorr_tiled_device(const cf32* a, const cf32* b, cf32* r, int num_a, int num_b);

// The cuFFT-based cross-correlation for fixed input lengths. Construction is setup: it
// allocates four FFT-length work buffers (zero-padded once) and a cuFFT plan. run() enqueues
// the device-to-device copies into the padded buffers, two forward FFTs, the conjugate
// multiply, the inverse FFT, and the copy of the num_a + num_b - 1 lags into r.
class XcorrFft {
public:
    XcorrFft(int num_a, int num_b);
    void run(const cf32* a, const cf32* b, cf32* r) const;
    // GPU memory this object holds: work buffers plus the plan's work area.
    std::size_t bytes() const;

private:
    int num_a_;
    int num_b_;
    std::size_t fft_len_;
    DeviceBuffer<cf32> padded_a_;
    DeviceBuffer<cf32> padded_b_;
    DeviceBuffer<cf32> spectrum_a_;
    DeviceBuffer<cf32> spectrum_b_;
    FftPlan plan_;
    std::size_t plan_bytes_ = 0;
};

// --- spectrum (spectrum.cu) ----------------------------------------------------------------

// The batched forward FFT `plan` (an FftPlan of nfft points times num_frames) from `in` to
// `out`. `in` must hold num_frames * nfft samples with the padding past the input's end already
// zero.
void spectrum_device(cufftHandle plan, const cf32* in, cf32* out);
