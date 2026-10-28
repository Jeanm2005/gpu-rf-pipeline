#pragma once

// Device-pointer entry points for the GPU operations, for callers that keep their data on the
// GPU between operations (the code sigflowc generates with --emit=cuda). Each one only enqueues
// work on the default stream: no allocation, no host transfers, no synchronization. Launch
// configurations live here, next to their launch sites, and the host wrappers in ops.hpp call
// these same functions, so both paths run identical launches.

#include "types.hpp"

#include <cufft.h>

// fir.cu, the naive variant: y[n] = sum_k taps[k] * x[n - k] for n < num_samples.
void fir_naive_device(const cf32* x, const float* taps, cf32* y, int num_samples, int num_taps);

// magnitude.cu: mag[n] = |x[n]| for n < num_samples.
void magnitude_device(const cf32* x, float* mag, int num_samples);

// xcorr.cu, the naive variant: num_a + num_b - 1 lags into r.
void xcorr_naive_device(const cf32* a, const cf32* b, cf32* r, int num_a, int num_b);

// spectrum.cu: the batched forward FFT `plan` (an FftPlan of nfft points times num_frames) from
// `in` to `out`. `in` must hold num_frames * nfft samples with the padding past the input's
// end already zero.
void spectrum_device(cufftHandle plan, const cf32* in, cf32* out);
