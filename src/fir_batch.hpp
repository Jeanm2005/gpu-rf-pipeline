#pragma once

// Kernel parameter for the batched FIR kernels (fused.cu, fir.cu): up to kMaxFirBatch channels
// that share one set of taps, passed by value as a single kernel argument (160 bytes, well under
// the 4 KB parameter limit), so a launch needs no device-side pointer table.

#include "device_ops.hpp"

struct FirBatch {
    const cf32* x[kMaxFirBatch];
    cf32* y[kMaxFirBatch];
    int num_samples[kMaxFirBatch];
};

// Fills a FirBatch from the host-side arrays of a batched call, checking the channel count, and
// returns the longest channel's length (the grid has to cover it).
int make_fir_batch(const cf32* const* x, cf32* const* y, const int* num_samples, int count,
                   FirBatch& batch);

// fir.cu: the tiled variants' batched launches, dispatched from fir_batched_device (fused.cu).
void fir_tiled_batched_device(const cf32* const* x, cf32* const* y, const int* num_samples,
                              int count, const float* taps, int num_taps);
void fir_tiled_const_batched_device(const cf32* const* x, cf32* const* y,
                                    const int* num_samples, int count, const float* taps,
                                    int num_taps);
