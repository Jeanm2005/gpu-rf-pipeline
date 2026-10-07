#pragma once

// The GPU operations as main.cpp sees them. Each one takes host data, does its one-time setup
// (device allocation, cuFFT plans), passes its upload / compute / download phases to
// Bench::run(), and returns the result as host data.
//
// FIR and cross-correlation keep every implementation variant in a table, selected with
// --impl. To add a variant, write its function in the .cu file and add one row to the table
// there. Older variants stay: they are the baselines in docs/PROFILING_LOG.md.

#include "bench.hpp"
#include "types.hpp"

#include <vector>

template <class Fn>
struct Impl {
    const char* name;     // value of --impl
    const char* summary;  // one line for --help
    Fn fn;
};

// spectrum.cu. Framed forward FFT with cuFFT: ceil(x.size() / nfft) frames of nfft bins, the
// last frame zero-padded, unnormalized.
std::vector<cf32> spectrum_cufft(const std::vector<cf32>& x, int nfft, Bench& bench);

// fir.cu. y[n] = sum_k taps[k] * x[n - k], x[m] = 0 for m < 0. Output length x.size().
using FirFn = std::vector<cf32> (*)(const std::vector<cf32>& x, const std::vector<float>& taps,
                                    Bench& bench);
const std::vector<Impl<FirFn>>& fir_impls();  // first entry is the default

// xcorr.cu. r[k] = sum_n b[n + k] * conj(a[n]) for k = -(Na - 1) ... Nb - 1.
// Output length Na + Nb - 1; index i holds lag i - (Na - 1).
using XcorrFn = std::vector<cf32> (*)(const std::vector<cf32>& a, const std::vector<cf32>& b,
                                      Bench& bench);
const std::vector<Impl<XcorrFn>>& xcorr_impls();  // first entry is the default
