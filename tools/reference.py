#!/usr/bin/env python3
"""NumPy/SciPy CPU reference for every rfgpu operation.

This file is the definition of "correct": each GPU result is compared to the function of the
same name here. Everything is computed in float64, so the reference is more accurate than the
float32 GPU output and the test tolerances only have to cover the GPU's rounding.

Output conventions (the GPU kernels must match these exactly):

  spectrum  The input is cut into consecutive frames of `nfft` samples, no overlap and no
            window. A partial last frame is zero-padded, so there are ceil(N / nfft) frames.
            Each frame is a forward FFT, unnormalized (what cuFFT computes), with bin 0 first.
            Output: num_frames * nfft complex values, frame after frame.

  fir       y[n] = sum_{k=0}^{T-1} taps[k] * x[n - k], with x[m] = 0 for m < 0.
            Causal, zero initial state, real taps. Output length equals input length N, so
            the first T - 1 outputs are the start-up transient.

  xcorr     r[k] = sum_n b[n + k] * conj(a[n]) for every lag k from -(Na - 1) to Nb - 1,
            with samples outside either input taken as zero. Output length Na + Nb - 1, and
            output index i holds lag i - (Na - 1). If b is a delayed by d samples, |r| peaks
            at lag +d.

Usage (mirrors the rfgpu CLI):
    python tools/reference.py spectrum --in tests/fixtures/tone.cf32 --out out/ref_spectrum.cf32 --nfft 1024
    python tools/reference.py fir      --in tests/fixtures/tone.cf32 --taps tests/fixtures/lowpass.f32 --out out/ref_fir.cf32
    python tools/reference.py xcorr    --a tests/fixtures/ch0.cf32 --b tests/fixtures/ch1.cf32 --out out/ref_xcorr.cf32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal


# --- File I/O ------------------------------------------------------------------------------

def read_cf32(path: Path) -> np.ndarray:
    """Read interleaved little-endian float32 [I0, Q0, I1, Q1, ...] as complex128."""
    iq = np.fromfile(path, dtype="<f4")
    if len(iq) % 2 != 0:
        raise ValueError(f"{path}: odd number of float32 values, not interleaved IQ")
    return iq[0::2].astype(np.float64) + 1j * iq[1::2].astype(np.float64)


def write_cf32(path: Path, x: np.ndarray) -> None:
    iq = np.empty(2 * x.size, dtype="<f4")
    iq[0::2] = x.real.ravel()
    iq[1::2] = x.imag.ravel()
    path.parent.mkdir(parents=True, exist_ok=True)
    iq.tofile(path)


def read_f32(path: Path) -> np.ndarray:
    """Read a raw little-endian float32 array (filter taps) as float64."""
    return np.fromfile(path, dtype="<f4").astype(np.float64)


def write_f32(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x.ravel().astype("<f4").tofile(path)


# --- Operations ----------------------------------------------------------------------------

def spectrum(x: np.ndarray, nfft: int) -> np.ndarray:
    """Framed FFT. Returns shape (ceil(len(x) / nfft), nfft), complex128."""
    if nfft < 1:
        raise ValueError("nfft must be at least 1")
    num_frames = -(-len(x) // nfft)  # ceiling division
    padded = np.zeros(num_frames * nfft, dtype=np.complex128)
    padded[: len(x)] = x
    return np.fft.fft(padded.reshape(num_frames, nfft), axis=1)


def magnitude(frames: np.ndarray) -> np.ndarray:
    """|X| of each bin, float64, same shape as the input."""
    return np.abs(frames)


def fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Causal FIR filter with zero initial state. Returns len(x) complex128 samples."""
    if len(taps) < 1:
        raise ValueError("need at least one tap")
    if len(x) == 0:
        return np.zeros(0, dtype=np.complex128)
    # Full convolution has len(x) + len(taps) - 1 samples; the first len(x) are the causal
    # output. (np.convolve rather than scipy.signal.lfilter, which is slow for long filters.)
    return np.convolve(x.astype(np.complex128), taps.astype(np.float64))[: len(x)]


def xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cross-correlation of b against a. Returns len(a) + len(b) - 1 complex128 values."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("both inputs need at least one sample")
    # scipy conjugates its second argument: correlate(b, a)[i] = sum_n b[n + k] * conj(a[n])
    # with k = i - (len(a) - 1).
    return signal.correlate(b.astype(np.complex128), a.astype(np.complex128), mode="full")


def xcorr_lags(num_a: int, num_b: int) -> np.ndarray:
    """Lag of each xcorr output index: -(num_a - 1) ... num_b - 1."""
    return np.arange(-(num_a - 1), num_b)


# --- CLI -----------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="op", required=True)

    p = sub.add_parser("spectrum", help="framed FFT")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nfft", type=int, default=1024)

    p = sub.add_parser("fir", help="FIR filter")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--taps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("xcorr", help="cross-correlation")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.op == "spectrum":
        y = spectrum(read_cf32(args.inp), args.nfft)
    elif args.op == "fir":
        y = fir(read_cf32(args.inp), read_f32(args.taps))
    else:
        y = xcorr(read_cf32(args.a), read_cf32(args.b))
    write_cf32(args.out, y)
    print(f"{args.op}: wrote {y.size} complex samples to {args.out}")


if __name__ == "__main__":
    main()
