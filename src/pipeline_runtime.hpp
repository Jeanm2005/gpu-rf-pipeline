#pragma once

// Runtime support for the programs sigflowc generates with --emit=cuda. A generated program
// keeps every intermediate value on the GPU, in the buffer slots liveness analysis assigned,
// and calls the device-pointer entry points in device_ops.hpp. This class does the parts that
// are the same for every program:
//
//   setup     read the captures, size and allocate the slots, upload the taps, make the cuFFT
//             plans; all before timing starts, as the measurement rules in AGENTS.md require
//   run       the upload / compute / download phases, through Bench (warm-up, CUDA events)
//   report    one JSON object on stdout: launches, GPU memory, and phase times
//
// Command line of a generated program:
//   PROGRAM [--bench N] [--warmup W]    default: a single run with no warm-up, as in rfgpu

#include "bench.hpp"
#include "device_buffer.hpp"
#include "fft_plan.hpp"
#include "types.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sigflow {

// Samples in ceil(n / nfft) whole frames: the length an FFT of `n` samples produces and reads.
inline std::size_t padded_to_frames(std::size_t n, int nfft)
{
    const std::size_t frame = static_cast<std::size_t>(nfft);
    return (n + frame - 1) / frame * frame;
}

class Runtime {
public:
    // `program` names the source in the report. Parses --bench / --warmup, creates the CUDA
    // context, and records how much GPU memory was in use before the program allocated any.
    Runtime(const char* program, bool fused, bool reuse, int argc, char** argv);

    // --- setup -----------------------------------------------------------------------------

    std::vector<cf32> read_capture(const std::string& path);

    // Declares that buffer `slot` must hold at least `bytes`. Call for every value (and for an
    // FFT input, its zero-padded length) before allocate_slots().
    void need(int slot, std::size_t bytes);
    void allocate_slots();
    template <class T>
    T* slot(int index) const
    {
        return reinterpret_cast<T*>(slots_.at(static_cast<std::size_t>(index))->get());
    }

    // Taps from a .f32 file in GPU memory, uploaded once per distinct path. `count` receives
    // the number of taps.
    const float* taps(const std::string& path, int& count);

    // A batched forward-FFT plan, owned by the runtime.
    cufftHandle fft_plan(int nfft, std::size_t num_frames);

    // --- run -------------------------------------------------------------------------------

    // Records that `compute` enqueues one kernel launch or library call of this kind. Only the
    // first pass through `compute` is counted, so the report gives launches per pipeline run.
    void count(const char* kind)
    {
        if (counting_) {
            ++counts_[kind];
        }
    }

    void run(const std::function<void()>& upload, const std::function<void()>& compute,
             const std::function<void()>& download);

    // --- results ---------------------------------------------------------------------------

    void write_cf32(const std::string& path, const std::vector<cf32>& data);
    void write_f32(const std::string& path, const std::vector<float>& data);

    // Prints the JSON report on stdout.
    void report() const;

private:
    std::string program_;
    bool fused_;
    bool reuse_;
    BenchConfig config_;
    std::size_t used_before_ = 0;  // GPU memory in use before setup, from cudaMemGetInfo
    std::size_t used_after_ = 0;   // ... and after it
    std::vector<std::size_t> slot_bytes_;
    std::vector<std::unique_ptr<DeviceBuffer<unsigned char>>> slots_;
    std::map<std::string, std::unique_ptr<DeviceBuffer<float>>> taps_;
    std::vector<std::unique_ptr<FftPlan>> plans_;
    std::size_t taps_bytes_ = 0;
    std::size_t fft_work_bytes_ = 0;
    std::map<std::string, int> counts_;  // launches per pipeline run
    bool counting_ = false;
    bool counted_ = false;
    BenchResult result_;
};

}  // namespace sigflow
