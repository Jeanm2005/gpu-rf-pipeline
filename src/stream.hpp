#pragma once

// Streaming FIR: a capture processed in fixed-size chunks the way live radio data arrives,
// with transfers of one chunk overlapping the kernel of another.
//
// `depth` slots form a ring. Each slot has a pinned host input and output buffer, a device
// input and output buffer, and its own CUDA stream. Chunk i goes to slot i % depth: the host
// waits until that slot's previous chunk is done, copies the new samples into its pinned input
// buffer (where a receiver would write them), and enqueues upload, kernel, and download on the
// slot's stream. With depth 1 every step waits for the previous one; with depth >= 2 the copy
// engines and the SMs work on different chunks at the same time.
//
// The FIR needs num_taps - 1 samples of history before each chunk. Every chunk after the first
// is uploaded together with that history (re-read from the input), the kernel runs over history
// plus chunk, and only the chunk's own outputs are downloaded, so the result is identical to
// filtering the whole capture at once.
//
// With registered memory, the input and output arrays themselves are page-locked
// (cudaHostRegister) for the run, so each chunk is copied straight from the input and straight
// into the output, with no host-side copies through the ring: the case of a receiver that
// writes into pinned memory. That leaves the GPU pipeline alone on the clock.
//
// With zero_copy, the pinned buffers are also mapped into the GPU's address space and the
// kernel reads its input from host memory and writes its output there, with no copies at all.
// On a discrete GPU every such access crosses PCIe; on a Jetson, where CPU and GPU share one
// DRAM, it is the way to avoid copying within the same memory (docs/JETSON.md).

#include "device_ops.hpp"
#include "types.hpp"

#include <string>
#include <vector>

enum class StreamMemory {
    Copy,        // host copies into and out of the pinned ring, async copies to the device
    Registered,  // input and output page-locked in place, async copies straight to and from them
    ZeroCopy,    // kernel reads and writes mapped pinned memory, no copies to the device
};

struct StreamConfig {
    FirImpl impl = FirImpl::Tiled;
    int chunk = 1 << 20;  // samples per chunk
    int depth = 3;        // slots in the ring (and CUDA streams)
    StreamMemory memory = StreamMemory::Copy;
};

// "copy", "registered", or "zero-copy" (rfgpu stream --memory); false for anything else.
bool parse_stream_memory(const std::string& name, StreamMemory& memory);

struct StreamTiming {
    double total_ms = 0.0;  // wall clock from the first chunk to the last output on the host
    int chunks = 0;
};

// Output length x.size(). Allocation and the tap upload happen before the clock starts.
std::vector<cf32> fir_stream(const std::vector<cf32>& x, const std::vector<float>& taps,
                             const StreamConfig& config, StreamTiming& timing);
