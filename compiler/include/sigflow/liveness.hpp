#pragma once

// Liveness analysis and buffer reuse (phase B).
//
// Every value node (everything but Emit) needs a GPU buffer from the step that computes it to
// the last step that reads it. Liveness finds that interval on the schedule; buffer assignment
// then gives each value a buffer slot, handing a slot to a later value once every reader of its
// previous value has run. A slot freed at step t is reused from step t + 1 at the earliest, never
// by the node that reads the old value, so no operation ever writes the buffer it is reading.

#include "sigflow/ir.hpp"
#include "sigflow/passes.hpp"

#include <ostream>
#include <string>
#include <vector>

namespace sigflow {

struct LiveRange {
    NodeId node = 0;
    int def = 0;        // schedule step that writes the value
    int last_use = 0;   // last schedule step that reads it (== def when nothing reads it)
    int slot = 0;       // buffer slot the value lives in
};

struct Liveness {
    std::vector<LiveRange> ranges;  // one per value node, in schedule order
    std::vector<int> slot_of;       // indexed by NodeId; -1 for Emit and dead nodes
    int slot_count = 0;
};

// Live ranges on `schedule` plus a buffer assignment. With `reuse`, a value takes the free slot
// with the lowest index (first fit; sizes are not known at compile time in general, so they play
// no part and the assignment stays a pure function of the program). Without it (--no-reuse),
// every value gets its own slot.
Liveness analyze_liveness(const Graph& graph, const Schedule& schedule, bool reuse);

// Size in bytes of every node's value, indexed by NodeId, or -1 where it cannot be known: a
// capture's size is its file's size, with capture paths resolved against `input_dir`, and every
// other size follows from its inputs (fir keeps the length, fft pads to whole frames, magnitude
// halves the bytes, xcorr gives Na + Nb - 1 samples). Emit nodes get 0.
std::vector<long long> value_bytes(const Graph& graph, const std::string& input_dir);

// Planned GPU memory for one run, in bytes; -1 when any value's size is unknown.
struct MemoryPlan {
    long long no_reuse = -1;   // every value in its own buffer: the sum of all value sizes
    long long reuse = -1;      // one buffer per slot, each as large as the largest value in it
    long long live_peak = -1;  // the most bytes live at any one step: a lower bound for any plan
};

MemoryPlan plan_memory(const Liveness& liveness, const std::vector<long long>& bytes);

// Text form used by --emit=liveness: one line per value in schedule order (live steps, slot,
// bytes), then the slot count and the memory plan. `input_dir` is only named when sizes are
// unknown.
void print_liveness(const Graph& graph, const Liveness& liveness,
                    const std::vector<long long>& bytes, const std::string& input_dir,
                    std::ostream& out);

}  // namespace sigflow
