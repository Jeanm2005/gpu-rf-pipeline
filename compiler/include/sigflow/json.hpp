#pragma once

// --emit=json: the IR and everything the passes found about it, for the DAG viewer (viewer/).
//
//   {
//     "passes":   ["build", "dce", "fuse"],      the passes that ran, in order
//     "nodes":    [{"id", "op", "line", "column", "live", ...}],
//     "edges":    [{"from", "to", "input"}],
//     "schedule": [ids in launch order],
//     "slots":    number of buffer slots
//   }
//
// "nodes" holds every node the IR builder made, so the viewer can draw the graph before and
// after each pass: a node a pass removed has "live": false and "removed_by" naming the pass.
// Live value nodes also carry "step", "live_range" ([def, last use]), "slot", and "bytes" when
// the size is known. Output is deterministic: nodes and edges are in id order.

#include "sigflow/ir.hpp"
#include "sigflow/liveness.hpp"
#include "sigflow/passes.hpp"

#include <ostream>
#include <vector>

namespace sigflow {

// `graph` is the IR after the passes: dead-node elimination when `dce_ran`, then kernel fusion
// when `fuse_ran`, which removes no nodes but sets each fused node's "group". It still holds
// every node the builder made, with removed ones not live.
void print_json(const Graph& graph, bool dce_ran, bool fuse_ran, const Schedule& schedule,
                const Liveness& liveness, const std::vector<long long>& bytes, std::ostream& out);

}  // namespace sigflow
