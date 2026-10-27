#pragma once

// IR passes and the scheduler. Each pass takes a Graph and returns a Graph, keeping the
// --emit=ir dump valid, and can be turned off from the command line so its effect can be
// measured (compiler/AGENTS.md).

#include "sigflow/ir.hpp"

#include <vector>

namespace sigflow {

struct Schedule {
    // Live nodes in launch order: every node comes after all of its inputs.
    std::vector<NodeId> order;
    // Live nodes that could not be scheduled because they are on or behind a cycle, in id
    // order. Empty for any graph built from a program sema accepted.
    std::vector<NodeId> unscheduled;
};

// Topological sort with Kahn's algorithm. Whenever more than one node is ready, the one written
// first in the source (Graph::before) goes first, so the order is deterministic and reads like
// the program.
Schedule schedule(const Graph& graph);

}  // namespace sigflow
