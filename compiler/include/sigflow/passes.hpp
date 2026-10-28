#pragma once

// IR passes and the scheduler. Each pass takes a Graph and returns a Graph, keeping the
// --emit=ir dump valid, and can be turned off from the command line so its effect can be
// measured (compiler/AGENTS.md).

#include "sigflow/ir.hpp"

#include <vector>

namespace sigflow {

// Dead-node elimination (turned off by --no-dce): every node that no Emit depends on, directly
// or through other nodes, stops being live. Node ids are not renumbered.
Graph eliminate_dead_nodes(Graph graph);

// Kernel fusion (turned off by --no-fuse): puts FIR nodes with the same taps file into one
// fusion group when none of them depends on another, directly or through other nodes, and
// grouping them keeps the graph of launches acyclic. A group runs as one batched kernel launch
// (fir_naive_batched_device in ../src/device_ops.hpp), at most kMaxFusedFirs nodes per group.
// Groups are numbered from 0 in the source order of their first node.
constexpr int kMaxFusedFirs = 8;
Graph fuse_kernels(Graph graph);

struct Schedule {
    // Live nodes in launch order: every node comes after all of its inputs, and the nodes of a
    // fusion group are consecutive.
    std::vector<NodeId> order;
    // The launch each entry of `order` belongs to: launch[i] == i, except that every node of a
    // fusion group shares the launch number of the group's first node.
    std::vector<int> launch;
    // Live nodes that could not be scheduled because they are on or behind a cycle, in id
    // order. Empty for any graph built from a program sema accepted.
    std::vector<NodeId> unscheduled;
};

// Topological sort with Kahn's algorithm over launches: a node on its own, or a whole fusion
// group, which becomes ready once every input from outside the group is scheduled. Whenever more
// than one launch is ready, the one written first in the source (Graph::before on its earliest
// node) goes first, so the order is deterministic and reads like the program.
Schedule schedule(const Graph& graph);

}  // namespace sigflow
