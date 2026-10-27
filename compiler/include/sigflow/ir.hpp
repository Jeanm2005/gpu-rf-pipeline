#pragma once

// Graph IR: the program as a DAG (directed acyclic graph) of operations. Built from the checked
// AST by build_ir() (src/ir_builder.cpp); every later pass takes a Graph and returns a Graph
// (compiler/AGENTS.md: "IR is the contract"), and no stage after the IR builder reads the AST.

#include "sigflow/ast.hpp"
#include "sigflow/sema.hpp"
#include "sigflow/types.hpp"

#include <ostream>
#include <string>
#include <vector>

namespace sigflow {

enum class Op {
    Capture,    // reads `path`; no inputs
    Fir,        // inputs: signal; attrs: taps
    Fft,        // inputs: signal; attrs: n
    Magnitude,  // inputs: frames
    Xcorr,      // inputs: a, b
    Emit,       // inputs: the value written to `path`; a sink, produces nothing
};

const char* op_name(Op op);

using NodeId = int;

struct Node {
    NodeId id = 0;  // equal to the node's index in Graph::nodes
    Op op = Op::Capture;
    std::vector<NodeId> inputs;
    Type type;          // meaningless for Emit
    std::string name;   // the binding this node defines, or "" for an intermediate value
    std::string path;   // Capture: file read. Emit: file written.
    std::string taps;   // Fir: taps file
    int n = 0;          // Fft: frame size
    // Where the operation was written. The scheduler breaks ties by this (source order).
    SourceRange range;
};

struct Graph {
    // Indexed by NodeId. A pass that removes nodes leaves their ids unused rather than
    // renumbering, so ids in dumps stay comparable before and after the pass: `live[id]` is
    // false for a removed node.
    std::vector<Node> nodes;
    std::vector<bool> live;

    // Source-order comparison used for every deterministic tie-break.
    bool before(NodeId a, NodeId b) const;
};

// Builds the IR for a program that sema accepted (`sema` must come with no errors). Every call
// becomes one node, pipes become ordinary inputs, a name reference becomes an edge to the node of
// the binding it names, and each emit becomes an Emit node. Nodes are created depth-first,
// inputs first, visiting statements in source order, so every input has a smaller id than its
// user. Because a binding may be used before it is defined, id order follows first use rather
// than source order; the scheduler (src/schedule.cpp) produces the source-ordered launch order.
Graph build_ir(const Program& program, const SemaResult& sema);

// Text form used by --emit=ir, one live node per line:
//   %3 = fir %0 taps="fixtures/lowpass.f32" : stream<cf32> @2.4e6   # la
//   emit %7 -> "out/spectrum.f32"
void print_ir(const Graph& graph, std::ostream& out);

// JSON form used by --emit=json, for the DAG viewer: {"nodes": [...], "edges": [...]}.
void print_ir_json(const Graph& graph, std::ostream& out);

}  // namespace sigflow
