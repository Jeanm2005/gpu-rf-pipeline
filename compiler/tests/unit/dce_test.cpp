#include "sigflow/passes.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace sigflow {
namespace {

NodeId add(Graph& g, Op op, std::vector<NodeId> inputs)
{
    Node node;
    node.id = static_cast<NodeId>(g.nodes.size());
    node.op = op;
    node.inputs = std::move(inputs);
    node.range.loc = {"t.sf", node.id + 1, 1};
    g.nodes.push_back(node);
    g.live.push_back(true);
    return node.id;
}

std::vector<NodeId> live_ids(const Graph& g)
{
    std::vector<NodeId> ids;
    for (const Node& node : g.nodes) {
        if (g.live[static_cast<std::size_t>(node.id)]) {
            ids.push_back(node.id);
        }
    }
    return ids;
}

TEST(DeadNodeElimination, KeepsEverythingAnEmitReaches)
{
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    const NodeId b = add(g, Op::Capture, {});
    const NodeId x = add(g, Op::Xcorr, {a, b});
    const NodeId e = add(g, Op::Emit, {x});
    EXPECT_EQ(live_ids(eliminate_dead_nodes(g)), (std::vector<NodeId>{a, b, x, e}));
}

TEST(DeadNodeElimination, DropsUnusedBranchesAndUnusedCaptures)
{
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    const NodeId f = add(g, Op::Fir, {a});
    add(g, Op::Fft, {a});         // only feeds the dead magnitude below
    add(g, Op::Magnitude, {2});
    add(g, Op::Capture, {});      // never used at all
    const NodeId e = add(g, Op::Emit, {f});
    const Graph out = eliminate_dead_nodes(g);
    EXPECT_EQ(live_ids(out), (std::vector<NodeId>{a, f, e}));
    EXPECT_EQ(out.nodes.size(), g.nodes.size());  // ids are kept, not renumbered
}

TEST(DeadNodeElimination, AlreadyDeadEmitsKeepNothingAlive)
{
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    const NodeId e = add(g, Op::Emit, {a});
    g.live[static_cast<std::size_t>(e)] = false;
    EXPECT_TRUE(live_ids(eliminate_dead_nodes(g)).empty());
}

}  // namespace
}  // namespace sigflow
