#include "sigflow/passes.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace sigflow {
namespace {

// Hand-built graphs (compiler/AGENTS.md: scheduler unit tests don't go through the front end).
// Each node is given a source line; ties are broken by line, then id.
class GraphBuilder {
public:
    NodeId add(int line, std::vector<NodeId> inputs, Op op = Op::Fir)
    {
        Node node;
        node.id = static_cast<NodeId>(graph_.nodes.size());
        node.op = op;
        node.inputs = std::move(inputs);
        node.range.loc = {"t.sf", line, 1};
        graph_.nodes.push_back(node);
        graph_.live.push_back(true);
        return node.id;
    }
    // Adds an edge after the fact, so cycles can be built.
    void add_input(NodeId node, NodeId input)
    {
        graph_.nodes[static_cast<std::size_t>(node)].inputs.push_back(input);
    }
    Graph& graph() { return graph_; }

private:
    Graph graph_;
};

TEST(Schedule, ChainRunsInDependencyOrder)
{
    GraphBuilder b;
    const NodeId x = b.add(3, {});
    const NodeId y = b.add(2, {x});
    const NodeId z = b.add(1, {y});
    const Schedule s = schedule(b.graph());
    EXPECT_EQ(s.order, (std::vector<NodeId>{x, y, z}));  // source order can't override edges
    EXPECT_TRUE(s.unscheduled.empty());
}

TEST(Schedule, IndependentNodesFollowSourceOrderNotIdOrder)
{
    GraphBuilder b;
    const NodeId late = b.add(5, {});
    const NodeId early = b.add(1, {});
    const NodeId middle = b.add(3, {});
    const Schedule s = schedule(b.graph());
    EXPECT_EQ(s.order, (std::vector<NodeId>{early, middle, late}));
}

TEST(Schedule, SameLineTiesFallBackToColumnThenId)
{
    GraphBuilder b;
    const NodeId first = b.add(1, {});
    const NodeId second = b.add(1, {});
    b.graph().nodes[static_cast<std::size_t>(first)].range.loc.column = 9;
    b.graph().nodes[static_cast<std::size_t>(second)].range.loc.column = 2;
    const NodeId third = b.add(1, {});
    b.graph().nodes[static_cast<std::size_t>(third)].range.loc.column = 9;
    const Schedule s = schedule(b.graph());
    EXPECT_EQ(s.order, (std::vector<NodeId>{second, first, third}));
}

TEST(Schedule, DiamondWaitsForBothBranches)
{
    // a feeds bb and c, and d = xcorr(bb, c) must come after both. c is written first.
    GraphBuilder b;
    const NodeId a = b.add(1, {});
    const NodeId c = b.add(2, {a});
    const NodeId bb = b.add(3, {a});
    const NodeId d = b.add(4, {bb, c}, Op::Xcorr);
    const Schedule s = schedule(b.graph());
    EXPECT_EQ(s.order, (std::vector<NodeId>{a, c, bb, d}));
}

TEST(Schedule, DuplicateInputEdgeIsCountedTwice)
{
    GraphBuilder b;
    const NodeId a = b.add(1, {});
    const NodeId x = b.add(2, {a, a}, Op::Xcorr);
    const Schedule s = schedule(b.graph());
    EXPECT_EQ(s.order, (std::vector<NodeId>{a, x}));
    EXPECT_TRUE(s.unscheduled.empty());
}

TEST(Schedule, CycleLeavesItsNodesAndTheirUsersUnscheduled)
{
    GraphBuilder b;
    const NodeId root = b.add(1, {});
    const NodeId p = b.add(2, {root});
    const NodeId q = b.add(3, {p});
    b.add_input(p, q);  // p <-> q
    const NodeId after = b.add(4, {q});
    const NodeId other = b.add(5, {root});
    const Schedule s = schedule(b.graph());
    EXPECT_EQ(s.order, (std::vector<NodeId>{root, other}));
    EXPECT_EQ(s.unscheduled, (std::vector<NodeId>{p, q, after}));
}

TEST(Schedule, SelfLoopIsACycle)
{
    GraphBuilder b;
    const NodeId a = b.add(1, {});
    b.add_input(a, a);
    const Schedule s = schedule(b.graph());
    EXPECT_TRUE(s.order.empty());
    EXPECT_EQ(s.unscheduled, std::vector<NodeId>{a});
}

TEST(Schedule, DeadNodesAreIgnored)
{
    GraphBuilder b;
    const NodeId a = b.add(1, {});
    const NodeId dead = b.add(2, {a});
    const NodeId c = b.add(3, {a});
    b.graph().live[static_cast<std::size_t>(dead)] = false;
    const Schedule s = schedule(b.graph());
    EXPECT_EQ(s.order, (std::vector<NodeId>{a, c}));
}

}  // namespace
}  // namespace sigflow
