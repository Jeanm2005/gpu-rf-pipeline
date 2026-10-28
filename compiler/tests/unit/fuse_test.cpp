#include "sigflow/liveness.hpp"
#include "sigflow/passes.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace sigflow {
namespace {

// Hand-built graphs (compiler/AGENTS.md: fusion unit tests don't go through the front end).
// Every node sits on its own source line, in id order.
class GraphBuilder {
public:
    NodeId add(Op op, std::vector<NodeId> inputs, const std::string& taps = "h.f32")
    {
        Node node;
        node.id = static_cast<NodeId>(graph_.nodes.size());
        node.op = op;
        node.inputs = std::move(inputs);
        node.taps = op == Op::Fir ? taps : "";
        node.range.loc = {"t.sf", node.id + 1, 1};
        graph_.nodes.push_back(node);
        graph_.live.push_back(true);
        return node.id;
    }
    Graph& graph() { return graph_; }

private:
    Graph graph_;
};

int group_of(const Graph& g, NodeId id) { return g.nodes[static_cast<std::size_t>(id)].group; }

TEST(Fuse, SiblingFirsWithTheSameTapsShareAGroup)
{
    GraphBuilder b;
    const NodeId a = b.add(Op::Capture, {});
    const NodeId c = b.add(Op::Capture, {});
    const NodeId fa = b.add(Op::Fir, {a});
    const NodeId fc = b.add(Op::Fir, {c});
    b.add(Op::Xcorr, {fa, fc});
    const Graph g = fuse_kernels(b.graph());
    EXPECT_EQ(group_of(g, fa), 0);
    EXPECT_EQ(group_of(g, fc), 0);
    EXPECT_EQ(group_of(g, a), -1);
}

TEST(Fuse, DifferentTapsAreNotFused)
{
    GraphBuilder b;
    const NodeId a = b.add(Op::Capture, {});
    const NodeId f1 = b.add(Op::Fir, {a}, "low.f32");
    const NodeId f2 = b.add(Op::Fir, {a}, "high.f32");
    const Graph g = fuse_kernels(b.graph());
    EXPECT_EQ(group_of(g, f1), -1);
    EXPECT_EQ(group_of(g, f2), -1);
}

TEST(Fuse, AChainOfFirsIsNotFused)
{
    // f2 reads f1, even through another node: one launch cannot hold both.
    GraphBuilder b;
    const NodeId a = b.add(Op::Capture, {});
    const NodeId f1 = b.add(Op::Fir, {a});
    const NodeId m = b.add(Op::Xcorr, {f1, a});
    const NodeId f2 = b.add(Op::Fir, {m});
    const Graph g = fuse_kernels(b.graph());
    EXPECT_EQ(group_of(g, f1), -1);
    EXPECT_EQ(group_of(g, f2), -1);
}

TEST(Fuse, AGroupThatWouldCreateACycleAmongLaunchesIsRefused)
{
    // A crossing: x1 feeds y2 and y1 feeds x2. {x1, x2} and {y1, y2} are each fusable on their
    // own, but fusing both would need the x launch before the y launch and the reverse. The
    // greedy pass fuses {y1, y2} first (y2 is written before x2) and must then refuse x2.
    GraphBuilder b;
    const NodeId a = b.add(Op::Capture, {});
    const NodeId x1 = b.add(Op::Fir, {a}, "x.f32");
    const NodeId y1 = b.add(Op::Fir, {a}, "y.f32");
    const NodeId y2 = b.add(Op::Fir, {x1}, "y.f32");
    const NodeId x2 = b.add(Op::Fir, {y1}, "x.f32");
    const Graph g = fuse_kernels(b.graph());
    const bool xs = group_of(g, x1) >= 0 && group_of(g, x1) == group_of(g, x2);
    const bool ys = group_of(g, y1) >= 0 && group_of(g, y1) == group_of(g, y2);
    EXPECT_TRUE(xs != ys) << "exactly one of the two pairs can be fused";
    EXPECT_TRUE(schedule(g).unscheduled.empty());
}

TEST(Fuse, GroupsHoldAtMostEightFirs)
{
    GraphBuilder b;
    const NodeId a = b.add(Op::Capture, {});
    std::vector<NodeId> firs;
    for (int i = 0; i < kMaxFusedFirs + 2; ++i) {
        firs.push_back(b.add(Op::Fir, {a}));
    }
    const Graph g = fuse_kernels(b.graph());
    for (int i = 0; i < kMaxFusedFirs; ++i) {
        EXPECT_EQ(group_of(g, firs[static_cast<std::size_t>(i)]), 0);
    }
    EXPECT_EQ(group_of(g, firs[kMaxFusedFirs]), 1);
    EXPECT_EQ(group_of(g, firs[kMaxFusedFirs + 1]), 1);
}

TEST(Fuse, DeadNodesAreLeftAlone)
{
    GraphBuilder b;
    const NodeId a = b.add(Op::Capture, {});
    const NodeId f1 = b.add(Op::Fir, {a});
    const NodeId dead = b.add(Op::Fir, {a});
    b.graph().live[static_cast<std::size_t>(dead)] = false;
    const Graph g = fuse_kernels(b.graph());
    EXPECT_EQ(group_of(g, f1), -1);
    EXPECT_EQ(group_of(g, dead), -1);
}

TEST(FusedSchedule, AGroupIsOneLaunchWithConsecutiveMembers)
{
    // The two FIRs are written apart (lines 3 and 5) with an unrelated capture in between; the
    // group still launches as one unit once both inputs are ready.
    GraphBuilder b;
    const NodeId a = b.add(Op::Capture, {});
    const NodeId c = b.add(Op::Capture, {});
    const NodeId fa = b.add(Op::Fir, {a});
    const NodeId d = b.add(Op::Capture, {});
    const NodeId fc = b.add(Op::Fir, {c});
    b.add(Op::Xcorr, {fa, fc});
    b.add(Op::Fft, {d});
    const Schedule s = schedule(fuse_kernels(b.graph()));
    ASSERT_EQ(s.order.size(), 7u);
    EXPECT_EQ(s.order, (std::vector<NodeId>{a, c, fa, fc, d, 5, 6}));
    EXPECT_EQ(s.launch, (std::vector<int>{0, 1, 2, 2, 3, 4, 5}));
}

TEST(FusedLiveness, NoMemberOfALaunchReusesABufferAnotherMemberReads)
{
    // Unfused, fc could take fa's input slot (a is last read by fa, one step earlier). Fused,
    // fa and fc run in the same kernel, so that would overwrite a while fa still reads it.
    GraphBuilder b;
    const NodeId a = b.add(Op::Capture, {});
    const NodeId c = b.add(Op::Capture, {});
    const NodeId fa = b.add(Op::Fir, {a});
    const NodeId fc = b.add(Op::Fir, {c});
    b.add(Op::Emit, {fa});
    b.add(Op::Emit, {fc});

    const Liveness unfused = analyze_liveness(b.graph(), schedule(b.graph()), true);
    EXPECT_EQ(unfused.slot_of[static_cast<std::size_t>(fc)],
              unfused.slot_of[static_cast<std::size_t>(a)]);

    const Graph g = fuse_kernels(b.graph());
    const Liveness fused = analyze_liveness(g, schedule(g), true);
    for (NodeId member : {fa, fc}) {
        for (NodeId input : {a, c}) {
            EXPECT_NE(fused.slot_of[static_cast<std::size_t>(member)],
                      fused.slot_of[static_cast<std::size_t>(input)]);
        }
    }
}

}  // namespace
}  // namespace sigflow
