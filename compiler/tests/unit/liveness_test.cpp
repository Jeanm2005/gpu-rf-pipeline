#include "sigflow/liveness.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace sigflow {
namespace {

// Hand-built graphs, scheduled for real, so the tests exercise the same path as the driver.
// Every node sits on its own source line, in id order, so the schedule is the id order whenever
// dependencies allow it.
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

const LiveRange& range_of(const Liveness& l, NodeId id)
{
    for (const LiveRange& range : l.ranges) {
        if (range.node == id) {
            return range;
        }
    }
    ADD_FAILURE() << "no live range for %" << id;
    return l.ranges.front();
}

TEST(Liveness, RangeRunsFromDefinitionToLastReader)
{
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    const NodeId f = add(g, Op::Fir, {a});
    const NodeId x = add(g, Op::Xcorr, {a, f});
    add(g, Op::Emit, {x});
    const Liveness l = analyze_liveness(g, schedule(g), true);

    ASSERT_EQ(l.ranges.size(), 3u);  // emits hold no value
    EXPECT_EQ(range_of(l, a).def, 0);
    EXPECT_EQ(range_of(l, a).last_use, 2);  // read by xcorr at step 2, not just fir at step 1
    EXPECT_EQ(range_of(l, f).last_use, 2);
    EXPECT_EQ(range_of(l, x).last_use, 3);  // the emit
}

TEST(Liveness, ChainReusesSlotsTwoStepsLater)
{
    // a -> f1 -> f2 -> f3 -> emit. a is last read at step 1 (by f1), so its slot is free from
    // step 2, where f2 takes it; f1 is last read at step 2, so f3 takes its slot at step 3.
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    const NodeId f1 = add(g, Op::Fir, {a});
    const NodeId f2 = add(g, Op::Fir, {f1});
    const NodeId f3 = add(g, Op::Fir, {f2});
    add(g, Op::Emit, {f3});
    const Liveness l = analyze_liveness(g, schedule(g), true);

    EXPECT_EQ(l.slot_count, 2);
    EXPECT_EQ(range_of(l, a).slot, 0);
    EXPECT_EQ(range_of(l, f1).slot, 1);
    EXPECT_EQ(range_of(l, f2).slot, 0);
    EXPECT_EQ(range_of(l, f3).slot, 1);
}

TEST(Liveness, NoValueEverSharesASlotWithAnInputOfItsOwnStep)
{
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    const NodeId b = add(g, Op::Capture, {});
    const NodeId fa = add(g, Op::Fir, {a});
    const NodeId fb = add(g, Op::Fir, {b});
    const NodeId x = add(g, Op::Xcorr, {fa, fb});
    const NodeId s = add(g, Op::Fft, {fa});
    const NodeId m = add(g, Op::Magnitude, {s});
    add(g, Op::Emit, {x});
    add(g, Op::Emit, {m});
    const Liveness l = analyze_liveness(g, schedule(g), true);

    // Two values whose live ranges overlap must be in different slots.
    for (const LiveRange& p : l.ranges) {
        for (const LiveRange& q : l.ranges) {
            if (p.node != q.node && p.def <= q.last_use && q.def <= p.last_use) {
                EXPECT_NE(p.slot, q.slot) << "%" << p.node << " and %" << q.node;
            }
        }
    }
    EXPECT_LT(l.slot_count, static_cast<int>(l.ranges.size()));
}

TEST(Liveness, NoReuseGivesEveryValueItsOwnSlot)
{
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    const NodeId f1 = add(g, Op::Fir, {a});
    const NodeId f2 = add(g, Op::Fir, {f1});
    add(g, Op::Emit, {f2});
    const Liveness l = analyze_liveness(g, schedule(g), false);
    EXPECT_EQ(l.slot_count, 3);
    EXPECT_EQ(l.slot_of[static_cast<std::size_t>(f2)], 2);
}

TEST(Liveness, UnreadValueIsLiveOnlyAtItsOwnStep)
{
    Graph g;  // as with --no-dce: f is never read
    const NodeId a = add(g, Op::Capture, {});
    const NodeId f = add(g, Op::Fir, {a});
    add(g, Op::Emit, {a});
    const Liveness l = analyze_liveness(g, schedule(g), true);
    EXPECT_EQ(range_of(l, f).def, range_of(l, f).last_use);
}

TEST(MemoryPlan, SumsSlotsAndFindsTheBusiestStep)
{
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    const NodeId f1 = add(g, Op::Fir, {a});
    const NodeId f2 = add(g, Op::Fir, {f1});
    add(g, Op::Emit, {f2});
    const Liveness l = analyze_liveness(g, schedule(g), true);  // slots: a, f1, f2 -> 0, 1, 0

    std::vector<long long> bytes = {100, 300, 200, 0};
    const MemoryPlan plan = plan_memory(l, bytes);
    EXPECT_EQ(plan.no_reuse, 600);
    EXPECT_EQ(plan.reuse, 200 + 300);   // slot 0 holds a (100) then f2 (200)
    EXPECT_EQ(plan.live_peak, 500);     // step 2: f1 and f2

    bytes[1] = -1;
    EXPECT_EQ(plan_memory(l, bytes).reuse, -1);
}

TEST(ValueBytes, FollowsEachOperationsOutputLength)
{
    const std::filesystem::path dir = ::testing::TempDir();
    {
        std::ofstream file(dir / "five.cf32", std::ios::binary);
        file << std::string(5 * 8, '\0');  // 5 complex samples
    }
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    g.nodes[static_cast<std::size_t>(a)].path = "five.cf32";
    const NodeId f = add(g, Op::Fir, {a});
    const NodeId s = add(g, Op::Fft, {f});
    g.nodes[static_cast<std::size_t>(s)].n = 4;
    const NodeId m = add(g, Op::Magnitude, {s});
    g.nodes[static_cast<std::size_t>(m)].type.element = ElementType::F32;
    const NodeId x = add(g, Op::Xcorr, {a, f});
    const NodeId e = add(g, Op::Emit, {m});

    const std::vector<long long> bytes = value_bytes(g, dir.string());
    EXPECT_EQ(bytes[static_cast<std::size_t>(a)], 5 * 8);
    EXPECT_EQ(bytes[static_cast<std::size_t>(f)], 5 * 8);
    EXPECT_EQ(bytes[static_cast<std::size_t>(s)], 8 * 8);      // two frames of 4, zero-padded
    EXPECT_EQ(bytes[static_cast<std::size_t>(m)], 8 * 4);      // same count, real float32
    EXPECT_EQ(bytes[static_cast<std::size_t>(x)], 9 * 8);      // 5 + 5 - 1 lags
    EXPECT_EQ(bytes[static_cast<std::size_t>(e)], 0);
}

TEST(ValueBytes, MissingCaptureFileIsUnknownAndSpreads)
{
    Graph g;
    const NodeId a = add(g, Op::Capture, {});
    g.nodes[static_cast<std::size_t>(a)].path = "missing.cf32";
    const NodeId f = add(g, Op::Fir, {a});
    const std::vector<long long> bytes = value_bytes(g, "/nonexistent");
    EXPECT_EQ(bytes[static_cast<std::size_t>(a)], -1);
    EXPECT_EQ(bytes[static_cast<std::size_t>(f)], -1);
}

}  // namespace
}  // namespace sigflow
