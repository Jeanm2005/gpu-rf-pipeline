#include "sigflow/liveness.hpp"

#include <algorithm>
#include <filesystem>
#include <set>

namespace sigflow {

Liveness analyze_liveness(const Graph& graph, const Schedule& schedule, bool reuse)
{
    const std::size_t count = graph.nodes.size();

    // Step of every scheduled node, then the last step that reads each value.
    std::vector<int> step_of(count, -1);
    for (std::size_t step = 0; step < schedule.order.size(); ++step) {
        step_of[static_cast<std::size_t>(schedule.order[step])] = static_cast<int>(step);
    }
    std::vector<int> last_use(step_of);
    for (NodeId id : schedule.order) {
        for (NodeId input : graph.nodes[static_cast<std::size_t>(id)].inputs) {
            int& last = last_use[static_cast<std::size_t>(input)];
            last = std::max(last, step_of[static_cast<std::size_t>(id)]);
        }
    }

    Liveness result;
    result.slot_of.assign(count, -1);

    // Linear scan over the schedule. `expiring` holds (last use, slot) of values in a slot; a
    // value whose last use is before the current step has released its slot.
    std::set<int> free_slots;
    std::set<std::pair<int, int>> expiring;
    for (NodeId id : schedule.order) {
        const Node& node = graph.nodes[static_cast<std::size_t>(id)];
        if (node.op == Op::Emit) {
            continue;
        }
        const int step = step_of[static_cast<std::size_t>(id)];
        while (!expiring.empty() && expiring.begin()->first < step) {
            free_slots.insert(expiring.begin()->second);
            expiring.erase(expiring.begin());
        }

        int slot = result.slot_count;
        if (reuse && !free_slots.empty()) {
            slot = *free_slots.begin();
            free_slots.erase(free_slots.begin());
        } else {
            ++result.slot_count;
        }
        const int last = last_use[static_cast<std::size_t>(id)];
        expiring.insert({last, slot});
        result.slot_of[static_cast<std::size_t>(id)] = slot;
        result.ranges.push_back(LiveRange{id, step, last, slot});
    }
    return result;
}

std::vector<long long> value_bytes(const Graph& graph, const std::string& input_dir)
{
    namespace fs = std::filesystem;
    constexpr long long kCf32 = 8;  // bytes per complex float32 sample
    constexpr long long kF32 = 4;

    // Sizes are computed in id order: inputs always have smaller ids (see build_ir()).
    std::vector<long long> samples(graph.nodes.size(), -1);
    std::vector<long long> bytes(graph.nodes.size(), -1);
    for (const Node& node : graph.nodes) {
        const auto in = [&](std::size_t i) {
            return samples[static_cast<std::size_t>(node.inputs[i])];
        };
        long long n = -1;
        switch (node.op) {
        case Op::Capture: {
            std::error_code ec;
            const std::uintmax_t size = fs::file_size(fs::path(input_dir) / node.path, ec);
            if (!ec && size % kCf32 == 0) {
                n = static_cast<long long>(size) / kCf32;
            }
            break;
        }
        case Op::Fir:
        case Op::Magnitude:
            n = in(0);
            break;
        case Op::Fft:
            n = in(0) < 0 ? -1 : (in(0) + node.n - 1) / node.n * node.n;
            break;
        case Op::Xcorr:
            n = in(0) < 0 || in(1) < 0 ? -1 : in(0) + in(1) - 1;
            break;
        case Op::Emit:
            bytes[static_cast<std::size_t>(node.id)] = 0;
            continue;
        }
        samples[static_cast<std::size_t>(node.id)] = n;
        const long long width = node.type.element == ElementType::Cf32 ? kCf32 : kF32;
        bytes[static_cast<std::size_t>(node.id)] = n < 0 ? -1 : n * width;
    }
    return bytes;
}

MemoryPlan plan_memory(const Liveness& liveness, const std::vector<long long>& bytes)
{
    MemoryPlan plan;
    int steps = 0;
    for (const LiveRange& range : liveness.ranges) {
        if (bytes[static_cast<std::size_t>(range.node)] < 0) {
            return plan;  // unknown sizes: nothing can be planned
        }
        steps = std::max(steps, range.last_use + 1);
    }

    std::vector<long long> slot_size(static_cast<std::size_t>(liveness.slot_count), 0);
    std::vector<long long> live_at(static_cast<std::size_t>(steps), 0);
    plan.no_reuse = 0;
    for (const LiveRange& range : liveness.ranges) {
        const long long size = bytes[static_cast<std::size_t>(range.node)];
        plan.no_reuse += size;
        long long& slot = slot_size[static_cast<std::size_t>(range.slot)];
        slot = std::max(slot, size);
        for (int step = range.def; step <= range.last_use; ++step) {
            live_at[static_cast<std::size_t>(step)] += size;
        }
    }
    plan.reuse = 0;
    for (long long size : slot_size) {
        plan.reuse += size;
    }
    plan.live_peak = live_at.empty() ? 0 : *std::max_element(live_at.begin(), live_at.end());
    return plan;
}

void print_liveness(const Graph& graph, const Liveness& liveness,
                    const std::vector<long long>& bytes, const std::string& input_dir,
                    std::ostream& out)
{
    for (const LiveRange& range : liveness.ranges) {
        const Node& node = graph.nodes[static_cast<std::size_t>(range.node)];
        const long long size = bytes[static_cast<std::size_t>(range.node)];
        out << "%" << range.node << " " << op_name(node.op) << ": live " << range.def << ".."
            << range.last_use << ", slot " << range.slot << ", "
            << (size < 0 ? std::string("?") : std::to_string(size)) << " bytes";
        if (!node.name.empty()) {
            out << "   # " << node.name;
        }
        out << "\n";
    }
    out << liveness.slot_count << " slots for " << liveness.ranges.size() << " values\n";

    const MemoryPlan plan = plan_memory(liveness, bytes);
    if (plan.reuse < 0) {
        out << "planned GPU memory: unknown (a capture file is missing under '" << input_dir
            << "'; see --input-dir)\n";
        return;
    }
    out << "planned GPU memory: " << plan.no_reuse << " bytes without reuse, " << plan.reuse
        << " with reuse, " << plan.live_peak << " live at the busiest step\n";
}

}  // namespace sigflow
