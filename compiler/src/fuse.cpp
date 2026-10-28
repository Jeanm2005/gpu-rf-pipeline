#include "sigflow/passes.hpp"

#include <algorithm>
#include <string>

namespace sigflow {

namespace {

// True when `to` can be reached from `from` along edges (input -> user) through live nodes.
bool reaches(const Graph& graph, const std::vector<std::vector<NodeId>>& users, NodeId from,
             NodeId to)
{
    std::vector<bool> seen(graph.nodes.size(), false);
    std::vector<NodeId> stack = {from};
    while (!stack.empty()) {
        const NodeId id = stack.back();
        stack.pop_back();
        if (id == to) {
            return true;
        }
        if (seen[static_cast<std::size_t>(id)]) {
            continue;
        }
        seen[static_cast<std::size_t>(id)] = true;
        for (NodeId user : users[static_cast<std::size_t>(id)]) {
            stack.push_back(user);
        }
    }
    return false;
}

// True when the graph of launches is acyclic, with `unit_of` mapping each node to its launch
// (itself, or the representative of its group): Kahn's algorithm must consume every unit.
bool launches_acyclic(const Graph& graph, const std::vector<NodeId>& unit_of)
{
    const std::size_t count = graph.nodes.size();
    std::vector<int> in_degree(count, 0);
    std::vector<std::vector<NodeId>> users(count);
    std::vector<bool> is_unit(count, false);
    for (const Node& node : graph.nodes) {
        if (!graph.live[static_cast<std::size_t>(node.id)]) {
            continue;
        }
        const NodeId unit = unit_of[static_cast<std::size_t>(node.id)];
        is_unit[static_cast<std::size_t>(unit)] = true;
        for (NodeId input : node.inputs) {
            const NodeId from = unit_of[static_cast<std::size_t>(input)];
            if (from != unit) {
                ++in_degree[static_cast<std::size_t>(unit)];
                users[static_cast<std::size_t>(from)].push_back(unit);
            }
        }
    }
    std::vector<NodeId> ready;
    std::size_t units = 0;
    for (std::size_t i = 0; i < count; ++i) {
        units += is_unit[i] ? 1 : 0;
        if (is_unit[i] && in_degree[i] == 0) {
            ready.push_back(static_cast<NodeId>(i));
        }
    }
    std::size_t done = 0;
    while (!ready.empty()) {
        const NodeId unit = ready.back();
        ready.pop_back();
        ++done;
        for (NodeId user : users[static_cast<std::size_t>(unit)]) {
            if (--in_degree[static_cast<std::size_t>(user)] == 0) {
                ready.push_back(user);
            }
        }
    }
    return done == units;
}

}  // namespace

Graph fuse_kernels(Graph graph)
{
    const std::size_t count = graph.nodes.size();
    std::vector<std::vector<NodeId>> users(count);
    std::vector<NodeId> firs;
    for (const Node& node : graph.nodes) {
        if (!graph.live[static_cast<std::size_t>(node.id)]) {
            continue;
        }
        if (node.op == Op::Fir) {
            firs.push_back(node.id);
        }
        for (NodeId input : node.inputs) {
            users[static_cast<std::size_t>(input)].push_back(node.id);
        }
    }
    std::sort(firs.begin(), firs.end(),
              [&graph](NodeId a, NodeId b) { return graph.before(a, b); });

    // Greedy, in source order: each FIR joins the first group it is compatible with, or starts
    // a new one. Compatible means the same taps file, room in the group, no path between it and
    // any member (a member would have to finish before another starts), and no cycle among the
    // launches once it joins (group {a, b} and group {c, d} with a -> c and d -> b would each
    // have to run before the other).
    std::vector<NodeId> unit_of(count);
    for (std::size_t i = 0; i < count; ++i) {
        unit_of[i] = static_cast<NodeId>(i);
    }
    std::vector<std::vector<NodeId>> groups;
    for (NodeId fir : firs) {
        const std::string& taps = graph.nodes[static_cast<std::size_t>(fir)].taps;
        bool placed = false;
        for (std::vector<NodeId>& group : groups) {
            const Node& head = graph.nodes[static_cast<std::size_t>(group.front())];
            if (head.taps != taps || static_cast<int>(group.size()) >= kMaxFusedFirs) {
                continue;
            }
            const bool independent = std::none_of(group.begin(), group.end(), [&](NodeId m) {
                return reaches(graph, users, m, fir) || reaches(graph, users, fir, m);
            });
            if (!independent) {
                continue;
            }
            unit_of[static_cast<std::size_t>(fir)] = group.front();
            if (!launches_acyclic(graph, unit_of)) {
                unit_of[static_cast<std::size_t>(fir)] = fir;
                continue;
            }
            group.push_back(fir);
            placed = true;
            break;
        }
        if (!placed) {
            groups.push_back({fir});
        }
    }

    int next = 0;
    for (const std::vector<NodeId>& group : groups) {
        if (group.size() < 2) {
            continue;  // nothing to fuse with
        }
        for (NodeId id : group) {
            graph.nodes[static_cast<std::size_t>(id)].group = next;
        }
        ++next;
    }
    return graph;
}

}  // namespace sigflow
