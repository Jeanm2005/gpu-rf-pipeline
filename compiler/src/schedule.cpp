#include "sigflow/passes.hpp"

#include <algorithm>
#include <map>
#include <set>

namespace sigflow {

Schedule schedule(const Graph& graph)
{
    const std::size_t count = graph.nodes.size();
    const auto live = [&](NodeId id) { return graph.live[static_cast<std::size_t>(id)]; };

    // Launch units: every fusion group is one unit, every other node its own. A unit is named by
    // its earliest node in source order, which is also what the ready set sorts by.
    std::map<int, std::vector<NodeId>> groups;  // group -> members, in id order
    for (const Node& node : graph.nodes) {
        if (live(node.id) && node.group >= 0) {
            groups[node.group].push_back(node.id);
        }
    }
    std::vector<NodeId> unit_of(count);
    std::vector<std::vector<NodeId>> members(count);
    for (const Node& node : graph.nodes) {
        unit_of[static_cast<std::size_t>(node.id)] = node.id;
    }
    for (auto& [group, ids] : groups) {
        NodeId head = ids.front();
        for (NodeId id : ids) {
            head = graph.before(id, head) ? id : head;
        }
        for (NodeId id : ids) {
            unit_of[static_cast<std::size_t>(id)] = head;
        }
    }
    for (const Node& node : graph.nodes) {
        if (live(node.id)) {
            members[static_cast<std::size_t>(unit_of[static_cast<std::size_t>(node.id)])]
                .push_back(node.id);
        }
    }

    // In-degree counts every edge into a unit from outside it, so xcorr(a, a) waits on two edges
    // from a, and `users` lists a user unit once per edge to match. Fusion never groups a node
    // with its own input, so an edge inside a unit can only be a self-loop, which is a cycle and
    // counts like any other edge.
    std::vector<int> in_degree(count, 0);
    std::vector<std::vector<NodeId>> users(count);
    for (const Node& node : graph.nodes) {
        if (!live(node.id)) {
            continue;
        }
        const NodeId unit = unit_of[static_cast<std::size_t>(node.id)];
        for (NodeId input : node.inputs) {
            const NodeId from = unit_of[static_cast<std::size_t>(input)];
            if (live(input) && (from != unit || input == node.id)) {
                ++in_degree[static_cast<std::size_t>(unit)];
                users[static_cast<std::size_t>(from)].push_back(unit);
            }
        }
    }

    // The ready set, ordered by source position: begin() is always the next unit to launch.
    const auto earlier = [&graph](NodeId a, NodeId b) { return graph.before(a, b); };
    std::set<NodeId, decltype(earlier)> ready(earlier);
    for (const Node& node : graph.nodes) {
        if (live(node.id) && unit_of[static_cast<std::size_t>(node.id)] == node.id
            && in_degree[static_cast<std::size_t>(node.id)] == 0) {
            ready.insert(node.id);
        }
    }

    Schedule result;
    int launches = 0;
    while (!ready.empty()) {
        const NodeId unit = *ready.begin();
        ready.erase(ready.begin());
        std::vector<NodeId> ids = members[static_cast<std::size_t>(unit)];
        std::sort(ids.begin(), ids.end(), earlier);
        for (NodeId id : ids) {
            result.order.push_back(id);
            result.launch.push_back(launches);
        }
        ++launches;
        for (NodeId user : users[static_cast<std::size_t>(unit)]) {
            if (--in_degree[static_cast<std::size_t>(user)] == 0) {
                ready.insert(user);
            }
        }
    }

    // Anything live that never became ready is on a cycle or downstream of one.
    for (const Node& node : graph.nodes) {
        const NodeId unit = unit_of[static_cast<std::size_t>(node.id)];
        if (live(node.id) && in_degree[static_cast<std::size_t>(unit)] > 0) {
            result.unscheduled.push_back(node.id);
        }
    }
    return result;
}

}  // namespace sigflow
