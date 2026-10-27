#include "sigflow/passes.hpp"

#include <set>

namespace sigflow {

Schedule schedule(const Graph& graph)
{
    const std::size_t count = graph.nodes.size();
    const auto live = [&](NodeId id) { return graph.live[static_cast<std::size_t>(id)]; };

    // In-degree counts every input edge, so xcorr(a, a) waits on two edges from a, and `users`
    // lists a user once per edge to match.
    std::vector<int> in_degree(count, 0);
    std::vector<std::vector<NodeId>> users(count);
    for (const Node& node : graph.nodes) {
        if (!live(node.id)) {
            continue;
        }
        for (NodeId input : node.inputs) {
            if (live(input)) {
                ++in_degree[static_cast<std::size_t>(node.id)];
                users[static_cast<std::size_t>(input)].push_back(node.id);
            }
        }
    }

    // The ready set, ordered by source position: begin() is always the next node to launch.
    const auto earlier = [&graph](NodeId a, NodeId b) { return graph.before(a, b); };
    std::set<NodeId, decltype(earlier)> ready(earlier);
    for (const Node& node : graph.nodes) {
        if (live(node.id) && in_degree[static_cast<std::size_t>(node.id)] == 0) {
            ready.insert(node.id);
        }
    }

    Schedule result;
    while (!ready.empty()) {
        const NodeId next = *ready.begin();
        ready.erase(ready.begin());
        result.order.push_back(next);
        for (NodeId user : users[static_cast<std::size_t>(next)]) {
            if (--in_degree[static_cast<std::size_t>(user)] == 0) {
                ready.insert(user);
            }
        }
    }

    // Anything live that never became ready is on a cycle or downstream of one.
    for (const Node& node : graph.nodes) {
        if (live(node.id) && in_degree[static_cast<std::size_t>(node.id)] > 0) {
            result.unscheduled.push_back(node.id);
        }
    }
    return result;
}

}  // namespace sigflow
