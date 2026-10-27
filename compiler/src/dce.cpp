#include "sigflow/passes.hpp"

namespace sigflow {

Graph eliminate_dead_nodes(Graph graph)
{
    // Walk backwards from every emit; whatever the walk never reaches is dead.
    std::vector<bool> reached(graph.nodes.size(), false);
    std::vector<NodeId> stack;
    for (const Node& node : graph.nodes) {
        if (node.op == Op::Emit && graph.live[static_cast<std::size_t>(node.id)]) {
            stack.push_back(node.id);
        }
    }
    while (!stack.empty()) {
        const NodeId id = stack.back();
        stack.pop_back();
        if (reached[static_cast<std::size_t>(id)]) {
            continue;
        }
        reached[static_cast<std::size_t>(id)] = true;
        for (NodeId input : graph.nodes[static_cast<std::size_t>(id)].inputs) {
            stack.push_back(input);
        }
    }
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        graph.live[i] = graph.live[i] && reached[i];
    }
    return graph;
}

}  // namespace sigflow
