#include "sigflow/json.hpp"

#include <string>

namespace sigflow {

namespace {

std::string json_escape(const std::string& text)
{
    std::string out;
    for (char c : text) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                static const char* hex = "0123456789abcdef";
                out += "\\u00";
                out += hex[(c >> 4) & 0xf];
                out += hex[c & 0xf];
            } else {
                out += c;
            }
        }
    }
    return out;
}

std::string quoted(const std::string& text) { return "\"" + json_escape(text) + "\""; }

}  // namespace

void print_json(const Graph& graph, bool dce_ran, const Schedule& schedule,
                const Liveness& liveness, const std::vector<long long>& bytes, std::ostream& out)
{
    std::vector<int> step_of(graph.nodes.size(), -1);
    for (std::size_t step = 0; step < schedule.order.size(); ++step) {
        step_of[static_cast<std::size_t>(schedule.order[step])] = static_cast<int>(step);
    }
    std::vector<const LiveRange*> range_of(graph.nodes.size(), nullptr);
    for (const LiveRange& range : liveness.ranges) {
        range_of[static_cast<std::size_t>(range.node)] = &range;
    }

    out << "{\n  \"passes\": [\"build\"" << (dce_ran ? ", \"dce\"" : "") << "],\n";

    out << "  \"nodes\": [";
    for (const Node& node : graph.nodes) {
        const std::size_t i = static_cast<std::size_t>(node.id);
        out << (node.id == 0 ? "\n" : ",\n") << "    {\"id\": " << node.id << ", \"op\": \""
            << op_name(node.op) << "\"";
        if (node.op != Op::Emit) {
            out << ", \"type\": " << quoted(type_name(node.type))
                << ", \"rate\": " << quoted(node.type.rate_text);
        }
        if (!node.name.empty()) {
            out << ", \"name\": " << quoted(node.name);
        }
        if (!node.path.empty()) {
            out << ", \"path\": " << quoted(node.path);
        }
        if (node.op == Op::Fir) {
            out << ", \"taps\": " << quoted(node.taps);
        }
        if (node.op == Op::Fft) {
            out << ", \"n\": " << node.n;
        }
        out << ", \"line\": " << node.range.loc.line << ", \"column\": " << node.range.loc.column;
        if (!graph.live[i]) {
            // The builder makes every node live and dead-node elimination is the only pass that
            // removes any, so far.
            out << ", \"live\": false, \"removed_by\": \"dce\"}";
            continue;
        }
        out << ", \"live\": true, \"step\": " << step_of[i];
        if (const LiveRange* range = range_of[i]) {
            out << ", \"live_range\": [" << range->def << ", " << range->last_use
                << "], \"slot\": " << range->slot;
            if (bytes[i] >= 0) {
                out << ", \"bytes\": " << bytes[i];
            }
        }
        out << "}";
    }
    out << "\n  ],\n";

    out << "  \"edges\": [";
    bool first = true;
    for (const Node& node : graph.nodes) {
        for (std::size_t input = 0; input < node.inputs.size(); ++input) {
            out << (first ? "\n" : ",\n") << "    {\"from\": " << node.inputs[input]
                << ", \"to\": " << node.id << ", \"input\": " << input << "}";
            first = false;
        }
    }
    out << "\n  ],\n";

    out << "  \"schedule\": [";
    for (std::size_t step = 0; step < schedule.order.size(); ++step) {
        out << (step == 0 ? "" : ", ") << schedule.order[step];
    }
    out << "],\n  \"slots\": " << liveness.slot_count << "\n}\n";
}

}  // namespace sigflow
