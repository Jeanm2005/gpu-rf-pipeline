#include "sigflow/ir.hpp"

#include <cassert>
#include <map>
#include <sstream>

namespace sigflow {

namespace {

Op op_for(const std::string& callee)
{
    if (callee == "capture") {
        return Op::Capture;
    }
    if (callee == "fir") {
        return Op::Fir;
    }
    if (callee == "fft") {
        return Op::Fft;
    }
    if (callee == "magnitude") {
        return Op::Magnitude;
    }
    assert(callee == "xcorr" && "sema accepted an operation the IR builder does not know");
    return Op::Xcorr;
}

class IrBuilder {
public:
    IrBuilder(const Program& program, const SemaResult& sema) : program_(program), sema_(sema) {}

    Graph run()
    {
        for (const Stmt& stmt : program_.statements) {
            if (stmt.kind == StmtKind::Binding) {
                binding_node(stmt.name);
            } else {
                Node emit;
                emit.op = Op::Emit;
                emit.inputs = {binding_node(stmt.name)};
                emit.path = stmt.path.text;
                emit.range = stmt.range;
                add(std::move(emit));
            }
        }
        graph_.live.assign(graph_.nodes.size(), true);
        return std::move(graph_);
    }

private:
    NodeId add(Node node)
    {
        node.id = static_cast<NodeId>(graph_.nodes.size());
        graph_.nodes.push_back(std::move(node));
        return graph_.nodes.back().id;
    }

    NodeId binding_node(const std::string& name)
    {
        const auto done = binding_nodes_.find(name);
        if (done != binding_nodes_.end()) {
            return done->second;
        }
        const Stmt& stmt = program_.statements[sema_.bindings.at(name)];
        const NodeId id = expr_node(*stmt.value);
        // `d = c` adds no node; `c`'s node keeps its own name.
        if (graph_.nodes[static_cast<std::size_t>(id)].name.empty()) {
            graph_.nodes[static_cast<std::size_t>(id)].name = name;
        }
        binding_nodes_.emplace(name, id);
        return id;
    }

    NodeId expr_node(const Expr& expr)
    {
        switch (expr.kind) {
        case ExprKind::Name:
            return binding_node(static_cast<const NameExpr&>(expr).name);
        case ExprKind::Call:
            return call_node(static_cast<const CallExpr&>(expr), nullptr, expr);
        case ExprKind::Pipe: {
            const auto& pipe = static_cast<const PipeExpr&>(expr);
            return call_node(*pipe.call, pipe.input.get(), expr);
        }
        case ExprKind::Literal:
            break;
        }
        assert(false && "sema rejects literals used as signals");
        return -1;
    }

    // `typed` is the expression sema recorded the result type on: the call itself, or the pipe
    // that feeds it.
    NodeId call_node(const CallExpr& call, const Expr* piped, const Expr& typed)
    {
        Node node;
        node.op = op_for(call.callee);
        node.type = sema_.types.at(&typed);
        node.range = call.callee_range;

        std::vector<const Expr*> positional;
        if (piped != nullptr) {
            positional.push_back(piped);
        }
        for (const ExprPtr& arg : call.args) {
            positional.push_back(arg.get());
        }

        if (node.op == Op::Capture) {
            node.path = static_cast<const LiteralExpr&>(*positional[0]).value.text;
        } else {
            for (const Expr* arg : positional) {
                node.inputs.push_back(expr_node(*arg));
            }
        }
        for (const KeywordArg& kwarg : call.kwargs) {
            if (kwarg.name == "taps") {
                node.taps = kwarg.value.text;
            } else if (kwarg.name == "n") {
                node.n = static_cast<int>(kwarg.value.number);
            }
        }
        return add(std::move(node));
    }

    const Program& program_;
    const SemaResult& sema_;
    Graph graph_;
    std::map<std::string, NodeId> binding_nodes_;
};

}  // namespace

const char* op_name(Op op)
{
    switch (op) {
    case Op::Capture:
        return "capture";
    case Op::Fir:
        return "fir";
    case Op::Fft:
        return "fft";
    case Op::Magnitude:
        return "magnitude";
    case Op::Xcorr:
        return "xcorr";
    case Op::Emit:
        return "emit";
    }
    return "?";
}

bool Graph::before(NodeId a, NodeId b) const
{
    const SourceLocation& la = nodes[static_cast<std::size_t>(a)].range.loc;
    const SourceLocation& lb = nodes[static_cast<std::size_t>(b)].range.loc;
    if (la.line != lb.line) {
        return la.line < lb.line;
    }
    if (la.column != lb.column) {
        return la.column < lb.column;
    }
    return a < b;
}

Graph build_ir(const Program& program, const SemaResult& sema)
{
    return IrBuilder(program, sema).run();
}

std::string format_node(const Node& node)
{
    std::ostringstream out;
    if (node.op == Op::Emit) {
        out << "emit %" << node.inputs[0] << " -> \"" << node.path << "\"";
        return out.str();
    }
    out << "%" << node.id << " = " << op_name(node.op);
    for (NodeId input : node.inputs) {
        out << " %" << input;
    }
    if (node.op == Op::Capture) {
        out << " \"" << node.path << "\"";
    }
    if (node.op == Op::Fir) {
        out << " taps=\"" << node.taps << "\"";
    }
    if (node.op == Op::Fft) {
        out << " n=" << node.n;
    }
    out << " : " << type_name(node.type) << " @" << node.type.rate_text;
    if (node.group >= 0) {
        out << " [fused " << node.group << "]";
    }
    if (!node.name.empty()) {
        out << "   # " << node.name;
    }
    return out.str();
}

void print_ir(const Graph& graph, std::ostream& out)
{
    for (const Node& node : graph.nodes) {
        if (graph.live[static_cast<std::size_t>(node.id)]) {
            out << format_node(node) << "\n";
        }
    }
}

}  // namespace sigflow
