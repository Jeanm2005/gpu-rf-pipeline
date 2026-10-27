#include "sigflow/ast.hpp"

#include <string>

namespace sigflow {

namespace {

std::string at(const SourceRange& range)
{
    return " @" + std::to_string(range.loc.line) + ":" + std::to_string(range.loc.column);
}

std::string literal_text(const Literal& lit)
{
    if (lit.kind == Literal::Kind::String) {
        return "string \"" + lit.text + "\"";
    }
    return "number " + lit.text;
}

void print_expr(const Expr& expr, int depth, std::ostream& out)
{
    const std::string indent(static_cast<std::size_t>(depth) * 2, ' ');
    switch (expr.kind) {
    case ExprKind::Name: {
        const auto& name = static_cast<const NameExpr&>(expr);
        out << indent << "name " << name.name << at(name.range) << "\n";
        break;
    }
    case ExprKind::Literal: {
        const auto& lit = static_cast<const LiteralExpr&>(expr);
        out << indent << literal_text(lit.value) << at(lit.range) << "\n";
        break;
    }
    case ExprKind::Call: {
        const auto& call = static_cast<const CallExpr&>(expr);
        out << indent << "call " << call.callee << (call.bare ? " (bare)" : "")
            << at(call.callee_range) << "\n";
        for (const ExprPtr& arg : call.args) {
            print_expr(*arg, depth + 1, out);
        }
        for (const KeywordArg& kwarg : call.kwargs) {
            out << indent << "  kwarg " << kwarg.name << " = " << literal_text(kwarg.value)
                << at(kwarg.name_range) << "\n";
        }
        break;
    }
    case ExprKind::Pipe: {
        const auto& pipe = static_cast<const PipeExpr&>(expr);
        out << indent << "pipe" << at(pipe.range) << "\n";
        print_expr(*pipe.input, depth + 1, out);
        print_expr(*pipe.call, depth + 1, out);
        break;
    }
    }
}

}  // namespace

void print_ast(const Program& program, std::ostream& out)
{
    out << "program\n";
    for (const Stmt& stmt : program.statements) {
        if (stmt.kind == StmtKind::Binding) {
            out << "  binding " << stmt.name << at(stmt.name_range) << "\n";
            print_expr(*stmt.value, 2, out);
        } else {
            out << "  emit " << stmt.name << " -> \"" << stmt.path.text << "\""
                << at(stmt.range) << "\n";
        }
    }
}

}  // namespace sigflow
