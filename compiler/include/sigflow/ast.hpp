#pragma once

// AST (abstract syntax tree) produced by the parser (src/parser.cpp) and checked by sema.
//
// The tree mirrors the grammar in compiler/README.md closely, including the pipe: `a |> f(x)`
// stays a PipeExpr here and is only rewritten to f(a, x) by the IR builder, so --emit=ast shows
// what the user wrote. Every node carries the location and width of the text it came from, for
// diagnostic carets.

#include "sigflow/diagnostics.hpp"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace sigflow {

// A position plus how many characters a diagnostic caret should underline.
struct SourceRange {
    SourceLocation loc;
    int length = 1;
};

struct Literal {
    enum class Kind { Number, String };
    Kind kind = Kind::Number;
    // Number: the lexeme as written ("2.4e6"), kept so diagnostics quote the user's spelling.
    // String: the contents, without quotes.
    std::string text;
    double number = 0.0;  // Number only
    SourceRange range;
};

enum class ExprKind {
    Name,     // a reference to a binding: `a`
    Literal,  // "fixtures/ch0.cf32", 1024
    Call,     // capture("x.cf32", rate=2.4e6), or a bare `magnitude` on the right of |>
    Pipe,     // a |> fir(taps="h.f32")
};

struct Expr {
    explicit Expr(ExprKind k) : kind(k) {}
    virtual ~Expr() = default;
    ExprKind kind;
    SourceRange range;
};
using ExprPtr = std::unique_ptr<Expr>;

struct NameExpr : Expr {
    NameExpr() : Expr(ExprKind::Name) {}
    std::string name;
};

struct LiteralExpr : Expr {
    LiteralExpr() : Expr(ExprKind::Literal) {}
    Literal value;
};

struct KeywordArg {
    std::string name;
    SourceRange name_range;
    Literal value;
};

struct CallExpr : Expr {
    CallExpr() : Expr(ExprKind::Call) {}
    std::string callee;
    SourceRange callee_range;  // just the operation name, where most call diagnostics point
    std::vector<ExprPtr> args;
    std::vector<KeywordArg> kwargs;
    // True for `x |> magnitude` (no parentheses). Only affects --emit=ast output.
    bool bare = false;
};

struct PipeExpr : Expr {
    PipeExpr() : Expr(ExprKind::Pipe) {}
    ExprPtr input;
    std::unique_ptr<CallExpr> call;  // receives `input` as its first positional argument
};

enum class StmtKind { Binding, Emit };

struct Stmt {
    StmtKind kind = StmtKind::Binding;
    std::string name;  // Binding: the name defined. Emit: the binding emitted.
    SourceRange name_range;
    ExprPtr value;     // Binding only
    Literal path;      // Emit only: the output file
    SourceRange range; // the whole statement's first token, for statement-level diagnostics
};

struct Program {
    std::vector<Stmt> statements;
};

// Indented text form of the tree, one node per line, used by --emit=ast.
void print_ast(const Program& program, std::ostream& out);

}  // namespace sigflow
