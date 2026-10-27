#include "sigflow/lexer.hpp"
#include "sigflow/parser.hpp"

#include <gtest/gtest.h>

#include <string>

namespace sigflow {
namespace {

struct Parsed {
    Program program;
    DiagnosticEngine diagnostics;
};

Parsed parse_text(const std::string& text)
{
    Parsed result;
    const SourceFile source("t.sf", text);
    const std::vector<Token> tokens = lex(source, result.diagnostics);
    result.program = parse(tokens, source, result.diagnostics);
    return result;
}

const CallExpr& as_call(const Expr& expr)
{
    EXPECT_EQ(expr.kind, ExprKind::Call);
    return static_cast<const CallExpr&>(expr);
}

TEST(Parser, CallWithPositionalAndKeywordArguments)
{
    const Parsed r = parse_text("a = capture(\"x.cf32\", rate=2.4e6)\n");
    ASSERT_FALSE(r.diagnostics.has_errors());
    ASSERT_EQ(r.program.statements.size(), 1u);
    const Stmt& stmt = r.program.statements[0];
    EXPECT_EQ(stmt.kind, StmtKind::Binding);
    EXPECT_EQ(stmt.name, "a");

    const CallExpr& call = as_call(*stmt.value);
    EXPECT_EQ(call.callee, "capture");
    ASSERT_EQ(call.args.size(), 1u);
    EXPECT_EQ(call.args[0]->kind, ExprKind::Literal);
    ASSERT_EQ(call.kwargs.size(), 1u);
    EXPECT_EQ(call.kwargs[0].name, "rate");
    EXPECT_EQ(call.kwargs[0].value.kind, Literal::Kind::Number);
    EXPECT_DOUBLE_EQ(call.kwargs[0].value.number, 2.4e6);
    EXPECT_EQ(call.kwargs[0].value.text, "2.4e6");
}

TEST(Parser, PipeIsLeftAssociative)
{
    // a |> fft(n=8) |> magnitude  ==  (a |> fft(n=8)) |> magnitude
    const Parsed r = parse_text("s = a |> fft(n=8) |> magnitude\n");
    ASSERT_FALSE(r.diagnostics.has_errors());
    const Expr& value = *r.program.statements[0].value;
    ASSERT_EQ(value.kind, ExprKind::Pipe);
    const auto& outer = static_cast<const PipeExpr&>(value);
    EXPECT_EQ(outer.call->callee, "magnitude");
    EXPECT_TRUE(outer.call->bare);
    ASSERT_EQ(outer.input->kind, ExprKind::Pipe);
    const auto& inner = static_cast<const PipeExpr&>(*outer.input);
    EXPECT_EQ(inner.call->callee, "fft");
    EXPECT_FALSE(inner.call->bare);
    EXPECT_EQ(inner.input->kind, ExprKind::Name);
}

TEST(Parser, ParenthesesGroupAPipeInsideAnArgument)
{
    const Parsed r = parse_text("c = xcorr((a |> fir(taps=\"h\")), b)\n");
    ASSERT_FALSE(r.diagnostics.has_errors());
    const CallExpr& call = as_call(*r.program.statements[0].value);
    ASSERT_EQ(call.args.size(), 2u);
    EXPECT_EQ(call.args[0]->kind, ExprKind::Pipe);
    EXPECT_EQ(call.args[1]->kind, ExprKind::Name);
}

TEST(Parser, EmitStatement)
{
    const Parsed r = parse_text("emit spec -> \"out/s.f32\"\n");
    ASSERT_FALSE(r.diagnostics.has_errors());
    const Stmt& stmt = r.program.statements[0];
    EXPECT_EQ(stmt.kind, StmtKind::Emit);
    EXPECT_EQ(stmt.name, "spec");
    EXPECT_EQ(stmt.path.text, "out/s.f32");
    EXPECT_EQ(stmt.name_range.loc.column, 6);
}

TEST(Parser, RecoversAtTheNextStatement)
{
    // Two broken statements around a good one: two errors, one statement kept.
    const Parsed r = parse_text("a capture()\nb = c\nemit -> \"x\"\n");
    EXPECT_EQ(r.diagnostics.diagnostics().size(), 2u);
    ASSERT_EQ(r.program.statements.size(), 1u);
    EXPECT_EQ(r.program.statements[0].name, "b");
}

TEST(Parser, PositionalAfterKeywordIsAnError)
{
    const Parsed r = parse_text("x = fir(taps=\"h\", a)\n");
    ASSERT_EQ(r.diagnostics.diagnostics().size(), 1u);
    EXPECT_EQ(r.diagnostics.diagnostics()[0].message,
              "positional argument after keyword argument 'taps'");
    EXPECT_EQ(r.diagnostics.diagnostics()[0].loc.column, 19);
}

TEST(Parser, KeywordValueMustBeALiteral)
{
    const Parsed r = parse_text("x = fir(a, taps=h)\n");
    ASSERT_EQ(r.diagnostics.diagnostics().size(), 1u);
    EXPECT_EQ(r.diagnostics.diagnostics()[0].loc.column, 17);
}

TEST(Parser, EmptyProgramParsesToNoStatements)
{
    const Parsed r = parse_text("# nothing here\n");
    EXPECT_FALSE(r.diagnostics.has_errors());
    EXPECT_TRUE(r.program.statements.empty());
}

}  // namespace
}  // namespace sigflow
