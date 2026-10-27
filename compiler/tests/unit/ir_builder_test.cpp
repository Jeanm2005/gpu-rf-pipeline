#include "sigflow/ir.hpp"
#include "sigflow/lexer.hpp"
#include "sigflow/parser.hpp"
#include "sigflow/sema.hpp"

#include <gtest/gtest.h>

#include <string>

namespace sigflow {
namespace {

Graph build_text(const std::string& text)
{
    DiagnosticEngine diagnostics;
    const SourceFile source("t.sf", text);
    const std::vector<Token> tokens = lex(source, diagnostics);
    const Program program = parse(tokens, source, diagnostics);
    const SemaResult sema = analyze(program, source, diagnostics);
    EXPECT_FALSE(diagnostics.has_errors()) << "test input must pass sema";
    return build_ir(program, sema);
}

const Node& named(const Graph& g, const std::string& name)
{
    for (const Node& node : g.nodes) {
        if (node.name == name) {
            return node;
        }
    }
    ADD_FAILURE() << "no node named " << name;
    return g.nodes.front();
}

TEST(IrBuilder, PipesBecomeOrdinaryInputs)
{
    const Graph g = build_text("a = capture(\"a.cf32\", rate=1e6)\n"
                               "s = a |> fft(n=64) |> magnitude\n"
                               "emit s -> \"s.f32\"\n");
    ASSERT_EQ(g.nodes.size(), 4u);  // capture, fft, magnitude, emit
    const Node& s = named(g, "s");
    EXPECT_EQ(s.op, Op::Magnitude);
    ASSERT_EQ(s.inputs.size(), 1u);
    const Node& fft = g.nodes[static_cast<std::size_t>(s.inputs[0])];
    EXPECT_EQ(fft.op, Op::Fft);
    EXPECT_EQ(fft.n, 64);
    EXPECT_TRUE(fft.name.empty());
    EXPECT_EQ(fft.inputs, std::vector<NodeId>{named(g, "a").id});
}

TEST(IrBuilder, ASharedBindingIsOneNode)
{
    const Graph g = build_text("a = capture(\"a.cf32\", rate=1e6)\n"
                               "c = xcorr(a, a)\nemit c -> \"c.cf32\"\n");
    const Node& c = named(g, "c");
    EXPECT_EQ(c.inputs, (std::vector<NodeId>{0, 0}));
}

TEST(IrBuilder, InputsAlwaysHaveSmallerIdsEvenWhenUsedBeforeDefinition)
{
    const Graph g = build_text("emit y -> \"y.cf32\"\n"
                               "y = x |> fir(taps=\"h.f32\")\n"
                               "x = capture(\"x.cf32\", rate=1e6)\n");
    for (const Node& node : g.nodes) {
        for (NodeId input : node.inputs) {
            EXPECT_LT(input, node.id);
        }
    }
    EXPECT_EQ(named(g, "y").taps, "h.f32");
    EXPECT_EQ(named(g, "x").path, "x.cf32");
}

TEST(IrBuilder, AnAliasAddsNoNode)
{
    const Graph g = build_text("a = capture(\"a.cf32\", rate=1e6)\nb = a\nemit b -> \"b.cf32\"\n");
    EXPECT_EQ(g.nodes.size(), 2u);
    EXPECT_EQ(g.nodes[0].name, "a");
}

TEST(IrBuilder, BeforeComparesSourcePosition)
{
    const Graph g = build_text("emit y -> \"y.cf32\"\n"
                               "y = x |> fir(taps=\"h.f32\")\n"
                               "x = capture(\"x.cf32\", rate=1e6)\n");
    const NodeId x = named(g, "x").id;  // built first (id 0), written last (line 3)
    const NodeId y = named(g, "y").id;
    EXPECT_LT(x, y);
    EXPECT_TRUE(g.before(y, x));
    EXPECT_FALSE(g.before(x, y));
}

}  // namespace
}  // namespace sigflow
