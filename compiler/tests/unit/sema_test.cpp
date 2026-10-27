#include "sigflow/lexer.hpp"
#include "sigflow/parser.hpp"
#include "sigflow/sema.hpp"

#include <gtest/gtest.h>

#include <string>

namespace sigflow {
namespace {

struct Checked {
    Program program;
    SemaResult sema;
    DiagnosticEngine diagnostics;
};

Checked check_text(const std::string& text)
{
    Checked result;
    const SourceFile source("t.sf", text);
    const std::vector<Token> tokens = lex(source, result.diagnostics);
    result.program = parse(tokens, source, result.diagnostics);
    EXPECT_FALSE(result.diagnostics.has_errors()) << "test input must lex and parse cleanly";
    result.sema = analyze(result.program, source, result.diagnostics);
    return result;
}

const Type& binding_type(const Checked& c, const std::string& name)
{
    const Stmt& stmt = c.program.statements[c.sema.bindings.at(name)];
    return c.sema.types.at(stmt.value.get());
}

std::string first_message(const Checked& c)
{
    return c.diagnostics.diagnostics().empty() ? "" : c.diagnostics.diagnostics()[0].message;
}

const char* kTwoChannel = "a    = capture(\"ch0.cf32\", rate=2.4e6)\n"
                          "b    = capture(\"ch1.cf32\", rate=2.4e6)\n"
                          "la   = a |> fir(taps=\"lp.f32\")\n"
                          "spec = la |> fft(n=1024) |> magnitude\n"
                          "corr = xcorr(la, b)\n"
                          "emit spec -> \"spec.f32\"\n"
                          "emit corr -> \"corr.cf32\"\n";

TEST(Sema, TypesFollowTheBuiltInSignatures)
{
    const Checked c = check_text(kTwoChannel);
    EXPECT_FALSE(c.diagnostics.has_errors()) << first_message(c);
    EXPECT_EQ(type_name(binding_type(c, "a")), "stream<cf32>");
    EXPECT_EQ(binding_type(c, "a").rate_text, "2.4e6");
    EXPECT_EQ(type_name(binding_type(c, "la")), "stream<cf32>");
    EXPECT_EQ(type_name(binding_type(c, "spec")), "frames<f32, 1024>");
    EXPECT_EQ(type_name(binding_type(c, "corr")), "stream<cf32>");
}

TEST(Sema, NamesMayBeUsedBeforeTheirDefinition)
{
    const Checked c = check_text("emit y -> \"y.cf32\"\n"
                                 "y = x |> fir(taps=\"h.f32\")\n"
                                 "x = capture(\"x.cf32\", rate=1e6)\n");
    EXPECT_FALSE(c.diagnostics.has_errors()) << first_message(c);
    EXPECT_EQ(type_name(binding_type(c, "y")), "stream<cf32>");
}

TEST(Sema, UnknownName)
{
    const Checked c = check_text("y = q |> fir(taps=\"h\")\nemit y -> \"y.cf32\"\n");
    EXPECT_EQ(first_message(c), "unknown name 'q'");
}

TEST(Sema, WrongArgumentCountCountsThePipedInput)
{
    const Checked c = check_text("a = capture(\"a\", rate=1)\n"
                                 "y = a |> xcorr(a, a)\nemit y -> \"y.cf32\"\n");
    EXPECT_EQ(first_message(c), "xcorr expects 2 positional arguments, got 3 (counting the input "
                                "piped in with '|>')");
}

TEST(Sema, UnknownAndMissingKeywordArguments)
{
    const Checked c = check_text("a = capture(\"a\", rate=1)\n"
                                 "y = a |> fir(tap=\"h\")\nemit y -> \"y.cf32\"\n");
    ASSERT_EQ(c.diagnostics.diagnostics().size(), 1u);  // the missing 'taps' is not piled on
    EXPECT_EQ(first_message(c), "fir has no keyword argument 'tap' (expected: taps)");
}

TEST(Sema, MagnitudeOfAStreamIsATypeMismatch)
{
    const Checked c = check_text("a = capture(\"a\", rate=1)\n"
                                 "m = a |> magnitude\nemit m -> \"m.f32\"\n");
    EXPECT_EQ(first_message(c),
              "magnitude expects frames<cf32, N>, got stream<cf32>; apply fft(n=...) first");
}

TEST(Sema, XcorrRateMismatchQuotesTheSourceSpelling)
{
    const Checked c = check_text("a = capture(\"a\", rate=2.4e6)\n"
                                 "b = capture(\"b\", rate=1.2e6)\n"
                                 "c = xcorr(a, b)\nemit c -> \"c.cf32\"\n");
    EXPECT_EQ(first_message(c), "xcorr inputs have different sample rates (2.4e6 vs 1.2e6)");
    EXPECT_EQ(c.diagnostics.diagnostics()[0].loc.column, 5);
}

TEST(Sema, EqualRatesSpelledDifferentlyAreAccepted)
{
    const Checked c = check_text("a = capture(\"a\", rate=2.4e6)\n"
                                 "b = capture(\"b\", rate=2400000)\n"
                                 "c = xcorr(a, b)\nemit c -> \"c.cf32\"\n");
    EXPECT_FALSE(c.diagnostics.has_errors()) << first_message(c);
}

TEST(Sema, CycleIsReportedOnceWithItsPath)
{
    const Checked c = check_text("a = b |> fir(taps=\"h\")\n"
                                 "b = a |> fir(taps=\"h\")\nemit a -> \"a.cf32\"\n");
    ASSERT_EQ(c.diagnostics.diagnostics().size(), 1u);
    EXPECT_EQ(first_message(c), "cycle: 'a' depends on itself (a -> b -> a)");
    EXPECT_EQ(c.diagnostics.diagnostics()[0].loc.line, 2);
}

TEST(Sema, Redefinition)
{
    const Checked c = check_text("a = capture(\"a\", rate=1)\na = capture(\"b\", rate=1)\n"
                                 "emit a -> \"a.cf32\"\n");
    EXPECT_EQ(first_message(c), "redefinition of 'a' (first defined on line 1)");
}

TEST(Sema, EmptyProgramHasNothingToCompute)
{
    const Checked c = check_text("");
    EXPECT_EQ(first_message(c), "program has no 'emit' statements, so there is nothing to compute");
}

TEST(Sema, MismatchedExtensionIsOnlyAWarning)
{
    const Checked c = check_text("a = capture(\"a\", rate=1)\nemit a -> \"a.f32\"\n");
    EXPECT_FALSE(c.diagnostics.has_errors());
    ASSERT_EQ(c.diagnostics.diagnostics().size(), 1u);
    EXPECT_EQ(c.diagnostics.diagnostics()[0].severity, Severity::Warning);
}

}  // namespace
}  // namespace sigflow
