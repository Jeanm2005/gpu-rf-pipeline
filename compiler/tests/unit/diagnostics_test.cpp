#include "sigflow/diagnostics.hpp"

#include <gtest/gtest.h>

#include <sstream>

namespace sigflow {
namespace {

TEST(Diagnostics, FormatsFileLineColumnMessageSourceLineAndCaret)
{
    Diagnostic d;
    d.severity = Severity::Error;
    d.loc = {"examples/bad_rate.sf", 4, 5};  // column 5 is where "xcorr" starts
    d.message = "xcorr inputs have different sample rates (2.4e6 vs 1.2e6)";
    d.source_line = "c = xcorr(a, b)";
    d.length = 5;  // spans "xcorr"

    EXPECT_EQ(format_diagnostic(d),
             "examples/bad_rate.sf:4:5: error: xcorr inputs have different sample rates "
             "(2.4e6 vs 1.2e6)\n"
             "  c = xcorr(a, b)\n"
             "      ^~~~~\n");
}

TEST(Diagnostics, WarningUsesWarningSeverityName)
{
    Diagnostic d;
    d.severity = Severity::Warning;
    d.loc = {"a.sf", 1, 1};
    d.message = "unused binding 'x'";
    d.source_line = "x = capture(\"a.cf32\")";

    EXPECT_EQ(format_diagnostic(d), "a.sf:1:1: warning: unused binding 'x'\n"
                                    "  x = capture(\"a.cf32\")\n"
                                    "  ^\n");
}

TEST(Diagnostics, LengthOneIsJustACaretWithNoTildes)
{
    Diagnostic d;
    d.loc = {"a.sf", 1, 3};
    d.message = "m";
    d.source_line = "ab";
    d.length = 1;

    EXPECT_EQ(format_diagnostic(d), "a.sf:1:3: error: m\n"
                                    "  ab\n"
                                    "    ^\n");
}

TEST(DiagnosticEngineTest, HasErrorsIsFalseUntilAnErrorIsRecorded)
{
    DiagnosticEngine engine;
    EXPECT_FALSE(engine.has_errors());

    engine.warning({"a.sf", 1, 1}, "just a warning", "x = 1");
    EXPECT_FALSE(engine.has_errors());

    engine.error({"a.sf", 2, 1}, "now an error", "y = 2");
    EXPECT_TRUE(engine.has_errors());
}

TEST(DiagnosticEngineTest, CollectsSeveralDiagnosticsInOrderAndPrintsThemAll)
{
    // "Collect several diagnostics per run when it's safe to continue" (compiler/AGENTS.md).
    DiagnosticEngine engine;
    engine.error({"a.sf", 1, 1}, "first problem", "a = 1");
    engine.error({"a.sf", 2, 1}, "second problem", "b = 2");

    ASSERT_EQ(engine.diagnostics().size(), 2u);
    EXPECT_EQ(engine.diagnostics()[0].message, "first problem");
    EXPECT_EQ(engine.diagnostics()[1].message, "second problem");

    std::ostringstream out;
    engine.print(out);
    EXPECT_EQ(out.str(), format_diagnostic(engine.diagnostics()[0])
                            + format_diagnostic(engine.diagnostics()[1]));
}

}  // namespace
}  // namespace sigflow
