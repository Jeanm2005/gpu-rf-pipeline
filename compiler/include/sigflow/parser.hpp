#pragma once

// Recursive-descent parser: tokens to AST, following the grammar in compiler/README.md.
//
// One function per grammar rule. A syntax error is reported through the DiagnosticEngine, the
// rest of that statement is skipped up to the next Newline, and parsing resumes with the next
// statement, so one run reports at most one error per statement.

#include "sigflow/ast.hpp"
#include "sigflow/diagnostics.hpp"
#include "sigflow/source.hpp"
#include "sigflow/token.hpp"

#include <vector>

namespace sigflow {

// `tokens` must come from lex() (ending with Eof, Newline before it). Statements with syntax
// errors are left out of the returned program.
Program parse(const std::vector<Token>& tokens, const SourceFile& source,
              DiagnosticEngine& diagnostics);

}  // namespace sigflow
