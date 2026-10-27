#pragma once

// Lexer: SigFlow source text to tokens, following the lexical rules in compiler/README.md.
//
// Every token carries its 1-based line and column. Errors (an unterminated string, a stray
// character, a malformed number) are reported through the DiagnosticEngine and the lexer keeps
// going, so one run can report several of them.

#include "sigflow/diagnostics.hpp"
#include "sigflow/source.hpp"
#include "sigflow/token.hpp"

#include <vector>

namespace sigflow {

// Returns the tokens of `source`, always ending with Eof. A Newline token ends every non-empty
// line; one is synthesized before Eof when the last line has no line terminator, so the parser
// can rely on every statement being followed by Newline. Comments ("# ...") are dropped.
std::vector<Token> lex(const SourceFile& source, DiagnosticEngine& diagnostics);

}  // namespace sigflow
