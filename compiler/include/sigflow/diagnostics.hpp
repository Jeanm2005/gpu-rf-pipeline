#pragma once

// Diagnostics: how every user-facing error and warning in sigflowc is reported.
//
// Per compiler/AGENTS.md: "No exceptions for user errors. Every user mistake becomes a
// diagnostic with file, line, column, a message, and the source line with a caret under the
// problem. Collect several diagnostics per run when it's safe to continue." This header is
// deliberately independent of the token/AST/IR types (not yet designed): the lexer, parser,
// and sema passes will each report through the same DiagnosticEngine once they exist.

#include <ostream>
#include <string>
#include <vector>

namespace sigflow {

// A 1-based position in a source file, as every diagnostic and (later) every AST node carries.
struct SourceLocation {
    std::string file;
    int line = 1;
    int column = 1;
};

enum class Severity {
    Error,
    Warning,
};

struct Diagnostic {
    Severity severity = Severity::Error;
    SourceLocation loc;
    std::string message;
    // The exact text of the line `loc` points into, so it can be printed under the message
    // without the engine re-reading the source file.
    std::string source_line;
    // How many characters starting at loc.column the caret marker spans. At least 1.
    int length = 1;
};

// Formats one diagnostic as:
//   <file>:<line>:<column>: error: <message>
//     <source_line>
//     <spaces>^~~~
// (compiler/README.md's example errors follow this shape). Ends with a trailing newline.
std::string format_diagnostic(const Diagnostic& diagnostic);

// Collects diagnostics for one compiler run. A pass reports every problem it finds through
// error()/warning() and keeps going where it safely can, rather than stopping at the first one.
class DiagnosticEngine {
public:
    void error(SourceLocation loc, std::string message, std::string source_line, int length = 1);
    void warning(SourceLocation loc, std::string message, std::string source_line,
                 int length = 1);

    // True once any Severity::Error diagnostic has been recorded. The driver (src/main.cpp,
    // once it exists) uses this to decide whether to stop the pipeline after this pass.
    bool has_errors() const;

    const std::vector<Diagnostic>& diagnostics() const { return diagnostics_; }

    // Writes every diagnostic to `out`, in the order they were recorded.
    void print(std::ostream& out) const;

private:
    void add(Severity severity, SourceLocation loc, std::string message, std::string source_line,
             int length);

    std::vector<Diagnostic> diagnostics_;
};

}  // namespace sigflow
