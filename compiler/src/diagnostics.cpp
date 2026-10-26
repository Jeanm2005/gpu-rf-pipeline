#include "sigflow/diagnostics.hpp"

#include <sstream>

namespace sigflow {

namespace {

const char* severity_name(Severity severity)
{
    switch (severity) {
    case Severity::Error:
        return "error";
    case Severity::Warning:
        return "warning";
    }
    return "error";  // unreachable; avoids a default case that would hide a missing enumerator
}

}  // namespace

std::string format_diagnostic(const Diagnostic& diagnostic)
{
    std::ostringstream out;
    out << diagnostic.loc.file << ":" << diagnostic.loc.line << ":" << diagnostic.loc.column
        << ": " << severity_name(diagnostic.severity) << ": " << diagnostic.message << "\n";
    out << "  " << diagnostic.source_line << "\n";

    const int column = diagnostic.loc.column > 0 ? diagnostic.loc.column : 1;
    const int length = diagnostic.length > 1 ? diagnostic.length : 1;
    out << "  " << std::string(static_cast<std::size_t>(column - 1), ' ') << "^"
        << std::string(static_cast<std::size_t>(length - 1), '~') << "\n";
    return out.str();
}

void DiagnosticEngine::add(Severity severity, SourceLocation loc, std::string message,
                           std::string source_line, int length)
{
    diagnostics_.push_back(
        Diagnostic{severity, std::move(loc), std::move(message), std::move(source_line), length});
}

void DiagnosticEngine::error(SourceLocation loc, std::string message, std::string source_line,
                             int length)
{
    add(Severity::Error, std::move(loc), std::move(message), std::move(source_line), length);
}

void DiagnosticEngine::warning(SourceLocation loc, std::string message, std::string source_line,
                               int length)
{
    add(Severity::Warning, std::move(loc), std::move(message), std::move(source_line), length);
}

bool DiagnosticEngine::has_errors() const
{
    for (const Diagnostic& diagnostic : diagnostics_) {
        if (diagnostic.severity == Severity::Error) {
            return true;
        }
    }
    return false;
}

void DiagnosticEngine::print(std::ostream& out) const
{
    for (const Diagnostic& diagnostic : diagnostics_) {
        out << format_diagnostic(diagnostic);
    }
}

}  // namespace sigflow
