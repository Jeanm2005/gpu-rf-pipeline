#pragma once

// Semantic analysis: name resolution, argument checks, and type and sample-rate checks on the
// AST (compiler/README.md, "Types" and "Built-in operations").
//
// Bindings are declarative: a name may be used before the line that defines it, so the program
// is a graph rather than a sequence. Sema resolves every name against the whole program, checks
// bindings depth-first in dependency order, and reports a cycle when a binding depends on
// itself. An expression that fails a check gets no type, and anything built on it is skipped
// silently, so one mistake produces one diagnostic.

#include "sigflow/ast.hpp"
#include "sigflow/diagnostics.hpp"
#include "sigflow/source.hpp"
#include "sigflow/types.hpp"

#include <map>
#include <string>

namespace sigflow {

struct SemaResult {
    // Every binding name to the index of its statement in Program::statements.
    std::map<std::string, std::size_t> bindings;
    // The type of every signal-valued expression that checked successfully. Keyed by node
    // address; never iterated to produce output.
    std::map<const Expr*, Type> types;
};

// Checks `program`. When diagnostics.has_errors() is false afterwards, every binding and every
// call in the program has an entry in `types` and every name resolves.
SemaResult analyze(const Program& program, const SourceFile& source,
                   DiagnosticEngine& diagnostics);

}  // namespace sigflow
