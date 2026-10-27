#pragma once

// Phase A backend: turns the scheduled IR into a bash script of rfgpu commands, one per node,
// in launch order. Intermediate values live in a temporary directory that the script removes on
// exit; each emit copies its value to the requested path.

#include "sigflow/ir.hpp"
#include "sigflow/passes.hpp"

#include <ostream>
#include <string>

namespace sigflow {

// `program_name` only appears in the header comment. The script runs `$RFGPU` (default:
// `rfgpu` on PATH), and relative paths resolve against the directory it is run from.
void emit_script(const Graph& graph, const Schedule& schedule, const std::string& program_name,
                 std::ostream& out);

}  // namespace sigflow
