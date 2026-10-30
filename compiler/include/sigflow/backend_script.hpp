#pragma once

// Phase A backend: turns the scheduled IR into a bash script of rfgpu commands, one per node,
// in launch order. Intermediate values live in a temporary directory that the script removes on
// exit, one file per buffer slot from liveness analysis, so a dead value's file is overwritten
// by a later one (all distinct with --no-reuse). Each emit copies its value to the requested
// path.

#include "sigflow/ir.hpp"
#include "sigflow/kernels.hpp"
#include "sigflow/liveness.hpp"
#include "sigflow/passes.hpp"

#include <ostream>
#include <string>

namespace sigflow {

// `program_name` only appears in the header comment. The script runs `$RFGPU` (default:
// `rfgpu` on PATH), and relative paths resolve against the directory it is run from. FIR and
// cross-correlation commands get `--impl` for any variant in `kernels` other than naive.
void emit_script(const Graph& graph, const Schedule& schedule, const Liveness& liveness,
                 const KernelChoice& kernels, const std::string& program_name,
                 std::ostream& out);

}  // namespace sigflow
