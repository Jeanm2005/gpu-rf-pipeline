#pragma once

// Phase C backend: turns the scheduled IR into a CUDA C++ program that keeps every value on the
// GPU, in the buffer slots liveness analysis assigned, and calls the device-pointer entry
// points in ../src/device_ops.hpp in launch order. The program links against rfgpu_ops (the
// rfgpu kernels and src/pipeline_runtime.hpp); compiler/tools/build_cuda.sh builds it.
//
// Unlike the phase A script, intermediate values never leave the GPU: the captures are uploaded
// once, every operation reads and writes slots, and only emitted values are downloaded. That is
// what makes buffer reuse and kernel fusion measurable.

#include "sigflow/ir.hpp"
#include "sigflow/liveness.hpp"
#include "sigflow/passes.hpp"

#include <ostream>
#include <string>

namespace sigflow {

// `program_name` names the program in its header comment and its report; `reuse` is recorded in
// the report so measurements say which plan they ran.
void emit_cuda(const Graph& graph, const Schedule& schedule, const Liveness& liveness,
               const std::string& program_name, bool reuse, std::ostream& out);

}  // namespace sigflow
