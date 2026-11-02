// libFuzzer target for the whole sigflowc pipeline on arbitrary source text: lexer, parser,
// semantic analysis, and, for programs that pass it, the IR, every pass, and both backends.
//
// Built only with -DSIGFLOWC_FUZZ=ON and Clang (see compiler/README.md, "Fuzzing"). A crash,
// a sanitizer report, or a failed assertion is a bug: every input, however malformed, must end
// in diagnostics or in output.

#include "sigflow/ast.hpp"
#include "sigflow/backend_cuda.hpp"
#include "sigflow/backend_script.hpp"
#include "sigflow/diagnostics.hpp"
#include "sigflow/ir.hpp"
#include "sigflow/json.hpp"
#include "sigflow/lexer.hpp"
#include "sigflow/liveness.hpp"
#include "sigflow/parser.hpp"
#include "sigflow/passes.hpp"
#include "sigflow/sema.hpp"
#include "sigflow/source.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Every pass after sema, with and without each optional one, as the driver runs them.
void compile_graph(const sigflow::Program& program, const sigflow::SemaResult& sema, bool dce,
                   bool fuse, bool reuse)
{
    using namespace sigflow;
    Graph graph = build_ir(program, sema);
    if (dce) {
        graph = eliminate_dead_nodes(std::move(graph));
    }
    if (fuse) {
        graph = fuse_kernels(std::move(graph));
    }
    const Schedule order = schedule(graph);
    if (!order.unscheduled.empty()) {
        __builtin_trap();  // sema accepted a cyclic program: a compiler bug
    }
    const Liveness liveness = analyze_liveness(graph, order, reuse);
    // Unknown sizes everywhere: value_bytes would read capture files from disk.
    const std::vector<long long> bytes(graph.nodes.size(), -1);
    std::ostringstream out;
    print_ir(graph, out);
    print_liveness(graph, liveness, bytes, ".", out);
    print_json(graph, dce, fuse, order, liveness, bytes, out);
    const KernelChoice kernels;
    emit_script(graph, order, liveness, kernels, "fuzz.sf", out);
    emit_cuda(graph, order, liveness, kernels, "fuzz.sf", reuse, out);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    using namespace sigflow;
    const SourceFile source("fuzz.sf", std::string(reinterpret_cast<const char*>(data), size));
    DiagnosticEngine diagnostics;
    std::ostringstream sink;

    const std::vector<Token> tokens = lex(source, diagnostics);
    if (diagnostics.has_errors()) {
        diagnostics.print(sink);
        return 0;
    }
    const Program program = parse(tokens, source, diagnostics);
    print_ast(program, sink);
    if (diagnostics.has_errors()) {
        diagnostics.print(sink);
        return 0;
    }
    const SemaResult sema = analyze(program, source, diagnostics);
    diagnostics.print(sink);
    if (diagnostics.has_errors()) {
        return 0;
    }
    compile_graph(program, sema, true, true, true);
    compile_graph(program, sema, false, false, false);
    return 0;
}
