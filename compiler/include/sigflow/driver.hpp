#pragma once

// The compiler driver: runs the pipeline (lexer -> parser -> ...) on one file and writes the
// requested --emit output. Kept apart from src/main.cpp so that --update-golden can run it
// in-process with both streams captured.

#include "sigflow/kernels.hpp"

#include <ostream>
#include <string>
#include <vector>

namespace sigflow {

enum class EmitMode {
    Tokens,
    Ast,
    Check,  // run every check and print nothing; the exit code says whether it passed
    Ir,
    Schedule,
    Liveness,  // live ranges, buffer slots, and planned GPU memory
    Json,
    Script,  // the phase A backend: a bash script of rfgpu commands (the default)
    Cuda,    // the phase C backend: a CUDA C++ program that keeps values on the GPU
};

struct Options {
    std::string input_path;    // where to read the program from
    std::string display_path;  // what diagnostics call it; usually the same as input_path
    EmitMode emit = EmitMode::Script;
    std::string output_path;   // -o: write the output here instead of stdout (src/main.cpp)
    bool dce = true;    // dead-node elimination; --no-dce turns it off
    bool fuse = true;   // kernel fusion; --no-fuse turns it off
    bool reuse = true;  // buffer reuse after liveness analysis; --no-reuse turns it off
    // --input-dir=DIR: where capture paths resolve when sizing buffers for --emit=liveness.
    std::string input_dir = ".";
    KernelChoice kernels;  // --fir-impl / --xcorr-impl
};

// Exit codes shared by the driver and src/main.cpp.
constexpr int kExitOk = 0;
constexpr int kExitCompileError = 1;  // the program has errors (diagnostics were printed)
constexpr int kExitUsage = 2;         // bad command line or unreadable input

// Parses the compile flags in `args` (everything after argv[0] except --update-golden) into
// `options`. On failure writes a message to `err` and returns false.
bool parse_options(const std::vector<std::string>& args, Options& options, std::ostream& err);

// Compiles one program. Requested output goes to `out`, diagnostics to `err`. Returns one of the
// exit codes above.
int compile(const Options& options, std::ostream& out, std::ostream& err);

// Regenerates every golden file in `dir`: for each <name>.sf, reads the compile flags from its
// first line ("# args: --emit=tokens"), compiles it with `dir` as the working directory, and
// writes the combined output to <name>.expected. Returns kExitOk or kExitUsage.
int update_golden(const std::string& dir, std::ostream& log);

// The text a golden file holds for one run: everything written to the output and diagnostic
// streams, then "[exit N]" on its own line when the exit code is not 0. tests/golden's CMake
// runner (cmake/RunGoldenTest.cmake) produces exactly the same text.
std::string golden_text(const std::string& combined_output, int exit_code);

}  // namespace sigflow
