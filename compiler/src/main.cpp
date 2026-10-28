// sigflowc: the SigFlow compiler command line.
//
//   sigflowc PROGRAM.sf -o run.sh                  phase A backend: a script of rfgpu commands
//   sigflowc PROGRAM.sf --emit=tokens|ast|check|ir|schedule|liveness|json|script|cuda
//   sigflowc --update-golden tests/golden
//
// See compiler/README.md for the language and compiler/AGENTS.md for every --emit mode planned.

#include "sigflow/driver.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(std::ostream& out)
{
    out << "usage:\n"
           "  sigflowc PROGRAM.sf [--emit=MODE] [-o FILE] [--no-dce] [--no-fuse] [--no-reuse]\n"
           "  sigflowc --update-golden DIR    regenerate DIR/*.expected from DIR/*.sf\n"
           "\n"
           "--emit modes:\n"
           "  script    bash script of rfgpu commands (the default)\n"
           "  cuda      CUDA C++ program that keeps values on the GPU (build it with\n"
           "            compiler/tools/build_cuda.sh)\n"
           "  tokens    lexer output, one token per line\n"
           "  ast       parser output, one node per line\n"
           "  check     run every check, print only diagnostics\n"
           "  ir        graph IR, one node per line\n"
           "  schedule  launch order (topological sort, ties broken by source order)\n"
           "  liveness  live range and buffer slot of every value, and planned GPU memory\n"
           "  json      graph IR as JSON, for the DAG viewer\n"
           "\n"
           "-o FILE     write the output to FILE instead of stdout (only if compiling succeeds);\n"
           "            a script written this way is made executable\n"
           "\n"
           "passes:\n"
           "  --no-dce    keep nodes that no emit depends on\n"
           "  --no-fuse   launch each FIR on its own instead of batching same-taps FIRs\n"
           "  --no-reuse  give every value its own buffer instead of reusing dead ones\n"
           "\n"
           "--input-dir=DIR   resolve capture paths against DIR when sizing buffers for\n"
           "                  --emit=liveness (default: the current directory)\n";
}

// Writes `text` to `path`, creating its directory and marking it executable when it is a script.
int write_output(const sigflow::Options& options, const std::string& text)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path parent = fs::path(options.output_path).parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
    }
    std::ofstream file(options.output_path, std::ios::binary);
    if (!file || !(file << text) || !file.flush()) {
        std::cerr << "sigflowc: error: cannot write '" << options.output_path << "'\n";
        return sigflow::kExitUsage;
    }
    file.close();
    if (options.emit == sigflow::EmitMode::Script) {
        fs::permissions(options.output_path,
                        fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                        fs::perm_options::add, ec);
    }
    return sigflow::kExitOk;
}

}  // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        print_usage(args.empty() ? std::cerr : std::cout);
        return args.empty() ? sigflow::kExitUsage : sigflow::kExitOk;
    }
    if (args[0] == "--update-golden") {
        if (args.size() != 2) {
            std::cerr << "sigflowc: error: --update-golden takes exactly one directory\n";
            return sigflow::kExitUsage;
        }
        return sigflow::update_golden(args[1], std::cout);
    }

    sigflow::Options options;
    if (!sigflow::parse_options(args, options, std::cerr)) {
        print_usage(std::cerr);
        return sigflow::kExitUsage;
    }
    if (options.output_path.empty()) {
        return sigflow::compile(options, std::cout, std::cerr);
    }
    std::ostringstream output;
    const int code = sigflow::compile(options, output, std::cerr);
    return code == sigflow::kExitOk ? write_output(options, output.str()) : code;
}
