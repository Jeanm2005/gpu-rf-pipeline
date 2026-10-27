// sigflowc: the SigFlow compiler command line.
//
//   sigflowc PROGRAM.sf --emit=tokens|ast|check|ir|json
//   sigflowc --update-golden tests/golden
//
// See compiler/README.md for the language and compiler/AGENTS.md for every --emit mode planned.

#include "sigflow/driver.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(std::ostream& out)
{
    out << "usage:\n"
           "  sigflowc PROGRAM.sf --emit=MODE\n"
           "  sigflowc --update-golden DIR    regenerate DIR/*.expected from DIR/*.sf\n"
           "\n"
           "--emit modes:\n"
           "  tokens    lexer output, one token per line\n"
           "  ast       parser output, one node per line\n"
           "  check     run every check, print only diagnostics\n"
           "  ir        graph IR, one node per line\n"
           "  json      graph IR as JSON, for the DAG viewer\n";
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
    return sigflow::compile(options, std::cout, std::cerr);
}
