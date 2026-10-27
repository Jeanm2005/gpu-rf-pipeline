#include "sigflow/driver.hpp"

#include "sigflow/diagnostics.hpp"
#include "sigflow/ir.hpp"
#include "sigflow/lexer.hpp"
#include "sigflow/parser.hpp"
#include "sigflow/passes.hpp"
#include "sigflow/sema.hpp"
#include "sigflow/source.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace sigflow {

namespace fs = std::filesystem;

namespace {

void print_tokens(const std::vector<Token>& tokens, std::ostream& out)
{
    for (const Token& token : tokens) {
        out << token.loc.line << ":" << token.loc.column << " " << token_kind_name(token.kind);
        switch (token.kind) {
        case TokenKind::Newline:
        case TokenKind::Eof:
            break;
        case TokenKind::String:
            out << " \"" << token.text << "\"";
            break;
        default:
            out << " " << token.text;
            break;
        }
        out << "\n";
    }
}

bool parse_emit(const std::string& value, EmitMode& mode)
{
    if (value == "tokens") {
        mode = EmitMode::Tokens;
        return true;
    }
    if (value == "ast") {
        mode = EmitMode::Ast;
        return true;
    }
    if (value == "check") {
        mode = EmitMode::Check;
        return true;
    }
    if (value == "ir") {
        mode = EmitMode::Ir;
        return true;
    }
    if (value == "schedule") {
        mode = EmitMode::Schedule;
        return true;
    }
    if (value == "json") {
        mode = EmitMode::Json;
        return true;
    }
    return false;
}

// "# args: --emit=ast --no-dce" -> {"--emit=ast", "--no-dce"}. Empty when the line has no
// "# args:" prefix.
std::vector<std::string> golden_args(const std::string& first_line)
{
    const std::string prefix = "# args:";
    std::vector<std::string> args;
    if (first_line.compare(0, prefix.size(), prefix) != 0) {
        return args;
    }
    std::istringstream words(first_line.substr(prefix.size()));
    std::string word;
    while (words >> word) {
        args.push_back(word);
    }
    return args;
}

}  // namespace

bool parse_options(const std::vector<std::string>& args, Options& options, std::ostream& err)
{
    bool have_input = false;
    for (const std::string& arg : args) {
        if (arg.compare(0, 7, "--emit=") == 0) {
            if (!parse_emit(arg.substr(7), options.emit)) {
                err << "sigflowc: error: unknown --emit mode '" << arg.substr(7)
                    << "' (available: tokens, ast, check, ir, schedule, json)\n";
                return false;
            }
        } else if (arg == "--no-dce") {
            options.dce = false;
        } else if (!arg.empty() && arg[0] == '-') {
            err << "sigflowc: error: unknown option '" << arg << "'\n";
            return false;
        } else if (have_input) {
            err << "sigflowc: error: more than one input file ('" << options.input_path
                << "' and '" << arg << "')\n";
            return false;
        } else {
            options.input_path = arg;
            have_input = true;
        }
    }
    if (!have_input) {
        err << "sigflowc: error: no input file\n";
        return false;
    }
    if (options.display_path.empty()) {
        options.display_path = options.input_path;
    }
    return true;
}

int compile(const Options& options, std::ostream& out, std::ostream& err)
{
    std::string text;
    if (!read_file(options.input_path, text)) {
        err << "sigflowc: error: cannot read '" << options.display_path << "'\n";
        return kExitUsage;
    }
    const SourceFile source(options.display_path, std::move(text));
    DiagnosticEngine diagnostics;

    // Each stage stops the pipeline when it reports an error, so later stages never see a
    // broken input and the user sees the first failing stage's errors only. Diagnostics are
    // printed once, when the run ends, and always before any --emit output.
    const auto failed = [&] {
        if (!diagnostics.has_errors()) {
            return false;
        }
        diagnostics.print(err);
        return true;
    };

    const std::vector<Token> tokens = lex(source, diagnostics);
    if (failed()) {
        return kExitCompileError;
    }
    if (options.emit == EmitMode::Tokens) {
        diagnostics.print(err);
        print_tokens(tokens, out);
        return kExitOk;
    }

    const Program program = parse(tokens, source, diagnostics);
    if (failed()) {
        return kExitCompileError;
    }
    if (options.emit == EmitMode::Ast) {
        diagnostics.print(err);
        print_ast(program, out);
        return kExitOk;
    }

    const SemaResult sema = analyze(program, source, diagnostics);
    if (failed()) {
        return kExitCompileError;
    }
    if (options.emit == EmitMode::Check) {
        diagnostics.print(err);
        return kExitOk;
    }

    // From here on only the IR is used (compiler/AGENTS.md: no backend code reads the AST).
    Graph graph = build_ir(program, sema);
    if (options.dce) {
        graph = eliminate_dead_nodes(std::move(graph));
    }
    const Schedule order = schedule(graph);
    if (!order.unscheduled.empty()) {
        // Sema rejects cyclic programs, so this is a compiler bug rather than a user error;
        // it is still reported like any other diagnostic, with the node it starts at.
        const Node& first = graph.nodes[static_cast<std::size_t>(order.unscheduled[0])];
        std::string nodes;
        for (NodeId id : order.unscheduled) {
            nodes += (nodes.empty() ? "%" : ", %") + std::to_string(id);
        }
        diagnostics.error(first.range.loc, "internal error: the IR has a cycle through " + nodes,
                          source.line_text(first.range.loc.line), first.range.length);
        diagnostics.print(err);
        return kExitCompileError;
    }
    diagnostics.print(err);

    switch (options.emit) {
    case EmitMode::Json:
        print_ir_json(graph, out);
        break;
    case EmitMode::Schedule:
        for (std::size_t step = 0; step < order.order.size(); ++step) {
            const Node& node = graph.nodes[static_cast<std::size_t>(order.order[step])];
            out << step << ": " << format_node(node) << "\n";
        }
        break;
    default:
        print_ir(graph, out);
        break;
    }
    return kExitOk;
}

std::string golden_text(const std::string& combined_output, int exit_code)
{
    std::string text = combined_output;
    if (exit_code != 0) {
        text += "[exit " + std::to_string(exit_code) + "]\n";
    }
    return text;
}

int update_golden(const std::string& dir, std::ostream& log)
{
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        log << "sigflowc: error: '" << dir << "' is not a directory\n";
        return kExitUsage;
    }
    std::vector<fs::path> programs;
    for (const fs::directory_entry& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".sf") {
            programs.push_back(entry.path());
        }
    }
    std::sort(programs.begin(), programs.end());

    for (const fs::path& program : programs) {
        std::string text;
        read_file(program.string(), text);
        const std::string first_line = text.substr(0, text.find('\n'));

        // Same command line the CMake runner uses: the bare file name, run from `dir`, so the
        // paths in diagnostics don't depend on where the repository is checked out.
        std::vector<std::string> args = golden_args(first_line);
        args.push_back(program.filename().string());
        Options options;
        std::ostringstream combined;
        int code = kExitUsage;
        if (parse_options(args, options, combined)) {
            options.input_path = program.string();
            code = compile(options, combined, combined);
        }

        fs::path expected = program;
        expected.replace_extension(".expected");
        std::ofstream file(expected, std::ios::binary);
        file << golden_text(combined.str(), code);
        log << "updated " << expected.string() << "\n";
    }
    return kExitOk;
}

}  // namespace sigflow
