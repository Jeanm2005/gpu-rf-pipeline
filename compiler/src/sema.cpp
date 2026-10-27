#include "sigflow/sema.hpp"

#include <cmath>
#include <optional>
#include <vector>

namespace sigflow {

std::string type_name(const Type& type)
{
    const char* element = type.element == ElementType::Cf32 ? "cf32" : "f32";
    if (type.kind == Type::Kind::Stream) {
        return std::string("stream<") + element + ">";
    }
    return std::string("frames<") + element + ", " + std::to_string(type.frame_size) + ">";
}

const char* file_extension(const Type& type)
{
    return type.element == ElementType::Cf32 ? ".cf32" : ".f32";
}

namespace {

struct KeywordSpec {
    const char* name;
    Literal::Kind kind;
};

// The built-in operations (compiler/README.md, "Built-in operations"). Every keyword argument
// is required. Sorted by name, which is also the order the "available" list prints in.
struct OpSpec {
    const char* name;
    int positional;
    std::vector<KeywordSpec> keywords;
};

const std::vector<OpSpec>& op_specs()
{
    static const std::vector<OpSpec> specs = {
        {"capture", 1, {{"rate", Literal::Kind::Number}}},
        {"fft", 1, {{"n", Literal::Kind::Number}}},
        {"fir", 1, {{"taps", Literal::Kind::String}}},
        {"magnitude", 1, {}},
        {"xcorr", 2, {}},
    };
    return specs;
}

const OpSpec* find_op(const std::string& name)
{
    for (const OpSpec& spec : op_specs()) {
        if (name == spec.name) {
            return &spec;
        }
    }
    return nullptr;
}

std::string op_names()
{
    std::string names;
    for (const OpSpec& spec : op_specs()) {
        names += names.empty() ? "" : ", ";
        names += spec.name;
    }
    return names;
}

const char* kind_name(Literal::Kind kind)
{
    return kind == Literal::Kind::Number ? "number" : "string";
}

std::string plural(int n, const char* word)
{
    return std::to_string(n) + " " + word + (n == 1 ? "" : "s");
}

class Sema {
public:
    Sema(const Program& program, const SourceFile& source, DiagnosticEngine& diagnostics)
        : program_(program), source_(source), diagnostics_(diagnostics)
    {
    }

    SemaResult run()
    {
        collect_bindings();
        bool any_emit = false;
        for (const Stmt& stmt : program_.statements) {
            if (stmt.kind == StmtKind::Binding) {
                binding_type(stmt.name, stmt.name_range);
            } else {
                check_emit(stmt);
                any_emit = true;
            }
        }
        if (!any_emit) {
            diagnostics_.error({source_.path(), 1, 1},
                               "program has no 'emit' statements, so there is nothing to compute",
                               source_.line_text(1), 1);
        }
        return std::move(result_);
    }

private:
    enum class State { Unvisited, InProgress, Done };

    void error(const SourceRange& range, std::string message)
    {
        diagnostics_.error(range.loc, std::move(message), source_.line_text(range.loc.line),
                           range.length);
    }
    void warning(const SourceRange& range, std::string message)
    {
        diagnostics_.warning(range.loc, std::move(message), source_.line_text(range.loc.line),
                             range.length);
    }

    void collect_bindings()
    {
        for (std::size_t i = 0; i < program_.statements.size(); ++i) {
            const Stmt& stmt = program_.statements[i];
            if (stmt.kind != StmtKind::Binding) {
                continue;
            }
            const auto [it, inserted] = result_.bindings.emplace(stmt.name, i);
            if (!inserted) {
                const Stmt& first = program_.statements[it->second];
                error(stmt.name_range, "redefinition of '" + stmt.name + "' (first defined on line "
                                           + std::to_string(first.name_range.loc.line) + ")");
            }
        }
    }

    // The type of binding `name`, checking its definition first if needed. `use` is where the
    // name is referenced, for the cycle diagnostic.
    std::optional<Type> binding_type(const std::string& name, const SourceRange& use)
    {
        const std::size_t index = result_.bindings.at(name);
        const Stmt& stmt = program_.statements[index];
        State& state = states_[index];
        if (state == State::Done) {
            const auto it = result_.types.find(stmt.value.get());
            return it == result_.types.end() ? std::nullopt : std::optional<Type>(it->second);
        }
        if (state == State::InProgress) {
            report_cycle(name, use);
            return std::nullopt;
        }
        state = State::InProgress;
        in_progress_.push_back(name);
        const std::optional<Type> type = expr_type(*stmt.value);
        in_progress_.pop_back();
        state = State::Done;  // std::map references stay valid across the inserts above
        return type;
    }

    void report_cycle(const std::string& name, const SourceRange& use)
    {
        std::string path;
        bool on_cycle = false;
        for (const std::string& step : in_progress_) {
            on_cycle = on_cycle || step == name;
            if (on_cycle) {
                path += step + " -> ";
            }
        }
        path += name;
        error(use, "cycle: '" + name + "' depends on itself (" + path + ")");
    }

    std::optional<Type> expr_type(const Expr& expr)
    {
        std::optional<Type> type;
        switch (expr.kind) {
        case ExprKind::Name:
            type = name_type(static_cast<const NameExpr&>(expr));
            break;
        case ExprKind::Literal: {
            const auto& lit = static_cast<const LiteralExpr&>(expr);
            error(lit.range, std::string("expected a signal, found a ") + kind_name(lit.value.kind)
                                 + " literal");
            break;
        }
        case ExprKind::Call:
            type = call_type(static_cast<const CallExpr&>(expr), nullptr);
            break;
        case ExprKind::Pipe: {
            const auto& pipe = static_cast<const PipeExpr&>(expr);
            type = call_type(*pipe.call, pipe.input.get());
            if (type) {
                result_.types[pipe.call.get()] = *type;
            }
            break;
        }
        }
        if (type) {
            result_.types[&expr] = *type;
        }
        return type;
    }

    std::optional<Type> name_type(const NameExpr& ref)
    {
        if (result_.bindings.count(ref.name) != 0) {
            return binding_type(ref.name, ref.range);
        }
        if (find_op(ref.name) != nullptr) {
            error(ref.range, "'" + ref.name + "' is an operation, not a binding; call it as "
                                 + ref.name + "(...) or after '|>'");
        } else {
            error(ref.range, "unknown name '" + ref.name + "'");
        }
        return std::nullopt;
    }

    // Checks a call. `piped` is the left side of `piped |> call`, which becomes the first
    // positional argument.
    std::optional<Type> call_type(const CallExpr& call, const Expr* piped)
    {
        if (result_.bindings.count(call.callee) != 0) {
            error(call.callee_range, "'" + call.callee + "' is a binding, not an operation");
            return std::nullopt;
        }
        const OpSpec* spec = find_op(call.callee);
        if (spec == nullptr) {
            error(call.callee_range,
                  "unknown operation '" + call.callee + "' (available: " + op_names() + ")");
            return std::nullopt;
        }

        std::vector<const Expr*> positional;
        if (piped != nullptr) {
            positional.push_back(piped);
        }
        for (const ExprPtr& arg : call.args) {
            positional.push_back(arg.get());
        }

        bool ok = true;
        if (static_cast<int>(positional.size()) != spec->positional) {
            error(call.callee_range,
                  call.callee + " expects " + plural(spec->positional, "positional argument")
                      + ", got " + std::to_string(positional.size())
                      + (piped != nullptr ? " (counting the input piped in with '|>')" : ""));
            ok = false;
        }
        std::map<std::string, const Literal*> keywords;
        ok = check_keywords(call, *spec, keywords) && ok;
        if (!ok) {
            return std::nullopt;
        }

        if (call.callee == "capture") {
            return capture_type(*positional[0], *keywords.at("rate"));
        }
        std::vector<Type> inputs;
        for (std::size_t i = 0; i < positional.size(); ++i) {
            const std::optional<Type> input = argument_type(call, i, *positional[i]);
            if (!input) {
                return std::nullopt;
            }
            inputs.push_back(*input);
        }
        if (call.callee == "fir") {
            return inputs[0];
        }
        if (call.callee == "fft") {
            return fft_type(inputs[0], *keywords.at("n"));
        }
        if (call.callee == "magnitude") {
            return magnitude_type(call, inputs[0], *positional[0]);
        }
        return xcorr_type(call, inputs[0], inputs[1]);
    }

    bool check_keywords(const CallExpr& call, const OpSpec& spec,
                        std::map<std::string, const Literal*>& keywords)
    {
        bool ok = true;
        for (const KeywordArg& kwarg : call.kwargs) {
            const KeywordSpec* expected = nullptr;
            for (const KeywordSpec& candidate : spec.keywords) {
                if (kwarg.name == candidate.name) {
                    expected = &candidate;
                }
            }
            if (expected == nullptr) {
                std::string known;
                for (const KeywordSpec& candidate : spec.keywords) {
                    known += (known.empty() ? "" : ", ") + std::string(candidate.name);
                }
                error(kwarg.name_range,
                      call.callee + " has no keyword argument '" + kwarg.name + "'"
                          + (known.empty() ? " (it takes none)" : " (expected: " + known + ")"));
                ok = false;
            } else if (!keywords.emplace(kwarg.name, &kwarg.value).second) {
                error(kwarg.name_range, "duplicate keyword argument '" + kwarg.name + "'");
                ok = false;
            } else if (kwarg.value.kind != expected->kind) {
                error(kwarg.value.range, "keyword argument '" + kwarg.name + "' of "
                                             + call.callee + " must be a "
                                             + kind_name(expected->kind) + ", got a "
                                             + kind_name(kwarg.value.kind));
                ok = false;
            }
        }
        for (const KeywordSpec& candidate : spec.keywords) {
            if (keywords.count(candidate.name) == 0 && ok) {
                error(call.callee_range, call.callee + " requires keyword argument '"
                                             + candidate.name + "'");
                ok = false;
            }
        }
        return ok;
    }

    std::optional<Type> capture_type(const Expr& path, const Literal& rate)
    {
        bool ok = true;
        if (path.kind != ExprKind::Literal
            || static_cast<const LiteralExpr&>(path).value.kind != Literal::Kind::String) {
            error(path.range, "capture expects a string literal path as argument 1");
            ok = false;
        }
        if (!(rate.number > 0.0) || !std::isfinite(rate.number)) {
            error(rate.range, "sample rate must be a positive number, got " + rate.text);
            ok = false;
        }
        if (!ok) {
            return std::nullopt;
        }
        return Type::stream(rate.number, rate.text);
    }

    // The type of positional argument `index` of a signal operation (everything but capture).
    std::optional<Type> argument_type(const CallExpr& call, std::size_t index, const Expr& arg)
    {
        if (arg.kind == ExprKind::Literal) {
            const auto& lit = static_cast<const LiteralExpr&>(arg);
            error(arg.range, call.callee + " expects a signal as argument "
                                 + std::to_string(index + 1) + ", got a "
                                 + kind_name(lit.value.kind) + " literal");
            return std::nullopt;
        }
        const std::optional<Type> type = expr_type(arg);
        if (!type) {
            return std::nullopt;  // already reported
        }
        // magnitude takes frames and checks them itself (magnitude_type); the rest take streams.
        if (call.callee != "magnitude" && type->kind != Type::Kind::Stream) {
            error(arg.range, call.callee + " expects stream<cf32> as argument "
                                 + std::to_string(index + 1) + ", got " + type_name(*type));
            return std::nullopt;
        }
        return type;
    }

    std::optional<Type> fft_type(const Type& input, const Literal& n)
    {
        if (!(n.number >= 1.0) || n.number != std::floor(n.number) || n.number > 1 << 30) {
            error(n.range, "fft size n must be a positive integer, got " + n.text);
            return std::nullopt;
        }
        return Type::frames(ElementType::Cf32, static_cast<int>(n.number), input.rate,
                            input.rate_text);
    }

    std::optional<Type> magnitude_type(const CallExpr& call, const Type& input, const Expr& arg)
    {
        if (input.kind != Type::Kind::Frames || input.element != ElementType::Cf32) {
            error(arg.range, call.callee + " expects frames<cf32, N>, got " + type_name(input)
                                 + (input.kind == Type::Kind::Stream
                                        ? "; apply fft(n=...) first"
                                        : ""));
            return std::nullopt;
        }
        return Type::frames(ElementType::F32, input.frame_size, input.rate, input.rate_text);
    }

    std::optional<Type> xcorr_type(const CallExpr& call, const Type& a, const Type& b)
    {
        if (a.rate != b.rate) {
            error(call.callee_range, "xcorr inputs have different sample rates (" + a.rate_text
                                         + " vs " + b.rate_text + ")");
            return std::nullopt;
        }
        return a;
    }

    void check_emit(const Stmt& stmt)
    {
        const auto it = result_.bindings.find(stmt.name);
        if (it == result_.bindings.end()) {
            error(stmt.name_range, "unknown name '" + stmt.name + "'");
            return;
        }
        const auto [previous, inserted] = emitted_paths_.emplace(stmt.path.text, &stmt);
        if (!inserted) {
            error(stmt.path.range, "\"" + stmt.path.text + "\" is already written by the emit on "
                                       "line "
                                       + std::to_string(previous->second->range.loc.line));
            return;
        }
        const std::optional<Type> type = binding_type(stmt.name, stmt.name_range);
        if (!type) {
            return;
        }
        const std::string extension = file_extension(*type);
        const std::string& path = stmt.path.text;
        if (path.size() < extension.size()
            || path.compare(path.size() - extension.size(), extension.size(), extension) != 0) {
            warning(stmt.path.range, "'" + stmt.name + "' is " + type_name(*type)
                                         + ", which is written as raw " + extension.substr(1)
                                         + "; the path does not end in " + extension);
        }
    }

    const Program& program_;
    const SourceFile& source_;
    DiagnosticEngine& diagnostics_;
    SemaResult result_;
    std::map<std::size_t, State> states_;
    std::vector<std::string> in_progress_;
    std::map<std::string, const Stmt*> emitted_paths_;
};

}  // namespace

SemaResult analyze(const Program& program, const SourceFile& source,
                   DiagnosticEngine& diagnostics)
{
    return Sema(program, source, diagnostics).run();
}

}  // namespace sigflow
