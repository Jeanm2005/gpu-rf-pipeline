#include "sigflow/parser.hpp"

#include <cstdlib>

namespace sigflow {

namespace {

// Thrown to abandon the current statement after a syntax error has been reported. Never escapes
// parse(): it is the parser's internal recovery mechanism, not a way of reporting user errors
// (those always go through the DiagnosticEngine, per compiler/AGENTS.md).
struct SyntaxError {};

class Parser {
public:
    Parser(const std::vector<Token>& tokens, const SourceFile& source,
           DiagnosticEngine& diagnostics)
        : tokens_(tokens), source_(source), diagnostics_(diagnostics)
    {
    }

    // program = { statement } EOF ;
    Program program()
    {
        Program result;
        while (peek().kind != TokenKind::Eof) {
            try {
                result.statements.push_back(statement());
            } catch (const SyntaxError&) {
                synchronize();
            }
        }
        return result;
    }

private:
    const Token& peek(std::size_t ahead = 0) const
    {
        const std::size_t i = pos_ + ahead;
        return i < tokens_.size() ? tokens_[i] : tokens_.back();
    }
    bool check(TokenKind kind) const { return peek().kind == kind; }
    const Token& advance()
    {
        const Token& token = peek();
        if (pos_ < tokens_.size() - 1) {
            ++pos_;
        }
        return token;
    }

    static SourceRange range_of(const Token& token) { return {token.loc, token.length}; }

    [[noreturn]] void fail(const Token& at, const std::string& message)
    {
        diagnostics_.error(at.loc, message, source_.line_text(at.loc.line), at.length);
        throw SyntaxError{};
    }

    [[noreturn]] void fail_expected(const std::string& what)
    {
        fail(peek(), "expected " + what + ", found " + describe_token(peek()));
    }

    const Token& expect(TokenKind kind, const std::string& what)
    {
        if (!check(kind)) {
            fail_expected(what);
        }
        return advance();
    }

    // Skips the rest of a broken statement, including its Newline.
    void synchronize()
    {
        while (!check(TokenKind::Newline) && !check(TokenKind::Eof)) {
            advance();
        }
        if (check(TokenKind::Newline)) {
            advance();
        }
    }

    // statement = ( binding | emit ) NEWLINE ;
    Stmt statement()
    {
        Stmt stmt;
        if (check(TokenKind::KwEmit)) {
            stmt = emit();
        } else if (check(TokenKind::Ident)) {
            stmt = binding();
        } else {
            fail_expected("a binding ('name = ...') or an 'emit' statement");
        }
        expect(TokenKind::Newline, "end of line after the statement");
        return stmt;
    }

    // binding = IDENT "=" expr ;
    Stmt binding()
    {
        const Token& name = advance();
        Stmt stmt;
        stmt.kind = StmtKind::Binding;
        stmt.name = name.text;
        stmt.name_range = range_of(name);
        stmt.range = range_of(name);
        expect(TokenKind::Equals, "'=' after '" + name.text + "'");
        stmt.value = expr();
        return stmt;
    }

    // emit = "emit" IDENT "->" STRING ;
    Stmt emit()
    {
        const Token& keyword = advance();
        Stmt stmt;
        stmt.kind = StmtKind::Emit;
        stmt.range = range_of(keyword);
        const Token& name = expect(TokenKind::Ident, "the name of a binding after 'emit'");
        stmt.name = name.text;
        stmt.name_range = range_of(name);
        expect(TokenKind::Arrow, "'->' after 'emit " + name.text + "'");
        const Token& path = expect(TokenKind::String, "an output path string after '->'");
        stmt.path = literal_from(path);
        return stmt;
    }

    // expr = term { "|>" call } ;
    ExprPtr expr()
    {
        ExprPtr result = term();
        while (check(TokenKind::Pipe)) {
            advance();
            if (!check(TokenKind::Ident)) {
                fail_expected("an operation name after '|>'");
            }
            auto pipe = std::make_unique<PipeExpr>();
            pipe->range = result->range;
            pipe->input = std::move(result);
            pipe->call = call();
            result = std::move(pipe);
        }
        return result;
    }

    // term = call | IDENT | literal | "(" expr ")" ;
    //
    // `literal` is not in the README's first draft of the grammar, but capture's path is a
    // positional string, so a term has to accept one (recorded in compiler/README.md).
    ExprPtr term()
    {
        if (check(TokenKind::Ident)) {
            if (peek(1).kind == TokenKind::LParen) {
                return call();
            }
            const Token& name = advance();
            auto ref = std::make_unique<NameExpr>();
            ref->name = name.text;
            ref->range = range_of(name);
            return ref;
        }
        if (check(TokenKind::Number) || check(TokenKind::String)) {
            auto lit = std::make_unique<LiteralExpr>();
            lit->value = literal_from(advance());
            lit->range = lit->value.range;
            return lit;
        }
        if (check(TokenKind::LParen)) {
            advance();
            ExprPtr inner = expr();
            expect(TokenKind::RParen, "')' to close the parenthesized expression");
            return inner;
        }
        fail_expected("an expression");
    }

    // call = IDENT "(" [ arg_list ] ")" | IDENT ;
    // arg_list = arg { "," arg } ;
    std::unique_ptr<CallExpr> call()
    {
        const Token& callee = advance();
        auto node = std::make_unique<CallExpr>();
        node->callee = callee.text;
        node->callee_range = range_of(callee);
        node->range = range_of(callee);
        if (!check(TokenKind::LParen)) {
            node->bare = true;
            return node;
        }
        advance();
        if (!check(TokenKind::RParen)) {
            arg(*node);
            while (check(TokenKind::Comma)) {
                advance();
                arg(*node);
            }
        }
        expect(TokenKind::RParen, "',' or ')' in the arguments of '" + node->callee + "'");
        return node;
    }

    // arg = IDENT "=" literal | expr ;   (keyword args come after positional)
    void arg(CallExpr& call)
    {
        if (check(TokenKind::Ident) && peek(1).kind == TokenKind::Equals) {
            const Token& name = advance();
            advance();  // =
            if (!check(TokenKind::Number) && !check(TokenKind::String)) {
                fail_expected("a number or a string as the value of keyword argument '"
                              + name.text + "'");
            }
            call.kwargs.push_back(KeywordArg{name.text, range_of(name), literal_from(advance())});
            return;
        }
        if (!call.kwargs.empty()) {
            fail(peek(), "positional argument after keyword argument '"
                             + call.kwargs.back().name + "'");
        }
        call.args.push_back(expr());
    }

    static Literal literal_from(const Token& token)
    {
        Literal lit;
        lit.range = range_of(token);
        lit.text = token.text;
        if (token.kind == TokenKind::Number) {
            lit.kind = Literal::Kind::Number;
            lit.number = std::strtod(token.text.c_str(), nullptr);
        } else {
            lit.kind = Literal::Kind::String;
        }
        return lit;
    }

    const std::vector<Token>& tokens_;
    const SourceFile& source_;
    DiagnosticEngine& diagnostics_;
    std::size_t pos_ = 0;
};

}  // namespace

Program parse(const std::vector<Token>& tokens, const SourceFile& source,
              DiagnosticEngine& diagnostics)
{
    return Parser(tokens, source, diagnostics).program();
}

}  // namespace sigflow
