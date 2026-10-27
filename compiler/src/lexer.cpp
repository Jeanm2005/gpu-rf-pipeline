#include "sigflow/lexer.hpp"

#include <cctype>

namespace sigflow {

const char* token_kind_name(TokenKind kind)
{
    switch (kind) {
    case TokenKind::Ident:
        return "IDENT";
    case TokenKind::Number:
        return "NUMBER";
    case TokenKind::String:
        return "STRING";
    case TokenKind::KwEmit:
        return "EMIT";
    case TokenKind::Equals:
        return "EQUALS";
    case TokenKind::Pipe:
        return "PIPE";
    case TokenKind::Arrow:
        return "ARROW";
    case TokenKind::LParen:
        return "LPAREN";
    case TokenKind::RParen:
        return "RPAREN";
    case TokenKind::Comma:
        return "COMMA";
    case TokenKind::Newline:
        return "NEWLINE";
    case TokenKind::Eof:
        return "EOF";
    }
    return "?";  // unreachable; no default case so a new enumerator triggers -Wswitch
}

std::string describe_token(const Token& token)
{
    switch (token.kind) {
    case TokenKind::Ident:
        return "identifier '" + token.text + "'";
    case TokenKind::Number:
        return "number " + token.text;
    case TokenKind::String:
        return "string \"" + token.text + "\"";
    case TokenKind::KwEmit:
        return "keyword 'emit'";
    case TokenKind::Newline:
        return "end of line";
    case TokenKind::Eof:
        return "end of file";
    case TokenKind::Equals:
    case TokenKind::Pipe:
    case TokenKind::Arrow:
    case TokenKind::LParen:
    case TokenKind::RParen:
    case TokenKind::Comma:
        return "'" + token.text + "'";
    }
    return "token";
}

namespace {

bool is_letter(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }
bool is_digit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }

class Lexer {
public:
    Lexer(const SourceFile& source, DiagnosticEngine& diagnostics)
        : source_(source), text_(source.text()), diagnostics_(diagnostics)
    {
    }

    std::vector<Token> run()
    {
        while (!at_end()) {
            lex_one();
        }
        add_newline(location());  // a last line with no line terminator still ends a statement
        tokens_.push_back(Token{TokenKind::Eof, "", location(), 1});
        return std::move(tokens_);
    }

private:
    bool at_end() const { return pos_ >= text_.size(); }
    char peek(std::size_t ahead = 0) const
    {
        return pos_ + ahead < text_.size() ? text_[pos_ + ahead] : '\0';
    }
    SourceLocation location() const { return SourceLocation{source_.path(), line_, column_}; }

    void advance()
    {
        if (text_[pos_] == '\n') {
            ++line_;
            column_ = 1;
        } else {
            ++column_;
        }
        ++pos_;
    }

    void error(const SourceLocation& loc, std::string message, int length)
    {
        diagnostics_.error(loc, std::move(message), source_.line_text(loc.line), length);
    }

    void add(TokenKind kind, std::string text, const SourceLocation& loc, int length)
    {
        tokens_.push_back(Token{kind, std::move(text), loc, length});
    }

    // Newlines are collapsed: the token list never starts with one or has two in a row, so
    // blank and comment-only lines are invisible to the parser.
    void add_newline(const SourceLocation& loc)
    {
        if (!tokens_.empty() && tokens_.back().kind != TokenKind::Newline) {
            add(TokenKind::Newline, "", loc, 1);
        }
    }

    void lex_one()
    {
        const char c = peek();
        const SourceLocation start = location();

        if (c == ' ' || c == '\t' || c == '\r') {
            advance();
        } else if (c == '\n') {
            add_newline(start);
            advance();
        } else if (c == '#') {
            while (!at_end() && peek() != '\n') {
                advance();
            }
        } else if (is_letter(c)) {
            lex_identifier(start);
        } else if (is_digit(c)) {
            lex_number(start);
        } else if (c == '"') {
            lex_string(start);
        } else if (c == '|' && peek(1) == '>') {
            advance();
            advance();
            add(TokenKind::Pipe, "|>", start, 2);
        } else if (c == '-' && peek(1) == '>') {
            advance();
            advance();
            add(TokenKind::Arrow, "->", start, 2);
        } else if (c == '=' || c == '(' || c == ')' || c == ',') {
            advance();
            const TokenKind kind = c == '='   ? TokenKind::Equals
                                   : c == '(' ? TokenKind::LParen
                                   : c == ')' ? TokenKind::RParen
                                              : TokenKind::Comma;
            add(kind, std::string(1, c), start, 1);
        } else {
            lex_stray(start);
        }
    }

    void lex_identifier(const SourceLocation& start)
    {
        const std::size_t begin = pos_;
        while (is_letter(peek()) || is_digit(peek()) || peek() == '_') {
            advance();
        }
        std::string word = text_.substr(begin, pos_ - begin);
        const int length = static_cast<int>(word.size());
        const TokenKind kind = word == "emit" ? TokenKind::KwEmit : TokenKind::Ident;
        add(kind, std::move(word), start, length);
    }

    // NUMBER = digit { digit } [ "." digit { digit } ] [ ( "e" | "E" ) [ "-" ] digit { digit } ]
    // A "+" after the exponent marker is accepted too ("1e+6"), as every other language does.
    void lex_number(const SourceLocation& start)
    {
        const std::size_t begin = pos_;
        bool ok = true;
        while (is_digit(peek())) {
            advance();
        }
        if (peek() == '.') {
            advance();
            if (!is_digit(peek())) {
                ok = false;
            }
            while (is_digit(peek())) {
                advance();
            }
        }
        if (ok && (peek() == 'e' || peek() == 'E')) {
            advance();
            if (peek() == '-' || peek() == '+') {
                advance();
            }
            if (!is_digit(peek())) {
                ok = false;
            }
            while (is_digit(peek())) {
                advance();
            }
        }
        std::string lexeme = text_.substr(begin, pos_ - begin);
        const int length = static_cast<int>(lexeme.size());
        if (!ok) {
            error(start, "malformed number '" + lexeme + "': expected a digit after '"
                             + std::string(1, lexeme.back()) + "'",
                  length);
        }
        add(TokenKind::Number, std::move(lexeme), start, length);
    }

    // STRING = '"' { any character except '"' or newline } '"'
    void lex_string(const SourceLocation& start)
    {
        advance();  // opening quote
        const std::size_t begin = pos_;
        while (!at_end() && peek() != '"' && peek() != '\n') {
            advance();
        }
        std::string contents = text_.substr(begin, pos_ - begin);
        if (!contents.empty() && contents.back() == '\r') {
            contents.pop_back();
        }
        if (peek() == '"') {
            advance();
            add(TokenKind::String, contents, start, static_cast<int>(contents.size()) + 2);
            return;
        }
        // Still produce the token so the rest of the line lexes normally; the driver stops
        // after the lexer anyway once there is an error.
        const int length = static_cast<int>(contents.size()) + 1;
        error(start, "unterminated string", length);
        add(TokenKind::String, std::move(contents), start, length);
    }

    void lex_stray(const SourceLocation& start)
    {
        const char c = peek();
        advance();
        std::string message = "unexpected character '" + std::string(1, c) + "'";
        if (c == '|') {
            message += "; did you mean '|>'?";
        } else if (c == '-') {
            message += "; did you mean '->'? (negative numbers are not supported)";
        }
        error(start, std::move(message), 1);
    }

    const SourceFile& source_;
    const std::string& text_;
    DiagnosticEngine& diagnostics_;
    std::vector<Token> tokens_;
    std::size_t pos_ = 0;
    int line_ = 1;
    int column_ = 1;
};

}  // namespace

std::vector<Token> lex(const SourceFile& source, DiagnosticEngine& diagnostics)
{
    return Lexer(source, diagnostics).run();
}

}  // namespace sigflow
