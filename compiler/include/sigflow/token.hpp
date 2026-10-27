#pragma once

// Tokens produced by the lexer (src/lexer.cpp) and consumed by the parser.

#include "sigflow/diagnostics.hpp"

#include <string>

namespace sigflow {

enum class TokenKind {
    Ident,    // capture, a, two_channel
    Number,   // 1024, 2.4e6
    String,   // "fixtures/ch0.cf32" (text holds the contents, without the quotes)
    KwEmit,   // emit
    Equals,   // =
    Pipe,     // |>
    Arrow,    // ->
    LParen,   // (
    RParen,   // )
    Comma,    // ,
    Newline,  // end of a statement; blank lines and comments produce one too
    Eof,
};

struct Token {
    TokenKind kind = TokenKind::Eof;
    // The lexeme as written, except for String, where it is the contents between the quotes.
    std::string text;
    SourceLocation loc;
    // Width of the lexeme in the source, quotes included, for diagnostic carets. At least 1.
    int length = 1;
};

// Upper-case name used by --emit=tokens and in diagnostics ("IDENT", "NEWLINE", ...).
const char* token_kind_name(TokenKind kind);

// How a token is described in a parser diagnostic: "'='", "identifier 'a'", "end of line", ...
std::string describe_token(const Token& token);

}  // namespace sigflow
