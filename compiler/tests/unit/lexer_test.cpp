#include "sigflow/lexer.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace sigflow {
namespace {

struct Lexed {
    std::vector<Token> tokens;
    DiagnosticEngine diagnostics;
};

Lexed lex_text(const std::string& text)
{
    Lexed result;
    const SourceFile source("t.sf", text);
    result.tokens = lex(source, result.diagnostics);
    return result;
}

std::vector<TokenKind> kinds(const std::vector<Token>& tokens)
{
    std::vector<TokenKind> out;
    for (const Token& token : tokens) {
        out.push_back(token.kind);
    }
    return out;
}

using K = TokenKind;

TEST(Lexer, EmptyProgramIsJustEof)
{
    const Lexed r = lex_text("");
    EXPECT_EQ(kinds(r.tokens), (std::vector<K>{K::Eof}));
    EXPECT_FALSE(r.diagnostics.has_errors());
}

TEST(Lexer, BlankAndCommentLinesProduceNoTokens)
{
    const Lexed r = lex_text("\n# only a comment\n\n   \n");
    EXPECT_EQ(kinds(r.tokens), (std::vector<K>{K::Eof}));
}

TEST(Lexer, BindingWithPipeCallAndKeywordArgument)
{
    const Lexed r = lex_text("la = a |> fir(taps=\"lp.f32\")\n");
    EXPECT_EQ(kinds(r.tokens), (std::vector<K>{K::Ident, K::Equals, K::Ident, K::Pipe, K::Ident,
                                               K::LParen, K::Ident, K::Equals, K::String,
                                               K::RParen, K::Newline, K::Eof}));
    EXPECT_FALSE(r.diagnostics.has_errors());
}

TEST(Lexer, TracksLineAndColumn)
{
    const Lexed r = lex_text("a = b\n  emit a -> \"x\"\n");
    ASSERT_GE(r.tokens.size(), 6u);
    EXPECT_EQ(r.tokens[2].loc.line, 1);
    EXPECT_EQ(r.tokens[2].loc.column, 5);  // b
    EXPECT_EQ(r.tokens[4].kind, K::KwEmit);
    EXPECT_EQ(r.tokens[4].loc.line, 2);
    EXPECT_EQ(r.tokens[4].loc.column, 3);
    EXPECT_EQ(r.tokens[6].kind, K::Arrow);
    EXPECT_EQ(r.tokens[6].loc.column, 10);
}

TEST(Lexer, StringTextExcludesQuotesButLengthIncludesThem)
{
    const Lexed r = lex_text("\"out/x.f32\"");
    ASSERT_EQ(r.tokens[0].kind, K::String);
    EXPECT_EQ(r.tokens[0].text, "out/x.f32");
    EXPECT_EQ(r.tokens[0].length, 11);
}

TEST(Lexer, NumberForms)
{
    const Lexed r = lex_text("1024 2.4e6 1E-3 1e+6 0.5");
    ASSERT_EQ(r.tokens.size(), 7u);  // 5 numbers, synthesized NEWLINE, EOF
    EXPECT_EQ(r.tokens[0].text, "1024");
    EXPECT_EQ(r.tokens[1].text, "2.4e6");
    EXPECT_EQ(r.tokens[2].text, "1E-3");
    EXPECT_EQ(r.tokens[3].text, "1e+6");
    EXPECT_EQ(r.tokens[4].text, "0.5");
    EXPECT_FALSE(r.diagnostics.has_errors());
}

TEST(Lexer, MalformedExponentIsAnError)
{
    const Lexed r = lex_text("n = 1e\n");
    ASSERT_EQ(r.diagnostics.diagnostics().size(), 1u);
    EXPECT_EQ(r.diagnostics.diagnostics()[0].loc.column, 5);
    EXPECT_EQ(r.diagnostics.diagnostics()[0].length, 2);
}

TEST(Lexer, EmitIsAKeywordButEmitterIsAnIdentifier)
{
    const Lexed r = lex_text("emit emitter");
    EXPECT_EQ(r.tokens[0].kind, K::KwEmit);
    EXPECT_EQ(r.tokens[1].kind, K::Ident);
    EXPECT_EQ(r.tokens[1].text, "emitter");
}

TEST(Lexer, NewlineIsSynthesizedWhenTheLastLineHasNoTerminator)
{
    const Lexed r = lex_text("a = b");
    EXPECT_EQ(kinds(r.tokens),
              (std::vector<K>{K::Ident, K::Equals, K::Ident, K::Newline, K::Eof}));
}

TEST(Lexer, CrlfLineEndingsAreAccepted)
{
    const Lexed r = lex_text("a = \"x\"\r\nb = a\r\n");
    EXPECT_FALSE(r.diagnostics.has_errors());
    EXPECT_EQ(r.tokens[2].text, "x");
    EXPECT_EQ(r.tokens[4].loc.line, 2);
}

TEST(Lexer, UnterminatedStringReportsAndKeepsLexingNextLine)
{
    const Lexed r = lex_text("a = \"abc\nb = c\n");
    ASSERT_EQ(r.diagnostics.diagnostics().size(), 1u);
    const Diagnostic& d = r.diagnostics.diagnostics()[0];
    EXPECT_EQ(d.message, "unterminated string");
    EXPECT_EQ(d.loc.line, 1);
    EXPECT_EQ(d.loc.column, 5);
    EXPECT_EQ(d.length, 4);
    EXPECT_EQ(d.source_line, "a = \"abc");
    EXPECT_EQ(r.tokens[4].kind, K::Ident);  // b, on line 2
    EXPECT_EQ(r.tokens[4].loc.line, 2);
}

TEST(Lexer, StrayCharactersAreEachReported)
{
    const Lexed r = lex_text("a = b $ c @\n");
    ASSERT_EQ(r.diagnostics.diagnostics().size(), 2u);
    EXPECT_EQ(r.diagnostics.diagnostics()[0].message, "unexpected character '$'");
    EXPECT_EQ(r.diagnostics.diagnostics()[0].loc.column, 7);
    EXPECT_EQ(r.diagnostics.diagnostics()[1].loc.column, 11);
}

TEST(Lexer, LoneBarSuggestsPipe)
{
    const Lexed r = lex_text("a | f\n");
    ASSERT_EQ(r.diagnostics.diagnostics().size(), 1u);
    EXPECT_EQ(r.diagnostics.diagnostics()[0].message,
              "unexpected character '|'; did you mean '|>'?");
}

}  // namespace
}  // namespace sigflow
