//===- Lexer.h - Jules Language Lexer --------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Lexer for the Jules language. The lexer transforms
// a source buffer into a stream of Tokens, handling:
//   - Whitespace and comments (-- line comments)
//   - Numeric literals (integers and floats, including scientific notation)
//   - String identifiers and keywords
//   - All punctuation and operator tokens
//   - Source location tracking for every token
//
//===----------------------------------------------------------------------===//

#ifndef JULES_LEXER_H
#define JULES_LEXER_H

#include "jules/Token.h"
#include "jules/Diagnostics.h"
#include <memory>
#include <string>
#include <vector>

namespace jules {

/// The Lexer transforms a source buffer into a sequence of Tokens.
///
/// Usage:
///   DiagnosticsEngine diag;
///   Lexer lexer(sourceCode, diag);
///   while (true) {
///     Token tok = lexer.nextToken();
///     if (tok.is(TokenKind::Eof)) break;
///     ...
///   }
///
/// The lexer does not own the source buffer. The caller must ensure it
/// outlives the Lexer.
class Lexer {
public:
  /// Construct a lexer over the given source buffer.
  /// \p sourceName is used in diagnostics (e.g. the file name).
  Lexer(const std::string &source, DiagnosticsEngine &diag,
        std::string sourceName = "<input>");

  /// Lex the next token from the source buffer.
  Token nextToken();

  /// Peek at the next token without consuming it.
  const Token &peekToken();

  /// Peek at the token after the next one (2-token lookahead).
  const Token &peekToken2();

  /// Get the current source location.
  SourceLocation getCurrentLocation() const;

  /// Get the source name (e.g. file name).
  const std::string &getSourceName() const { return sourceName_; }

  /// Lex all tokens at once (useful for testing).
  std::vector<Token> lexAll();

private:
  // ── Character access ─────────────────────────────────────────────────────
  char peekChar() const;
  char peekChar(int offset) const;
  char consumeChar();
  bool isEOF() const;
  bool matchChar(char expected);

  // ── Whitespace & comments ────────────────────────────────────────────────
  void skipWhitespace();
  void skipLineComment();

  // ── Token lexers ─────────────────────────────────────────────────────────
  Token lexNumber();         // integer or float
  Token lexIdentifier();     // identifier or keyword
  Token lexPunctuation();    // symbols and operators

  // ── Helpers ──────────────────────────────────────────────────────────────
  Token formToken(TokenKind kind, SourceLocation loc, const std::string &text);

  // ── State ────────────────────────────────────────────────────────────────
  const std::string    &source_;
  DiagnosticsEngine    &diag_;
  std::string           sourceName_;
  size_t                cursor_;    // byte offset into source_
  SourceLocation        curLoc_;    // location of cursor_
  size_t                lineStart_; // byte offset of the start of the current line

  // Lookahead buffer (at most 2 tokens ahead).
  std::vector<Token>    lookahead_;
};

} // namespace jules

#endif // JULES_LEXER_H
