//===- Token.h - Jules Language Token Types --------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Token type and TokenKind enum for the Jules language
// lexer. Every lexical unit produced by the lexer is represented as a Token
// carrying its kind, source location, and literal text.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_TOKEN_H
#define JULES_TOKEN_H

#include <cstdint>
#include <string>
#include <ostream>

namespace jules {

/// Classification of every lexical unit the Jules lexer can produce.
enum class TokenKind : uint16_t {
  // ── Markers ──────────────────────────────────────────────────────────────
  Unknown = 0,
  Eof,

  // ── Literals ─────────────────────────────────────────────────────────────
  FloatLiteral,   // e.g. 3.14, 1.0e-5
  IntLiteral,     // e.g. 42, 0xFF
  BoolLiteral,    // true | false

  // ── Identifiers & Keywords ───────────────────────────────────────────────
  Identifier,     // any non-keyword identifier

  // Keywords
  KwFn,           // fn
  KwLet,          // let
  KwIn,           // in
  KwIf,           // if
  KwThen,         // then
  KwElse,         // else
  KwGrad,         // grad
  KwWhere,        // where
  KwCast,         // cast
  KwExtern,       // extern

  // Type keywords
  KwF32,          // f32
  KwF64,          // f64
  KwI32,          // i32
  KwI64,          // i64
  KwBool,         // bool
  KwUnit,         // unit

  // ── Punctuation ──────────────────────────────────────────────────────────
  LParen,         // (
  RParen,         // )
  LBracket,       // [
  RBracket,       // ]
  LBrace,         // {
  RBrace,         // }
  Comma,          // ,
  Colon,          // :
  Semicolon,      // ;
  Dot,            // .
  Equals,         // =
  Backslash,      // backslash

  // ── Operators ────────────────────────────────────────────────────────────
  Arrow,          // ->
  FatArrow,       // =>
  DoubleStar,     // **  (matrix multiply)
  Caret,          // ^   (power)
  Plus,           // +
  Minus,          // -
  Star,           // *  (element-wise multiply)
  Slash,          // /
  Percent,        // %

  // Comparison
  EqEq,           // ==
  BangEq,         // !=
  Lt,             // <
  Gt,             // >
  LtEq,           // <=
  GtEq,           // >=

  // Logical
  AmpAmp,         // &&
  PipePipe,       // ||
  Bang,           // !

  // ── Sentinel ─────────────────────────────────────────────────────────────
  NUM_TOKEN_KINDS
};

/// Return the string name of a TokenKind for diagnostics.
const char *tokenKindToString(TokenKind kind);

/// Try to map a keyword string to its TokenKind. Returns Identifier if the
/// string is not a recognized keyword.
TokenKind classifyKeyword(const std::string &text);

/// A single lexical token with source location information.
struct SourceLocation {
  uint32_t line   = 1;
  uint32_t column = 1;
  uint32_t offset = 0;   // byte offset into the source buffer

  bool operator==(const SourceLocation &other) const {
    return line == other.line && column == other.column && offset == other.offset;
  }
  bool operator!=(const SourceLocation &other) const { return !(*this == other); }
};

struct Token {
  TokenKind    kind     = TokenKind::Unknown;
  SourceLocation loc;
  std::string   text;    // the raw text of the token

  Token() = default;
  Token(TokenKind k, SourceLocation l, std::string t)
      : kind(k), loc(l), text(std::move(t)) {}

  bool is(TokenKind k)       const { return kind == k; }
  bool isNot(TokenKind k)    const { return kind != k; }
  bool isAny(TokenKind k1, TokenKind k2) const { return is(k1) || is(k2); }

  template <typename... Kinds>
  bool isAny(TokenKind k1, TokenKind k2, Kinds... rest) const {
    return is(k1) || isAny(k2, rest...);
  }

  /// Is this token one of the type keywords (f32, f64, i32, i64, bool, unit)?
  bool isTypeKeyword() const;

  /// Is this a binary operator token?
  bool isBinaryOp() const;

  /// Get the binary operator precedence (higher = tighter binding).
  /// Returns -1 if not a binary operator.
  int binaryPrecedence() const;

  /// Is this a right-associative binary operator?
  bool isRightAssociative() const;
};

} // namespace jules

std::ostream &operator<<(std::ostream &os, jules::TokenKind kind);
std::ostream &operator<<(std::ostream &os, const jules::Token &tok);

#endif // JULES_TOKEN_H
