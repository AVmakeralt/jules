//===- Parser.h - Jules Language Parser ------------------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the recursive-descent Parser for the Jules language.
// The parser consumes a token stream from the Lexer and produces a fully
// structured AST (Program).
//
// Grammar overview:
//
//   program      ::= (functionDecl | externDecl)*
//   functionDecl ::= identifier ':' typeSig identifier params '=' expr
//   typeSig      ::= type ('->' type)*
//   type         ::= scalarType | tensorType | '(' ')' | '(' typeSig ')'
//   expr         ::= letExpr | lambdaExpr | ifExpr | binOpExpr
//   letExpr      ::= 'let' identifier '=' expr 'in' expr
//   lambdaExpr   ::= '\' params '->' expr
//   ifExpr       ::= 'if' expr 'then' expr 'else' expr
//   binOpExpr    ::= unaryExpr (op unaryExpr)*   [precedence climbing]
//   unaryExpr    ::= ('-' | '!') postfixExpr | postfixExpr
//   postfixExpr  ::= primaryExpr ('[' expr ']' | '(' args ')')*
//   primaryExpr  ::= literal | identifier | '(' expr ')'
//
//===----------------------------------------------------------------------===//

#ifndef JULES_PARSER_H
#define JULES_PARSER_H

#include "jules/AST.h"
#include "jules/Lexer.h"
#include "jules/Diagnostics.h"
#include <memory>
#include <string>
#include <vector>

namespace jules {

class Parser {
public:
  Parser(Lexer &lexer, DiagnosticsEngine &diag);

  /// Parse the entire input into a Program.
  std::unique_ptr<Program> parseProgram();

  /// Parse a single top-level function declaration.
  std::unique_ptr<FunctionDecl> parseFunctionDecl();

  /// Parse a single extern declaration.
  std::unique_ptr<ExternDecl> parseExternDecl();

private:
  // ── Token management ─────────────────────────────────────────────────────
  const Token &currentToken() const;
  const Token &peekToken() const;
  Token consumeToken();
  Token consumeToken(TokenKind expected);
  bool consumeIf(TokenKind expected);
  bool check(TokenKind kind) const;
  SourceLocation currentLocation() const;

  // ── Error recovery ───────────────────────────────────────────────────────
  Token error(TokenKind expected, const std::string &msg);
  void synchronize();

  // ── Type parsing ─────────────────────────────────────────────────────────
  std::unique_ptr<TypeNode> parseType();
  std::unique_ptr<ScalarType> parseScalarType();
  std::unique_ptr<TensorType> parseTensorType();
  std::unique_ptr<FunctionType> parseFunctionTypeSig();

  // ── Expression parsing (precedence climbing) ─────────────────────────────
  std::unique_ptr<Expr> parseExpr();
  std::unique_ptr<Expr> parseLetExpr();
  std::unique_ptr<Expr> parseLambdaExpr();
  std::unique_ptr<Expr> parseIfExpr();
  std::unique_ptr<Expr> parseBinOpExpr(int minPrec = 0);
  std::unique_ptr<Expr> parseUnaryExpr();
  std::unique_ptr<Expr> parsePostfixExpr();
  std::unique_ptr<Expr> parsePrimaryExpr();

  // ── Helpers ──────────────────────────────────────────────────────────────
  std::unique_ptr<Expr> parseBlockOrExpr();
  std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>> parseParamList();
  std::vector<Dimension> parseDimensions();

  // ── State ────────────────────────────────────────────────────────────────
  Lexer             &lexer_;
  DiagnosticsEngine &diag_;
  Token              curTok_;
  bool               hasError_;
};

} // namespace jules

#endif // JULES_PARSER_H
