//===- Parser.cpp - Jules Language Parser Implementation --------------------===//
//
// Recursive-descent parser with precedence climbing for binary operators.
//
//===----------------------------------------------------------------------===//

#include "jules/Parser.h"
#include "jules/Diagnostics.h"
#include <cassert>
#include <sstream>

namespace jules {

Parser::Parser(Lexer &lexer, DiagnosticsEngine &diag)
    : lexer_(lexer), diag_(diag), hasError_(false) {
  curTok_ = lexer_.nextToken();
}

// ── Token management ────────────────────────────────────────────────────────

const Token &Parser::currentToken() const { return curTok_; }

const Token &Parser::peekToken() const { return lexer_.peekToken(); }

Token Parser::consumeToken() {
  Token tok = std::move(curTok_);
  curTok_ = lexer_.nextToken();
  return tok;
}

Token Parser::consumeToken(TokenKind expected) {
  if (curTok_.isNot(expected)) {
    return error(expected, std::string("expected '") +
                              tokenKindToString(expected) + "', got '" +
                              curTok_.text + "'");
  }
  return consumeToken();
}

bool Parser::consumeIf(TokenKind expected) {
  if (curTok_.is(expected)) {
    consumeToken();
    return true;
  }
  return false;
}

bool Parser::check(TokenKind kind) const { return curTok_.is(kind); }

SourceLocation Parser::currentLocation() const { return curTok_.loc; }

Token Parser::error(TokenKind expected, const std::string &msg) {
  diag_.error(curTok_.loc, msg);
  hasError_ = true;
  return curTok_;
}

/// Skip tokens until we find a likely recovery point.
void Parser::synchronize() {
  while (!curTok_.is(TokenKind::Eof)) {
    // Stop at tokens that could start a new top-level declaration.
    if (curTok_.isAny(TokenKind::KwFn, TokenKind::KwExtern, TokenKind::Identifier)) {
      // But only if the next token is a colon (type annotation) — this
      // indicates a top-level function declaration.
      if (curTok_.is(TokenKind::Identifier) &&
          peekToken().isNot(TokenKind::Colon)) {
        consumeToken();
        continue;
      }
      return;
    }
    consumeToken();
  }
}

// ── Program ─────────────────────────────────────────────────────────────────

std::unique_ptr<Program> Parser::parseProgram() {
  auto program = std::make_unique<Program>();

  while (!curTok_.is(TokenKind::Eof)) {
    // Skip any top-level semicolons
    if (consumeIf(TokenKind::Semicolon)) continue;

    // extern declaration
    if (check(TokenKind::KwExtern)) {
      auto ext = parseExternDecl();
      if (ext) program->addExtern(std::move(ext));
      else     synchronize();
      continue;
    }

    // Function declaration (always starts with identifier : type)
    if (check(TokenKind::Identifier)) {
      auto fn = parseFunctionDecl();
      if (fn) program->addFunction(std::move(fn));
      else    synchronize();
      continue;
    }

    diag_.error(curTok_.loc,
                "expected function declaration or extern declaration");
    synchronize();
  }

  return program;
}

// ── Extern declaration ──────────────────────────────────────────────────────

std::unique_ptr<ExternDecl> Parser::parseExternDecl() {
  SourceLocation loc = curTok_.loc;
  consumeToken(TokenKind::KwExtern);

  // Parse the name
  Token nameTok = consumeToken(TokenKind::Identifier);

  // Parse the colon and type
  consumeToken(TokenKind::Colon);
  auto type = parseType();

  if (!type) {
    diag_.error(loc, "expected type after 'extern name :'");
    return nullptr;
  }

  return std::make_unique<ExternDecl>(nameTok.text, std::move(type), loc);
}

// ── Function declaration ────────────────────────────────────────────────────
//
// Grammar:
//   name : Type1 -> Type2 -> ... -> ResultType
//   name param1 param2 ... = body
//
// The first line is the type signature. The second line is the implementation
// with explicit parameter names and the body expression.
//
// We parse this as:
//   1. Read the name (identifier)
//   2. Read the colon
//   3. Parse the type signature (a chain of types separated by ->)
//   4. Read the name again (same identifier)
//   5. Parse the parameter list
//   6. Read the equals sign
//   7. Parse the body expression

std::unique_ptr<FunctionDecl> Parser::parseFunctionDecl() {
  SourceLocation loc = curTok_.loc;

  // 1. Name
  Token nameTok = consumeToken(TokenKind::Identifier);
  const std::string funcName = nameTok.text;

  // 2. Colon
  consumeToken(TokenKind::Colon);

  // 3. Type signature
  auto funcType = parseFunctionTypeSig();
  if (!funcType) {
    diag_.error(loc, "expected function type signature after ':'");
    return nullptr;
  }

  // 4. Name again (definition)
  Token defNameTok = consumeToken(TokenKind::Identifier);
  if (defNameTok.text != funcName) {
    diag_.warning(defNameTok.loc,
                  "definition name '" + defNameTok.text +
                  "' does not match declaration name '" + funcName + "'");
  }

  // 5. Parse parameter names (with optional type annotations)
  auto params = parseParamList();

  // 6. Equals sign
  consumeToken(TokenKind::Equals);

  // 7. Body
  auto body = parseExpr();
  if (!body) {
    diag_.error(loc, "expected function body after '='");
    return nullptr;
  }

  return std::make_unique<FunctionDecl>(
      funcName, std::move(funcType), std::move(params), std::move(body), loc);
}

// ── Parameter list ──────────────────────────────────────────────────────────

std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>>
Parser::parseParamList() {
  std::vector<std::pair<std::string, std::unique_ptr<TypeNode>>> params;

  while (check(TokenKind::Identifier) || check(TokenKind::LParen)) {
    std::string paramName;
    std::unique_ptr<TypeNode> paramType;

    // Unit parameter: ()
    if (check(TokenKind::LParen)) {
      consumeToken(); // (
      if (!consumeIf(TokenKind::RParen)) {
        diag_.error(curTok_.loc, "expected ')' for unit parameter");
        break;
      }
      paramName = "()";
      paramType = std::make_unique<ScalarType>(ScalarType::SK_Unit);
    } else {
      // Named parameter
      Token paramTok = consumeToken(TokenKind::Identifier);
      paramName = paramTok.text;
    }

    params.emplace_back(std::move(paramName), std::move(paramType));
  }

  return params;
}

// ── Type parsing ────────────────────────────────────────────────────────────

std::unique_ptr<TypeNode> Parser::parseType() {
  // Tensor type: [D1, D2, ...]ElementType
  if (check(TokenKind::LBracket)) {
    return parseTensorType();
  }

  // Unit type: ()
  if (check(TokenKind::LParen)) {
    SourceLocation loc = curTok_.loc;
    consumeToken(); // (
    if (consumeIf(TokenKind::RParen)) {
      return std::make_unique<ScalarType>(ScalarType::SK_Unit, loc);
    }
    // Could be a parenthesized function type
    auto inner = parseFunctionTypeSig();
    consumeToken(TokenKind::RParen);
    return inner;
  }

  // Scalar type keyword
  if (curTok_.isTypeKeyword()) {
    return parseScalarType();
  }

  // Identifier as symbolic type variable (uncommon but allowed)
  if (check(TokenKind::Identifier)) {
    // This could be a named type reference
    Token tok = consumeToken();
    diag_.warning(tok.loc, "unknown type '" + tok.text + "', defaulting to f32");
    return std::make_unique<ScalarType>(ScalarType::SK_F32, tok.loc);
  }

  diag_.error(curTok_.loc, "expected type");
  return nullptr;
}

std::unique_ptr<ScalarType> Parser::parseScalarType() {
  SourceLocation loc = curTok_.loc;
  TokenKind kw = curTok_.kind;
  consumeToken();
  return std::make_unique<ScalarType>(ScalarType::keywordToScalarKind(kw), loc);
}

std::unique_ptr<TensorType> Parser::parseTensorType() {
  SourceLocation loc = curTok_.loc;
  consumeToken(TokenKind::LBracket);

  auto dims = parseDimensions();

  consumeToken(TokenKind::RBracket);

  // Element type follows
  if (!curTok_.isTypeKeyword()) {
    diag_.error(curTok_.loc, "expected element type after tensor dimensions");
    return nullptr;
  }

  auto elemKind = ScalarType::keywordToScalarKind(curTok_.kind);
  consumeToken();

  return std::make_unique<TensorType>(std::move(dims), elemKind, loc);
}

std::vector<Dimension> Parser::parseDimensions() {
  std::vector<Dimension> dims;

  while (true) {
    if (check(TokenKind::IntLiteral)) {
      // Concrete dimension
      int64_t size = std::stoll(curTok_.text);
      consumeToken();
      dims.push_back(Dimension::concrete(size));
    } else if (check(TokenKind::Identifier)) {
      // Symbolic dimension
      std::string name = curTok_.text;
      consumeToken();
      dims.push_back(Dimension::symbolic(name));
    } else {
      // No more dimensions
      break;
    }

    // Comma between dimensions
    if (!consumeIf(TokenKind::Comma)) {
      break;
    }
  }

  return dims;
}

std::unique_ptr<FunctionType> Parser::parseFunctionTypeSig() {
  SourceLocation loc = curTok_.loc;
  std::vector<std::unique_ptr<TypeNode>> paramTypes;

  // Parse the first type
  auto firstType = parseType();
  if (!firstType) return nullptr;
  paramTypes.push_back(std::move(firstType));

  // Parse additional parameter types separated by ->
  while (consumeIf(TokenKind::Arrow)) {
    auto paramType = parseType();
    if (!paramType) {
      diag_.error(curTok_.loc, "expected type after '->'");
      return nullptr;
    }
    paramTypes.push_back(std::move(paramType));
  }

  // The last type in the chain is the result type.
  // Function type: T1 -> T2 -> ... -> Tn-1 -> Tn
  // Where T1..Tn-1 are parameter types, Tn is result.
  if (paramTypes.size() < 2) {
    // A single type — it's just a value, not really a function.
    // Wrap it as a zero-parameter function.
    auto result = std::move(paramTypes.back());
    paramTypes.pop_back();
    return std::make_unique<FunctionType>(std::move(paramTypes),
                                           std::move(result), loc);
  }

  auto result = std::move(paramTypes.back());
  paramTypes.pop_back();
  return std::make_unique<FunctionType>(std::move(paramTypes),
                                         std::move(result), loc);
}

// ── Expression parsing ──────────────────────────────────────────────────────

std::unique_ptr<Expr> Parser::parseExpr() {
  // Let expression
  if (check(TokenKind::KwLet)) {
    return parseLetExpr();
  }

  // Lambda expression
  if (check(TokenKind::Backslash)) {
    return parseLambdaExpr();
  }

  // If expression
  if (check(TokenKind::KwIf)) {
    return parseIfExpr();
  }

  return parseBinOpExpr();
}

std::unique_ptr<Expr> Parser::parseLetExpr() {
  SourceLocation loc = curTok_.loc;
  consumeToken(TokenKind::KwLet);

  // Name
  Token nameTok = consumeToken(TokenKind::Identifier);

  // Optional type annotation: let x : Type = ...
  std::unique_ptr<TypeNode> typeAnnotation;
  if (consumeIf(TokenKind::Colon)) {
    typeAnnotation = parseType();
  }

  // =
  consumeToken(TokenKind::Equals);

  // Value
  auto value = parseExpr();

  // 'in' keyword
  consumeToken(TokenKind::KwIn);

  // Body
  auto body = parseExpr();

  auto let = std::make_unique<LetExpr>(nameTok.text, std::move(value),
                                        std::move(body), loc);
  if (typeAnnotation) {
    let->setTypeAnnotation(std::move(typeAnnotation));
  }
  return let;
}

std::unique_ptr<Expr> Parser::parseLambdaExpr() {
  SourceLocation loc = curTok_.loc;
  consumeToken(TokenKind::Backslash);

  // Parameters
  std::vector<LambdaExpr::Param> params;
  while (check(TokenKind::Identifier)) {
    LambdaExpr::Param p;
    p.name = curTok_.text;
    consumeToken();

    // Optional type annotation: \x : Type -> ...
    if (consumeIf(TokenKind::Colon)) {
      p.type = parseType();
    }

    params.push_back(std::move(p));
  }

  // Arrow
  consumeToken(TokenKind::Arrow);

  // Body
  auto body = parseExpr();

  return std::make_unique<LambdaExpr>(std::move(params), std::move(body), loc);
}

std::unique_ptr<Expr> Parser::parseIfExpr() {
  SourceLocation loc = curTok_.loc;
  consumeToken(TokenKind::KwIf);

  auto cond = parseExpr();
  consumeToken(TokenKind::KwThen);

  auto trueBranch = parseExpr();
  consumeToken(TokenKind::KwElse);

  auto falseBranch = parseExpr();

  return std::make_unique<IfExpr>(std::move(cond), std::move(trueBranch),
                                   std::move(falseBranch), loc);
}

// ── Precedence-climbing binary operator parsing ─────────────────────────────

std::unique_ptr<Expr> Parser::parseBinOpExpr(int minPrec) {
  auto lhs = parseUnaryExpr();
  if (!lhs) return nullptr;

  while (curTok_.isBinaryOp()) {
    int prec = curTok_.binaryPrecedence();
    if (prec < minPrec) break;

    Token opTok = consumeToken();
    BinaryExpr::Op binOp;

    switch (opTok.kind) {
    case TokenKind::Plus:       binOp = BinaryExpr::Add;    break;
    case TokenKind::Minus:      binOp = BinaryExpr::Sub;    break;
    case TokenKind::Star:       binOp = BinaryExpr::Mul;    break;
    case TokenKind::Slash:      binOp = BinaryExpr::Div;    break;
    case TokenKind::Percent:    binOp = BinaryExpr::Mod;    break;
    case TokenKind::DoubleStar: binOp = BinaryExpr::MatMul; break;
    case TokenKind::Caret:      binOp = BinaryExpr::Pow;    break;
    case TokenKind::EqEq:       binOp = BinaryExpr::Eq;     break;
    case TokenKind::BangEq:     binOp = BinaryExpr::Neq;    break;
    case TokenKind::Lt:         binOp = BinaryExpr::Lt;     break;
    case TokenKind::Gt:         binOp = BinaryExpr::Gt;     break;
    case TokenKind::LtEq:       binOp = BinaryExpr::Leq;    break;
    case TokenKind::GtEq:       binOp = BinaryExpr::Geq;    break;
    case TokenKind::AmpAmp:     binOp = BinaryExpr::And;    break;
    case TokenKind::PipePipe:   binOp = BinaryExpr::Or;     break;
    default:
      diag_.error(opTok.loc, "unexpected binary operator");
      return lhs;
    }

    // Compute the next minimum precedence.
    int nextMinPrec = opTok.isRightAssociative() ? prec : prec + 1;

    auto rhs = parseBinOpExpr(nextMinPrec);
    if (!rhs) {
      diag_.error(opTok.loc, "expected expression after binary operator");
      return lhs;
    }

    lhs = std::make_unique<BinaryExpr>(binOp, std::move(lhs),
                                        std::move(rhs), opTok.loc);
  }

  return lhs;
}

std::unique_ptr<Expr> Parser::parseUnaryExpr() {
  SourceLocation loc = curTok_.loc;

  // Negate
  if (check(TokenKind::Minus)) {
    consumeToken();
    auto operand = parseUnaryExpr();
    return std::make_unique<UnaryExpr>(UnaryExpr::Negate,
                                        std::move(operand), loc);
  }

  // Logical not
  if (check(TokenKind::Bang)) {
    consumeToken();
    auto operand = parseUnaryExpr();
    return std::make_unique<UnaryExpr>(UnaryExpr::Not,
                                        std::move(operand), loc);
  }

  return parsePostfixExpr();
}

std::unique_ptr<Expr> Parser::parsePostfixExpr() {
  auto expr = parsePrimaryExpr();
  if (!expr) return nullptr;

  while (true) {
    // Function call: expr(args...)
    if (check(TokenKind::LParen)) {
      SourceLocation loc = curTok_.loc;
      consumeToken(); // (

      std::vector<std::unique_ptr<Expr>> args;
      if (!check(TokenKind::RParen)) {
        args.push_back(parseExpr());
        while (consumeIf(TokenKind::Comma)) {
          args.push_back(parseExpr());
        }
      }
      consumeToken(TokenKind::RParen);

      expr = std::make_unique<CallExpr>(std::move(expr), std::move(args), loc);
      continue;
    }

    // Index expression: expr[indices...]
    if (check(TokenKind::LBracket)) {
      SourceLocation loc = curTok_.loc;
      consumeToken(); // [

      std::vector<std::unique_ptr<Expr>> indices;
      if (!check(TokenKind::RBracket)) {
        indices.push_back(parseExpr());
        while (consumeIf(TokenKind::Comma)) {
          indices.push_back(parseExpr());
        }
      }
      consumeToken(TokenKind::RBracket);

      expr = std::make_unique<IndexExpr>(std::move(expr), std::move(indices), loc);
      continue;
    }

    break;
  }

  return expr;
}

std::unique_ptr<Expr> Parser::parsePrimaryExpr() {
  SourceLocation loc = curTok_.loc;

  // Float literal
  if (check(TokenKind::FloatLiteral)) {
    double val = std::stod(curTok_.text);
    consumeToken();
    return std::make_unique<FloatLiteralExpr>(val, loc);
  }

  // Int literal
  if (check(TokenKind::IntLiteral)) {
    int64_t val = std::stoll(curTok_.text, nullptr, 0);
    consumeToken();
    return std::make_unique<IntLiteralExpr>(val, loc);
  }

  // Bool literal
  if (check(TokenKind::BoolLiteral)) {
    bool val = (curTok_.text == "true");
    consumeToken();
    return std::make_unique<BoolLiteralExpr>(val, loc);
  }

  // Unit literal: ()
  if (check(TokenKind::LParen) && peekToken().is(TokenKind::RParen)) {
    consumeToken(); // (
    consumeToken(); // )
    return std::make_unique<UnitLiteralExpr>(loc);
  }

  // Parenthesized expression
  if (check(TokenKind::LParen)) {
    consumeToken(); // (
    auto expr = parseExpr();
    consumeToken(TokenKind::RParen);
    return expr;
  }

  // Tensor literal: [e1, e2, ...]
  if (check(TokenKind::LBracket)) {
    consumeToken(); // [
    std::vector<std::unique_ptr<Expr>> elements;
    if (!check(TokenKind::RBracket)) {
      elements.push_back(parseExpr());
      while (consumeIf(TokenKind::Comma)) {
        elements.push_back(parseExpr());
      }
    }
    consumeToken(TokenKind::RBracket);
    return std::make_unique<TensorLiteralExpr>(std::move(elements), loc);
  }

  // grad built-in: grad(fn_expr, var_name)
  if (check(TokenKind::KwGrad)) {
    consumeToken(); // grad
    consumeToken(TokenKind::LParen);

    auto fnExpr = parseExpr();
    consumeToken(TokenKind::Comma);

    Token varTok = consumeToken(TokenKind::Identifier);
    consumeToken(TokenKind::RParen);

    return std::make_unique<GRADExpr>(std::move(fnExpr), varTok.text, loc);
  }

  // cast built-in: cast(expr, Type)
  if (check(TokenKind::KwCast)) {
    consumeToken(); // cast
    consumeToken(TokenKind::LParen);

    auto expr = parseExpr();
    consumeToken(TokenKind::Comma);

    auto targetType = parseType();
    consumeToken(TokenKind::RParen);

    return std::make_unique<CastExpr>(std::move(expr), std::move(targetType), loc);
  }

  // Identifier (variable reference or function name)
  if (check(TokenKind::Identifier)) {
    std::string name = curTok_.text;
    consumeToken();
    return std::make_unique<IdentifierExpr>(std::move(name), loc);
  }

  diag_.error(curTok_.loc, "expected expression, got '" + curTok_.text + "'");
  return nullptr;
}

} // namespace jules
