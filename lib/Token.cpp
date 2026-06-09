//===- Token.cpp - Jules Token Implementation -------------------------------===//

#include "jules/Token.h"
#include <cstring>

namespace jules {

namespace {
struct KeywordEntry {
  const char *text;
  TokenKind   kind;
};

const KeywordEntry keywords[] = {
  {"fn",    TokenKind::KwFn},
  {"let",   TokenKind::KwLet},
  {"in",    TokenKind::KwIn},
  {"if",    TokenKind::KwIf},
  {"then",  TokenKind::KwThen},
  {"else",  TokenKind::KwElse},
  {"grad",  TokenKind::KwGrad},
  {"where", TokenKind::KwWhere},
  {"cast",  TokenKind::KwCast},
  {"extern",TokenKind::KwExtern},
  {"f32",   TokenKind::KwF32},
  {"f64",   TokenKind::KwF64},
  {"i32",   TokenKind::KwI32},
  {"i64",   TokenKind::KwI64},
  {"bool",  TokenKind::KwBool},
  {"unit",  TokenKind::KwUnit},
  {"true",  TokenKind::BoolLiteral},
  {"false", TokenKind::BoolLiteral},
};

constexpr size_t numKeywords = sizeof(keywords) / sizeof(KeywordEntry);
} // anonymous namespace

const char *tokenKindToString(TokenKind kind) {
  switch (kind) {
  case TokenKind::Unknown:       return "unknown";
  case TokenKind::Eof:           return "eof";
  case TokenKind::FloatLiteral:  return "float_literal";
  case TokenKind::IntLiteral:    return "int_literal";
  case TokenKind::BoolLiteral:   return "bool_literal";
  case TokenKind::Identifier:    return "identifier";
  case TokenKind::KwFn:          return "fn";
  case TokenKind::KwLet:         return "let";
  case TokenKind::KwIn:          return "in";
  case TokenKind::KwIf:          return "if";
  case TokenKind::KwThen:        return "then";
  case TokenKind::KwElse:        return "else";
  case TokenKind::KwGrad:        return "grad";
  case TokenKind::KwWhere:       return "where";
  case TokenKind::KwCast:        return "cast";
  case TokenKind::KwExtern:      return "extern";
  case TokenKind::KwF32:         return "f32";
  case TokenKind::KwF64:         return "f64";
  case TokenKind::KwI32:         return "i32";
  case TokenKind::KwI64:         return "i64";
  case TokenKind::KwBool:        return "bool";
  case TokenKind::KwUnit:        return "unit";
  case TokenKind::LParen:        return "(";
  case TokenKind::RParen:        return ")";
  case TokenKind::LBracket:      return "[";
  case TokenKind::RBracket:      return "]";
  case TokenKind::LBrace:        return "{";
  case TokenKind::RBrace:        return "}";
  case TokenKind::Comma:         return ",";
  case TokenKind::Colon:         return ":";
  case TokenKind::Semicolon:     return ";";
  case TokenKind::Dot:           return ".";
  case TokenKind::Equals:        return "=";
  case TokenKind::Backslash:     return "\\";
  case TokenKind::Arrow:         return "->";
  case TokenKind::FatArrow:      return "=>";
  case TokenKind::DoubleStar:    return "**";
  case TokenKind::Caret:         return "^";
  case TokenKind::Plus:          return "+";
  case TokenKind::Minus:         return "-";
  case TokenKind::Star:          return "*";
  case TokenKind::Slash:         return "/";
  case TokenKind::Percent:       return "%";
  case TokenKind::EqEq:          return "==";
  case TokenKind::BangEq:        return "!=";
  case TokenKind::Lt:            return "<";
  case TokenKind::Gt:            return ">";
  case TokenKind::LtEq:          return "<=";
  case TokenKind::GtEq:          return ">=";
  case TokenKind::AmpAmp:        return "&&";
  case TokenKind::PipePipe:      return "||";
  case TokenKind::Bang:          return "!";
  case TokenKind::NUM_TOKEN_KINDS: return "NUM_TOKEN_KINDS";
  }
  return "invalid";
}

TokenKind classifyKeyword(const std::string &text) {
  for (size_t i = 0; i < numKeywords; ++i) {
    if (text == keywords[i].text) {
      return keywords[i].kind;
    }
  }
  return TokenKind::Identifier;
}

bool Token::isTypeKeyword() const {
  return isAny(TokenKind::KwF32, TokenKind::KwF64,
               TokenKind::KwI32, TokenKind::KwI64,
               TokenKind::KwBool, TokenKind::KwUnit);
}

bool Token::isBinaryOp() const {
  return isAny(TokenKind::Plus, TokenKind::Minus, TokenKind::Star,
               TokenKind::Slash, TokenKind::Percent, TokenKind::DoubleStar,
               TokenKind::Caret, TokenKind::EqEq, TokenKind::BangEq,
               TokenKind::Lt, TokenKind::Gt, TokenKind::LtEq, TokenKind::GtEq,
               TokenKind::AmpAmp, TokenKind::PipePipe);
}

int Token::binaryPrecedence() const {
  switch (kind) {
  // Logical OR (lowest precedence)
  case TokenKind::PipePipe:   return 1;
  // Logical AND
  case TokenKind::AmpAmp:     return 2;
  // Equality
  case TokenKind::EqEq:
  case TokenKind::BangEq:     return 3;
  // Comparison
  case TokenKind::Lt:
  case TokenKind::Gt:
  case TokenKind::LtEq:
  case TokenKind::GtEq:       return 4;
  // Addition / subtraction
  case TokenKind::Plus:
  case TokenKind::Minus:      return 5;
  // Multiplication / division / modulo
  case TokenKind::Star:
  case TokenKind::Slash:
  case TokenKind::Percent:    return 6;
  // Power (right-associative)
  case TokenKind::Caret:      return 7;
  // Matrix multiplication
  case TokenKind::DoubleStar: return 8;
  default:                    return -1;
  }
}

bool Token::isRightAssociative() const {
  return kind == TokenKind::Caret;
}

} // namespace jules

std::ostream &operator<<(std::ostream &os, jules::TokenKind kind) {
  return os << jules::tokenKindToString(kind);
}

std::ostream &operator<<(std::ostream &os, const jules::Token &tok) {
  os << tok.loc.line << ":" << tok.loc.column << " "
     << jules::tokenKindToString(tok.kind);
  if (!tok.text.empty()) {
    os << " '" << tok.text << "'";
  }
  return os;
}
