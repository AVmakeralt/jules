//===- Lexer.cpp - Jules Language Lexer Implementation ----------------------===//
//
// The lexer implements a standard character-by-character scanner. It supports:
//   - Line comments starting with --
//   - Integer and float literals (including hex integers and scientific notation)
//   - All keywords and identifiers
//   - Multi-character operators: ->, =>, **, ==, !=, <=, >=, &&, ||
//
//===----------------------------------------------------------------------===//

#include "jules/Lexer.h"
#include "jules/Diagnostics.h"
#include <cctype>
#include <sstream>

namespace jules {

Lexer::Lexer(const std::string &source, DiagnosticsEngine &diag,
             std::string sourceName)
    : source_(source), diag_(diag), sourceName_(std::move(sourceName)),
      cursor_(0), lineStart_(0) {
  curLoc_.line = 1;
  curLoc_.column = 1;
  curLoc_.offset = 0;
}

// ── Character access ────────────────────────────────────────────────────────

char Lexer::peekChar() const {
  if (cursor_ >= source_.size()) return '\0';
  return source_[cursor_];
}

char Lexer::peekChar(int offset) const {
  size_t pos = cursor_ + offset;
  if (pos >= source_.size()) return '\0';
  return source_[pos];
}

char Lexer::consumeChar() {
  if (cursor_ >= source_.size()) return '\0';
  char ch = source_[cursor_++];
  curLoc_.offset = cursor_;

  if (ch == '\n') {
    ++curLoc_.line;
    curLoc_.column = 1;
    lineStart_ = cursor_;
  } else {
    ++curLoc_.column;
  }
  return ch;
}

bool Lexer::isEOF() const { return cursor_ >= source_.size(); }

bool Lexer::matchChar(char expected) {
  if (peekChar() == expected) {
    consumeChar();
    return true;
  }
  return false;
}

// ── Whitespace & comments ───────────────────────────────────────────────────

void Lexer::skipWhitespace() {
  while (!isEOF()) {
    char ch = peekChar();
    if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
      consumeChar();
    } else if (ch == '-' && peekChar(1) == '-') {
      // Line comment: -- ...
      skipLineComment();
    } else {
      break;
    }
  }
}

void Lexer::skipLineComment() {
  // Consume the -- and everything until end of line
  consumeChar(); // first -
  consumeChar(); // second -
  while (!isEOF() && peekChar() != '\n') {
    consumeChar();
  }
}

// ── Token lexers ────────────────────────────────────────────────────────────

Token Lexer::lexNumber() {
  SourceLocation startLoc = curLoc_;
  size_t startPos = cursor_;

  bool isFloat = false;

  // Hex prefix
  if (peekChar() == '0' && (peekChar(1) == 'x' || peekChar(1) == 'X')) {
    consumeChar(); // 0
    consumeChar(); // x/X
    while (!isEOF() && std::isxdigit(static_cast<unsigned char>(peekChar()))) {
      consumeChar();
    }
    std::string text = source_.substr(startPos, cursor_ - startPos);
    return formToken(TokenKind::IntLiteral, startLoc, text);
  }

  // Decimal digits
  while (!isEOF() && std::isdigit(static_cast<unsigned char>(peekChar()))) {
    consumeChar();
  }

  // Fractional part
  if (peekChar() == '.' && std::isdigit(static_cast<unsigned char>(peekChar(1)))) {
    isFloat = true;
    consumeChar(); // .
    while (!isEOF() && std::isdigit(static_cast<unsigned char>(peekChar()))) {
      consumeChar();
    }
  }

  // Exponent part
  if (peekChar() == 'e' || peekChar() == 'E') {
    isFloat = true;
    consumeChar(); // e/E
    if (peekChar() == '+' || peekChar() == '-') {
      consumeChar();
    }
    if (!isEOF() && std::isdigit(static_cast<unsigned char>(peekChar()))) {
      while (!isEOF() && std::isdigit(static_cast<unsigned char>(peekChar()))) {
        consumeChar();
      }
    } else {
      diag_.error(curLoc_, "expected digit after exponent");
    }
  }

  std::string text = source_.substr(startPos, cursor_ - startPos);
  return formToken(isFloat ? TokenKind::FloatLiteral : TokenKind::IntLiteral,
                   startLoc, text);
}

Token Lexer::lexIdentifier() {
  SourceLocation startLoc = curLoc_;
  size_t startPos = cursor_;

  while (!isEOF()) {
    char ch = peekChar();
    if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_') {
      consumeChar();
    } else {
      break;
    }
  }

  std::string text = source_.substr(startPos, cursor_ - startPos);
  TokenKind kind = classifyKeyword(text);
  return formToken(kind, startLoc, text);
}

Token Lexer::lexPunctuation() {
  SourceLocation startLoc = curLoc_;
  char ch = peekChar();

  switch (ch) {
  // Single-character tokens
  case '(': consumeChar(); return formToken(TokenKind::LParen, startLoc, "(");
  case ')': consumeChar(); return formToken(TokenKind::RParen, startLoc, ")");
  case '[': consumeChar(); return formToken(TokenKind::LBracket, startLoc, "[");
  case ']': consumeChar(); return formToken(TokenKind::RBracket, startLoc, "]");
  case '{': consumeChar(); return formToken(TokenKind::LBrace, startLoc, "{");
  case '}': consumeChar(); return formToken(TokenKind::RBrace, startLoc, "}");
  case ',': consumeChar(); return formToken(TokenKind::Comma, startLoc, ",");
  case ':': consumeChar(); return formToken(TokenKind::Colon, startLoc, ":");
  case ';': consumeChar(); return formToken(TokenKind::Semicolon, startLoc, ";");
  case '.': consumeChar(); return formToken(TokenKind::Dot, startLoc, ".");
  case '=':
    consumeChar();
    if (matchChar('='))  return formToken(TokenKind::EqEq, startLoc, "==");
    if (matchChar('>'))  return formToken(TokenKind::FatArrow, startLoc, "=>");
    return formToken(TokenKind::Equals, startLoc, "=");
  case '\\': consumeChar(); return formToken(TokenKind::Backslash, startLoc, "\\");
  case '+': consumeChar(); return formToken(TokenKind::Plus, startLoc, "+");
  case '-':
    consumeChar();
    if (matchChar('>'))  return formToken(TokenKind::Arrow, startLoc, "->");
    return formToken(TokenKind::Minus, startLoc, "-");
  case '*':
    consumeChar();
    if (matchChar('*'))  return formToken(TokenKind::DoubleStar, startLoc, "**");
    return formToken(TokenKind::Star, startLoc, "*");
  case '/': consumeChar(); return formToken(TokenKind::Slash, startLoc, "/");
  case '%': consumeChar(); return formToken(TokenKind::Percent, startLoc, "%");
  case '^': consumeChar(); return formToken(TokenKind::Caret, startLoc, "^");
  case '<':
    consumeChar();
    if (matchChar('='))  return formToken(TokenKind::LtEq, startLoc, "<=");
    return formToken(TokenKind::Lt, startLoc, "<");
  case '>':
    consumeChar();
    if (matchChar('='))  return formToken(TokenKind::GtEq, startLoc, ">=");
    return formToken(TokenKind::Gt, startLoc, ">");
  case '!':
    consumeChar();
    if (matchChar('='))  return formToken(TokenKind::BangEq, startLoc, "!=");
    return formToken(TokenKind::Bang, startLoc, "!");
  case '&':
    consumeChar();
    if (matchChar('&'))  return formToken(TokenKind::AmpAmp, startLoc, "&&");
    diag_.error(startLoc, "expected '&&' for logical AND; single '&' is not supported");
    return formToken(TokenKind::Unknown, startLoc, "&");
  case '|':
    consumeChar();
    if (matchChar('|'))  return formToken(TokenKind::PipePipe, startLoc, "||");
    diag_.error(startLoc, "expected '||' for logical OR; single '|' is not supported");
    return formToken(TokenKind::Unknown, startLoc, "|");
  default:
    consumeChar();
    diag_.error(startLoc, std::string("unexpected character '") + ch + "'");
    return formToken(TokenKind::Unknown, startLoc, std::string(1, ch));
  }
}

// ── Main token lexer ────────────────────────────────────────────────────────

Token Lexer::nextToken() {
  // Return a lookahead token if we have one.
  if (!lookahead_.empty()) {
    Token tok = std::move(lookahead_.front());
    lookahead_.erase(lookahead_.begin());
    curTok_ = tok;
    return tok;
  }

  skipWhitespace();

  if (isEOF()) {
    return formToken(TokenKind::Eof, curLoc_, "");
  }

  char ch = peekChar();

  if (std::isdigit(static_cast<unsigned char>(ch))) {
    return lexNumber();
  }
  if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
    return lexIdentifier();
  }
  return lexPunctuation();
}

const Token &Lexer::peekToken() {
  if (lookahead_.empty()) {
    lookahead_.push_back(nextToken());
  }
  return lookahead_.front();
}

const Token &Lexer::peekToken2() {
  while (lookahead_.size() < 2) {
    // Temporarily get tokens without consuming them
    skipWhitespace();
    if (isEOF()) {
      lookahead_.push_back(formToken(TokenKind::Eof, curLoc_, ""));
      break;
    }
    char ch = peekChar();
    Token tok;
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      tok = lexNumber();
    } else if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
      tok = lexIdentifier();
    } else {
      tok = lexPunctuation();
    }
    lookahead_.push_back(std::move(tok));
  }
  return lookahead_[1];
}

SourceLocation Lexer::getCurrentLocation() const { return curLoc_; }

std::vector<Token> Lexer::lexAll() {
  std::vector<Token> tokens;
  while (true) {
    Token tok = nextToken();
    tokens.push_back(std::move(tok));
    if (tokens.back().is(TokenKind::Eof)) break;
  }
  return tokens;
}

Token Lexer::formToken(TokenKind kind, SourceLocation loc,
                       const std::string &text) {
  return Token(kind, loc, text);
}

} // namespace jules
