//===- Diagnostics.h - Jules Compiler Diagnostics --------------------------===//
//
// Copyright (c) 2025 Jules Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the diagnostic infrastructure for the Jules compiler.
// Diagnostics carry source locations and severity levels. A DiagnosticsEngine
// collects them and can render them to stderr or a stream.
//
//===----------------------------------------------------------------------===//

#ifndef JULES_DIAGNOSTICS_H
#define JULES_DIAGNOSTICS_H

#include "jules/Token.h"
#include <cassert>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace jules {

enum class DiagnosticSeverity {
  Note,
  Warning,
  Error,
  Fatal,
};

/// A single diagnostic message with location, severity, and text.
struct Diagnostic {
  DiagnosticSeverity severity;
  SourceLocation     loc;
  std::string        message;

  std::string toString(const std::string &sourceName = "<input>") const;
};

/// Collects diagnostics and emits them through a user-provided callback.
class DiagnosticsEngine {
public:
  using EmitCallback = std::function<void(const Diagnostic &)>;

  explicit DiagnosticsEngine(EmitCallback cb = defaultEmit)
      : callback_(std::move(cb)), errorCount_(0), fatalCount_(0) {}

  /// Emit a diagnostic.
  void emit(DiagnosticSeverity severity, SourceLocation loc,
            const std::string &msg);

  /// Convenience methods.
  void note(SourceLocation loc, const std::string &msg) {
    emit(DiagnosticSeverity::Note, loc, msg);
  }
  void warning(SourceLocation loc, const std::string &msg) {
    emit(DiagnosticSeverity::Warning, loc, msg);
  }
  void error(SourceLocation loc, const std::string &msg) {
    emit(DiagnosticSeverity::Error, loc, msg);
  }
  void fatal(SourceLocation loc, const std::string &msg) {
    emit(DiagnosticSeverity::Fatal, loc, msg);
  }

  /// How many errors (including fatal) have been emitted?
  unsigned getErrorCount() const { return errorCount_; }
  unsigned getFatalCount() const { return fatalCount_; }

  /// Has any error been emitted?
  bool hasErrors() const { return errorCount_ > 0; }

  /// Reset the engine (e.g. between compilation units).
  void reset() {
    diagnostics_.clear();
    errorCount_ = 0;
    fatalCount_ = 0;
  }

  /// Access all collected diagnostics.
  const std::vector<Diagnostic> &getDiagnostics() const { return diagnostics_; }

  /// Default emitter that prints to stderr.
  static void defaultEmit(const Diagnostic &diag);

private:
  EmitCallback              callback_;
  std::vector<Diagnostic>   diagnostics_;
  unsigned                  errorCount_;
  unsigned                  fatalCount_;
};

/// RAII helper for building a diagnostic message with <<.
class DiagnosticBuilder {
public:
  DiagnosticBuilder(DiagnosticsEngine &engine, DiagnosticSeverity severity,
                    SourceLocation loc)
      : engine_(engine), severity_(severity), loc_(loc) {}

  ~DiagnosticBuilder() {
    engine_.emit(severity_, loc_, stream_.str());
  }

  // Non-copyable, movable.
  DiagnosticBuilder(const DiagnosticBuilder &) = delete;
  DiagnosticBuilder &operator=(const DiagnosticBuilder &) = delete;
  DiagnosticBuilder(DiagnosticBuilder &&o) noexcept
      : engine_(o.engine_), severity_(o.severity_), loc_(o.loc_),
        stream_(std::move(o.stream_)) {}

  template <typename T> DiagnosticBuilder &operator<<(const T &val) {
    stream_ << val;
    return *this;
  }

private:
  DiagnosticsEngine   &engine_;
  DiagnosticSeverity   severity_;
  SourceLocation       loc_;
  std::stringstream    stream_;
};

/// Create a DiagnosticBuilder for an error at \p loc.
inline DiagnosticBuilder diagError(DiagnosticsEngine &engine, SourceLocation loc) {
  return DiagnosticBuilder(engine, DiagnosticSeverity::Error, loc);
}

/// Create a DiagnosticBuilder for a warning at \p loc.
inline DiagnosticBuilder diagWarning(DiagnosticsEngine &engine, SourceLocation loc) {
  return DiagnosticBuilder(engine, DiagnosticSeverity::Warning, loc);
}

/// Create a DiagnosticBuilder for a note at \p loc.
inline DiagnosticBuilder diagNote(DiagnosticsEngine &engine, SourceLocation loc) {
  return DiagnosticBuilder(engine, DiagnosticSeverity::Note, loc);
}

/// Create a DiagnosticBuilder for a fatal error at \p loc.
inline DiagnosticBuilder diagFatal(DiagnosticsEngine &engine, SourceLocation loc) {
  return DiagnosticBuilder(engine, DiagnosticSeverity::Fatal, loc);
}

} // namespace jules

#endif // JULES_DIAGNOSTICS_H
