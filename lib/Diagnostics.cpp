//===- Diagnostics.cpp - Jules Diagnostics Implementation ------------------===//

#include "jules/Diagnostics.h"
#include <iostream>
#include <sstream>

namespace jules {

std::string Diagnostic::toString(const std::string &sourceName) const {
  const char *severityStr = "";
  switch (severity) {
  case DiagnosticSeverity::Note:    severityStr = "note"; break;
  case DiagnosticSeverity::Warning: severityStr = "warning"; break;
  case DiagnosticSeverity::Error:   severityStr = "error"; break;
  case DiagnosticSeverity::Fatal:   severityStr = "fatal error"; break;
  }
  std::ostringstream oss;
  oss << sourceName << ":" << loc.line << ":" << loc.column
      << ": " << severityStr << ": " << message;
  return oss.str();
}

void DiagnosticsEngine::defaultEmit(const Diagnostic &diag) {
  std::cerr << diag.toString() << std::endl;
}

void DiagnosticsEngine::emit(DiagnosticSeverity severity, SourceLocation loc,
                             const std::string &msg) {
  Diagnostic diag{severity, loc, msg};
  diagnostics_.push_back(std::move(diag));

  if (severity == DiagnosticSeverity::Error ||
      severity == DiagnosticSeverity::Fatal) {
    ++errorCount_;
  }
  if (severity == DiagnosticSeverity::Fatal) {
    ++fatalCount_;
  }

  if (callback_) {
    callback_(diagnostics_.back());
  }
}

} // namespace jules
