#include "Reporter.h"
#include "Helpers.h"
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

namespace stable_abi {

void Reporter::addFinding(FindingKind kind, const clang::SourceManager &SM,
                          clang::SourceLocation loc, std::string_view old_text,
                          std::string_view new_text, FindingAction action) {
    if (loc.isInvalid())
        return;

    auto ploc = SM.getPresumedLoc(loc);
    if (ploc.isInvalid())
        return;

    addFinding(kind, ploc.getFilename(), ploc.getLine(), ploc.getColumn(),
               old_text, new_text, action);
}

void Reporter::addFinding(FindingKind kind, std::string_view file,
                           unsigned line, unsigned col,
                           std::string_view old_text,
                           std::string_view new_text, FindingAction action) {
    if (!seen_.emplace(std::string(file), line, col, std::string(old_text)).second)
        return;

    findings_.push_back(
        {kind, std::string(file), line, col, std::string(old_text),
         std::string(new_text), action});

    if (action == FindingAction::Flag)
        ++flag_count_;
    else
        ++rewrite_count_;
}

void Reporter::printReport() const {
    for (const auto &f : findings_) {
        auto label = (f.action == FindingAction::Flag)
            ? std::string_view("FLAG")
            : kindLabel(f.kind);
        llvm::outs() << llvm::format("[%-5s] ", label.data())
                      << f.file << ":" << f.line << ":" << f.col << "  "
                      << f.old_text << " -> "
                      << (f.action == FindingAction::Flag ? "(manual review) " : "")
                      << f.new_text << "\n";
    }
}

void Reporter::printSummary() const {
    llvm::outs() << "\nSummary: " << rewrite_count_ << " auto-rewritable, "
                 << flag_count_ << " flagged for manual review\n";
}

void Reporter::printJson() const {
    llvm::outs() << "{\n  \"findings\": [\n";
    for (size_t i = 0; i < findings_.size(); ++i) {
        const auto &f = findings_[i];
        llvm::outs() << "    {";
        llvm::outs() << "\"kind\": \"" << kindLabel(f.kind) << "\", ";
        llvm::outs() << "\"file\": \"" << jsonEscape(f.file) << "\", ";
        llvm::outs() << "\"line\": " << f.line << ", ";
        llvm::outs() << "\"col\": " << f.col << ", ";
        llvm::outs() << "\"old\": \"" << jsonEscape(f.old_text) << "\", ";
        llvm::outs() << "\"new\": \"" << jsonEscape(f.new_text) << "\", ";
        llvm::outs() << "\"flag\": "
                      << (f.action == FindingAction::Flag ? "true" : "false");
        llvm::outs() << "}";
        if (i + 1 < findings_.size())
            llvm::outs() << ",";
        llvm::outs() << "\n";
    }
    llvm::outs() << "  ],\n";
    llvm::outs() << "  \"rewrites\": " << rewrite_count_ << ",\n";
    llvm::outs() << "  \"flags\": " << flag_count_ << ",\n";
    llvm::outs() << "  \"parse_errors\": " << parse_error_count_ << "\n";
    llvm::outs() << "}\n";
}

void Reporter::recordParseError(const std::string &file) {
    ++parse_error_count_;
    ++parse_errors_by_file_[file];
}

void Reporter::printParseWarnings() const {
    if (parse_error_count_ == 0)
        return;
    llvm::errs() << "\nwarning: parse errors in "
                 << parse_errors_by_file_.size()
                 << " file(s) — results may be incomplete\n";
    for (const auto &[file, count] : parse_errors_by_file_)
        llvm::errs() << "  " << file << ": " << count << " error(s)\n";
}

void Reporter::suppressRedundantFlags() {
    std::set<std::pair<std::string, unsigned>> covered;
    for (const auto &f : findings_)
        if (f.action == FindingAction::Rewrite)
            covered.insert({f.file, f.line});

    size_t removed = 0;
    auto it = std::remove_if(findings_.begin(), findings_.end(),
        [&covered, &removed](const Finding &f) {
            if (f.action == FindingAction::Flag && covered.count({f.file, f.line})) {
                ++removed;
                return true;
            }
            return false;
        });
    findings_.erase(it, findings_.end());
    flag_count_ -= removed;
}

bool Reporter::hasNonIncludeFindingsForFile(std::string_view filename) const {
    for (const auto &f : findings_) {
        if (f.kind != FindingKind::Include && f.file == filename)
            return true;
    }
    return false;
}

std::string_view Reporter::kindLabel(FindingKind kind) {
    switch (kind) {
    case FindingKind::Include:
        return "INCL";
    case FindingKind::Macro:
        return "MACRO";
    case FindingKind::Type:
        return "TYPE";
    case FindingKind::ScalarType:
        return "STYPE";
    case FindingKind::DataPtr:
        return "DPTR";
    case FindingKind::CudaStream:
        return "STRM";
    case FindingKind::DeviceGuard:
        return "GUARD";
    case FindingKind::MethodToFunc:
        return "M2F";
    case FindingKind::FreeFunc:
        return "FUNC";
    case FindingKind::Flag:
        return "FLAG";
    }
    return "?????";
}

} // namespace stable_abi
