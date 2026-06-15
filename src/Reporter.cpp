#include "Reporter.h"
#include "Helpers.h"
#include <fstream>
#include <llvm/Support/FileSystem.h>
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

    std::lock_guard<std::mutex> lock(mutex_);
    addFindingLocked(kind, ploc.getFilename(), ploc.getLine(), ploc.getColumn(),
                     old_text, new_text, action);
}

void Reporter::addFinding(FindingKind kind, std::string_view file,
                          unsigned line, unsigned col,
                          std::string_view old_text, std::string_view new_text,
                          FindingAction action) {
    std::lock_guard<std::mutex> lock(mutex_);
    addFindingLocked(kind, file, line, col, old_text, new_text, action);
}

void Reporter::addFindingLocked(FindingKind kind, std::string_view file,
                                unsigned line, unsigned col,
                                std::string_view old_text,
                                std::string_view new_text,
                                FindingAction action) {
    if (!seen_.emplace(std::string(file), line, col, std::string(old_text))
             .second)
        return;

    findings_.push_back({kind, std::string(file), line, col,
                         std::string(old_text), std::string(new_text), action});

    if (action == FindingAction::Flag)
        ++flag_count_;
    else
        ++rewrite_count_;
}

static std::string_view shortenPath(std::string_view path,
                                    std::string_view root) {
    if (!root.empty() && path.starts_with(root)) {
        path.remove_prefix(root.size());
        if (!path.empty() && path[0] == '/')
            path.remove_prefix(1);
    }
    return path;
}

void Reporter::printReport(std::string_view projectRoot) const {
    if (findings_.empty())
        return;

    bool hasFlags = flag_count_ > 0;
    bool hasRewrites = rewrite_count_ > 0;

    if (hasFlags) {
        bool hasAstFlags = false;
        bool hasTextFlags = false;
        for (const auto &f : findings_) {
            if (f.action != FindingAction::Flag)
                continue;
            bool isText =
                f.new_text.find("not analyzed by AST") != std::string::npos;
            if (isText)
                hasTextFlags = true;
            else
                hasAstFlags = true;
        }

        if (hasAstFlags) {
            llvm::outs() << "Manual review needed:\n";
            for (const auto &f : findings_) {
                if (f.action != FindingAction::Flag)
                    continue;
                if (f.new_text.find("not analyzed by AST") != std::string::npos)
                    continue;
                auto path = shortenPath(f.file, projectRoot);
                llvm::outs() << "  " << path << ":" << f.line << "  "
                             << f.old_text << "\n";
            }
        }

        if (hasTextFlags) {
            llvm::outs() << "\nAdditional patterns (not fully analyzed by "
                            "AST):\n";
            for (const auto &f : findings_) {
                if (f.action != FindingAction::Flag)
                    continue;
                if (f.new_text.find("not analyzed by AST") == std::string::npos)
                    continue;
                auto path = shortenPath(f.file, projectRoot);
                llvm::outs() << "  " << path << ":" << f.line << "  "
                             << f.old_text << "\n";
            }
        }
    }

    if (hasRewrites) {
        llvm::outs() << "\nAuto-rewritable (" << rewrite_count_ << "):\n";
        std::map<FindingKind, unsigned> byKind;
        for (const auto &f : findings_)
            if (f.action == FindingAction::Rewrite)
                ++byKind[f.kind];
        auto desc = [](FindingKind k) -> const char * {
            switch (k) {
            case FindingKind::Include:
                return "includes";
            case FindingKind::Macro:
                return "macros";
            case FindingKind::Type:
                return "types";
            case FindingKind::ScalarType:
                return "scalar types";
            case FindingKind::DataPtr:
                return "data_ptr";
            case FindingKind::CudaStream:
                return "CUDA streams";
            case FindingKind::DeviceGuard:
                return "device guards";
            case FindingKind::MethodToFunc:
                return "method → function";
            case FindingKind::FreeFunc:
                return "free functions";
            default:
                return "other";
            }
        };
        for (const auto &[kind, count] : byKind)
            llvm::outs() << "  " << count << " " << desc(kind) << "\n";
    }
}

void Reporter::printSummary() const {
    auto total = rewrite_count_ + flag_count_;
    if (total == 0) {
        llvm::outs() << "\nNo unstable API usage found.\n";
        return;
    }
    size_t astFlags = 0, textFlags = 0;
    for (const auto &f : findings_) {
        if (f.action != FindingAction::Flag)
            continue;
        if (f.new_text.find("not analyzed by AST") != std::string::npos)
            ++textFlags;
        else
            ++astFlags;
    }
    llvm::outs() << "\nSummary: " << rewrite_count_ << " auto-rewritable, "
                 << astFlags << " flagged";
    if (textFlags > 0)
        llvm::outs() << ", " << textFlags << " in incomplete files";
    if (rewrite_count_ > 0 && astFlags == 0 && textFlags == 0)
        llvm::outs() << " — ready for --mode=rewrite";
    llvm::outs() << "\n";
}

void Reporter::printFileReport(
    std::string_view projectRoot,
    const std::set<std::string> &incompleteFiles) const {
    std::map<std::string, unsigned> flagsByFile;
    for (const auto &f : findings_) {
        if (f.action == FindingAction::Flag)
            ++flagsByFile[f.file];
    }
    if (flagsByFile.empty())
        return;

    std::vector<std::pair<unsigned, std::string>> sorted;
    for (const auto &[file, count] : flagsByFile)
        sorted.push_back({count, file});
    std::sort(sorted.begin(), sorted.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    llvm::outs() << "\nFiles needing attention:\n";
    for (const auto &[count, file] : sorted) {
        auto path = shortenPath(file, projectRoot);
        llvm::outs() << "  " << path << "  " << count << " flag(s)";
        if (incompleteFiles.count(file))
            llvm::outs() << " (incomplete analysis)";
        llvm::outs() << "\n";
    }
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
    llvm::outs() << "  \"parse_errors\": " << parse_error_count_ << ",\n";
    llvm::outs() << "  \"parse_errors_by_file\": {";
    {
        size_t i = 0;
        for (const auto &[file, count] : parse_errors_by_file_) {
            if (i++ > 0)
                llvm::outs() << ",";
            llvm::outs() << "\n    \"" << jsonEscape(file) << "\": " << count;
        }
    }
    if (!parse_errors_by_file_.empty())
        llvm::outs() << "\n  ";
    llvm::outs() << "}\n";
    llvm::outs() << "}\n";
}

std::map<std::string, std::vector<Finding>> Reporter::findingsByFile() const {
    std::map<std::string, std::vector<Finding>> result;
    for (const auto &f : findings_)
        result[f.file].push_back(f);
    return result;
}

void Reporter::recordParseError(const std::string &file) {
    std::lock_guard<std::mutex> lock(mutex_);
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
    llvm::errs()
        << "  hint: for accurate analysis, generate a compilation database:\n"
        << "        cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build\n";
}

void Reporter::runTextScanComplement(const std::vector<std::string> &sources) {
    std::set<std::pair<std::string, unsigned>> coveredLines;
    for (const auto &f : findings_)
        coveredLines.insert({f.file, f.line});

    const auto &patterns = getUnstablePatterns();
    for (const auto &src : sources) {
        llvm::SmallString<256> realPath;
        if (llvm::sys::fs::real_path(src, realPath))
            continue;
        std::string canonical(realPath);

        std::ifstream file(src);
        std::string line;
        unsigned lineNo = 0;
        while (std::getline(file, line)) {
            ++lineNo;
            if (coveredLines.count({canonical, lineNo}))
                continue;
            for (const auto &p : patterns) {
                auto pos = line.find(p);
                if (pos == std::string::npos)
                    continue;
                if (pos > 0 &&
                    (std::isalnum(static_cast<unsigned char>(line[pos - 1])) ||
                     line[pos - 1] == '_'))
                    continue;
                addFinding(FindingKind::UnstableRef, canonical, lineNo, 0, p,
                           p + " (not analyzed by AST)", FindingAction::Flag);
                break;
            }
        }
    }
}

void Reporter::sortFindings() {
    std::sort(findings_.begin(), findings_.end(),
              [](const Finding &a, const Finding &b) {
                  if (a.file != b.file)
                      return a.file < b.file;
                  if (a.line != b.line)
                      return a.line < b.line;
                  if (a.col != b.col)
                      return a.col < b.col;
                  return a.old_text < b.old_text;
              });
}

void Reporter::suppressRedundantFlags() {
    std::set<std::pair<std::string, unsigned>> coveredLines;
    for (const auto &f : findings_)
        if (f.action == FindingAction::Rewrite)
            coveredLines.insert({f.file, f.line});

    size_t removed = 0;
    auto it = std::remove_if(findings_.begin(), findings_.end(),
                             [&coveredLines, &removed](const Finding &f) {
                                 if (f.action == FindingAction::Flag &&
                                     coveredLines.count({f.file, f.line})) {
                                     ++removed;
                                     return true;
                                 }
                                 return false;
                             });
    findings_.erase(it, findings_.end());
    flag_count_ -= removed;
}

bool Reporter::hasNonIncludeFindingsForFile(std::string_view filename) const {
    std::lock_guard<std::mutex> lock(mutex_);
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
    case FindingKind::UnstableType:
        return "UTYPE";
    case FindingKind::UnstableRef:
        return "UREF";
    case FindingKind::UnstableMethod:
        return "UMETH";
    }
    return "?????";
}

} // namespace stable_abi
