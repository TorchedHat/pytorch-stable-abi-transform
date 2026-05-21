#pragma once

#include <clang/Basic/SourceLocation.h>
#include <clang/Basic/SourceManager.h>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace stable_abi {

enum class FindingAction { Rewrite, Flag };

enum class FindingKind {
    Include,
    Macro,
    Type,
    ScalarType,
    DataPtr,
    CudaStream,
    DeviceGuard,
    MethodToFunc,
    FreeFunc,
    Flag,
};

struct Finding {
    FindingKind kind;
    std::string file;
    unsigned line;
    unsigned col;
    std::string old_text;
    std::string new_text;
    FindingAction action;
};

class Reporter {
public:
    void addFinding(FindingKind kind, const clang::SourceManager &SM,
                    clang::SourceLocation loc, std::string_view old_text,
                    std::string_view new_text,
                    FindingAction action = FindingAction::Rewrite);

    void addFinding(FindingKind kind, std::string_view file, unsigned line,
                    unsigned col, std::string_view old_text,
                    std::string_view new_text,
                    FindingAction action = FindingAction::Rewrite);

    void printReport() const;
    void printSummary() const;
    void printJson() const;

    void suppressRedundantFlags();

    [[nodiscard]] size_t rewriteCount() const { return rewrite_count_; }
    [[nodiscard]] size_t flagCount() const { return flag_count_; }

    [[nodiscard]] bool hasNonIncludeFindingsForFile(std::string_view filename) const;

    void recordParseError(const std::string &file);
    [[nodiscard]] size_t parseErrorCount() const { return parse_error_count_; }
    void printParseWarnings() const;

private:
    std::vector<Finding> findings_;
    std::set<std::tuple<std::string, unsigned, unsigned, std::string>> seen_;
    size_t rewrite_count_ = 0;
    size_t flag_count_ = 0;
    size_t parse_error_count_ = 0;
    std::map<std::string, size_t> parse_errors_by_file_;

    static std::string_view kindLabel(FindingKind kind);
};

} // namespace stable_abi
