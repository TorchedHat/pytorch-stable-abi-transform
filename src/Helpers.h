#pragma once

#include "Rules.h"
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Expr.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/ArgumentsAdjusters.h>
#include <clang/Tooling/Core/Replacement.h>
#include <llvm/ADT/StringRef.h>
#include <map>
#include <set>
#include <string>

namespace stable_abi {

using IncludeGraph = std::map<std::string, std::set<std::string>>;
using FileReplacements = std::map<std::string, clang::tooling::Replacements>;

inline std::string getSourceText(clang::SourceRange range,
                                 const clang::SourceManager &SM,
                                 const clang::LangOptions &LO) {
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(range), SM, LO)
        .str();
}

inline std::string getIndent(clang::SourceLocation loc,
                             const clang::SourceManager &SM) {
    auto lineNo = SM.getSpellingLineNumber(loc);
    auto fileID = SM.getFileID(loc);
    auto lineStart = SM.translateLineCol(fileID, lineNo, 1);
    auto colOffset = SM.getSpellingColumnNumber(loc) - 1;
    if (colOffset == 0)
        return "";
    auto lineText = clang::Lexer::getSourceText(
        clang::CharSourceRange::getCharRange(lineStart, loc), SM,
        clang::LangOptions());
    std::string indent;
    for (char c : lineText) {
        if (c == ' ' || c == '\t')
            indent += c;
        else
            break;
    }
    return indent;
}

[[nodiscard]] inline bool isInProjectScope(const clang::SourceManager &SM,
                                           clang::SourceLocation Loc,
                                           llvm::StringRef projectRoot) {
    if (projectRoot.empty())
        return SM.isWrittenInMainFile(Loc);
    auto spelling = SM.getSpellingLoc(Loc);
    if (spelling.isInvalid())
        return false;
    auto filename = SM.getFilename(spelling);
    return filename.starts_with(projectRoot);
}

inline void addReplacement(FileReplacements &fileRepls,
                           const clang::SourceManager &SM,
                           clang::SourceLocation loc, unsigned len,
                           llvm::StringRef text) {
    clang::tooling::Replacement R(SM, loc, len, text);
    auto &repls = fileRepls[R.getFilePath().str()];
    if (auto err = repls.add(R)) {
        llvm::errs() << "warning: conflicting replacement at "
                     << R.getFilePath() << ":" << R.getOffset() << " -- "
                     << llvm::toString(std::move(err)) << "\n";
    }
}

inline void addReplacement(FileReplacements &fileRepls,
                           const clang::SourceManager &SM,
                           clang::CharSourceRange range, llvm::StringRef text,
                           const clang::LangOptions &LO) {
    clang::tooling::Replacement R(SM, range, text, LO);
    auto &repls = fileRepls[R.getFilePath().str()];
    if (auto err = repls.add(R)) {
        llvm::errs() << "warning: conflicting replacement at "
                     << R.getFilePath() << ":" << R.getOffset() << " -- "
                     << llvm::toString(std::move(err)) << "\n";
    }
}

[[nodiscard]] inline bool isTensorType(const clang::Expr *obj) {
    if (!obj)
        return false;
    auto objType = obj->getType().getNonReferenceType().getUnqualifiedType();
    if (objType->isPointerType())
        objType = objType->getPointeeType()
                      .getNonReferenceType()
                      .getUnqualifiedType();
    const auto *RD = objType->getAsCXXRecordDecl();
    if (!RD)
        return false;
    auto qualified = RD->getQualifiedNameAsString();
    return qualified == "at::Tensor" || qualified == "at::TensorBase" ||
           qualified == "torch::Tensor" || qualified == "c10::TensorImpl" ||
           qualified == "torch::stable::Tensor";
}

inline std::string jsonEscape(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

// Strip gcc/nvcc-specific flags. ClangTool's built-in adjusters already
// handle -o, -MF/-MT/-MQ (output/deps), -c → -fsyntax-only, and compiler
// path replacement. This adjuster only strips flags unknown to Clang.
inline clang::tooling::ArgumentsAdjuster makeStripNonClangAdjuster() {
    return [](const clang::tooling::CommandLineArguments &Args,
              llvm::StringRef /*Filename*/) {
        clang::tooling::CommandLineArguments result;
        bool skipNext = false;
        for (const auto &arg : Args) {
            if (skipNext) {
                skipNext = false;
                continue;
            }
            llvm::StringRef a(arg);
            if (a.starts_with("-Xcudafe") || a.starts_with("--expt-") ||
                a.starts_with("-forward-unknown") ||
                a.starts_with("-Xcompiler") ||
                a.starts_with("--diag_suppress") ||
                a.starts_with("-static-global-template-stub") || a == "-fPIC" ||
                a == "-fPIE" || a == "-shared")
                continue;
            if (a == "-gencode" || a == "--generate-code" || a == "-x") {
                skipNext = true;
                continue;
            }
            if (a.starts_with("arch=") || a.starts_with("code="))
                continue;
            if (a.starts_with("-std=gnu++")) {
                result.push_back("-std=c++" +
                                 std::string(a.substr(strlen("-std=gnu++"))));
                continue;
            }
            result.push_back(arg);
        }
        return result;
    };
}

// Add tool-specific flags: CUDA handling, arch, resource-dir, error limit.
inline clang::tooling::ArgumentsAdjuster
makeToolAdjuster(const std::string &resourceDir, const std::string &cudaPath,
                 bool hasArchFlag) {
    return [resourceDir, cudaPath,
            hasArchFlag](const clang::tooling::CommandLineArguments &Args,
                         llvm::StringRef Filename) {
        auto result = Args;
        clang::tooling::CommandLineArguments extra;
        extra.push_back("-ferror-limit=0");
        if (!hasArchFlag)
            extra.push_back("-march=native");
        if (!resourceDir.empty()) {
            extra.push_back("-resource-dir");
            extra.push_back(resourceDir);
        }
        if (Filename.ends_with(".cu") || Filename.ends_with(".cuh")) {
            if (Filename.ends_with(".cuh"))
                extra.push_back("-xcuda");
            extra.push_back("--cuda-host-only");
            extra.push_back("-nocudalib");
            extra.push_back("-nogpulib");
            extra.push_back("-Wno-unknown-cuda-version");
            extra.push_back("-DUSE_CUDA");
            if (!cudaPath.empty())
                extra.push_back("--cuda-path=" + cudaPath);
        }
        return clang::tooling::getInsertArgumentAdjuster(
            extra, clang::tooling::ArgumentInsertPosition::BEGIN)(result,
                                                                  Filename);
    };
}

inline const std::vector<std::string> &getUnstablePatterns() {
    static const auto patterns = [] {
        std::set<std::string> unique;
        for (const auto &r : kTypeRules)
            if (!r.from.empty())
                unique.insert(std::string(r.from));
        for (const auto &r : kMacroRules)
            unique.insert(std::string(r.from));
        for (const auto &r : kComparisonMacroRules)
            unique.insert(std::string(r.name));
        for (const auto &r : kDispatchConvRules)
            unique.insert(std::string(r.old_name));
        for (const auto &r : kFreeFuncRules)
            unique.insert(std::string(r.from));
        for (const auto &r : kMethodToFreeFuncRules) {
            unique.insert("." + std::string(r.from) + "(");
            unique.insert("." + std::string(r.from) + "<");
        }
        for (const auto &r : kMethodRenameRules) {
            unique.insert("." + std::string(r.from) + "(");
            unique.insert("." + std::string(r.from) + "<");
        }
        for (const auto &r : kNamespaceRules)
            unique.insert(std::string(r.from));
        for (const auto &p : kDedicatedAstPatterns)
            unique.insert(std::string(p));
        std::vector<std::string> result(unique.begin(), unique.end());
        std::sort(
            result.begin(), result.end(),
            [](const auto &a, const auto &b) { return a.size() > b.size(); });
        return result;
    }();
    return patterns;
}

} // namespace stable_abi
