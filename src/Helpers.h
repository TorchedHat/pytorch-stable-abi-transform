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

// Strip gcc/nvcc-specific flags from compilation database commands.
// Keeps: -D*, -I*, -isystem, -std=*, -m* (arch), -W*
// Strips: compiler paths, -o, -c, -fPIC, nvcc-specific flags
inline clang::tooling::ArgumentsAdjuster makeStripNonClangAdjuster() {
    return [](const clang::tooling::CommandLineArguments &Args,
              llvm::StringRef /*Filename*/) {
        clang::tooling::CommandLineArguments result;
        bool skipNext = false;
        for (size_t i = 0; i < Args.size(); ++i) {
            if (skipNext) {
                skipNext = false;
                continue;
            }
            llvm::StringRef a(Args[i]);
            // Skip compiler path (first arg or path-like)
            if (i == 0 &&
                (a.contains('/') || a.ends_with("g++") || a.ends_with("gcc") ||
                 a.ends_with("nvcc") || a.ends_with("clang++")))
                continue;
            // Skip output/object flags and their arguments
            if (a == "-o" || a == "-c" || a == "-MF" || a == "-MT" ||
                a == "-MQ") {
                skipNext = true;
                continue;
            }
            // Skip nvcc-specific flags
            if (a.starts_with("-Xcudafe") || a.starts_with("--expt-") ||
                a.starts_with("-forward-unknown") ||
                a.starts_with("-gencode") || a.starts_with("--generate-code") ||
                a == "-x" || a.starts_with("-Xcompiler") ||
                a.starts_with("--diag_suppress"))
                continue;
            if (a.starts_with("arch=") || a.starts_with("code="))
                continue;
            // Skip -x argument value (e.g., "cu" after "-x")
            if (i > 0 && Args[i - 1] == "-x")
                continue;
            // Skip compilation-only flags
            if (a == "-fPIC" || a == "-fPIE" || a == "-shared" || a == "-c")
                continue;
            // Keep everything else
            result.push_back(Args[i]);
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
        clang::tooling::CommandLineArguments result = Args;
        result.push_back("-ferror-limit=0");
        if (!hasArchFlag)
            result.push_back("-march=native");
        if (!resourceDir.empty()) {
            result.push_back("-resource-dir");
            result.push_back(resourceDir);
        }
        if (Filename.ends_with(".cu") || Filename.ends_with(".cuh")) {
            if (Filename.ends_with(".cuh")) {
                auto it =
                    std::find(result.begin(), result.end(), Filename.str());
                if (it != result.end())
                    result.insert(it, "-xcuda");
                else
                    result.push_back("-xcuda");
            }
            result.push_back("--cuda-host-only");
            result.push_back("-nocudalib");
            result.push_back("-nogpulib");
            result.push_back("-Wno-unknown-cuda-version");
            result.push_back("-DUSE_CUDA");
            if (!cudaPath.empty())
                result.push_back("--cuda-path=" + cudaPath);
        }
        return result;
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
