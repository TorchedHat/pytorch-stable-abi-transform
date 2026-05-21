#pragma once

#include <clang/AST/Expr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/Core/Replacement.h>
#include <llvm/ADT/StringRef.h>
#include <map>
#include <string>

namespace stable_abi {

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
                     << R.getFilePath() << ":" << R.getOffset()
                     << " -- " << llvm::toString(std::move(err)) << "\n";
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
                     << R.getFilePath() << ":" << R.getOffset()
                     << " -- " << llvm::toString(std::move(err)) << "\n";
    }
}

[[nodiscard]] inline bool isTensorType(const clang::Expr *obj) {
    if (!obj)
        return false;
    auto objType =
        obj->getType().getNonReferenceType().getUnqualifiedType();
    if (objType->isPointerType())
        objType = objType->getPointeeType().getNonReferenceType().getUnqualifiedType();
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
    for (char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c;
        }
    }
    return out;
}

} // namespace stable_abi
