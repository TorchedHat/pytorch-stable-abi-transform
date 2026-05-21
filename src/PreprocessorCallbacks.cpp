#include "PreprocessorCallbacks.h"
#include "Helpers.h"
#include "Rules.h"
#include <clang/Basic/IdentifierTable.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/MacroArgs.h>
#include <clang/Lex/MacroInfo.h>

namespace stable_abi {

void PreprocessorCallbacks::InclusionDirective(
    clang::SourceLocation HashLoc, const clang::Token &IncludeTok,
    llvm::StringRef FileName, bool IsAngled,
    clang::CharSourceRange FilenameRange, clang::OptionalFileEntryRef File,
    llvm::StringRef SearchPath, llvm::StringRef RelativePath,
    const clang::Module *SuggestedModule, bool ModuleImported,
    clang::SrcMgr::CharacteristicKind FileType) {

    if (!isInProjectScope(SM_, HashLoc, project_root_))
        return;

    for (const auto &rule : kIncludeRules) {
        if (FileName != llvm::StringRef(rule.old_path))
            continue;

        auto lineBegin = SM_.getSpellingLoc(HashLoc);
        unsigned lineNo = SM_.getSpellingLineNumber(lineBegin);
        auto fileID = SM_.getFileID(lineBegin);
        auto lineStart = SM_.translateLineCol(fileID, lineNo, 1);
        auto nextLineStart = SM_.translateLineCol(fileID, lineNo + 1, 1);

        if (rule.remove_only) {
            reporter_.addFinding(FindingKind::Include, SM_, HashLoc,
                                 std::string("#include <") +
                                     std::string(rule.old_path) + ">",
                                 "(removed)");
            if (rewrite_mode_ && nextLineStart.isValid()) {
                unsigned len = SM_.getFileOffset(nextLineStart) -
                               SM_.getFileOffset(lineStart);
                addReplacement(file_repls_, SM_, lineStart, len, "");
            }
            return;
        }

        auto filename = SM_.getFilename(lineBegin);
        pending_includes_.push_back(
            {std::string(filename), HashLoc, lineStart,
             nextLineStart, &rule});
        return;
    }

    emitted_includes_.insert(FileName.str());

    if (FileName.starts_with("ATen/") ||
        FileName.starts_with("c10/") ||
        (FileName.starts_with("torch/") &&
         !FileName.starts_with("torch/csrc/stable/") &&
         !FileName.starts_with("torch/csrc/inductor/aoti_torch/") &&
         !FileName.starts_with("torch/headeronly/"))) {
        std::string includeText =
            std::string("#include <") + FileName.str() + ">";
        reporter_.addFinding(FindingKind::Include, SM_, HashLoc,
                             includeText,
                             "unstable include — replace with stable equivalent "
                             "or remove if unused",
                             FindingAction::Flag);
    }
}

void PreprocessorCallbacks::finalizeIncludes() {
    for (const auto &pi : pending_includes_) {
        const auto &rule = *pi.rule;
        std::string oldInclude =
            std::string("#include <") + std::string(rule.old_path) + ">";

        bool isMainFile = SM_.isWrittenInMainFile(pi.hashLoc);
        if (!isMainFile &&
            !reporter_.hasNonIncludeFindingsForFile(pi.containingFile)) {
            reporter_.addFinding(FindingKind::Include, SM_, pi.hashLoc,
                                 oldInclude, "(removed, no torch API usage)");
            if (rewrite_mode_ && pi.nextLineStart.isValid()) {
                unsigned len = SM_.getFileOffset(pi.nextLineStart) -
                               SM_.getFileOffset(pi.lineStart);
                addReplacement(file_repls_, SM_, pi.lineStart, len, "");
            }
            continue;
        }

        std::string replacement;
        for (const auto &new_path : rule.new_paths) {
            if (new_path.empty())
                break;
            if (emitted_includes_.count(std::string(new_path)))
                continue;
            emitted_includes_.insert(std::string(new_path));
            replacement += "#include <";
            replacement += new_path;
            replacement += ">\n";
        }

        if (replacement.empty()) {
            reporter_.addFinding(FindingKind::Include, SM_, pi.hashLoc,
                                 oldInclude, "(removed, already emitted)");
            if (rewrite_mode_ && pi.nextLineStart.isValid()) {
                unsigned len = SM_.getFileOffset(pi.nextLineStart) -
                               SM_.getFileOffset(pi.lineStart);
                addReplacement(file_repls_, SM_, pi.lineStart, len, "");
            }
            continue;
        }

        reporter_.addFinding(FindingKind::Include, SM_, pi.hashLoc,
                             oldInclude, replacement);
        if (rewrite_mode_ && pi.nextLineStart.isValid()) {
            unsigned len = SM_.getFileOffset(pi.nextLineStart) -
                           SM_.getFileOffset(pi.lineStart);
            addReplacement(file_repls_, SM_, pi.lineStart, len, replacement);
        }
    }
    pending_includes_.clear();
}

void PreprocessorCallbacks::MacroExpands(const clang::Token &MacroNameTok,
                                         const clang::MacroDefinition &MD,
                                         clang::SourceRange Range,
                                         const clang::MacroArgs *Args) {
    auto loc = MacroNameTok.getLocation();
    auto name = MacroNameTok.getIdentifierInfo()->getName();

    if (SM_.isMacroBodyExpansion(loc)) {
        auto spellLoc = SM_.getSpellingLoc(loc);
        bool isUnstablePrefix =
            name.starts_with("TORCH_") || name.starts_with("AT_") ||
            name.starts_with("C10_") || name == "PYBIND11_MODULE";
        bool isDefinedInStableHeader = false;
        if (isUnstablePrefix) {
            if (auto *MI = MD.getMacroInfo()) {
                auto defFile = SM_.getFilename(MI->getDefinitionLoc());
                isDefinedInStableHeader =
                    defFile.contains("torch/csrc/stable/") ||
                    defFile.contains("torch/headeronly/") ||
                    defFile.contains("torch/csrc/inductor/aoti_torch/");
            }
        }
        if (isInProjectScope(SM_, spellLoc, project_root_) &&
            isUnstablePrefix && !isDefinedInStableHeader) {
            reporter_.addFinding(FindingKind::Macro, SM_, spellLoc, name,
                                 "unstable macro in macro body",
                                 FindingAction::Flag);
        }
        return;
    }

    if (!isInProjectScope(SM_, loc, project_root_))
        return;

    if (name == "PYBIND11_MODULE") {
        reporter_.addFinding(FindingKind::Macro, SM_, SM_.getSpellingLoc(loc),
                             "PYBIND11_MODULE",
                             "migrate to STABLE_TORCH_LIBRARY + TORCH_BOX",
                             FindingAction::Flag);
        return;
    }

    for (const auto &cmp : kComparisonMacroRules) {
        if (name != llvm::StringRef(cmp.name))
            continue;
        if (!Args)
            break;

        auto getArgText = [&](unsigned idx) -> std::string {
            const clang::Token *tok = Args->getUnexpArgument(idx);
            if (!tok || tok->is(clang::tok::eof))
                return {};
            auto start = SM_.getSpellingLoc(tok->getLocation());
            auto end = start;
            while (tok->isNot(clang::tok::eof)) {
                end = SM_.getSpellingLoc(tok->getLocation());
                end = end.getLocWithOffset(tok->getLength());
                ++tok;
            }
            return clang::Lexer::getSourceText(
                       clang::CharSourceRange::getCharRange(start, end),
                       SM_, lang_opts_)
                .str();
        };

        std::string lhs = getArgText(0);
        std::string rhs = getArgText(1);
        if (lhs.empty() || rhs.empty()) {
            reporter_.addFinding(FindingKind::Macro, SM_, loc,
                                 cmp.name,
                                 "could not extract arguments — rewrite manually",
                                 FindingAction::Flag);
            return;
        }

        std::string repl = "STD_TORCH_CHECK((" + lhs + ") " +
                           std::string(cmp.op) + " (" + rhs + "))";

        reporter_.addFinding(FindingKind::Macro, SM_, loc,
                             cmp.name, "STD_TORCH_CHECK");
        if (rewrite_mode_) {
            auto beginLoc = SM_.getSpellingLoc(MacroNameTok.getLocation());
            auto endLoc =
                SM_.getSpellingLoc(Range.getEnd()).getLocWithOffset(1);
            unsigned len =
                SM_.getFileOffset(endLoc) - SM_.getFileOffset(beginLoc);
            addReplacement(file_repls_, SM_, beginLoc, len, repl);
        }
        return;
    }

    for (const auto &conv : kDispatchConvRules) {
        if (name != llvm::StringRef(conv.old_name))
            continue;
        std::string desc = std::string("THO_DISPATCH_V2(..., ") +
                           std::string(conv.type_collection) + ")";
        reporter_.addFinding(FindingKind::Macro, SM_, loc,
                             conv.old_name, desc);
        if (rewrite_mode_) {
            auto nameLen = MacroNameTok.getLength();
            addReplacement(file_repls_, SM_, loc, nameLen, "THO_DISPATCH_V2");
            auto closeParen = SM_.getSpellingLoc(Range.getEnd());
            std::string insert = std::string(", ") + std::string(conv.type_collection);
            addReplacement(file_repls_, SM_, closeParen, 0, insert);
        }
        return;
    }

    // AT_ERROR(msg, ...) → STD_TORCH_CHECK(false, msg, ...)
    if (name == "AT_ERROR") {
        reporter_.addFinding(FindingKind::Macro, SM_, loc,
                             "AT_ERROR", "STD_TORCH_CHECK(false, ...)");
        if (rewrite_mode_ && Args) {
            auto nameLen = MacroNameTok.getLength();
            addReplacement(file_repls_, SM_, loc, nameLen, "STD_TORCH_CHECK");
            const auto *firstArgTok = Args->getUnexpArgument(0);
            if (firstArgTok) {
                auto argLoc = SM_.getSpellingLoc(firstArgTok->getLocation());
                addReplacement(file_repls_, SM_, argLoc, 0, "false, ");
            }
        }
        return;
    }

    for (const auto &rule : kMacroRules) {
        if (name != llvm::StringRef(rule.from))
            continue;

        if (rule.flag_only) {
            std::string suggestion;
            if (!rule.to.empty()) {
                suggestion =
                    std::string("use ") + std::string(rule.to);
            } else {
                suggestion =
                    "no stable equivalent — rewrite with STD_TORCH_CHECK";
            }
            reporter_.addFinding(FindingKind::Macro, SM_, loc,
                                 std::string(rule.from), suggestion,
                                 FindingAction::Flag);
            return;
        }

        reporter_.addFinding(FindingKind::Macro, SM_, loc,
                             std::string(rule.from),
                             std::string(rule.to));

        if (rewrite_mode_) {
            auto nameLen = MacroNameTok.getLength();
            addReplacement(file_repls_, SM_, loc, nameLen,
                           rule.to);
        }
        return;
    }
}

} // namespace stable_abi
