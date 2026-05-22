#pragma once

#include "Helpers.h"
#include "Reporter.h"
#include "Rules.h"
#include <clang/Lex/PPCallbacks.h>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace stable_abi {

using IncludeGraph = std::map<std::string, std::set<std::string>>;

class PreprocessorCallbacks : public clang::PPCallbacks {
public:
    PreprocessorCallbacks(FileReplacements &fileRepls,
                          Reporter &reporter, clang::SourceManager &SM,
                          const clang::LangOptions &langOpts,
                          bool rewrite_mode,
                          const std::string &projectRoot = "",
                          IncludeGraph *includeGraph = nullptr)
        : file_repls_(fileRepls), reporter_(reporter), SM_(SM),
          lang_opts_(langOpts),
          rewrite_mode_(rewrite_mode), project_root_(projectRoot),
          include_graph_(includeGraph) {}

    void InclusionDirective(clang::SourceLocation HashLoc,
                            const clang::Token &IncludeTok,
                            llvm::StringRef FileName, bool IsAngled,
                            clang::CharSourceRange FilenameRange,
                            clang::OptionalFileEntryRef File,
                            llvm::StringRef SearchPath,
                            llvm::StringRef RelativePath,
                            const clang::Module *SuggestedModule,
                            bool ModuleImported,
                            clang::SrcMgr::CharacteristicKind FileType) override;

    void MacroExpands(const clang::Token &MacroNameTok,
                      const clang::MacroDefinition &MD,
                      clang::SourceRange Range,
                      const clang::MacroArgs *Args) override;

    void finalizeIncludes();

private:
    struct PendingInclude {
        std::string containingFile;
        clang::SourceLocation hashLoc;
        clang::SourceLocation lineStart;
        clang::SourceLocation nextLineStart;
        const IncludeRule *rule;
    };

    FileReplacements &file_repls_;
    Reporter &reporter_;
    clang::SourceManager &SM_;
    const clang::LangOptions &lang_opts_;
    bool rewrite_mode_;
    std::string project_root_;
    std::set<std::string> emitted_includes_;
    std::vector<PendingInclude> pending_includes_;
    IncludeGraph *include_graph_ = nullptr;
};

} // namespace stable_abi
