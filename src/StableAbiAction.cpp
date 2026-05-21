#include "StableAbiAction.h"
#include "TransformerRules.h"
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/Core/Replacement.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>

namespace stable_abi {

StableAbiConsumer::StableAbiConsumer(FileReplacements &fileRepls,
                                     Reporter &rep,
                                     const ActionOptions &opts,
                                     PreprocessorCallbacks *ppCallbacks)
    : file_repls_(fileRepls), opts_(opts), pp_callbacks_(ppCallbacks),
      transformer_(
          buildTransformerRules(rep, opts_.generates_edits(), opts_.project_root),
          [this](llvm::Expected<llvm::MutableArrayRef<
                     clang::tooling::AtomicChange>> C) {
              if (C) {
                  for (auto &change : *C)
                      changes_.push_back(std::move(change));
              } else {
                  llvm::consumeError(C.takeError());
              }
          }),
      guardCallback_(fileRepls, rep, opts_.generates_edits(), opts_.project_root),
      streamCallback_(fileRepls, rep, opts_.generates_edits(), guardCallback_, opts_.project_root) {
    transformer_.registerMatchers(&finder_);
    registerManualMatchers(finder_, streamCallback_, guardCallback_);
}

void StableAbiConsumer::HandleTranslationUnit(clang::ASTContext &Context) {
    finder_.matchAST(Context);

    if (pp_callbacks_)
        pp_callbacks_->finalizeIncludes();

    if (opts_.generates_edits()) {
        for (const auto &change : changes_) {
            for (const auto &r : change.getReplacements()) {
                auto &repls = file_repls_[r.getFilePath().str()];
                if (auto err = repls.add(r))
                    llvm::errs() << "warning: conflicting replacement -- "
                                 << llvm::toString(std::move(err)) << "\n";
            }
        }
    }
}

std::unique_ptr<clang::ASTConsumer>
StableAbiFrontendAction::CreateASTConsumer(clang::CompilerInstance &CI,
                                           llvm::StringRef InFile) {
    rewriter_.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());

    auto ppCallbacks = std::make_unique<PreprocessorCallbacks>(
        file_repls_, reporter_, CI.getSourceManager(), CI.getLangOpts(),
        opts_.generates_edits(), opts_.project_root);
    auto *ppRaw = ppCallbacks.get();
    CI.getPreprocessor().addPPCallbacks(std::move(ppCallbacks));

    return std::make_unique<StableAbiConsumer>(file_repls_, reporter_,
                                               opts_, ppRaw);
}

static void printUnifiedDiff(const std::string &filename,
                             const std::string &original,
                             const std::string &modified) {
    auto splitLines = [](const std::string &s) {
        std::vector<std::string> lines;
        std::istringstream stream(s);
        std::string line;
        while (std::getline(stream, line))
            lines.push_back(line);
        return lines;
    };

    auto origLines = splitLines(original);
    auto modLines = splitLines(modified);

    llvm::outs() << "--- a/" << filename << "\n";
    llvm::outs() << "+++ b/" << filename << "\n";

    size_t i = 0, j = 0;
    while (i < origLines.size() || j < modLines.size()) {
        size_t hunkStart_i = i, hunkStart_j = j;
        // Find next difference
        while (i < origLines.size() && j < modLines.size() &&
               origLines[i] == modLines[j]) {
            ++i; ++j;
        }
        if (i >= origLines.size() && j >= modLines.size())
            break;

        // Context: up to 3 lines before
        size_t ctx = std::min(i - hunkStart_i, size_t{3});
        size_t ctxStart_i = i - ctx, ctxStart_j = j - ctx;

        // Find end of different region
        size_t diffEnd_i = i, diffEnd_j = j;
        size_t matchCount = 0;
        while ((diffEnd_i < origLines.size() || diffEnd_j < modLines.size()) &&
               matchCount < 3) {
            if (diffEnd_i < origLines.size() && diffEnd_j < modLines.size() &&
                origLines[diffEnd_i] == modLines[diffEnd_j]) {
                ++matchCount; ++diffEnd_i; ++diffEnd_j;
            } else {
                matchCount = 0;
                if (diffEnd_i < origLines.size()) ++diffEnd_i;
                if (diffEnd_j < modLines.size()) ++diffEnd_j;
            }
        }

        // Hunk header
        size_t origCount = diffEnd_i - ctxStart_i;
        size_t modCount = diffEnd_j - ctxStart_j;
        llvm::outs() << "@@ -" << (ctxStart_i + 1) << "," << origCount
                      << " +" << (ctxStart_j + 1) << "," << modCount << " @@\n";

        // Context before
        for (size_t k = ctxStart_i; k < i; ++k)
            llvm::outs() << " " << origLines[k] << "\n";

        // Changed lines
        size_t oi = i, oj = j;
        while (oi < diffEnd_i || oj < diffEnd_j) {
            if (oi < diffEnd_i && oj < diffEnd_j &&
                origLines[oi] == modLines[oj]) {
                llvm::outs() << " " << origLines[oi] << "\n";
                ++oi; ++oj;
            } else {
                if (oi < diffEnd_i) {
                    llvm::outs() << "-" << origLines[oi] << "\n";
                    ++oi;
                }
                if (oj < diffEnd_j) {
                    llvm::outs() << "+" << modLines[oj] << "\n";
                    ++oj;
                }
            }
        }

        i = diffEnd_i;
        j = diffEnd_j;
    }
}

void StableAbiFrontendAction::EndSourceFileAction() {
    if (opts_.generates_edits()) {
        for (auto &[filename, repls] : file_repls_) {
            if (!clang::tooling::applyAllReplacements(repls, rewriter_))
                llvm::errs() << "warning: failed to apply replacements to "
                             << filename << "\n";
        }
        if (opts_.write_mode == WriteMode::DryRun) {
            auto &SM = rewriter_.getSourceMgr();
            for (auto it = rewriter_.buffer_begin();
                 it != rewriter_.buffer_end(); ++it) {
                auto fileID = it->first;
                auto optRef = SM.getFileEntryRefForID(fileID);
                if (!optRef) continue;
                std::string filename = optRef->getName().str();

                std::string modified;
                llvm::raw_string_ostream os(modified);
                it->second.write(os);

                auto origBuf = SM.getBufferData(fileID);
                std::string original = origBuf.str();

                if (original != modified)
                    printUnifiedDiff(filename, original, modified);
            }
        } else {
            rewriter_.overwriteChangedFiles();
        }
        file_repls_.clear();
    }
}

} // namespace stable_abi
