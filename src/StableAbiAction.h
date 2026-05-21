#pragma once

#include "PreprocessorCallbacks.h"
#include "TransformerRules.h"
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Refactoring/AtomicChange.h>
#include <clang/Tooling/Transformer/Transformer.h>
#include <clang/Tooling/Tooling.h>

namespace stable_abi {

enum class WriteMode { Audit, Rewrite, DryRun };

struct ActionOptions {
    WriteMode write_mode = WriteMode::Audit;
    std::string project_root;

    bool generates_edits() const { return write_mode != WriteMode::Audit; }
};

class StableAbiConsumer : public clang::ASTConsumer {
public:
    StableAbiConsumer(FileReplacements &fileRepls, Reporter &rep,
                      const ActionOptions &opts,
                      PreprocessorCallbacks *ppCallbacks = nullptr);
    void HandleTranslationUnit(clang::ASTContext &Context) override;

private:
    FileReplacements &file_repls_;
    ActionOptions opts_;
    PreprocessorCallbacks *pp_callbacks_;
    std::vector<clang::tooling::AtomicChange> changes_;
    clang::ast_matchers::MatchFinder finder_;
    clang::tooling::Transformer transformer_;
    DeviceGuardCallback guardCallback_;
    CudaStreamCallback streamCallback_;
};

class StableAbiFrontendAction : public clang::ASTFrontendAction {
public:
    StableAbiFrontendAction(Reporter &reporter, const ActionOptions &opts)
        : reporter_(reporter), opts_(opts) {}

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &CI,
                      llvm::StringRef InFile) override;
    void EndSourceFileAction() override;

private:
    clang::Rewriter rewriter_;
    Reporter &reporter_;
    FileReplacements file_repls_;
    ActionOptions opts_;
};

class StableAbiActionFactory
    : public clang::tooling::FrontendActionFactory {
public:
    explicit StableAbiActionFactory(const ActionOptions &opts)
        : opts_(opts) {}

    std::unique_ptr<clang::FrontendAction> create() override {
        return std::make_unique<StableAbiFrontendAction>(reporter_, opts_);
    }

    Reporter &getReporter() { return reporter_; }

private:
    ActionOptions opts_;
    Reporter reporter_;
};

class ParseDiagConsumer : public clang::DiagnosticConsumer {
public:
    explicit ParseDiagConsumer(Reporter &reporter) : reporter_(reporter) {}

    void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                          const clang::Diagnostic &info) override {
        DiagnosticConsumer::HandleDiagnostic(level, info);
        if (level < clang::DiagnosticsEngine::Error)
            return;
        std::string file = "<unknown>";
        if (info.hasSourceManager()) {
            auto &SM = info.getSourceManager();
            auto loc = info.getLocation();
            if (loc.isValid()) {
                auto ploc = SM.getPresumedLoc(loc);
                if (ploc.isValid())
                    file = ploc.getFilename();
            }
        }
        llvm::SmallString<256> msg;
        info.FormatDiagnostic(msg);
        llvm::errs() << file << ": error: " << msg << "\n";
        reporter_.recordParseError(file);
    }

private:
    Reporter &reporter_;
};

} // namespace stable_abi
