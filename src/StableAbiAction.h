#pragma once

#include "PreprocessorCallbacks.h"
#include "TransformerRules.h"
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Refactoring/AtomicChange.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/Transformer/Transformer.h>
#include <mutex>

namespace stable_abi {

enum class WriteMode { Audit, Rewrite, DryRun };

struct ActionOptions {
    WriteMode write_mode = WriteMode::Audit;
    std::string project_root;
    std::string output_dir;
    IncludeGraph *include_graph = nullptr;
    std::mutex *include_graph_mutex = nullptr;
    std::mutex *write_mutex = nullptr;
    std::set<std::string> *incomplete_files = nullptr;
    std::mutex *incomplete_files_mutex = nullptr;

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
    NbytesCallback nbytesCallback_;
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

class StableAbiActionFactory : public clang::tooling::FrontendActionFactory {
  public:
    explicit StableAbiActionFactory(const ActionOptions &opts) : opts_(opts) {}

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
    ParseDiagConsumer(Reporter &reporter,
                      std::set<std::string> *incompleteFiles = nullptr,
                      std::mutex *incompleteFilesMutex = nullptr)
        : reporter_(reporter), incomplete_files_(incompleteFiles),
          incomplete_files_mutex_(incompleteFilesMutex) {}

    void setMainFile(std::string mainFile) { main_file_ = std::move(mainFile); }

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
        if (incomplete_files_ && !main_file_.empty()) {
            std::lock_guard<std::mutex> lock(*incomplete_files_mutex_);
            incomplete_files_->insert(main_file_);
        }
    }

  private:
    Reporter &reporter_;
    std::string main_file_;
    std::set<std::string> *incomplete_files_ = nullptr;
    std::mutex *incomplete_files_mutex_ = nullptr;
};

} // namespace stable_abi
