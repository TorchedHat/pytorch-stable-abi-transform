#include "Verifier.h"
#include "Helpers.h"
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <vector>

namespace stable_abi {

// ---------------------------------------------------------------------------
// Shadow include tree: expose only stable PyTorch headers
// ---------------------------------------------------------------------------

class ShadowIncludeTree {
  public:
    explicit ShadowIncludeTree(const std::string &pytorch_root) {
        llvm::SmallString<128> tmp;
        if (auto ec = llvm::sys::fs::createUniqueDirectory("stable-abi-verify",
                                                           tmp)) {
            llvm::errs() << "warning: failed to create shadow tree: "
                         << ec.message() << "\n";
            return;
        }
        root_ = std::string(tmp);

        mkpath("torch/csrc/stable");
        createSymlink(pytorch_root + "/torch/csrc/stable",
                      root_ + "/torch/csrc/stable");

        mkpath("torch/csrc/inductor/aoti_torch/c");
        createSymlink(pytorch_root + "/torch/csrc/inductor/aoti_torch/c",
                      root_ + "/torch/csrc/inductor/aoti_torch/c");
        mkpath("torch/csrc/inductor/aoti_torch/generated");
        createSymlink(pytorch_root +
                          "/torch/csrc/inductor/aoti_torch/generated",
                      root_ + "/torch/csrc/inductor/aoti_torch/generated");

        mkpath("torch");
        std::string headeronly_src =
            pytorch_root + "/torch/include/torch/headeronly";
        if (!llvm::sys::fs::is_directory(headeronly_src))
            headeronly_src = pytorch_root + "/torch/headeronly";
        createSymlink(headeronly_src, root_ + "/torch/headeronly");
    }

    ~ShadowIncludeTree() { cleanup(); }

    ShadowIncludeTree(const ShadowIncludeTree &) = delete;
    ShadowIncludeTree &operator=(const ShadowIncludeTree &) = delete;
    ShadowIncludeTree(ShadowIncludeTree &&other) noexcept
        : root_(std::move(other.root_)), symlinks_(std::move(other.symlinks_)) {
        other.root_.clear();
    }
    ShadowIncludeTree &operator=(ShadowIncludeTree &&other) noexcept {
        if (this != &other) {
            cleanup();
            root_ = std::move(other.root_);
            symlinks_ = std::move(other.symlinks_);
            other.root_.clear();
        }
        return *this;
    }

    const std::string &path() const { return root_; }

  private:
    std::string root_;
    std::vector<std::string> symlinks_;

    void cleanup() {
        if (root_.empty())
            return;
        for (const auto &link : symlinks_)
            llvm::sys::fs::remove(link);
        llvm::sys::fs::remove_directories(root_);
        root_.clear();
        symlinks_.clear();
    }

    void mkpath(const char *rel) {
        llvm::SmallString<256> p(root_);
        llvm::sys::path::append(p, rel);
        if (auto ec = llvm::sys::fs::create_directories(p))
            llvm::errs() << "warning: mkdir " << p << ": " << ec.message()
                         << "\n";
    }

    void createSymlink(const std::string &target, const std::string &link) {
        llvm::sys::fs::remove(link);
        if (auto ec = llvm::sys::fs::create_link(target, link))
            llvm::errs() << "warning: symlink " << link << " -> " << target
                         << ": " << ec.message() << "\n";
        else
            symlinks_.push_back(link);
    }
};

// ---------------------------------------------------------------------------
// Diagnostic consumer: capture compiler errors as Violations
// ---------------------------------------------------------------------------

class StableAbiDiagConsumer : public clang::DiagnosticConsumer {
  public:
    explicit StableAbiDiagConsumer(std::vector<Violation> &out)
        : violations_(out) {}

    void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                          const clang::Diagnostic &info) override {
        DiagnosticConsumer::HandleDiagnostic(level, info);

        if (level < clang::DiagnosticsEngine::Error)
            return;

        std::string file;
        unsigned line = 0, col = 0;
        if (info.hasSourceManager()) {
            auto &SM = info.getSourceManager();
            auto ploc = SM.getPresumedLoc(info.getLocation());
            if (ploc.isValid()) {
                file = ploc.getFilename();
                line = ploc.getLine();
                col = ploc.getColumn();
            }
        }

        llvm::SmallString<256> msg;
        info.FormatDiagnostic(msg);

        std::string reason = (level == clang::DiagnosticsEngine::Fatal)
                                 ? "fatal error"
                                 : "error";

        violations_.push_back({file, line, col, msg.str().str(), reason});
    }

  private:
    std::vector<Violation> &violations_;
};

// ---------------------------------------------------------------------------
// Compile-based verification
// ---------------------------------------------------------------------------

std::vector<Violation> verifyStableAbi(const std::string &filepath,
                                       const VerifyOptions &opts) {
    std::vector<Violation> violations;

    ShadowIncludeTree shadow(opts.pytorch_root);
    if (shadow.path().empty()) {
        violations.push_back(
            {filepath, 0, 0, "", "failed to create shadow include tree"});
        return violations;
    }

    std::string cudaPath;
    if (!opts.cuda_include.empty()) {
        llvm::SmallString<256> cp(opts.cuda_include);
        llvm::sys::path::remove_filename(cp);
        cudaPath = std::string(cp);
    }

    // Base args: just the standard, shadow tree, and project includes
    std::vector<std::string> args;
    args.push_back("-std=c++20");
    args.push_back("-I" + shadow.path());
    if (!opts.cuda_include.empty())
        args.push_back("-I" + opts.cuda_include);
    for (const auto &inc : opts.extra_includes)
        args.push_back("-I" + inc);

    auto db =
        std::make_unique<clang::tooling::FixedCompilationDatabase>(".", args);
    std::vector<std::string> sources = {filepath};

    clang::tooling::ClangTool tool(*db, sources);
    // Same adjusters as audit path
    tool.appendArgumentsAdjuster(
        makeToolAdjuster(opts.resource_dir, cudaPath, false));

    StableAbiDiagConsumer diagConsumer(violations);
    tool.setDiagnosticConsumer(&diagConsumer);

    tool.run(clang::tooling::newFrontendActionFactory<clang::SyntaxOnlyAction>()
                 .get());

    return violations;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void printViolations(const std::vector<Violation> &violations) {
    if (violations.empty()) {
        llvm::outs()
            << "ABI verification: PASS (no unstable API usage found)\n";
        return;
    }

    llvm::outs() << "ABI verification: FAIL (" << violations.size()
                 << " violations)\n";
    for (const auto &v : violations) {
        llvm::outs() << "  [UNSTABLE] " << v.file << ":" << v.line;
        if (v.col > 0)
            llvm::outs() << ":" << v.col;
        llvm::outs() << "  " << v.text << "\n"
                     << "             " << v.reason << "\n";
    }
}

void printViolationsJson(const std::vector<Violation> &violations) {
    llvm::outs() << "{\n  \"violations\": [\n";
    for (size_t i = 0; i < violations.size(); ++i) {
        const auto &v = violations[i];
        llvm::outs() << "    {\"file\": \"" << jsonEscape(v.file) << "\", "
                     << "\"line\": " << v.line << ", "
                     << "\"col\": " << v.col << ", "
                     << "\"text\": \"" << jsonEscape(v.text) << "\", "
                     << "\"reason\": \"" << jsonEscape(v.reason) << "\"}";
        if (i + 1 < violations.size())
            llvm::outs() << ",";
        llvm::outs() << "\n";
    }
    llvm::outs() << "  ],\n";
    llvm::outs() << "  \"count\": " << violations.size() << ",\n";
    llvm::outs() << "  \"pass\": " << (violations.empty() ? "true" : "false")
                 << "\n}\n";
}

} // namespace stable_abi
