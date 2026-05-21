#include "Verifier.h"
#include "Helpers.h"
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <fstream>
#include <regex>
#include <vector>

namespace stable_abi {

// ---------------------------------------------------------------------------
// Shadow include tree: expose only stable PyTorch headers
// ---------------------------------------------------------------------------

class ShadowIncludeTree {
public:
    explicit ShadowIncludeTree(const std::string &pytorch_root) {
        llvm::SmallString<128> tmp;
        if (auto ec = llvm::sys::fs::createUniqueDirectory("stable-abi-verify", tmp)) {
            llvm::errs() << "warning: failed to create shadow tree: " << ec.message() << "\n";
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
        createSymlink(pytorch_root + "/torch/csrc/inductor/aoti_torch/generated",
                      root_ + "/torch/csrc/inductor/aoti_torch/generated");

        mkpath("torch");
        std::string headeronly_src = pytorch_root + "/torch/include/torch/headeronly";
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
            llvm::errs() << "warning: mkdir " << p << ": " << ec.message() << "\n";
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

        std::string reason =
            (level == clang::DiagnosticsEngine::Fatal) ? "fatal error" : "error";

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
        violations.push_back({filepath, 0, 0, "", "failed to create shadow include tree"});
        return violations;
    }

    std::vector<std::string> args;
    args.push_back("-std=c++20");
    args.push_back("-fsyntax-only");
    args.push_back("-ferror-limit=0");

    if (!opts.resource_dir.empty()) {
        args.push_back("-resource-dir");
        args.push_back(opts.resource_dir);
    }

    args.push_back("-I" + shadow.path());

    if (!opts.cuda_include.empty()) {
        args.push_back("-I" + opts.cuda_include);
        args.push_back("-DUSE_CUDA");
    }

    for (const auto &inc : opts.extra_includes) {
        args.push_back("-I" + inc);
    }

    if (llvm::StringRef(filepath).ends_with(".cu") ||
        llvm::StringRef(filepath).ends_with(".cuh")) {
        args.push_back("--cuda-host-only");
    }

    auto db = std::make_unique<clang::tooling::FixedCompilationDatabase>(
        ".", args);
    std::vector<std::string> sources = {filepath};

    clang::tooling::ClangTool tool(*db, sources);

    StableAbiDiagConsumer diagConsumer(violations);
    tool.setDiagnosticConsumer(&diagConsumer);

    tool.run(
        clang::tooling::newFrontendActionFactory<clang::SyntaxOnlyAction>()
            .get());

    return violations;
}

// ---------------------------------------------------------------------------
// Regex-based verification (legacy fallback)
// ---------------------------------------------------------------------------

struct ForbiddenPattern {
    std::regex pattern;
    std::string_view reason;
};

static const std::vector<ForbiddenPattern> &getForbiddenPatterns() {
    static const std::vector<ForbiddenPattern> patterns = {
        {std::regex(R"(#include\s*[<"]torch/all\.h[">])"),
         "unstable include: torch/all.h"},
        {std::regex(R"(#include\s*[<"]torch/torch\.h[">])"),
         "unstable include: torch/torch.h"},
        {std::regex(R"(#include\s*[<"]torch/cuda\.h[">])"),
         "unstable include: torch/cuda.h"},
        {std::regex(R"(#include\s*[<"]torch/library\.h[">])"),
         "unstable include: torch/library.h"},
        {std::regex(R"(#include\s*[<"]ATen/)"),
         "unstable include: ATen/"},
        {std::regex(R"(#include\s*[<"]c10/)"),
         "unstable include: c10/"},
        {std::regex(R"(#include\s*[<"]torch/csrc/(?!stable/|inductor/aoti_torch/))"),
         "unstable include: torch/csrc/ (not stable or aoti)"},
        {std::regex(R"(\bat::Tensor\b)"),
         "unstable type: at::Tensor (use torch::stable::Tensor)"},
        {std::regex(R"(\btorch::Tensor\b)"),
         "unstable type: torch::Tensor (use torch::stable::Tensor)"},
        {std::regex(R"(\bat::ScalarType\b)"),
         "unstable type: at::ScalarType (use torch::headeronly::ScalarType)"},
        {std::regex(R"(\bc10::ScalarType\b)"),
         "unstable type: c10::ScalarType (use torch::headeronly::ScalarType)"},
        {std::regex(R"(\bat::Device\b)"),
         "unstable type: at::Device (use torch::stable::Device)"},
        {std::regex(R"(\bc10::Device\b)"),
         "unstable type: c10::Device (use torch::stable::Device)"},
        {std::regex(R"(\bat::DeviceType\b)"),
         "unstable type: at::DeviceType (use torch::headeronly::DeviceType)"},
        {std::regex(R"(\bc10::DeviceType\b)"),
         "unstable type: c10::DeviceType (use torch::headeronly::DeviceType)"},
        {std::regex(R"(\bat::k[A-Z]\w+\b)"),
         "unstable shorthand: at::k* (use torch::headeronly::ScalarType::*)"},
        {std::regex(R"(\btorch::k[A-Z]\w+\b)"),
         "unstable shorthand: torch::k* (use torch::headeronly::ScalarType::*)"},
        {std::regex(R"(\bTORCH_CHECK\s*\()"),
         "unstable macro: TORCH_CHECK (use STD_TORCH_CHECK)"},
        {std::regex(R"(\bTORCH_CHECK_NOT_IMPLEMENTED\s*\()"),
         "unstable macro: TORCH_CHECK_NOT_IMPLEMENTED (use STD_TORCH_CHECK_NOT_IMPLEMENTED)"},
        {std::regex(R"(\bTORCH_CHECK_(EQ|NE|LT|GT|GE|LE)\s*\()"),
         "unstable macro: TORCH_CHECK_EQ/NE/LT/GT/GE/LE (use STD_TORCH_CHECK with operator)"},
        {std::regex(R"(\bTORCH_LIBRARY\s*\()"),
         "unstable macro: TORCH_LIBRARY (use STABLE_TORCH_LIBRARY)"},
        {std::regex(R"(\bTORCH_LIBRARY_EXPAND\s*\()"),
         "unstable macro: TORCH_LIBRARY_EXPAND"},
        {std::regex(R"(\bTORCH_LIBRARY_IMPL\s*\()"),
         "unstable macro: TORCH_LIBRARY_IMPL (use STABLE_TORCH_LIBRARY_IMPL)"},
        {std::regex(R"(\bC10_CUDA_CHECK\s*\()"),
         "unstable macro: C10_CUDA_CHECK (use STD_CUDA_CHECK)"},
        {std::regex(R"(\bAT_CUDA_CHECK\s*\()"),
         "unstable macro: AT_CUDA_CHECK (use STD_CUDA_CHECK)"},
        {std::regex(R"(\bC10_CUDA_KERNEL_LAUNCH_CHECK\s*\()"),
         "unstable macro: C10_CUDA_KERNEL_LAUNCH_CHECK (use STD_CUDA_KERNEL_LAUNCH_CHECK)"},
        {std::regex(R"(\bat::cuda::)"),
         "unstable API: at::cuda:: (use torch::stable::accelerator::)"},
        {std::regex(R"(\bc10::cuda::)"),
         "unstable API: c10::cuda:: (use stable accelerator API)"},
        {std::regex(R"(\.data_ptr\s*[<(])"),
         "unstable method: .data_ptr (use .mutable_data_ptr or .const_data_ptr)"},
        {std::regex(R"(\.dtype\s*\(\))"),
         "unstable method: .dtype() (use .scalar_type())"},
        {std::regex(R"(\bc10::optional\b)"),
         "unstable type: c10::optional (use std::optional)"},
        {std::regex(R"(\bc10::nullopt\b)"),
         "unstable value: c10::nullopt (use std::nullopt)"},
        {std::regex(R"(\bc10::ArrayRef\b)"),
         "unstable type: c10::ArrayRef (use torch::headeronly::HeaderOnlyArrayRef)"},
        {std::regex(R"(\bc10::IntArrayRef\b)"),
         "unstable type: c10::IntArrayRef (use torch::headeronly::IntHeaderOnlyArrayRef)"},
        {std::regex(R"(\bat::IntArrayRef\b)"),
         "unstable type: at::IntArrayRef (use torch::headeronly::IntHeaderOnlyArrayRef)"},
        {std::regex(R"(\bc10::string_view\b)"),
         "unstable type: c10::string_view (use std::string_view)"},
        {std::regex(R"(\btorch::TensorOptions\b)"),
         "unstable type: torch::TensorOptions (decompose into scalar_type, layout, device args)"},
        {std::regex(R"(\bat::Half\b)"),
         "unstable type: at::Half (use torch::headeronly::Half)"},
        {std::regex(R"(\bc10::Half\b)"),
         "unstable type: c10::Half (use torch::headeronly::Half)"},
        {std::regex(R"(\bc10::BFloat16\b)"),
         "unstable type: c10::BFloat16 (use torch::headeronly::BFloat16)"},
        {std::regex(R"(\bc10::Float8_e4m3fn\b)"),
         "unstable type: c10::Float8_e4m3fn (use torch::headeronly::Float8_e4m3fn)"},
        {std::regex(R"(\bc10::Float8_e4m3fnuz\b)"),
         "unstable type: c10::Float8_e4m3fnuz (use torch::headeronly::Float8_e4m3fnuz)"},
        {std::regex(R"(\bAT_DISPATCH_)"),
         "unstable macro: AT_DISPATCH_* (use THO_DISPATCH_*)"},
        {std::regex(R"(\bc10::CppTypeToScalarType\b)"),
         "unstable type: c10::CppTypeToScalarType (use torch::headeronly::CppTypeToScalarType)"},
        {std::regex(R"(\bat::elementSize\b)"),
         "unstable function: at::elementSize (use tensor.element_size())"},
    };
    return patterns;
}

std::vector<Violation> verifyStableAbiRegex(const std::string &filepath) {
    std::vector<Violation> violations;

    std::ifstream file(filepath);
    if (!file.is_open())
        return violations;

    std::string line;
    unsigned lineNo = 0;
    bool in_block_comment = false;

    while (std::getline(file, line)) {
        ++lineNo;

        if (in_block_comment) {
            auto end = line.find("*/");
            if (end != std::string::npos) {
                in_block_comment = false;
                line = line.substr(end + 2);
            } else {
                continue;
            }
        }

        auto slashslash = line.find("//");
        std::string code = (slashslash != std::string::npos)
                               ? line.substr(0, slashslash)
                               : line;

        for (;;) {
            auto blockStart = code.find("/*");
            if (blockStart == std::string::npos)
                break;
            auto blockEnd = code.find("*/", blockStart + 2);
            if (blockEnd != std::string::npos) {
                code = code.substr(0, blockStart) + code.substr(blockEnd + 2);
            } else {
                code = code.substr(0, blockStart);
                in_block_comment = true;
                break;
            }
        }

        if (code.empty())
            continue;

        for (const auto &fp : getForbiddenPatterns()) {
            std::smatch match;
            if (std::regex_search(code, match, fp.pattern)) {
                violations.push_back(
                    {filepath, lineNo, 0, match[0].str(), std::string(fp.reason)});
            }
        }
    }

    return violations;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void printViolations(const std::vector<Violation> &violations) {
    if (violations.empty()) {
        llvm::outs() << "ABI verification: PASS (no unstable API usage found)\n";
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
