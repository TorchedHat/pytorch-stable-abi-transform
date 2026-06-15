#include "Config.h"
#include <cstdio>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

namespace stable_abi {

static void expandVar(std::string &s, llvm::StringRef varName,
                      llvm::StringRef value) {
    std::string token = "${" + varName.str() + "}";
    size_t pos = 0;
    while ((pos = s.find(token, pos)) != std::string::npos) {
        s.replace(pos, token.size(), value.data(), value.size());
        pos += value.size();
    }
}

static void expandVars(std::string &s, const Config &c) {
    expandVar(s, "pytorch_root", c.pytorch_root);
    expandVar(s, "project_root", c.project_root);
}

static void makeAbsolute(std::string &path, llvm::StringRef base) {
    if (path.empty())
        return;
    llvm::SmallString<256> abs(path);
    if (!llvm::sys::path::is_absolute(abs)) {
        llvm::SmallString<256> full(base);
        llvm::sys::path::append(full, abs);
        abs = full;
    }
    llvm::sys::path::remove_dots(abs, true);
    path = std::string(abs);
}

bool loadConfig(const std::string &path, Config &out, std::string &error) {
    auto bufOrErr = llvm::MemoryBuffer::getFile(path);
    if (!bufOrErr) {
        error = "cannot open config file: " + path + ": " +
                bufOrErr.getError().message();
        return false;
    }

    llvm::yaml::Input yamlIn(bufOrErr.get()->getBuffer());
    yamlIn >> out;
    if (yamlIn.error()) {
        error = "YAML parse error in " + path;
        return false;
    }

    llvm::SmallString<256> configDir(path);
    llvm::sys::path::remove_filename(configDir);
    if (configDir.empty())
        configDir = ".";
    llvm::sys::fs::make_absolute(configDir);
    auto base = llvm::StringRef(configDir);

    if (out.pytorch_root != "auto")
        makeAbsolute(out.pytorch_root, base);
    makeAbsolute(out.project_root, base);
    makeAbsolute(out.cuda_include, base);
    expandVars(out.output_dir, out);
    makeAbsolute(out.output_dir, base);

    auto warnUnexpanded = [](const std::string &s, llvm::StringRef field) {
        size_t pos = 0;
        while ((pos = s.find("${", pos)) != std::string::npos) {
            auto end = s.find('}', pos);
            if (end != std::string::npos) {
                llvm::errs()
                    << "warning: unexpanded variable in " << field << ": "
                    << llvm::StringRef(s).substr(pos, end - pos + 1) << "\n";
                pos = end + 1;
            } else {
                break;
            }
        }
    };

    for (auto &p : out.include_paths) {
        expandVars(p, out);
        makeAbsolute(p, base);
        warnUnexpanded(p, "include_paths");
    }
    for (auto &p : out.extra_includes) {
        expandVars(p, out);
        makeAbsolute(p, base);
        warnUnexpanded(p, "extra_includes");
    }
    for (auto &p : out.transform) {
        expandVars(p, out);
        makeAbsolute(p, base);
        warnUnexpanded(p, "transform");
    }
    warnUnexpanded(out.pytorch_root, "pytorch_root");
    warnUnexpanded(out.project_root, "project_root");
    warnUnexpanded(out.cuda_include, "cuda_include");
    warnUnexpanded(out.output_dir, "output_dir");

    return true;
}

static bool hasStableHeaders(const std::string &path) {
    return llvm::sys::fs::is_directory(path + "/torch/csrc/stable");
}

static std::string detectPytorchRoot() {
    if (const char *env = std::getenv("PYTORCH_ROOT")) {
        std::string root(env);
        if (hasStableHeaders(root))
            return root;
        std::string inc = root + "/torch/include";
        if (hasStableHeaders(inc))
            return inc;
    }

    FILE *pipe =
        popen("python3 -c \"import torch, os; "
              "print(os.path.join(torch.__path__[0], 'include'))\" 2>/dev/null",
              "r");
    if (!pipe)
        return "";
    char buf[512];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe))
        result += buf;
    int status = pclose(pipe);
    if (status != 0)
        return "";
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

std::vector<std::string> pytorchIncludePaths(const std::string &root) {
    if (root.empty())
        return {};
    std::vector<std::string> paths;
    paths.push_back(root);
    std::string apiInclude = root + "/torch/csrc/api/include";
    if (llvm::sys::fs::is_directory(apiInclude))
        paths.push_back(apiInclude);
    std::string torchInclude = root + "/torch/include";
    if (llvm::sys::fs::is_directory(torchInclude))
        paths.push_back(torchInclude);
    return paths;
}

bool resolvePytorchRoot(Config &cfg, std::string &error) {
    if (cfg.pytorch_root.empty() || cfg.pytorch_root == "auto") {
        auto detected = detectPytorchRoot();
        if (!detected.empty()) {
            cfg.pytorch_root = detected;
            llvm::errs() << "note: auto-detected pytorch_root: "
                         << cfg.pytorch_root << "\n";
        } else if (cfg.pytorch_root == "auto") {
            error = "pytorch_root=auto but python3 -c 'import torch' failed. "
                    "Install torch or set pytorch_root explicitly.";
            return false;
        }
    }
    if (!cfg.pytorch_root.empty()) {
        std::string stableDir = cfg.pytorch_root + "/torch/csrc/stable";
        if (!llvm::sys::fs::is_directory(stableDir)) {
            error = "pytorch_root does not contain stable ABI headers "
                    "(torch/csrc/stable/). Requires PyTorch >= 2.6. "
                    "Path: " +
                    cfg.pytorch_root;
            return false;
        }
    }
    return true;
}

void printExampleConfig() {
    llvm::outs() << R"(# .stable-abi.yaml — stable-abi-transform project config
#
# Usage:
#   stable-abi-transform --config=.stable-abi.yaml
#   stable-abi-transform   # auto-discovers .stable-abi.yaml in cwd

# Operating mode: audit (report findings), rewrite (transform in-place), verify, plan
mode: audit

# Output format: text or json
format: text

# Path to PyTorch root. PyTorch include paths are auto-derived from this.
#   "auto"               — detect from pip-installed torch (recommended)
#   /path/to/pytorch     — PyTorch source tree
#   /path/to/libtorch/include — libtorch download or pip site-packages
pytorch_root: auto

# Project root — rewrites files under this path (headers included).
# Also auto-discovers .cpp/.cu source files when 'transform' is omitted.
project_root: ./csrc

# Compiler flags passed to clang
compiler_flags:
  - -std=c++20

# Additional include paths for your project (PyTorch paths are auto-derived
# from pytorch_root — you only need project-specific paths here).
# include_paths:
#   - ./csrc/inc
#   - /usr/local/cuda/include

# Additional include paths for verification (project-specific headers)
# extra_includes:
#   - ./csrc

# Files or directories to transform (optional — auto-discovered from project_root if omitted).
# Directory entries are recursively walked for .cpp/.cu/.cuh files.
# Use this for incremental migration: set project_root for full include scope,
# then list only the files you want to transform in this run.
# transform:
#   - csrc/attention/           # walk entire directory
#   - csrc/cache.cu             # single file

# CUDA include path (auto-detected if omitted)
# cuda_include: /usr/local/cuda/include

# Path to directory containing compile_commands.json (optional).
# When set, per-file compiler flags are read from the database instead of
# compiler_flags/include_paths. Auto-detected from project_root if omitted.
# Generate with: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build
# compile_commands_dir: build

# Output directory for out-of-place rewrite (optional)
# When set, transformed files are written here instead of in-place.
# Only modified files are written, preserving relative paths from project_root.
# output_dir: ./stable-output

# Parallel TU processing (0 = auto-detect, 1 = sequential)
# jobs: 0
)";
}

} // namespace stable_abi
