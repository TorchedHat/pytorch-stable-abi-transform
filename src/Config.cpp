#include "Config.h"
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/FileSystem.h>
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

    makeAbsolute(out.pytorch_root, base);
    makeAbsolute(out.project_root, base);
    makeAbsolute(out.cuda_include, base);

    auto warnUnexpanded = [](const std::string &s, llvm::StringRef field) {
        size_t pos = 0;
        while ((pos = s.find("${", pos)) != std::string::npos) {
            auto end = s.find('}', pos);
            if (end != std::string::npos) {
                llvm::errs() << "warning: unexpanded variable in " << field
                             << ": "
                             << llvm::StringRef(s).substr(pos, end - pos + 1)
                             << "\n";
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
    for (auto &p : out.sources) {
        expandVars(p, out);
        makeAbsolute(p, base);
        warnUnexpanded(p, "sources");
    }
    warnUnexpanded(out.pytorch_root, "pytorch_root");
    warnUnexpanded(out.project_root, "project_root");
    warnUnexpanded(out.cuda_include, "cuda_include");

    return true;
}

void printExampleConfig() {
    llvm::outs() << R"(# .stable-abi.yaml — stable-abi-transform project config
#
# Usage:
#   stable-abi-transform --config=.stable-abi.yaml
#   stable-abi-transform   # auto-discovers .stable-abi.yaml in cwd

# Operating mode: audit (report findings), rewrite (transform in-place), verify
mode: audit

# Output format: text or json
format: text

# Path to PyTorch source/install root (required for compile-based verify)
pytorch_root: /path/to/pytorch

# Project root — rewrites files under this path (headers included).
# Also auto-discovers .cpp/.cu source files when 'sources' is omitted.
project_root: ./csrc

# Compiler flags passed to clang
compiler_flags:
  - -std=c++20

# Include paths (each becomes a -I flag)
# Use ${pytorch_root} and ${project_root} for variable expansion.
include_paths:
  - ${pytorch_root}/torch/csrc/api/include
  - ${pytorch_root}
  - ${pytorch_root}/torch/include
  - /usr/local/cuda/include
  # - ./csrc/inc

# Additional include paths for verification (project-specific headers)
# extra_includes:
#   - ./csrc

# Explicit source files (optional — auto-discovered from project_root if omitted)
# sources:
#   - csrc/file1.cpp
#   - csrc/file2.cu

# Verification method: compile (default, requires pytorch_root) or regex
verify_method: compile

# CUDA include path (auto-detected if omitted)
# cuda_include: /usr/local/cuda/include
)";
}

} // namespace stable_abi
