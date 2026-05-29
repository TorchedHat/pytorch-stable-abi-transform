# PyTorch Stable ABI Transform

Clang-based source-to-source rewriter that migrates PyTorch C++ extensions from the unstable ABI (`at::`, `c10::`, `torch::`) to the stable ABI (`torch::stable::`, `torch::headeronly::`).

The stable ABI lets extensions work across PyTorch versions without recompilation. This tool handles the mechanical migration: includes, types, macros, `data_ptr` disambiguation, CUDA guards, method-to-function rewrites. It is a deterministic CLI tool and no LLM is required. It operates on the abstract syntax tree and so has a richer semantic understanding than a regex "find and replace" level tool. 

It is likely that LLMs will be used in the migration effort. Rather than have the LLM probabilistically make the transformations and then you clean it up, let the LLM use the deterministic tool and then it can clean up. A CLAUDE.md file and a Claude skill are included to facilitate this workflow. An LLM is not required in order to use this migration tool.

The scope of this tool's "success" is compilation of the transformed code against the stable ABI headers. Correctness and performance are not in this tool's scope and are the true metrics of a "successful" migration. But you have to compile first...

## Quick start

```bash
# Install LLVM dev libraries (see docs/user-guide.md for details)
# Fedora: dnf install -y clang-devel llvm-devel ninja-build
# Ubuntu: apt install -y libclang-19-dev libclang-cpp19-dev llvm-19-dev ninja-build

# Build (tested with LLVM 19, CUDA 12.9)
mkdir -p build && cmake -GNinja -B build -S . && cmake --build build

# Generate config
./build/stable-abi-transform --init-config > .stable-abi.yaml
# Edit: set pytorch_root, project_root

# Audit (read-only)
./build/stable-abi-transform                        # mode: audit

# Migration plan (dependency-aware grouping)
./build/stable-abi-transform --mode=plan

# Preview changes
./build/stable-abi-transform --mode=rewrite --dry-run

# Rewrite + auto-verify
./build/stable-abi-transform --mode=rewrite

# Or without a config file — pass flags directly:
./build/stable-abi-transform --mode=audit file.cu -- -std=c++20 \
  -I/path/to/pytorch/torch/csrc/api/include -I/path/to/pytorch
```

Before/after — the tool rewrites this:

```cpp
#include <torch/all.h>
void foo(const at::Tensor& input) {
    TORCH_CHECK(input.dim() == 2);
    float* ptr = input.data_ptr<float>();
    auto result = input.clone();
}
```

To this:

```cpp
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
void foo(const torch::stable::Tensor& input) {
    STD_TORCH_CHECK(input.dim() == 2);
    const float* ptr = input.const_data_ptr<float>();
    auto result = torch::stable::clone(input);
}
```

Patterns it can't auto-rewrite (TensorOptions decomposition, PYBIND11_MODULE replacement, project dispatch macros) are flagged with actionable guidance.

Supports `.cpp`, `.cu`, and `.cuh` files. Audit and plan modes run in parallel (`--jobs`) for large projects. Use the `transform` config field to target specific files for incremental migration. See the [demo walkthrough](docs/demo-walkthrough.md) for a real-world example analyzing vLLM (~50k lines of C++/CUDA).

## Verification

Compile-based verification creates a shadow include tree with *only* stable headers. If the file compiles against it, migration is provably complete — zero false positives, zero false negatives. See [docs/user-guide.md](docs/user-guide.md) for details.

## Documentation

- **[Demo Walkthrough](docs/demo-walkthrough.md)** — real-world case study (vLLM, 235 files)
- **[User Guide](docs/user-guide.md)** — full reference: modes, config, flags, verification, architecture
- **[Claude Skill](docs/claude-skill.md)** — Claude Code skill for agent-assisted migration
- **[CLAUDE.md](CLAUDE.md)** — project context for Claude Code

## License

[MIT](LICENSE)
