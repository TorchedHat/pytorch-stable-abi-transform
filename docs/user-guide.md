# User Guide

## Installation

### Prerequisites

- Linux (tested on Fedora 41)
- LLVM/Clang 19 development libraries, CUDA 12.9 headers
- C++20 compiler
- PyTorch source tree or install root (for include paths and verification)

### Build

```bash
# Fedora
dnf install -y clang-devel llvm-devel ninja-build

# Ubuntu/Debian
apt install -y libclang-19-dev libclang-cpp19-dev llvm-19-dev ninja-build

# Build
mkdir -p build && cmake -GNinja -B build -S . && cmake --build build
```

## Configuration

Generate a starter config and edit it for your project:

```bash
stable-abi-transform --init-config > .stable-abi.yaml
```

Example config:

```yaml
mode: audit
pytorch_root: /path/to/pytorch
project_root: ./csrc

compiler_flags:
  - -std=c++20

include_paths:
  - ${pytorch_root}/torch/csrc/api/include
  - ${pytorch_root}
  - ${pytorch_root}/torch/include
  - /usr/local/cuda/include
  - ./csrc/inc                            # project-specific headers

extra_includes:                           # for verify mode
  - ./csrc
```

At minimum you need: `pytorch_root`, the three PyTorch include paths, CUDA include path, and the project's own header directories. The `${pytorch_root}` variable is expanded automatically.

When `.stable-abi.yaml` exists in the working directory, all commands auto-load it — no flags needed:

```bash
stable-abi-transform    # uses .stable-abi.yaml
```

## Modes

| Mode | What it does | Modifies files? |
|------|-------------|-----------------|
| `audit` (default) | Reports all unstable API usage with suggested replacements | No |
| `rewrite` | Transforms source files in-place, then auto-verifies | Yes |
| `rewrite` + `--dry-run` | Shows unified diff of what rewrite would produce | No |
| `verify` | Checks if a file uses only stable APIs | No |

Switch modes by editing `mode:` in `.stable-abi.yaml` or passing `--mode=` on the command line.

### Output formats

- `--format=text` (default) — human-readable report
- `--format=json` — machine-readable for CI integration

## What gets rewritten

| Category | Example | Auto-rewrite? |
|----------|---------|---------------|
| Includes | `torch/all.h` -> stable headers | Yes |
| Types | `at::Tensor` -> `torch::stable::Tensor` | Yes |
| Macros | `TORCH_CHECK` -> `STD_TORCH_CHECK` | Yes |
| Scalar types | `at::kFloat` -> `torch::headeronly::ScalarType::Float` | Yes |
| Device types | `at::kCUDA` -> `torch::headeronly::DeviceType::CUDA` | Yes |
| data_ptr | `t.data_ptr<T>()` -> `mutable_data_ptr` or `const_data_ptr` (AST const-analysis) | Yes |
| Methods | `t.clone()` -> `torch::stable::clone(t)` | Yes |
| Free functions | `torch::empty(...)` -> `torch::stable::empty(...)` | Yes |
| CUDA guards | `OptionalCUDAGuard` -> `DeviceGuard` | Yes |
| CUDA streams | `getCurrentCUDAStream()` -> `aoti_torch_get_current_cuda_stream()` | Yes |
| Op registration | `TORCH_LIBRARY` -> `STABLE_TORCH_LIBRARY` | Yes |
| Dispatch macros | `AT_DISPATCH_SWITCH` -> `THO_DISPATCH_SWITCH` | Yes |
| Comparison macros | `TORCH_CHECK_EQ(a, b)` -> `STD_TORCH_CHECK((a) == (b))` | Yes |
| `c10::optional<T>` | -> `std::optional<T>` | Yes |
| `c10::nullopt` | -> `std::nullopt` | Yes |
| `c10::ArrayRef<T>` | -> `torch::headeronly::HeaderOnlyArrayRef<T>` | Yes |
| `c10::IntArrayRef` | -> `torch::headeronly::IntHeaderOnlyArrayRef` | Yes |
| `c10::string_view` | -> `std::string_view` | Yes |
| TensorOptions | `torch::TensorOptions(...)` | Flagged |
| Unstable macros in macro bodies | `TORCH_CHECK` etc. inside `#define` | Flagged |
| PYBIND11_MODULE | Binding macro detection | Flagged |
| `elementSize(dtype)` | Standalone call (no tensor) | Flagged |
| Project dispatch macros | e.g. `VLLM_DISPATCH_*` | Flagged |

"Flagged" means the tool detects the pattern, reports it as `[FLAG]`, and tells you what to do — but the rewrite requires human judgment.

### Catch-all detection

The tool has AST-based catch-all matchers that flag ANY remaining `at::`, `c10::`, or unstable `torch::` usage. No unstable pattern can silently pass through unreported.

## Verification

### Compile-based (default)

Requires `pytorch_root` in config (or `--pytorch-root` flag).

Creates a temporary shadow include tree containing symlinks to only the stable PyTorch headers:

```
$TMP/torch/csrc/stable/               -> $PYTORCH/torch/csrc/stable/
$TMP/torch/csrc/inductor/aoti_torch/  -> $PYTORCH/torch/csrc/inductor/aoti_torch/
$TMP/torch/headeronly/                -> $PYTORCH/torch/include/torch/headeronly/
```

Then compiles the file with `-I $TMP` only. If it compiles, the file uses only stable API — provably. If not, the compiler errors ARE the gap analysis.

**Why this is sound**: Stable headers form a closed dependency set. They never include `c10/`, `ATen/`, or `torch/csrc/api/`. So compilation against only stable headers has zero false positives and zero false negatives.

Ensure `extra_includes` lists the project's own header directories (e.g., `./csrc`) — the verifier needs them to resolve project-internal `#include` directives.

### Regex-based (fallback)

```yaml
verify_method: regex    # in .stable-abi.yaml
```

Pattern-matches against known unstable namespaces. Faster but less precise — use compile-based when possible.

## CLI reference

| Flag | Description |
|------|-------------|
| `--mode` | `audit`, `rewrite`, or `verify` |
| `--pytorch-root` | Path to PyTorch source/install root |
| `--project-root` | Process all C++/CUDA files under this directory |
| `--extra-includes` | Project-specific `-I` paths for verify mode |
| `--cuda-include` | CUDA include path (auto-detected from `/usr/local/cuda/include`) |
| `--verify-method` | `compile` (default) or `regex` |
| `--format` | `text` (default) or `json` |
| `--dry-run` | Preview rewrite as unified diff (rewrite mode only) |
| `--config` | Path to YAML config file (default: `.stable-abi.yaml`) |
| `--init-config` | Generate a starter config and exit |

## CI integration

```bash
# Exit codes:
#   audit/dry-run: 0 = no unstable API found, 1 = findings exist
#   rewrite:       0 = all auto-fixed, 1 = flagged items remain
#   verify:        0 = stable ABI only, 1 = unstable API detected

stable-abi-transform --mode=audit --format=json src/*.cu -- [flags]

# Or verify already-migrated files
stable-abi-transform --mode=verify --pytorch-root=$PYTORCH src/*.cu -- -std=c++20
```

### Using a config file with CLI overrides

When `.stable-abi.yaml` exists, the tool auto-loads it. You can still override which files to process on the command line:

```bash
# Process a specific file using config's compiler settings
stable-abi-transform src/my_kernel.cu

# Config controls compiler_flags and include_paths;
# -- flags on the CLI are ignored in config mode
```

## Regenerating rules

Transformation rules are derived from PyTorch stable ABI headers:

```bash
python3 scripts/gen_rules.py /path/to/pytorch src/Rules.h
```

Parses `torch/csrc/stable/` and `torch/headeronly/` to extract the complete stable API surface. Run this when PyTorch adds new stable APIs.

## Architecture

```
src/
  main.cpp                  CLI, mode dispatch, ClangTool configuration
  StableAbiAction.{h,cpp}   Clang FrontendAction + ASTConsumer wiring
  TransformerRules.cpp       AST-based rewrite rules (Clang Transformer)
  AstCallbacks.{h,cpp}      Manual AST matchers (CUDA guard/stream)
  PreprocessorCallbacks.cpp  Include and macro rewrites
  Reporter.{h,cpp}          Finding accumulation, text/JSON output
  Verifier.{h,cpp}          Compile-based and regex-based verification
  Rules.h                   Auto-generated transformation tables
  Config.{h,cpp}            YAML config support
  Helpers.h                 Shared utilities
```

### How verification works internally

The compile-based verifier (`Verifier.cpp`):

1. Creates a temp directory
2. Symlinks only the three stable header trees into it
3. Runs `clang++ -fsyntax-only` with `-I $TMP` replacing PyTorch includes
4. If exit 0 → file is stable. If not → compiler errors identify remaining unstable API usage

This is sound because stable headers form a closed set — they never `#include` unstable code. Verified empirically: zero references to `c10/`, `ATen/`, or `torch/csrc/api/` from any stable header.

## Tests

```bash
PYTORCH_DIR=/path/to/pytorch RESOURCE_DIR=/usr/lib/clang/19 bash test/run_tests.sh
```

`PYTORCH_DIR` is required (points to your PyTorch source tree). `RESOURCE_DIR` defaults to `/usr/lib/clang/19`.

13 test cases with input/expected pairs in `test/inputs/` and `test/expected/`. The test suite runs four passes: rewrite correctness, regex verification, exit code validation, and compile-based verification.
