# User Guide

## Installation

### Prerequisites

- Linux (tested on Fedora 41)
- LLVM/Clang 19 development libraries
- C++20 compiler
- PyTorch headers (via `pip install torch`, libtorch download, or source tree)
- CUDA headers (optional, for `.cu`/`.cuh` analysis)

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
pytorch_root: auto                        # detects from pip-installed torch
project_root: ./csrc

compiler_flags:
  - -std=c++20

# include_paths:                          # only needed for project-specific paths
#   - ./csrc/inc                          # (PyTorch paths are auto-derived)

# extra_includes:                         # for verify mode
#   - ./csrc
```

PyTorch include paths are auto-derived from `pytorch_root` — you only need project-specific paths in `include_paths`. Set `pytorch_root` to `auto` (detects from `pip install torch`), or an explicit path to a PyTorch source tree or libtorch download.

When `.stable-abi.yaml` exists in the working directory, all commands auto-load it — no flags needed:

```bash
stable-abi-transform    # uses .stable-abi.yaml
```

CLI flags override the config file, so the YAML captures your project defaults and the CLI handles per-invocation choices.

### Incremental migration with `transform`

For large projects, you don't want to transform everything at once. Use the `transform` field to target specific files or directories while keeping the full project scope for include analysis:

```yaml
project_root: ./csrc            # full scope — all headers visible
transform:                       # only these get transformed
  - ./csrc/attention/
  - ./csrc/cache_kernels.cu
```

Or on the CLI (positional args serve as transform targets):

```bash
stable-abi-transform --mode=rewrite ./csrc/cache_kernels.cu
```

Headers included from the target files that live under `project_root` are still in rewrite scope — shared headers get transformed correctly. Files outside `transform` are untouched.

Omit `transform` to process everything under `project_root`.

## Modes

| Mode | What it does | Modifies files? |
|------|-------------|-----------------|
| `audit` (default) | Reports all unstable API usage with suggested replacements | No |
| `plan` | Analyzes include dependencies and partitions files into independent migration groups | No |
| `rewrite` | Transforms source files in-place, then auto-verifies | Yes |
| `rewrite` + `--dry-run` | Shows unified diff of what rewrite would produce | No |
| `verify` | Checks if a file uses only stable APIs | No |

Switch modes by editing `mode:` in `.stable-abi.yaml` or passing `--mode=` on the command line.

### Plan mode

Plan mode analyzes the include dependency graph and partitions files into groups that must migrate together (they share headers with unstable API usage):

```bash
stable-abi-transform --mode=plan
```

Groups are independent — they can be separate PRs merged in any order. Each group lists the specific unstable APIs it uses, so reviewers know the scope before starting.

Use `--format=json` for machine-readable plan output (dependency-aware PR grouping in CI).

### Output formats

- `--format=text` (default) — human-readable report
- `--format=json` — machine-readable for CI integration

## What gets rewritten

| Category | Example | Auto-rewrite? |
|----------|---------|---------------|
| Includes | `torch/all.h` -> stable headers | Yes |
| Types | `at::Tensor` -> `torch::stable::Tensor` | Yes |
| Macros | `TORCH_CHECK` -> `STD_TORCH_CHECK` | Yes |
| Comparison macros | `TORCH_CHECK_EQ(a, b)` -> `STD_TORCH_CHECK((a) == (b))` | Yes |
| Scalar types | `at::kFloat` -> `torch::headeronly::ScalarType::Float` | Yes |
| Device types | `at::kCUDA` -> `torch::headeronly::DeviceType::CUDA` | Yes |
| data_ptr | `t.data_ptr<T>()` -> `mutable_data_ptr` or `const_data_ptr` (AST const-analysis) | Yes |
| Methods | `t.clone()` -> `torch::stable::clone(t)` | Yes |
| Free functions | `torch::empty(...)` -> `torch::stable::empty(...)` | Yes |
| CUDA guards | `OptionalCUDAGuard` -> `DeviceGuard` | Yes |
| CUDA streams | `getCurrentCUDAStream()` -> `aoti_torch_get_current_cuda_stream()` | Yes |
| Op registration | `TORCH_LIBRARY` -> `STABLE_TORCH_LIBRARY` | Yes |
| Dispatch macros | `AT_DISPATCH_SWITCH` -> `THO_DISPATCH_SWITCH` | Yes |
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

## .cuh files

`.cuh` (CUDA header) files are automatically discovered and analyzed alongside `.cpp` and `.cu` files. They are parsed as C++ — no CUDA toolchain is required. CUDA device-only constructs may cause parse errors that are reported per-file but do not block analysis of host-side API calls.

## Parallel execution

Audit and plan modes process translation units in parallel using a thread pool:

```bash
stable-abi-transform --jobs=8           # 8 threads
stable-abi-transform --jobs=0           # auto-detect (default)
stable-abi-transform --jobs=1           # sequential
```

Or in config:

```yaml
jobs: 0    # 0 = auto-detect, 1 = sequential
```

Rewrite mode always runs sequentially (parallel rewrite with shared header deduplication is planned for v0.3.0).

## Out-of-place rewrite

To keep originals intact, write transformed files to a separate directory:

```yaml
mode: rewrite
output_dir: ./csrc-stable
```

Or on the CLI:

```bash
stable-abi-transform --mode=rewrite --output-dir=./csrc-stable
```

Only modified files are written, preserving directory structure relative to `project_root`.

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

## Migration progress

`scripts/migration_progress.py` measures how close a project is to stable ABI adoption:

```bash
# Audit then report progress
python3 scripts/migration_progress.py /path/to/project --source-dir csrc

# From pre-computed audit JSON
python3 scripts/migration_progress.py --audit-json audit.json --source-dir csrc

# With dependency-aware PR grouping (from --mode=plan --format=json)
python3 scripts/migration_progress.py --audit-json audit.json --source-dir csrc \
  --plan-json plan.json

# Machine-readable
python3 scripts/migration_progress.py --audit-json audit.json --source-dir csrc \
  --format json
```

## CLI reference

| Flag | Description |
|------|-------------|
| `--mode` | `audit` (default), `rewrite`, `verify`, or `plan` |
| `--pytorch-root` | Path to PyTorch source/install root, or `auto` to detect from pip |
| `--project-root` | Project root directory; auto-discovers source files and bounds rewrites |
| `--output-dir` | Write transformed files here instead of in-place (requires `--project-root`) |
| `--extra-includes` | Project-specific `-I` paths for verify mode |
| `--cuda-include` | CUDA include path (auto-detected from `/usr/local/cuda/include`) |
| `--verify-method` | `compile` (default) or `regex` |
| `--format` | `text` (default) or `json` |
| `--dry-run` | Preview rewrite as unified diff (rewrite mode only) |
| `--jobs` | Parallel TU threads: 0 = auto (default), 1 = sequential. Audit and plan only. |
| `--config` | Path to YAML config file (default: `.stable-abi.yaml`) |
| `--init-config` | Generate a starter config and exit |

Positional arguments are treated as transform targets (files or directories to process).

## CLI without a config file

The tool works without `.stable-abi.yaml` — pass source files and compiler flags directly:

```bash
TOOL=build/stable-abi-transform
PYTORCH=/path/to/pytorch

# Audit a single file
$TOOL file.cu -- -std=c++20 -I$PYTORCH/torch/csrc/api/include \
  -I$PYTORCH -I$PYTORCH/torch/include

# Rewrite in-place
$TOOL --mode=rewrite --pytorch-root=$PYTORCH file.cu -- -std=c++20 \
  -I$PYTORCH/torch/csrc/api/include -I$PYTORCH -I$PYTORCH/torch/include

# Verify
$TOOL --mode=verify --pytorch-root=$PYTORCH \
  --extra-includes=./csrc file.cu -- -std=c++20

# Plan (requires --project-root)
$TOOL --mode=plan --project-root=./csrc -- -std=c++20 \
  -I$PYTORCH/torch/csrc/api/include -I$PYTORCH -I$PYTORCH/torch/include
```

Everything after `--` is passed to clang. The config file eliminates the repeated flags — it's recommended for any project you'll run the tool on more than once.

## CI integration

```bash
# Exit codes:
#   audit/dry-run: 0 = no unstable API found, 1 = findings exist
#   rewrite:       0 = all auto-fixed, 1 = flagged items remain
#   verify:        0 = stable ABI only, 1 = unstable API detected
#   plan:          always 0

# With config (recommended)
stable-abi-transform --mode=audit --format=json

# Without config
stable-abi-transform --mode=audit --format=json src/*.cu -- -std=c++20 \
  -I$PYTORCH/torch/csrc/api/include -I$PYTORCH -I$PYTORCH/torch/include

# Verify already-migrated files
stable-abi-transform --mode=verify --pytorch-root=$PYTORCH src/*.cu -- -std=c++20
```

### CLI overrides

When `.stable-abi.yaml` exists, the tool auto-loads it. CLI flags override config values, and positional arguments override `transform`:

```bash
# Process a specific file using config's compiler settings
stable-abi-transform src/my_kernel.cu

# Override mode for this invocation
stable-abi-transform --mode=plan
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
  main.cpp                  CLI, mode dispatch, parallel execution, ClangTool config
  StableAbiAction.{h,cpp}   Clang FrontendAction + ASTConsumer wiring
  TransformerRules.cpp       AST-based rewrite rules (Clang Transformer)
  AstCallbacks.{h,cpp}      Manual AST matchers (CUDA guard/stream)
  PreprocessorCallbacks.cpp  Include and macro rewrites
  Reporter.{h,cpp}          Finding accumulation, text/JSON output
  Verifier.{h,cpp}          Compile-based and regex-based verification
  Config.{h,cpp}            YAML config support
  DepGraph.{h,cpp}          Include DAG analysis, migration partitioning, plan output
  Rules.h                   Auto-generated transformation tables
  Helpers.h                 Shared utilities (project scope check, path helpers)

scripts/
  gen_rules.py              Derives Rules.h from PyTorch stable ABI headers
  migration_progress.py     Migration progress reporting from audit/plan JSON
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

14 test cases with input/expected pairs in `test/inputs/` and `test/expected/`. The test suite runs four passes: rewrite correctness, regex verification, exit code validation, and compile-based verification.
