# stable-abi-transform: Demo Walkthrough

Analyzing vllm's C++ extension codebase (235 files, ~50k lines) as a real-world case study.

## Setup

### Generate a project config

From your project root:

```bash
cd /path/to/vllm
stable-abi-transform --init-config > .stable-abi.yaml
```

Edit `.stable-abi.yaml` to match your project:

```yaml
mode: audit
format: text
pytorch_root: auto            # detect from pip-installed torch
project_root: ./csrc

compiler_flags:
  - -std=c++20

# include_paths:              # only needed for project-specific paths
#   - ./csrc/inc              # (PyTorch paths are auto-derived from pytorch_root)
```

Now every invocation auto-loads this config — no long CLI flags needed:

```bash
stable-abi-transform                  # audit (default mode)
stable-abi-transform --mode=plan      # override just the mode
stable-abi-transform --format=json    # override just the format
```

CLI flags override the config file, so the YAML captures your project defaults and the CLI handles per-invocation choices.

### Incremental migration with `transform`

For a large project, you don't want to rewrite 200+ files at once. Use the `transform` field to target a subset while keeping the full project scope for include analysis:

```yaml
project_root: ./csrc           # full scope — all headers visible
transform:                      # only these get transformed
  - ./csrc/cache_kernels.cu
  - ./csrc/attention/
```

Or on the CLI (positional args serve as transform targets):

```bash
stable-abi-transform --mode=rewrite ./csrc/cache_kernels.cu
```

Headers included from the target files that live under `project_root` are still in rewrite scope — shared headers get transformed correctly. Files outside `transform` are untouched.

Omit `transform` to process everything under `project_root`.

## Step 1: Audit — What needs to change?

```bash
stable-abi-transform
```

The tool auto-discovers all `.cpp`, `.cu`, and `.cuh` files under `project_root` and reports every usage of unstable PyTorch internals:

```
[MACRO] csrc/cache_kernels.cu:45:5       TORCH_CHECK -> STD_TORCH_CHECK
[TYPE ] csrc/sampler.cu:12:18            torch::Tensor -> torch::stable::Tensor
[DPTR ] csrc/sampler.cu:30:22            .data_ptr<float>() -> .mutable_data_ptr<float>()
[GUARD] csrc/sampler.cu:8:5              at::cuda::OptionalCUDAGuard -> DeviceGuard
[STRM ] csrc/sampler.cu:9:22            at::cuda::getCurrentCUDAStream() -> aoti_torch_get_current_cuda_stream()
[M2F  ] csrc/attention/paged_attention_v2.cu:190:36  dtype -> scalar_type
[INCL ] csrc/moe/moe_ops.h:3:1          #include <torch/all.h> -> (removed, no torch API usage)
[FLAG ] csrc/cpu/sgl-kernels/vec.h:20:1  at::vec::Vectorized -> no stable equivalent
```

Each finding is either:
- **Auto-rewritable** — the tool can transform this automatically
- **Flagged** — requires manual review (no stable equivalent, or inside a macro body)

### vllm results

```
Summary: 2,554 auto-rewritable, 429 flagged for manual review
```

| Category | Description | Auto | Flag | Total |
|----------|-------------|------|------|-------|
| MACRO | `TORCH_CHECK` -> `STD_TORCH_CHECK`, etc. | 841 | 83 | 924 |
| DPTR | `.data_ptr<T>()` -> `.mutable_data_ptr<T>()` | 593 | 2 | 595 |
| TYPE | `at::Tensor` -> `torch::stable::Tensor`, etc. | 499 | 72 | 571 |
| FLAG | Unstable APIs with no stable equivalent | 0 | 209 | 209 |
| INCL | `#include <torch/all.h>` -> stable headers | 152 | 26 | 178 |
| STYPE | `at::kFloat` -> `torch::headeronly::ScalarType::Float` | 146 | 29 | 175 |
| FUNC | `at::empty(...)` -> `torch::stable::empty(...)` | 101 | 4 | 105 |
| M2F | `.clone()` -> `torch::stable::clone(t)` | 93 | 0 | 93 |
| STRM | CUDA stream acquisition | 69 | 4 | 73 |
| GUARD | `CUDAGuard` -> `DeviceGuard` | 60 | 0 | 60 |

For CI integration or migration tracking, use JSON output:

```bash
stable-abi-transform --format=json
```

## Step 2: Plan — How to sequence the migration

```bash
stable-abi-transform --mode=plan
```

The plan mode analyzes the include dependency graph and partitions files into groups that must migrate together (they share headers with unstable API usage):

```
Migration plan: 13 groups (all independent — transform in parallel)

Round 1:
  Group A (76 files, 1523 findings):
    csrc/activation_kernels.cu
    csrc/cache_kernels.cu
    csrc/sampler.cu
    csrc/torch_bindings.cpp
    ... (46 sources, 30 headers)

  Group B (1 file, 37 findings):
    csrc/attention/vertical_slash_index.cu

  Group C (14 files, 97 findings):
    csrc/cpu/activation.cpp
    csrc/cpu/shm.cpp
    ... (10 sources, 4 headers)

  Group D (15 files, 943 findings):
    csrc/cpu/sgl-kernels/conv.cpp
    csrc/cpu/sgl-kernels/gemm.cpp
    ... (10 sources, 5 headers)

  Group E (1 file, 20 findings):   csrc/cuda_view.cu
  Group F (1 file, 53 findings):   csrc/custom_all_reduce.cu
  Group G (1 file, 4 findings):    csrc/custom_quickreduce.cu
  Group H (2 files, 149 findings): csrc/mamba/mamba_ssm/selective_scan_fwd.cu
  Group I (1 file, 53 findings):   csrc/moe/dynamic_4bit_int_moe_cpu.cpp
  Group J (1 file, 35 findings):   csrc/moe/grouped_topk_kernels.cu
  Group K (1 file, 11 findings):   csrc/moe/moe_wna16.cu
  Group L (1 file, 11 findings):   csrc/quantization/gptq/q_gemm.cu
  Group M (1 file, 54 findings):   csrc/topk.cu
```

Each group lists the specific unstable APIs it uses, so reviewers know the scope before starting. Groups are independent — they can be separate PRs merged in any order.

Use `--format=json` for machine-readable plan output (dependency-aware PR grouping in CI).

## Step 3: Preview — Dry-run before committing

```bash
stable-abi-transform --mode=rewrite --dry-run
```

Shows a unified diff of every change the tool would make, without modifying any files:

```diff
-    TORCH_CHECK(group_size <= MAX_SHM_RANK_NUM);
+    STD_TORCH_CHECK(group_size <= MAX_SHM_RANK_NUM);
-    TORCH_CHECK_EQ(ptr->thread_num, thread_num);
+    STD_TORCH_CHECK((ptr->thread_num) == (thread_num));
```

To preview a single file instead of the full project:

```bash
stable-abi-transform --mode=rewrite --dry-run csrc/cpu/shm.cpp
```

## Step 4: Rewrite — Apply transformations

### In-place

```bash
stable-abi-transform --mode=rewrite
```

Transforms all files under `project_root` and runs post-rewrite verification automatically.

### Out-of-place (keep originals intact)

Add `output_dir` to your config:

```yaml
mode: rewrite
output_dir: ./csrc-stable
```

Or on the CLI:

```bash
stable-abi-transform --mode=rewrite --output-dir=./csrc-stable
```

Transformed files are written to `csrc-stable/` preserving directory structure. Originals are untouched.

## Step 5: Verify — Confirm no unstable API remains

After rewriting, verify the result:

```bash
stable-abi-transform --mode=verify
```

**Compile-based verification** (default, requires `pytorch_root`): Creates a shadow include tree with symlinks to only the stable PyTorch headers, then compiles the file against it. If it compiles, the file uses only stable ABI. If not, the compiler errors *are* the gap analysis — precise, zero false positives.

**Regex-based verification** (no PyTorch source needed):

```bash
stable-abi-transform --mode=verify --verify-method=regex
```

## What the tool does automatically

| Transformation | Before | After |
|----------------|--------|-------|
| Macro rename | `TORCH_CHECK(x)` | `STD_TORCH_CHECK(x)` |
| Comparison macro | `TORCH_CHECK_EQ(a, b)` | `STD_TORCH_CHECK((a) == (b))` |
| Type rename | `at::Tensor` | `torch::stable::Tensor` |
| data_ptr (mutable) | `t.data_ptr<float>()` | `t.mutable_data_ptr<float>()` |
| data_ptr (const) | `ct.data_ptr<float>()` | `ct.const_data_ptr<float>()` |
| Method to function | `t.clone()` | `torch::stable::clone(t)` |
| Method rename | `t.dtype()` | `t.scalar_type()` |
| nbytes | `t.nbytes()` | `t.numel() * t.element_size()` |
| sizes indexing | `t.sizes()[0]` | `t.size(0)` |
| Free function | `at::empty({3}, at::kFloat)` | `torch::stable::empty({3}, ...)` |
| Scalar type | `at::kFloat` | `torch::headeronly::ScalarType::Float` |
| Include | `#include <torch/all.h>` | stable header set |
| CUDA guard | `at::cuda::OptionalCUDAGuard` | `torch::stable::accelerator::DeviceGuard` |
| CUDA stream | `at::cuda::getCurrentCUDAStream()` | `aoti_torch_get_current_cuda_stream()` |
| nullopt/optional | `c10::optional<T>` / `c10::nullopt` | `std::optional<T>` / `std::nullopt` |

For vllm: **2,554 automatic transformations** across 235 files.

## What requires manual work

### No stable equivalent (209 findings in vllm)

PyTorch internal APIs that have not been stabilized yet:

- `at::vec::Vectorized<T>` — SIMD vectorization (CPU kernels)
- `at::native::cpublas::brgemm` — CPU BLAS routines
- `at::empty_like`, `at::zeros`, `at::ones` — factory functions not yet in stable ops
- `at::stack`, `at::clamp`, `at::sigmoid`, `at::silu` — higher-level ops
- `c10::InferenceMode` — inference mode guard

These require waiting for PyTorch to stabilize the API, or restructuring code to use stable primitives.

### Inside macro bodies (184 findings in vllm)

The tool detects unstable API usage inside `#define` bodies but cannot auto-rewrite them:

```cpp
#define MY_CHECK(x) TORCH_CHECK(x, "failed")  // tool flags but can't rewrite
```

The tool reports exactly which macro and what the replacement should be. The edit is manual but straightforward.

### Type boundaries across files

`at::Tensor` and `torch::stable::Tensor` are distinct C++ types. Changing a function's signature in one file without updating all callers causes compilation errors. The plan mode identifies which files share headers and must migrate together, but verifying that function signatures are consistent across translation units requires human review.

Use the plan groups as PR boundaries. Within each group, ensure functions declared in shared headers have consistent signatures.

## Typical workflow

```bash
# 1. One-time setup
stable-abi-transform --init-config > .stable-abi.yaml
vim .stable-abi.yaml               # set pytorch_root, project_root, transform

# 2. Assess the scope
stable-abi-transform                # audit — see what needs to change
stable-abi-transform --mode=plan    # plan — see how to group the work

# 3. Review before touching code
stable-abi-transform --mode=rewrite --dry-run   # preview all changes

# 4. Apply and verify
stable-abi-transform --mode=rewrite              # transform
stable-abi-transform --mode=verify               # confirm stable-only

# 5. Build and test
make -j30 && python -m pytest tests/
```
