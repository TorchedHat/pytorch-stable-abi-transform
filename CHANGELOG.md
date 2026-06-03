# Changelog

## v0.2.0

### New features

- **Plan mode** (`--mode=plan`) — analyzes include dependencies and partitions files into independent migration groups for PR-sized work
- **Parallel execution** (`--jobs`) — audit and plan modes process translation units in parallel (~18x speedup on large projects)
- **YAML config** (`--init-config`) — generate `.stable-abi.yaml` to replace repeated CLI flags; `pytorch_root: auto` detects pip-installed torch
- **Incremental migration** (`transform` config field) — target specific files/directories while keeping full project scope for include analysis
- **Out-of-place rewrite** (`--output-dir`) — write transformed files to a separate directory, preserving originals
- **Dry-run preview** (`--dry-run`) — unified diff output without modifying files
- **.cuh file support** — CUDA headers discovered and analyzed alongside .cpp and .cu
- **Migration progress script** (`scripts/migration_progress.py`) — measures stable ABI adoption from audit/plan JSON

### New rewrite rules

- Comparison macros: `TORCH_CHECK_EQ(a, b)` -> `STD_TORCH_CHECK((a) == (b))` (and NE/LT/GT/GE/LE)
- `c10::optional<T>` / `c10::nullopt` -> `std::optional<T>` / `std::nullopt`
- `c10::ArrayRef<T>` -> `torch::headeronly::HeaderOnlyArrayRef<T>`
- Dispatch macros: `AT_DISPATCH_FLOATING_TYPES` -> `THO_DISPATCH_V2` + type list
- DeviceGuard `.device()` -> `.current_device()`
- Expanded type, macro, and free function coverage from PyTorch 2.11 stable headers

### Bug fixes

- `jsonEscape`: handle control characters below 0x20 with `\u00xx` encoding
- Output-dir: fix path escape crash (SmallVector self-reference) and reject files outside project root
- Reporter: deterministic output ordering with `old_text` tiebreaker

### Infrastructure

- Pip-based CI pipeline (auto-detects PyTorch from `pip install torch`)
- Per-file parse error reporting in JSON output
- Macro body detection (flags unstable API inside `#define` bodies)
- Catch-all flagger for any unrecognized `at::`, `c10::`, or unstable `torch::` usage

## v0.1.0

Initial release. Audit and rewrite modes, compile-based and regex-based verification, 22 stable ops coverage.
