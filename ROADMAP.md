# Roadmap

## v0.2.0 (current)

- Parallel audit/plan via `--jobs` (~18x speedup on large projects)
- `.cuh` file support (parsed as C++ for host-side analysis)
- Per-file parse error reporting in JSON output
- Migration progress tracker (`scripts/migration_progress.py`)

## v0.3.0

- Parallel rewrite (shared header deduplication via include graph pre-scan)
- Per-TU parse error attribution
- `--mode=progress` integrated into the C++ tool
- CUDA-aware `.cuh` parsing (stub definitions for `__device__`/`__global__`)
- GitHub Actions CI template for stable ABI adoption

## v1.0.0

- Stable CLI and JSON schema (semver-guaranteed interface)
- Compile-based verification as default
- Full PyTorch stable API coverage
- Validated on 3+ major consumers (vLLM, SGLang, KVCached)
