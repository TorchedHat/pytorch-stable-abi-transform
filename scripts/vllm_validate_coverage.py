#!/usr/bin/env python3
"""
Measure stable-abi-transform coverage against ground-truth PRs.

Runs the tool's own AST-based audit on both "before" and "after" versions
of files from merged stable ABI migration PRs. Compares at the finding
level — not line-level — to compute detection rate and auto-fix rate.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PRSpec:
    commit: str
    pr_number: int
    description: str


@dataclass
class FilePair:
    before_path: str
    after_path: str
    change_type: str  # "renamed" | "modified"


@dataclass
class Finding:
    kind: str
    line: int
    col: int
    old: str
    new: str
    flag: bool


@dataclass
class FileAnalysis:
    pair: FilePair
    before_findings: list[Finding] = field(default_factory=list)
    after_findings: list[Finding] = field(default_factory=list)
    before_parse_ok: bool = True
    after_parse_ok: bool = True
    error: str = ""


@dataclass
class PRAnalysis:
    spec: PRSpec
    files: list[FileAnalysis] = field(default_factory=list)
    skipped_structural: list[str] = field(default_factory=list)
    skipped_non_cpp: list[str] = field(default_factory=list)


GROUND_TRUTH_PRS = [
    PRSpec("8b10e4fb3", 31509, "permute_cols"),
    PRSpec("bf4cc9ed2", 36058, "per_token_group_quant"),
    PRSpec("ab1a6a43f", 37221, "cutlass/scaled_mm"),
    PRSpec("7c080dd3c", 37503, "FP4/W4A8"),
]

CPP_EXTENSIONS = {".cu", ".cpp", ".cuh", ".hpp", ".h"}

STRUCTURAL_BASENAMES = {"torch_bindings.cpp", "ops.h"}


def extract_file_pairs(repo: str, commit: str) -> tuple[list[FilePair], list[str], list[str]]:
    """Extract before/after file pairs from a git commit."""
    result = subprocess.run(
        ["git", "diff", "--name-status", "-M10", f"{commit}^..{commit}"],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    if result.returncode != 0:
        print(f"  git diff failed: {result.stderr.strip()}", file=sys.stderr)
        return [], [], []

    pairs: list[FilePair] = []
    added: dict[str, bool] = {}
    deleted: dict[str, bool] = {}
    skipped_structural: list[str] = []
    skipped_non_cpp: list[str] = []

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]

        if status.startswith("R"):
            old_path, new_path = parts[1], parts[2]
            ext = Path(old_path).suffix
            if ext not in CPP_EXTENSIONS:
                skipped_non_cpp.append(old_path)
                continue
            if Path(old_path).name in STRUCTURAL_BASENAMES:
                skipped_structural.append(old_path)
                continue
            pairs.append(FilePair(old_path, new_path, "renamed"))

        elif status == "M":
            path = parts[1]
            ext = Path(path).suffix
            if ext not in CPP_EXTENSIONS:
                skipped_non_cpp.append(path)
                continue
            if Path(path).name in STRUCTURAL_BASENAMES:
                skipped_structural.append(path)
                continue
            pairs.append(FilePair(path, path, "modified"))

        elif status == "A":
            added[parts[1]] = True

        elif status == "D":
            deleted[parts[1]] = True

    # Reconstruct undetected renames: deleted csrc/X + added csrc/libtorch_stable/X
    for del_path in list(deleted):
        basename = Path(del_path).name
        ext = Path(del_path).suffix
        if ext not in CPP_EXTENSIONS:
            continue

        candidates = [a for a in added if Path(a).name == basename and "libtorch_stable" in a]
        if candidates:
            add_path = candidates[0]
            if Path(del_path).name in STRUCTURAL_BASENAMES:
                skipped_structural.append(del_path)
            else:
                pairs.append(FilePair(del_path, add_path, "renamed"))
            del deleted[del_path]
            del added[add_path]

    return pairs, skipped_structural, skipped_non_cpp


def extract_file_content(repo: str, commit: str, path: str, which: str) -> str | None:
    """Extract file content at a specific commit version."""
    ref = f"{commit}^" if which == "before" else commit
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def run_audit(
    tool: str,
    filepath: str,
    resource_dir: str,
    pytorch_dir: str,
    extra_includes: list[str],
    timeout: int,
) -> tuple[list[Finding], bool, str]:
    """Run the tool in audit+json mode on a file. Returns (findings, success, error)."""
    cmd = [
        tool,
        "--mode=audit",
        "--format=json",
        filepath,
        "--",
        "-std=c++20",
        f"-resource-dir={resource_dir}",
        f"-I{pytorch_dir}/torch/csrc/api/include",
        f"-I{pytorch_dir}",
        f"-I{pytorch_dir}/torch/include",
        "-I/usr/local/cuda/include",
    ]
    for inc in extra_includes:
        cmd.append(f"-I{inc}")

    is_cuda = filepath.endswith(".cu") or filepath.endswith(".cuh")
    if is_cuda:
        cmd.append("--cuda-host-only")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [], False, "timeout"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if not stdout or not stdout.startswith("{"):
        # Tool produced no JSON — likely a parse failure
        error_lines = [line for line in stderr.split("\n") if "error:" in line]
        short_error = error_lines[0] if error_lines else stderr[:200]
        return [], False, short_error

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return [], False, f"invalid JSON: {stdout[:100]}"

    findings = []
    for f in data.get("findings", []):
        findings.append(
            Finding(
                kind=f["kind"].strip(),
                line=f["line"],
                col=f["col"],
                old=f["old"],
                new=f["new"],
                flag=f["flag"],
            )
        )

    return findings, True, ""


def analyze_file(
    pair: FilePair,
    repo: str,
    commit: str,
    tool: str,
    resource_dir: str,
    pytorch_dir: str,
    extra_includes: list[str],
    timeout: int,
) -> FileAnalysis:
    """Run audit on both before and after versions of a file."""
    analysis = FileAnalysis(pair=pair)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Before
        before_content = extract_file_content(repo, commit, pair.before_path, "before")
        if before_content is None:
            analysis.before_parse_ok = False
            analysis.error = f"could not extract {pair.before_path} from {commit}^"
            return analysis

        before_file = os.path.join(tmpdir, Path(pair.before_path).name)
        with open(before_file, "w") as f:
            f.write(before_content)

        analysis.before_findings, analysis.before_parse_ok, err = run_audit(
            tool,
            before_file,
            resource_dir,
            pytorch_dir,
            extra_includes,
            timeout,
        )
        if not analysis.before_parse_ok:
            analysis.error = err
            return analysis

        # After
        after_content = extract_file_content(repo, commit, pair.after_path, "after")
        if after_content is None:
            analysis.after_parse_ok = False
            analysis.error = f"could not extract {pair.after_path} from {commit}"
            return analysis

        after_file = os.path.join(tmpdir, Path(pair.after_path).name)
        with open(after_file, "w") as f:
            f.write(after_content)

        analysis.after_findings, analysis.after_parse_ok, err = run_audit(
            tool,
            after_file,
            resource_dir,
            pytorch_dir,
            extra_includes,
            timeout,
        )
        if not analysis.after_parse_ok:
            analysis.error = err

    return analysis


def classify_file(a: FileAnalysis) -> str:
    """Classify a file as migrated/untouched/partial/no-api/failed."""
    if not a.before_parse_ok:
        return "failed"
    nb = len(a.before_findings)
    na = len(a.after_findings)
    if nb == 0 and na == 0:
        return "no-api"
    if nb > 0 and na == 0:
        return "migrated"
    if nb > 0 and na == nb:
        return "untouched"
    if nb > 0 and 0 < na < nb:
        return "partial"
    return "other"


def compute_metrics(analyses: list[FileAnalysis]) -> dict:
    """Compute aggregate metrics across all analyzed files."""
    parse_failures = 0
    categories: dict[str, int] = Counter()

    # Separate metrics for migrated-only vs all files
    migrated_auto = 0
    migrated_flag = 0
    migrated_total = 0
    all_auto = 0
    all_flag = 0
    all_total = 0
    all_after = 0
    flag_counter = Counter()

    for a in analyses:
        cat = classify_file(a)
        categories[cat] += 1
        if cat == "failed":
            parse_failures += 1
            continue

        all_total += len(a.before_findings)
        all_after += len(a.after_findings)
        for f in a.before_findings:
            if f.flag:
                all_flag += 1
                flag_counter[(f.kind, f.old)] += 1
            else:
                all_auto += 1

        if cat == "migrated":
            for f in a.before_findings:
                migrated_total += 1
                if f.flag:
                    migrated_flag += 1
                else:
                    migrated_auto += 1

    migrated_rate = (migrated_auto / migrated_total * 100) if migrated_total > 0 else 0
    all_rate = (all_auto / all_total * 100) if all_total > 0 else 0

    return {
        "categories": dict(categories),
        "parse_failures": parse_failures,
        "migrated_total": migrated_total,
        "migrated_auto": migrated_auto,
        "migrated_flag": migrated_flag,
        "migrated_rate": migrated_rate,
        "all_total": all_total,
        "all_auto": all_auto,
        "all_flag": all_flag,
        "all_after": all_after,
        "all_rate": all_rate,
        "flag_counter": flag_counter,
    }


def print_file_analysis(a: FileAnalysis, verbose: bool) -> None:
    """Print per-file analysis."""
    label = a.pair.before_path
    if a.pair.change_type == "renamed":
        label = f"{a.pair.before_path} -> {a.pair.after_path}"

    cat = classify_file(a)

    if cat == "failed":
        print(f"    {label}  [PARSE FAILURE]")
        print(f"      {a.error}")
        return

    if cat == "no-api" and not verbose:
        return

    n_auto = sum(1 for f in a.before_findings if not f.flag)
    n_flag = sum(1 for f in a.before_findings if f.flag)
    n_after = len(a.after_findings)

    tag = {
        "migrated": "MIGRATED",
        "untouched": "NOT MIGRATED",
        "partial": "PARTIAL",
        "no-api": "no PyTorch API",
    }.get(cat, cat)

    print(f"    {label}  [{tag}]")
    if cat == "no-api":
        return
    print(f"      Before: {len(a.before_findings)} findings ({n_auto} auto, {n_flag} flag)")
    if cat != "untouched":
        print(f"      After:  {n_after} findings")

    if not verbose:
        return

    if n_auto > 0:
        print(f"      Auto-rewritable ({n_auto}):")
        grouped = Counter()
        for f in a.before_findings:
            if not f.flag:
                grouped[(f.kind, f.old, f.new)] += 1
        for (kind, old, new), count in grouped.most_common():
            print(f"        {count}x {kind:5s}  {old} -> {new}")

    if n_flag > 0:
        print(f"      Flags ({n_flag}):")
        grouped = Counter()
        for f in a.before_findings:
            if f.flag:
                grouped[(f.kind, f.old, f.new)] += 1
        for (kind, old, new), count in grouped.most_common():
            print(f"        {count}x {kind:5s}  {old} -> {new}")

    if n_after > 0:
        print(f"      After-migration residual ({n_after}):")
        for f in a.after_findings:
            print(f"        {f.kind:5s}  L{f.line}:{f.col}  {f.old}")


def print_report(pr_analyses: list[PRAnalysis], verbose: bool) -> None:
    """Print the full text report."""
    all_files: list[FileAnalysis] = []

    for pr in pr_analyses:
        print(f"\n=== PR #{pr.spec.pr_number} ({pr.spec.commit[:9]}): {pr.spec.description} ===")

        analyzed = [a for a in pr.files if a.before_parse_ok]
        failed = [a for a in pr.files if not a.before_parse_ok]

        print(
            f"  Files: {len(analyzed)} analyzed, {len(failed)} parse failures, "
            f"{len(pr.skipped_structural)} structural, "
            f"{len(pr.skipped_non_cpp)} non-C++"
        )

        if pr.skipped_structural:
            print(
                f"  Structural (skipped): {', '.join(Path(p).name for p in pr.skipped_structural)}"
            )

        for a in pr.files:
            print_file_analysis(a, verbose)
            all_files.append(a)

    # Overall summary
    metrics = compute_metrics(all_files)
    cats = metrics["categories"]
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print("  Files:")
    print(f"    Migrated (before→0):      {cats.get('migrated', 0)}")
    print(f"    Not migrated by PR:       {cats.get('untouched', 0)}")
    print(f"    Partial:                  {cats.get('partial', 0)}")
    print(f"    No PyTorch API (0/0):     {cats.get('no-api', 0)}")
    print(f"    Parse failures:           {cats.get('failed', 0)}")
    print()
    print(
        f"  Migrated files ({cats.get('migrated', 0)} files, {metrics['migrated_total']} findings):"
    )
    print(f"    Auto-rewritable: {metrics['migrated_auto']}")
    print(f"    Flag-only:       {metrics['migrated_flag']}")
    print(
        f"    Auto-fix rate:   {metrics['migrated_rate']:.1f}%"
        f" ({metrics['migrated_auto']}/{metrics['migrated_total']})"
    )
    print()
    print(f"  All files ({metrics['all_total']} findings across all categories):")
    print(f"    Auto-rewritable: {metrics['all_auto']}")
    print(f"    Flag-only:       {metrics['all_flag']}")
    print(f"    Auto-fix rate:   {metrics['all_rate']:.1f}%")

    if metrics["flag_counter"]:
        print("\n  TOP FLAG PATTERNS (opportunities for new auto-rewrite rules):")
        for (kind, old), count in metrics["flag_counter"].most_common(15):
            print(f"    {count:3d}x {kind:5s}  {old}")


def write_json_report(pr_analyses: list[PRAnalysis], path: str) -> None:
    """Write structured JSON report."""
    all_files = [a for pr in pr_analyses for a in pr.files]
    metrics = compute_metrics(all_files)

    report = {
        "summary": {
            "file_categories": metrics["categories"],
            "migrated_findings": metrics["migrated_total"],
            "migrated_auto": metrics["migrated_auto"],
            "migrated_flag": metrics["migrated_flag"],
            "migrated_auto_fix_rate_pct": round(metrics["migrated_rate"], 1),
            "all_findings": metrics["all_total"],
            "all_auto": metrics["all_auto"],
            "all_flag": metrics["all_flag"],
            "all_auto_fix_rate_pct": round(metrics["all_rate"], 1),
        },
        "prs": [],
    }

    for pr in pr_analyses:
        pr_data = {
            "pr_number": pr.spec.pr_number,
            "commit": pr.spec.commit,
            "description": pr.spec.description,
            "skipped_structural": pr.skipped_structural,
            "files": [],
        }
        for a in pr.files:
            file_data = {
                "before_path": a.pair.before_path,
                "after_path": a.pair.after_path,
                "change_type": a.pair.change_type,
                "before_parse_ok": a.before_parse_ok,
                "after_parse_ok": a.after_parse_ok,
                "error": a.error,
                "before_findings": [
                    {
                        "kind": f.kind,
                        "line": f.line,
                        "col": f.col,
                        "old": f.old,
                        "new": f.new,
                        "flag": f.flag,
                    }
                    for f in a.before_findings
                ],
                "after_findings": [
                    {
                        "kind": f.kind,
                        "line": f.line,
                        "col": f.col,
                        "old": f.old,
                        "new": f.new,
                        "flag": f.flag,
                    }
                    for f in a.after_findings
                ],
            }
            pr_data["files"].append(file_data)
        report["prs"].append(pr_data)

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report written to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure stable-abi-transform coverage against ground-truth PRs",
    )
    parser.add_argument("--vllm-repo", required=True, help="Path to vLLM source tree")
    parser.add_argument("--tool", default=None)
    parser.add_argument("--pytorch-dir", required=True, help="Path to PyTorch source root")
    parser.add_argument("--resource-dir", default="/usr/lib/clang/19")
    parser.add_argument("--json", default=None, metavar="PATH")
    parser.add_argument("--pr", action="append", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    tool = args.tool or str(project_dir / "build" / "stable-abi-transform")

    if not os.path.isfile(tool):
        print(f"error: tool not found at {tool}", file=sys.stderr)
        return 1

    prs = GROUND_TRUTH_PRS
    if args.pr:
        prs = [p for p in prs if p.pr_number in args.pr]
        if not prs:
            print(f"error: no matching PRs for {args.pr}", file=sys.stderr)
            return 1

    extra_includes = [
        os.path.join(args.vllm_repo, "csrc"),
        os.path.join(args.vllm_repo, "csrc", "libtorch_stable"),
    ]

    pr_analyses: list[PRAnalysis] = []
    for pr in prs:
        print(f"Processing PR #{pr.pr_number} ({pr.description})...", file=sys.stderr)

        pairs, structural, non_cpp = extract_file_pairs(args.vllm_repo, pr.commit)
        analysis = PRAnalysis(
            spec=pr,
            skipped_structural=structural,
            skipped_non_cpp=non_cpp,
        )

        for pair in pairs:
            print(f"  {Path(pair.before_path).name}...", file=sys.stderr, end="", flush=True)
            fa = analyze_file(
                pair,
                args.vllm_repo,
                pr.commit,
                tool,
                args.resource_dir,
                args.pytorch_dir,
                extra_includes,
                args.timeout,
            )
            status = "ok" if fa.before_parse_ok else "FAIL"
            print(f" {status}", file=sys.stderr)
            analysis.files.append(fa)

        pr_analyses.append(analysis)

    print_report(pr_analyses, args.verbose)

    if args.json:
        write_json_report(pr_analyses, args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
