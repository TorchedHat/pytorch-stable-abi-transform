#!/usr/bin/env python3
"""Stable ABI migration progress tracker.

Measures how close a PyTorch C++ extension is to building against the stable
ABI — the point where you ship one binary for all PyTorch versions and drop
per-version builds from your CI matrix.

Exit codes (for CI gating):
    0  Fully migrated or no work needed
    1  Migration in progress (manual work remains)
    2  Error (audit failed, no source files found)

Usage:
    # Audit a project and show progress
    migration_progress.py /path/to/project --source-dir csrc

    # From pre-computed audit JSON
    migration_progress.py --audit-json audit.json --source-dir /path/to/csrc

    # Machine-readable for CI/dashboards
    migration_progress.py --audit-json audit.json --source-dir csrc --format json

    # Markdown for issue tracker / wiki
    migration_progress.py /path/to/project --source-dir csrc --format markdown
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

TOOL = str(Path(__file__).resolve().parent.parent / "build" / "stable-abi-transform")

SOURCE_EXTENSIONS = {".cpp", ".cu", ".cuh", ".h", ".hpp"}

# Map the C++ tool's FindingKind (JSON "kind" field) to blocker categories.
# The kind field is authoritative — no string-matching on old_text.
KIND_TO_BLOCKER = {
    "FLAG": "missing_api",
    "INCL": "unstable_include",
    "MACRO": "macro_in_body",
    "STYPE": "scalar_shorthand",
    "STRM": "cuda_stream",
    "GUARD": "device_guard",
    "TYPE": "type_decompose",
    "FUNC": "func_signature",
}

BLOCKER_LABELS = {
    "missing_api": "No stable equivalent (CPU vec/BLAS, Vectorized, etc.)",
    "unstable_include": "Unstable #include",
    "macro_in_body": "Macro inside macro body (needs extraction)",
    "scalar_shorthand": "Scalar type shorthand in macro body",
    "cuda_stream": "CUDA stream (manual device_index wiring)",
    "device_guard": "DeviceGuard (manual constructor rewrite)",
    "type_decompose": "Type shorthand needs decomposition",
    "func_signature": "Function signature change",
}

HARD_BLOCKERS = {"missing_api"}
MODERATE_BLOCKERS = {"macro_in_body", "scalar_shorthand"}

TIER_ORDER = ["green", "yellow", "orange", "red", "clean"]
TIER_LABELS = {
    "green": "Auto-rewritable",
    "yellow": "Auto-rewrite + small manual fixes",
    "orange": "Macro restructuring required",
    "red": "Blocked on PyTorch stable API expansion",
    "clean": "No findings",
}


@dataclass
class FileStatus:
    rewrites: int = 0
    flags: int = 0
    blockers: set = field(default_factory=set)

    @property
    def tier(self) -> str:
        if not self.blockers:
            return "green" if self.rewrites > 0 else "clean"
        if self.blockers & HARD_BLOCKERS:
            return "red"
        if self.blockers & MODERATE_BLOCKERS:
            return "orange"
        return "yellow"


def run_audit(project_root: str, source_dir: str, jobs: int = 0) -> dict:
    cmd = [TOOL, "--mode=audit", "--format=json", f"--project-root={source_dir}"]
    if jobs > 0:
        cmd.append(f"--jobs={jobs}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, timeout=7200)
    if result.returncode != 0:
        print(f"audit failed (exit {result.returncode}):", file=sys.stderr)
        if result.stderr:
            print(result.stderr[:2000], file=sys.stderr)
        sys.exit(2)
    return json.loads(result.stdout)


def discover_sources(root: str) -> set[str]:
    return {
        os.path.relpath(os.path.join(d, f), root)
        for d, _, files in os.walk(root)
        for f in files
        if Path(f).suffix in SOURCE_EXTENSIONS
    }


def analyze(audit: dict, source_root: str, all_files: set[str]) -> dict[str, FileStatus]:
    status: dict[str, FileStatus] = {}

    for finding in audit["findings"]:
        rel = os.path.normpath(os.path.relpath(finding["file"], source_root))
        if rel.startswith(".."):
            continue
        fs = status.setdefault(rel, FileStatus())
        if finding["flag"]:
            fs.flags += 1
            fs.blockers.add(KIND_TO_BLOCKER.get(finding["kind"], "missing_api"))
        else:
            fs.rewrites += 1

    for f in all_files:
        status.setdefault(f, FileStatus())

    return status


def load_plan_groups(plan_json_path: str, source_root: str) -> dict[str, list[str]] | None:
    """Load dependency-aware PR groups from --mode=plan JSON output."""
    with open(plan_json_path) as f:
        plan = json.load(f)
    groups: dict[str, list[str]] = {}
    for partition in plan.get("partitions", []):
        label = partition.get("label", "unnamed")
        files = []
        for fp in partition.get("sources", []) + partition.get("headers", []):
            rel = os.path.normpath(os.path.relpath(fp, source_root))
            if not rel.startswith(".."):
                files.append(rel)
        if files:
            groups[label] = sorted(files)
    return groups if groups else None


def build_report(
    status: dict[str, FileStatus],
    audit: dict,
    source_label: str,
    plan_groups: dict[str, list[str]] | None = None,
) -> dict:
    tiers = Counter(fs.tier for fs in status.values())
    blockers = Counter()
    for fs in status.values():
        for b in fs.blockers:
            blockers[b] += 1

    total = len(status)
    needs_work = sum(1 for fs in status.values() if fs.rewrites > 0 or fs.flags > 0)
    auto_ready = tiers.get("green", 0)

    if plan_groups is not None:
        green_dirs = {
            k: v
            for k, v in plan_groups.items()
            if any(status.get(f, FileStatus()).tier == "green" for f in v)
        }
        grouping_method = "plan"
    else:
        green_dirs: dict[str, list[str]] = {}
        for f, fs in sorted(status.items()):
            if fs.tier == "green":
                d = os.path.dirname(f) or "(root)"
                green_dirs.setdefault(d, []).append(f)
        grouping_method = "directory"

    yellow_files = sorted(f for f, fs in status.items() if fs.tier == "yellow")

    return {
        "source_dir": source_label,
        "total_files": total,
        "needs_work": needs_work,
        "auto_ready": auto_ready,
        "tiers": {t: tiers.get(t, 0) for t in TIER_ORDER},
        "blockers": dict(blockers.most_common()),
        "total_rewrites": audit["rewrites"],
        "total_flags": audit["flags"],
        "parse_errors": audit["parse_errors"],
        "green_dirs": green_dirs,
        "yellow_files": yellow_files,
        "file_status": status,
        "grouping_method": grouping_method,
    }


def format_score(r: dict) -> str:
    needs = r["needs_work"]
    if needs == 0:
        return "No migration needed"
    ready = r["auto_ready"]
    pct = round(ready * 100 / needs, 1)
    return f"{ready}/{needs} files auto-rewritable ({pct}%)"


def print_text(r: dict):
    total = r["total_files"]
    label = r["source_dir"]
    status = r["file_status"]

    print("=" * 70)
    print(f"  STABLE ABI MIGRATION — {label}/")
    print("=" * 70)
    print()
    print(f"  Progress:           {format_score(r)}")
    print(f"  Total files:        {total}")
    print(f"  Need migration:     {r['needs_work']}")
    print(
        f"  Auto-rewritable:    {r['total_rewrites']} findings across {r['tiers']['green']} files"
    )
    print(
        f"  Manual review:      {r['total_flags']} findings across "
        f"{r['tiers']['yellow'] + r['tiers']['orange'] + r['tiers']['red']}"
        f" files"
    )
    if r["parse_errors"]:
        print(
            f"  Parse errors:       {r['parse_errors']} "
            "(completeness not guaranteed — run --mode=verify after rewriting)"
        )
    print()

    if r["needs_work"] == 0:
        print("  No files need migration.")
        print()
        return

    print("  TIER BREAKDOWN (of files needing migration)")
    print("  " + "-" * 50)
    for tier in TIER_ORDER:
        c = r["tiers"][tier]
        if tier == "clean":
            continue
        if c == 0:
            continue
        pct = c * 100 // r["needs_work"] if r["needs_work"] else 0
        bar = "#" * (c * 40 // r["needs_work"]) if r["needs_work"] else ""
        print(f"  {tier:8s}  {c:3d} ({pct:2d}%)  {bar}")
    print(f"  {'clean':8s}  {r['tiers']['clean']:3d}       (no migration needed)")
    print()

    if r["blockers"]:
        print("  BLOCKERS")
        print("  " + "-" * 50)
        for cat, count in r["blockers"].items():
            print(f"  {count:3d} files  {BLOCKER_LABELS.get(cat, cat)}")
        print()

    pr_num = 0
    if r["green_dirs"]:
        grouping = r.get("grouping_method", "directory")
        heading = "GREEN TIER — AUTO-REWRITABLE"
        if grouping == "directory":
            heading += " (approximate — use --plan-json for dependency-aware groups)"
        print(f"  {heading}")
        print("  " + "-" * 50)
        for d, files in sorted(r["green_dirs"].items()):
            pr_num += 1
            n_rewrites = sum(status[f].rewrites for f in files if f in status)
            is_header = all(Path(f).suffix in {".h", ".hpp", ".cuh"} for f in files)
            ftype = "headers" if is_header else "files"
            print(f"  PR {pr_num:2d}: {label}/{d}/ — {len(files)} {ftype}, {n_rewrites} rewrites")

    if r["yellow_files"]:
        print()
        print("  YELLOW TIER — AUTO-REWRITE + SMALL MANUAL FIXES")
        print("  " + "-" * 50)
        for f in r["yellow_files"]:
            fs = status[f]
            pr_num += 1
            print(f"  PR {pr_num:2d}: {label}/{f} — {fs.rewrites} auto + {fs.flags} manual")

    print()
    print(f"  Actionable PRs: {pr_num}")
    red = r["tiers"]["red"]
    if red:
        print(f"  Red tier: {red} files blocked on PyTorch stable API expansion")
    print()
    print("  After rewriting, run: stable-abi-transform --mode=verify")
    print("  For PR sequencing:    stable-abi-transform --mode=plan")
    print()


def print_markdown(r: dict):
    total = r["total_files"]
    label = r["source_dir"]
    status = r["file_status"]

    print(f"# Stable ABI Migration — {label}/")
    print()
    print(f"> **{format_score(r)}**")
    print()
    print("| Metric | Count |")
    print("|--------|------:|")
    print(f"| Total files | {total} |")
    print(f"| Need migration | {r['needs_work']} |")
    print(f"| Auto-rewritable findings | {r['total_rewrites']} |")
    print(f"| Manual-review findings | {r['total_flags']} |")
    if r["parse_errors"]:
        print(f"| Parse errors | {r['parse_errors']} |")
    print()

    if r["parse_errors"]:
        print(
            "> **Note:** Parse errors mean some findings may be missing. "
            "Run `--mode=verify` after rewriting to confirm completeness.\n"
        )

    print("## Tier Breakdown\n")
    print("| Tier | Files | Description |")
    print("|------|------:|-------------|")
    for tier in TIER_ORDER:
        c = r["tiers"][tier]
        print(f"| {tier} | {c} | {TIER_LABELS[tier]} |")
    print()

    if r["blockers"]:
        print("## Blockers\n")
        print("| Category | Files | Description |")
        print("|----------|------:|-------------|")
        for cat, count in r["blockers"].items():
            print(f"| `{cat}` | {count} | {BLOCKER_LABELS.get(cat, cat)} |")
        print()

    pr_num = 0
    if r["green_dirs"]:
        grouping = r.get("grouping_method", "directory")
        print("## Green Tier PRs\n")
        if grouping == "directory":
            print(
                "*Grouped by directory (approximate). Use `--plan-json` "
                "for dependency-aware grouping.*\n"
            )
        print(
            "Auto-rewritable. Run `stable-abi-transform --mode=rewrite`, "
            "then `--mode=verify` to confirm.\n"
        )
        for d, files in sorted(r["green_dirs"].items()):
            pr_num += 1
            n_rewrites = sum(status[f].rewrites for f in files if f in status)
            print(f"### PR {pr_num}: `{label}/{d}/`")
            print(f"{len(files)} files, {n_rewrites} auto-rewrites\n")
            for f in sorted(files):
                rewrites = status[f].rewrites if f in status else 0
                print(f"- [ ] `{f}` ({rewrites} findings)")
            print()

    if r["yellow_files"]:
        print("## Yellow Tier PRs\n")
        print("Auto-rewrite covers most changes; small manual fixes per file.\n")
        for f in r["yellow_files"]:
            fs = status[f]
            pr_num += 1
            cats = ", ".join(sorted(fs.blockers))
            print(f"### PR {pr_num}: `{label}/{f}`")
            print(f"{fs.rewrites} auto + {fs.flags} manual (`{cats}`)\n")

    print(f"**Actionable PRs: {pr_num}** (green + yellow)")
    red = r["tiers"]["red"]
    if red:
        print(f"\nRed tier ({red} files) blocked on PyTorch stable API expansion — track upstream.")
    print(
        "\nAfter rewriting, verify with: `stable-abi-transform --mode=verify --pytorch-root=<path>`"
    )
    print("\nFor dependency-aware PR sequencing: `stable-abi-transform --mode=plan`")


def print_json(r: dict):
    needs = r["needs_work"]
    ready = r["auto_ready"]
    out = {
        "source_dir": r["source_dir"],
        "total_files": r["total_files"],
        "needs_migration": needs,
        "auto_ready": ready,
        "auto_ready_pct": round(ready * 100 / needs, 1) if needs else 100.0,
        "tiers": r["tiers"],
        "blockers": r["blockers"],
        "total_rewrites": r["total_rewrites"],
        "total_flags": r["total_flags"],
        "parse_errors": r["parse_errors"],
        "grouping_method": r["grouping_method"],
        "actionable_prs": len(r["green_dirs"]) + len(r["yellow_files"]),
        "files": {
            f: {
                "tier": fs.tier,
                "rewrites": fs.rewrites,
                "flags": fs.flags,
                "blockers": sorted(fs.blockers),
            }
            for f, fs in sorted(r["file_status"].items())
        },
    }
    json.dump(out, sys.stdout, indent=2)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Measure stable ABI migration progress for a PyTorch C++ extension project"
    )
    parser.add_argument("project_root", nargs="?", help="Path to the project root")
    parser.add_argument(
        "--source-dir", help="Source subdirectory (relative to project root, or absolute path)"
    )
    parser.add_argument("--audit-json", help="Pre-computed audit JSON (skip running the tool)")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel threads for audit (0=auto)")
    parser.add_argument(
        "--plan-json",
        help="Pre-computed plan JSON (--mode=plan --format=json) for dependency-aware PR grouping",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args()

    if not args.audit_json and not args.project_root:
        parser.error("project_root is required unless --audit-json is given")

    if args.source_dir:
        if os.path.isabs(args.source_dir):
            source_root = args.source_dir
        else:
            if not args.project_root:
                parser.error("project_root required with relative --source-dir")
            source_root = os.path.join(args.project_root, args.source_dir)
    elif args.project_root:
        source_root = args.project_root
    else:
        parser.error("--source-dir required with --audit-json and no project_root")

    source_root = os.path.normpath(source_root)

    if args.audit_json:
        with open(args.audit_json) as f:
            audit = json.load(f)
    else:
        print("Running audit...", file=sys.stderr)
        audit = run_audit(args.project_root, source_root, args.jobs)

    all_files = discover_sources(source_root)
    if not all_files:
        print(f"No source files in {source_root}", file=sys.stderr)
        sys.exit(2)

    plan_groups = None
    if args.plan_json:
        plan_groups = load_plan_groups(args.plan_json, source_root)
        if plan_groups is None:
            print(
                "warning: --plan-json contained no partitions, falling back to directory grouping",
                file=sys.stderr,
            )

    file_status = analyze(audit, source_root, all_files)
    report = build_report(
        file_status, audit, os.path.basename(source_root) or "project", plan_groups
    )

    {"text": print_text, "markdown": print_markdown, "json": print_json}[args.format](report)

    sys.exit(1 if report["total_flags"] > 0 else 0)


if __name__ == "__main__":
    main()
