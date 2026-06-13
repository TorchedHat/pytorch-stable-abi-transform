#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TOOL="$PROJECT_DIR/build/stable-abi-transform"
INPUTS="$SCRIPT_DIR/inputs"
EXPECTED="$SCRIPT_DIR/expected"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

if [ -z "${PYTORCH_DIR:-}" ]; then
    PYTORCH_DIR=$(python3 -c "import torch, os; print(os.path.join(torch.__path__[0], 'include'))" 2>/dev/null) \
        || { echo "error: Set PYTORCH_DIR or pip install torch"; exit 1; }
    echo "note: auto-detected PYTORCH_DIR=$PYTORCH_DIR"
fi
RESOURCE_DIR="${RESOURCE_DIR:-/usr/lib/clang/19}"

CUDA_INCLUDE="${CUDA_INCLUDE:-/usr/local/cuda/include}"

PYTORCH_ARGS=(-I"$PYTORCH_DIR" -I"$PYTORCH_DIR/torch/csrc/api/include")
[ -d "$PYTORCH_DIR/torch/include" ] && PYTORCH_ARGS+=(-I"$PYTORCH_DIR/torch/include")

COMMON_ARGS=(
    -- -std=c++20
    -resource-dir "$RESOURCE_DIR"
    -DC10_USING_CUSTOM_GENERATED_MACROS
    "${PYTORCH_ARGS[@]}"
    -I"$CUDA_INCLUDE"
)

echo "--- Input integrity check ---"
for input_file in "$INPUTS"/*.cpp "$INPUTS"/*.cu; do
    [ -f "$input_file" ] || continue
    basename="$(basename "$input_file")"
    expected_file="$EXPECTED/$basename"
    [ -f "$expected_file" ] || continue
    if diff -q "$input_file" "$expected_file" > /dev/null 2>&1; then
        echo "CORRUPT  $basename: input is identical to expected output"
        echo "         Run: git checkout HEAD -- test/inputs/$basename"
        exit 1
    fi
done
echo "All inputs differ from expected outputs."
echo ""

passed=0
failed=0
skipped=0

for input_file in "$INPUTS"/*.cpp "$INPUTS"/*.cu; do
    [ -f "$input_file" ] || continue

    basename="$(basename "$input_file")"
    expected_file="$EXPECTED/$basename"

    if [ ! -f "$expected_file" ]; then
        echo "SKIP  $basename (no expected file)"
        skipped=$((skipped + 1))
        continue
    fi

    # Copy input to temp dir
    cp "$input_file" "$WORK_DIR/$basename"

    # Run the tool in rewrite mode
    # Exit 0 = clean rewrite, exit 1 = flags remain for manual review, exit >1 = crash
    set +e
    "$TOOL" --mode=rewrite "$WORK_DIR/$basename" "${COMMON_ARGS[@]}" > "$WORK_DIR/$basename.stdout" 2>"$WORK_DIR/$basename.stderr"
    rc=$?
    set -e
    if [ $rc -gt 1 ]; then
        echo "FAIL  $basename (tool crashed with exit code $rc)"
        cat "$WORK_DIR/$basename.stderr"
        failed=$((failed + 1))
        continue
    fi

    # Compare output against expected
    if diff -u "$expected_file" "$WORK_DIR/$basename" > "$WORK_DIR/$basename.diff" 2>&1; then
        echo "PASS  $basename"
        passed=$((passed + 1))
    else
        echo "FAIL  $basename"
        cat "$WORK_DIR/$basename.diff"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Results: $passed passed, $failed failed, $skipped skipped"

if [ "$failed" -gt 0 ]; then
    exit 1
fi

# Verification tests: run verify mode on expected outputs (should all PASS)
echo ""
echo "--- Verification tests ---"
verify_passed=0
verify_failed=0

for expected_file in "$EXPECTED"/*.cpp "$EXPECTED"/*.cu; do
    [ -f "$expected_file" ] || continue
    basename="$(basename "$expected_file")"

    output=$("$TOOL" --mode=verify --verify-method=regex "$expected_file" "${COMMON_ARGS[@]}" 2>&1)
    if echo "$output" | grep -q "PASS"; then
        echo "VERIFY PASS  $basename (regex)"
        verify_passed=$((verify_passed + 1))
    else
        echo "VERIFY FAIL  $basename (regex)"
        echo "$output"
        verify_failed=$((verify_failed + 1))
    fi
done

echo ""
echo "Verification (regex): $verify_passed passed, $verify_failed failed"
[ "$verify_failed" -gt 0 ] && exit 1

# Exit code tests
echo ""
echo "--- Exit code tests ---"
exitcode_passed=0
exitcode_failed=0

# Audit mode should exit 1 on file with unstable API
if "$TOOL" --mode=audit "$INPUTS/smoke.cpp" "${COMMON_ARGS[@]}" > /dev/null 2>&1; then
    echo "FAIL  audit-exit-code: expected exit 1 on unstable input, got 0"
    exitcode_failed=$((exitcode_failed + 1))
else
    echo "PASS  audit-exit-code: exits 1 on unstable input"
    exitcode_passed=$((exitcode_passed + 1))
fi

# Audit mode should exit 0 on already-stable file
if "$TOOL" --mode=audit "$EXPECTED/smoke.cpp" "${COMMON_ARGS[@]}" > /dev/null 2>&1; then
    echo "PASS  audit-clean-exit-code: exits 0 on stable file"
    exitcode_passed=$((exitcode_passed + 1))
else
    echo "FAIL  audit-clean-exit-code: expected exit 0 on stable file, got non-zero"
    exitcode_failed=$((exitcode_failed + 1))
fi

# Dry-run mode should exit 1 when findings exist
if "$TOOL" --mode=rewrite --dry-run "$INPUTS/smoke.cpp" "${COMMON_ARGS[@]}" > /dev/null 2>&1; then
    echo "FAIL  dry-run-exit-code: expected exit 1 on unstable input, got 0"
    exitcode_failed=$((exitcode_failed + 1))
else
    echo "PASS  dry-run-exit-code: exits 1 on unstable input"
    exitcode_passed=$((exitcode_passed + 1))
fi

# Missing source file should exit 1 with clear error
if "$TOOL" --mode=audit /nonexistent/file.cpp "${COMMON_ARGS[@]}" 2>"$WORK_DIR/missing.stderr" > /dev/null; then
    echo "FAIL  missing-file: expected exit 1, got 0"
    exitcode_failed=$((exitcode_failed + 1))
elif grep -q "source file not found" "$WORK_DIR/missing.stderr"; then
    echo "PASS  missing-file: exits 1 with clear error"
    exitcode_passed=$((exitcode_passed + 1))
else
    echo "FAIL  missing-file: exits non-zero but error message unclear"
    cat "$WORK_DIR/missing.stderr"
    exitcode_failed=$((exitcode_failed + 1))
fi

echo ""
echo "Exit code tests: $exitcode_passed passed, $exitcode_failed failed"
[ "$exitcode_failed" -gt 0 ] && exit 1

# Feature tests: directory sources, output-dir, plan mode
echo ""
echo "--- Feature tests ---"
feat_passed=0
feat_failed=0

# 1. Directory source expansion: --project-root auto-discovers files
if "$TOOL" --mode=audit --project-root="$INPUTS" "${COMMON_ARGS[@]}" > /dev/null 2>&1; then
    echo "FAIL  dir-discovery: expected exit 1 (unstable inputs), got 0"
    feat_failed=$((feat_failed + 1))
else
    echo "PASS  dir-discovery: auto-discovers sources under --project-root"
    feat_passed=$((feat_passed + 1))
fi

# 2. Output-dir: out-of-place rewrite preserves originals
OUTDIR_WORK="$(mktemp -d)"
cp "$INPUTS/smoke.cpp" "$OUTDIR_WORK/smoke.cpp"
cp_md5=$(md5sum "$OUTDIR_WORK/smoke.cpp" | cut -d' ' -f1)
if "$TOOL" --mode=rewrite --project-root="$OUTDIR_WORK" \
    --output-dir="$OUTDIR_WORK/out" "$OUTDIR_WORK/smoke.cpp" \
    "${COMMON_ARGS[@]}" > /dev/null 2>"$OUTDIR_WORK/stderr"; then
    after_md5=$(md5sum "$OUTDIR_WORK/smoke.cpp" | cut -d' ' -f1)
    if [ "$cp_md5" != "$after_md5" ]; then
        echo "FAIL  output-dir-rewrite: original file was modified"
        feat_failed=$((feat_failed + 1))
    elif [ ! -f "$OUTDIR_WORK/out/smoke.cpp" ]; then
        echo "FAIL  output-dir-rewrite: output file not created"
        feat_failed=$((feat_failed + 1))
    elif diff -q "$EXPECTED/smoke.cpp" "$OUTDIR_WORK/out/smoke.cpp" > /dev/null 2>&1; then
        echo "PASS  output-dir-rewrite: original preserved, output matches expected"
        feat_passed=$((feat_passed + 1))
    else
        echo "FAIL  output-dir-rewrite: output differs from expected"
        diff -u "$EXPECTED/smoke.cpp" "$OUTDIR_WORK/out/smoke.cpp" | head -20
        feat_failed=$((feat_failed + 1))
    fi
else
    echo "FAIL  output-dir-rewrite: tool returned non-zero"
    cat "$OUTDIR_WORK/stderr"
    feat_failed=$((feat_failed + 1))
fi
rm -rf "$OUTDIR_WORK"

# 3. Output-dir without --project-root should error
if "$TOOL" --mode=rewrite --output-dir=/tmp/out "$INPUTS/smoke.cpp" \
    "${COMMON_ARGS[@]}" > /dev/null 2>"$WORK_DIR/outdir_err.stderr"; then
    echo "FAIL  output-dir-no-root: expected exit 1, got 0"
    feat_failed=$((feat_failed + 1))
elif grep -q "requires --project-root" "$WORK_DIR/outdir_err.stderr"; then
    echo "PASS  output-dir-no-root: requires --project-root"
    feat_passed=$((feat_passed + 1))
else
    echo "FAIL  output-dir-no-root: error message unclear"
    cat "$WORK_DIR/outdir_err.stderr"
    feat_failed=$((feat_failed + 1))
fi

# 4. Plan mode: single file, valid JSON with expected structure
plan_out="$WORK_DIR/plan.json"
if "$TOOL" --mode=plan --format=json --project-root="$INPUTS" \
    "$INPUTS/smoke.cpp" "${COMMON_ARGS[@]}" > "$plan_out" 2>/dev/null; then
    if python3 -m json.tool "$plan_out" > /dev/null 2>&1 &&
       python3 -c "
import json, sys
d = json.load(open('$plan_out'))
assert d['fully_parallel'] == True
assert len(d['groups']) == 1
assert d['groups'][0]['findings'] > 0
assert 'sources' in d['groups'][0]
assert 'headers' in d['groups'][0]
" 2>/dev/null; then
        echo "PASS  plan-json: valid JSON, 1 group, fully_parallel"
        feat_passed=$((feat_passed + 1))
    else
        echo "FAIL  plan-json: JSON structure invalid"
        cat "$plan_out"
        feat_failed=$((feat_failed + 1))
    fi
else
    echo "FAIL  plan-json: tool returned non-zero"
    feat_failed=$((feat_failed + 1))
fi

# 5. Plan mode: two unrelated files produce two independent groups
plan_multi="$WORK_DIR/plan_multi.json"
if "$TOOL" --mode=plan --format=json --project-root="$INPUTS" \
    "$INPUTS/smoke.cpp" "$INPUTS/types.cpp" "${COMMON_ARGS[@]}" \
    > "$plan_multi" 2>/dev/null; then
    if python3 -c "
import json, sys
d = json.load(open('$plan_multi'))
assert d['fully_parallel'] == True
assert len(d['groups']) == 2
" 2>/dev/null; then
        echo "PASS  plan-multi-group: 2 independent groups"
        feat_passed=$((feat_passed + 1))
    else
        echo "FAIL  plan-multi-group: expected 2 groups with fully_parallel"
        cat "$plan_multi"
        feat_failed=$((feat_failed + 1))
    fi
else
    echo "FAIL  plan-multi-group: tool returned non-zero"
    feat_failed=$((feat_failed + 1))
fi

# 6. Plan mode without --project-root should error
if "$TOOL" --mode=plan "$INPUTS/smoke.cpp" "${COMMON_ARGS[@]}" \
    > /dev/null 2>"$WORK_DIR/plan_err.stderr"; then
    echo "FAIL  plan-no-root: expected exit 1, got 0"
    feat_failed=$((feat_failed + 1))
elif grep -q "project-root required" "$WORK_DIR/plan_err.stderr"; then
    echo "PASS  plan-no-root: requires --project-root"
    feat_passed=$((feat_passed + 1))
else
    echo "FAIL  plan-no-root: error message unclear"
    cat "$WORK_DIR/plan_err.stderr"
    feat_failed=$((feat_failed + 1))
fi

# 7. Plan mode: text output format
plan_text="$WORK_DIR/plan_text.out"
if "$TOOL" --mode=plan --project-root="$INPUTS" "$INPUTS/smoke.cpp" \
    "${COMMON_ARGS[@]}" > "$plan_text" 2>/dev/null; then
    if grep -q "Migration plan:" "$plan_text" && grep -q "Group A" "$plan_text"; then
        echo "PASS  plan-text: text output has expected structure"
        feat_passed=$((feat_passed + 1))
    else
        echo "FAIL  plan-text: missing expected text"
        cat "$plan_text"
        feat_failed=$((feat_failed + 1))
    fi
else
    echo "FAIL  plan-text: tool returned non-zero"
    feat_failed=$((feat_failed + 1))
fi

echo ""
echo "Feature tests: $feat_passed passed, $feat_failed failed"
[ "$feat_failed" -gt 0 ] && exit 1

# Parallel determinism: --jobs=2 must produce identical findings to --jobs=1
echo ""
echo "--- Parallel determinism test ---"
par_passed=0
par_failed=0

seq_out="$WORK_DIR/seq_audit.json"
par_out="$WORK_DIR/par_audit.json"
"$TOOL" --mode=audit --format=json --jobs=1 \
    "$INPUTS/smoke.cpp" "$INPUTS/types.cpp" "${COMMON_ARGS[@]}" \
    > "$seq_out" 2>/dev/null || true
"$TOOL" --mode=audit --format=json --jobs=2 \
    "$INPUTS/smoke.cpp" "$INPUTS/types.cpp" "${COMMON_ARGS[@]}" \
    > "$par_out" 2>/dev/null || true

seq_rewrites=$(python3 -c "import json; print(json.load(open('$seq_out'))['rewrites'])")
par_rewrites=$(python3 -c "import json; print(json.load(open('$par_out'))['rewrites'])")
seq_flags=$(python3 -c "import json; print(json.load(open('$seq_out'))['flags'])")
par_flags=$(python3 -c "import json; print(json.load(open('$par_out'))['flags'])")

if [ "$seq_rewrites" = "$par_rewrites" ] && [ "$seq_flags" = "$par_flags" ]; then
    echo "PASS  parallel-determinism: jobs=1 ($seq_rewrites rewrites, $seq_flags flags) == jobs=2 ($par_rewrites rewrites, $par_flags flags)"
    par_passed=$((par_passed + 1))
else
    echo "FAIL  parallel-determinism: jobs=1 ($seq_rewrites/$seq_flags) != jobs=2 ($par_rewrites/$par_flags)"
    par_failed=$((par_failed + 1))
fi

echo ""
echo "Parallel tests: $par_passed passed, $par_failed failed"
[ "$par_failed" -gt 0 ] && exit 1

# Macro body detection: unstable API inside #define must be flagged, not silently dropped
echo ""
echo "--- Macro body detection test ---"
macro_passed=0
macro_failed=0

MACRO_INPUT="$WORK_DIR/macro_body_test.cpp"
cat > "$MACRO_INPUT" << 'MACROEOF'
#include <torch/all.h>

#define MY_TENSOR at::Tensor
#define GET_PTR(x) ((x).data_ptr<float>())
#define CLONE_IT(x) ((x).clone())

void foo(torch::Tensor& t) {
    MY_TENSOR a;
    float* p = GET_PTR(t);
    auto c = CLONE_IT(t);
}
MACROEOF

macro_out=$("$TOOL" --mode=audit --format=json "$MACRO_INPUT" "${COMMON_ARGS[@]}" 2>/dev/null || true)
macro_result=$(echo "$macro_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
mb = [f for f in d['findings'] if f['flag'] and 'macro body' in f['new']]
kinds = {f['kind'] for f in mb}
required = {'TYPE', 'DPTR', 'M2F'}
missing = required - kinds
if missing:
    print(f'FAIL missing kinds: {missing}')
elif len(mb) < 3:
    print(f'FAIL only {len(mb)} flags, expected >= 3')
else:
    print(f'PASS {len(mb)}')
")
if echo "$macro_result" | grep -q "^PASS"; then
    count=$(echo "$macro_result" | grep -oP '\d+')
    echo "PASS  macro-body-detection: $count flags across TYPE, DPTR, M2F"
    macro_passed=$((macro_passed + 1))
else
    echo "FAIL  macro-body-detection: $macro_result"
    echo "$macro_out"
    macro_failed=$((macro_failed + 1))
fi

echo ""
echo "Macro body tests: $macro_passed passed, $macro_failed failed"
[ "$macro_failed" -gt 0 ] && exit 1

# Completeness tests: every rule category fires, catch-all works
echo ""
echo "--- Completeness tests ---"
comp_passed=0
comp_failed=0

# 1. Every finding kind must appear when auditing completeness.cpp
comp_out=$("$TOOL" --mode=audit --format=json "$INPUTS/completeness.cpp" "${COMMON_ARGS[@]}" 2>/dev/null || true)
comp_result=$(echo "$comp_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
kinds = {f['kind'] for f in d['findings']}
required = {'TYPE', 'STYPE', 'DPTR', 'M2F', 'FUNC', 'MACRO', 'INCL'}
missing = required - kinds
if missing:
    print(f'FAIL missing: {missing}')
else:
    print(f'PASS {len(d[\"findings\"])} findings, all {len(required)} kinds covered')
")
if echo "$comp_result" | grep -q "^PASS"; then
    echo "PASS  rule-coverage: $comp_result"
    comp_passed=$((comp_passed + 1))
else
    echo "FAIL  rule-coverage: $comp_result"
    comp_failed=$((comp_failed + 1))
fi

# 2. Catch-all: unknown unstable APIs must be flagged
CATCHALL_INPUT="$WORK_DIR/catchall_test.cpp"
cat > "$CATCHALL_INPUT" << 'CATCHALLEOF'
#include <torch/all.h>

void foo() {
    auto a = at::randn({3});
    auto b = at::ones({3});
}
CATCHALLEOF

catchall_out=$("$TOOL" --mode=audit --format=json "$CATCHALL_INPUT" "${COMMON_ARGS[@]}" 2>/dev/null || true)
catchall_flags=$(echo "$catchall_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(sum(1 for f in d['findings'] if f['kind'] in ('UTYPE', 'UREF')))
")
if [ "$catchall_flags" -ge 2 ]; then
    echo "PASS  catch-all: $catchall_flags unknown unstable APIs flagged"
    comp_passed=$((comp_passed + 1))
else
    echo "FAIL  catch-all: expected >= 2 flags, got $catchall_flags"
    comp_failed=$((comp_failed + 1))
fi

# Method catch-all: unhandled methods on unstable types must be flagged
METHCATCH_INPUT="$WORK_DIR/method_catchall.cpp"
cat > "$METHCATCH_INPUT" << 'METHEOF'
#include <ATen/ATen.h>

void foo(at::Tensor& t) {
    auto opts = t.options();
    bool pinned = t.is_pinned();
}
METHEOF

methcatch_out=$("$TOOL" --mode=audit --format=json "$METHCATCH_INPUT" "${COMMON_ARGS[@]}" 2>/dev/null || true)
methcatch_flags=$(echo "$methcatch_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
flags = [f for f in d['findings'] if f['kind'] == 'UMETH']
print(len(flags))
")
if [ "$methcatch_flags" -eq 2 ]; then
    echo "PASS  method-catch-all: 2 unhandled methods flagged"
    comp_passed=$((comp_passed + 1))
else
    echo "FAIL  method-catch-all: expected 2 flags, got $methcatch_flags"
    echo "$methcatch_out" | python3 -m json.tool 2>/dev/null | head -30
    comp_failed=$((comp_failed + 1))
fi

# ifdef auto-reparse: finds unstable API inside #ifdef blocks
IFDEF_INPUT="$WORK_DIR/ifdef_reparse.cpp"
cat > "$IFDEF_INPUT" << 'IFDEFEOF'
#include <ATen/ATen.h>

void always_visible(at::Tensor& t) {
    auto x = t.clone();
}

#ifdef SOME_TEST_FLAG
void conditionally_visible(at::Tensor& t) {
    TORCH_CHECK(t.dim() > 0, "bad");
    auto y = t.clone();
    float* p = t.data_ptr<float>();
    auto z = t.contiguous();
    auto n = t.numel();
    auto s = t.sizes();
    auto d = t.device();
    auto sc = t.scalar_type();
    (void)p; (void)z; (void)n; (void)s; (void)d; (void)sc;
    (void)y;
}
#endif
IFDEFEOF

ifdef_out=$("$TOOL" --mode=audit --format=json "$IFDEF_INPUT" "${COMMON_ARGS[@]}" 2>/dev/null || true)
ifdef_check=$(echo "$ifdef_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
incl = sum(1 for f in d['findings'] if f['kind'] == 'INCL')
skipped = sum(1 for f in d['findings'] if 'not analyzed by AST' in f.get('new',''))
print(f'{incl} {skipped} {len(d[\"findings\"])}')
")
ifdef_incl=$(echo "$ifdef_check" | cut -d' ' -f1)
ifdef_skipped=$(echo "$ifdef_check" | cut -d' ' -f2)
ifdef_total=$(echo "$ifdef_check" | cut -d' ' -f3)
if [ "$ifdef_incl" -eq 1 ] && [ "$ifdef_skipped" -ge 2 ]; then
    echo "PASS  ifdef-skipped-scan: $ifdef_skipped findings in skipped regions, include not double-counted"
    comp_passed=$((comp_passed + 1))
else
    echo "FAIL  ifdef-skipped-scan: incl=$ifdef_incl (want 1), skipped=$ifdef_skipped (want >= 2)"
    echo "$ifdef_out" | python3 -m json.tool 2>/dev/null | head -30
    comp_failed=$((comp_failed + 1))
fi

# String literal: TORCH_CHECK inside a code generator string is flagged
STRLIT_INPUT="$WORK_DIR/string_literal.cpp"
cat > "$STRLIT_INPUT" << 'STRLITEOF'
#include <ATen/ATen.h>

const char* codegen = "TORCH_CHECK(x, msg)";

void f(at::Tensor& t) {
    auto x = t.clone();
}
STRLITEOF

strlit_out=$("$TOOL" --mode=audit --format=json "$STRLIT_INPUT" "${COMMON_ARGS[@]}" 2>/dev/null || true)
strlit_check=$(echo "$strlit_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ast_flags = sum(1 for f in d['findings'] if 'not analyzed by AST' in f.get('new',''))
print(ast_flags)
")
if [ "$strlit_check" -ge 1 ]; then
    echo "PASS  string-literal-scan: TORCH_CHECK in string flagged"
    comp_passed=$((comp_passed + 1))
else
    echo "FAIL  string-literal-scan: expected >= 1 flag in string literal"
    echo "$strlit_out" | python3 -m json.tool 2>/dev/null | head -20
    comp_failed=$((comp_failed + 1))
fi

# Comment: unstable API in comment is flagged
COMMENT_INPUT="$WORK_DIR/comment_scan.cpp"
cat > "$COMMENT_INPUT" << 'COMMENTEOF'
#include <ATen/ATen.h>

// This used to be at::Tensor before migration
void f(torch::stable::Tensor& t) {
    auto x = t.dim();
}
COMMENTEOF

comment_out=$("$TOOL" --mode=audit --format=json "$COMMENT_INPUT" "${COMMON_ARGS[@]}" 2>/dev/null || true)
comment_check=$(echo "$comment_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
comment_flags = [f for f in d['findings'] if f['line'] == 3 and 'not analyzed by AST' in f.get('new','')]
print(len(comment_flags))
")
if [ "$comment_check" -ge 1 ]; then
    echo "PASS  comment-scan: at::Tensor in comment flagged"
    comp_passed=$((comp_passed + 1))
else
    echo "FAIL  comment-scan: expected flag on comment line"
    echo "$comment_out" | python3 -m json.tool 2>/dev/null | head -20
    comp_failed=$((comp_failed + 1))
fi

# No double-counting: AST-covered lines should not get text-scan duplicates
DEDUP_INPUT="$WORK_DIR/dedup_scan.cpp"
cat > "$DEDUP_INPUT" << 'DEDUPEOF'
#include <ATen/ATen.h>

void f(at::Tensor& t) {
    auto x = t.clone();
}
DEDUPEOF

dedup_out=$("$TOOL" --mode=audit --format=json "$DEDUP_INPUT" "${COMMON_ARGS[@]}" 2>/dev/null || true)
dedup_check=$(echo "$dedup_out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
# Line 3 has at::Tensor — AST should find it, text scan should not duplicate
line3 = [f for f in d['findings'] if f['line'] == 3]
ast_only = all('not analyzed by AST' not in f.get('new','') for f in line3)
has_finding = len(line3) > 0
print(f'{has_finding} {ast_only} {len(line3)}')
")
dedup_found=$(echo "$dedup_check" | cut -d' ' -f1)
dedup_ast_only=$(echo "$dedup_check" | cut -d' ' -f2)
dedup_count=$(echo "$dedup_check" | cut -d' ' -f3)
if [ "$dedup_found" = "True" ] && [ "$dedup_ast_only" = "True" ]; then
    echo "PASS  no-double-count: line 3 has $dedup_count finding(s), all from AST"
    comp_passed=$((comp_passed + 1))
else
    echo "FAIL  no-double-count: found=$dedup_found ast_only=$dedup_ast_only count=$dedup_count"
    echo "$dedup_out" | python3 -m json.tool 2>/dev/null | head -20
    comp_failed=$((comp_failed + 1))
fi

echo ""
echo "Completeness tests: $comp_passed passed, $comp_failed failed"
[ "$comp_failed" -gt 0 ] && exit 1

# Compile-based verification: stricter check, may expose issues regex misses
echo ""
echo "--- Compile-based verification tests ---"
compile_passed=0
compile_failed=0
for expected_file in "$EXPECTED"/*.cpp; do
    [ -f "$expected_file" ] || continue
    basename="$(basename "$expected_file")"

    set +e
    timeout 30 "$TOOL" --mode=verify --pytorch-root="$PYTORCH_DIR" "$expected_file" -- -std=c++20 > "$WORK_DIR/verify_out" 2>&1
    rc=$?
    set -e
    if [ $rc -eq 124 ]; then
        echo "COMPILE TIMEOUT $basename"
        compile_failed=$((compile_failed + 1))
        continue
    fi
    if [ $rc -eq 0 ] && grep -q "PASS" "$WORK_DIR/verify_out"; then
        echo "COMPILE PASS  $basename"
        compile_passed=$((compile_passed + 1))
    else
        echo "COMPILE FAIL  $basename"
        cat "$WORK_DIR/verify_out"
        compile_failed=$((compile_failed + 1))
    fi
done

echo ""
echo "Compile verification: $compile_passed passed, $compile_failed failed"
[ "$compile_failed" -gt 0 ] && exit 1

# Idempotence: rewriting already-rewritten output must produce zero changes
echo ""
echo "--- Idempotence tests ---"
idemp_passed=0
idemp_failed=0
for expected_file in "$EXPECTED"/*.cpp; do
    [ -f "$expected_file" ] || continue
    basename="$(basename "$expected_file")"

    cp "$expected_file" "$WORK_DIR/idemp_$basename"
    set +e
    "$TOOL" --mode=rewrite "$WORK_DIR/idemp_$basename" "${COMMON_ARGS[@]}" > /dev/null 2>"$WORK_DIR/idemp_$basename.stderr"
    rc=$?
    set -e
    if [ $rc -gt 1 ]; then
        echo "IDEMP FAIL  $basename (tool crashed with exit code $rc)"
        cat "$WORK_DIR/idemp_$basename.stderr"
        idemp_failed=$((idemp_failed + 1))
        continue
    fi

    if diff -q "$expected_file" "$WORK_DIR/idemp_$basename" > /dev/null 2>&1; then
        echo "IDEMP PASS  $basename"
        idemp_passed=$((idemp_passed + 1))
    else
        echo "IDEMP FAIL  $basename (rewrite changed already-stable output)"
        diff -u "$expected_file" "$WORK_DIR/idemp_$basename" | head -20
        idemp_failed=$((idemp_failed + 1))
    fi
done

echo ""
echo "Idempotence: $idemp_passed passed, $idemp_failed failed"
[ "$idemp_failed" -gt 0 ] && exit 1
exit 0
