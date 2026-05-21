#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TOOL="$PROJECT_DIR/build/stable-abi-transform"
INPUTS="$SCRIPT_DIR/inputs"
EXPECTED="$SCRIPT_DIR/expected"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PYTORCH_DIR="${PYTORCH_DIR:?Set PYTORCH_DIR to your PyTorch source root}"
RESOURCE_DIR="${RESOURCE_DIR:-/usr/lib/clang/19}"

CUDA_INCLUDE="${CUDA_INCLUDE:-/usr/local/cuda/include}"

COMMON_ARGS=(
    -- -std=c++20
    -resource-dir "$RESOURCE_DIR"
    -DC10_USING_CUSTOM_GENERATED_MACROS
    -I"$PYTORCH_DIR/torch/csrc/api/include"
    -I"$PYTORCH_DIR"
    -I"$PYTORCH_DIR/torch/include"
    -I"$CUDA_INCLUDE"
)

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
    if ! "$TOOL" --mode=rewrite "$WORK_DIR/$basename" "${COMMON_ARGS[@]}" > "$WORK_DIR/$basename.stdout" 2>"$WORK_DIR/$basename.stderr"; then
        echo "FAIL  $basename (tool returned non-zero)"
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

# Compile-based verification: stricter check, may expose issues regex misses
echo ""
echo "--- Compile-based verification tests ---"
compile_passed=0
compile_failed=0
for expected_file in "$EXPECTED"/*.cpp; do
    [ -f "$expected_file" ] || continue
    basename="$(basename "$expected_file")"

    if timeout 30 "$TOOL" --mode=verify --pytorch-root="$PYTORCH_DIR" "$expected_file" -- -std=c++20 > "$WORK_DIR/verify_out" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    if [ $rc -eq 124 ]; then
        echo "COMPILE TIMEOUT $basename"
        compile_failed=$((compile_failed + 1))
        continue
    fi
    if grep -q "PASS" "$WORK_DIR/verify_out"; then
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
exit 0
