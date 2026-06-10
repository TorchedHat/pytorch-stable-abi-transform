#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
echo "clang-format..."
clang-format --dry-run --Werror src/*.cpp src/*.h
echo "ruff..."
ruff check scripts/
ruff format --check scripts/
if [ -d build ]; then
    echo "clang-tidy..."
    clang-tidy -p build --warnings-as-errors='*' --quiet src/*.cpp 2>/dev/null
fi
echo "All checks passed."
