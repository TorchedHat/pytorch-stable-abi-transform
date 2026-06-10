#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
clang-format -i src/*.cpp src/*.h
ruff check --fix scripts/ 2>/dev/null || true
ruff format scripts/
echo "Formatted."
