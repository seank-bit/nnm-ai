#!/usr/bin/env bash
set -euo pipefail
ruff check src tests
ruff format --check src tests
mypy src
pytest tests/unit -v
echo "✓ all checks passed"
