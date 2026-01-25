#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/test.sh <command>

Commands:
  unit       Run unified orchestrator unit tests
  coverage   Run unified orchestrator unit tests with coverage report
EOF
}

if [[ "${1:-}" == "" ]]; then
  usage
  exit 1
fi

case "${1}" in
  unit)
    pytest tests/unified_orchestrator -v
    ;;
  coverage)
    python -m coverage run --source=orc/unified_orchestrator -m pytest tests/unified_orchestrator
    python -m coverage report -m
    ;;
  *)
    usage
    exit 1
    ;;
 esac
