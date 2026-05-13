#!/bin/bash
# Compatibility wrapper. The active Euler workflow submits one model-size job
# that runs all configured properties across all five property_regression views.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/submit_property_workflow_euler.sh <small|medium|large|xl>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_euler.sh" "$1"
