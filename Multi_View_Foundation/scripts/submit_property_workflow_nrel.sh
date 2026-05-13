#!/bin/bash
# Submit MVF property_regression jobs on NREL.
#
# Usage:
#   bash scripts/submit_property_workflow_nrel.sh [model_size] [views] [partition] [qos]
# Example:
#   bash scripts/submit_property_workflow_nrel.sh medium smiles_bpe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${REPO_DIR}/configs/config.yaml"

MODEL_SIZE="${1:-small}"
PROPERTY_VIEWS="${2:-${MVF_PROPERTY_VIEWS:-}}"
PARTITION="${3:-}"
QOS="${4:-}"

LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

mapfile -t PROPERTIES < <(
  python - "${CONFIG_PATH}" <<'PY'
import os
import sys
from pathlib import Path
import yaml

forced = os.environ.get("MVF_PROPERTY_FILES")
if forced:
    files = [x.strip() for x in forced.split(",") if x.strip()]
else:
    with Path(sys.argv[1]).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    files = (cfg.get("property", {}) or {}).get("files") or ["Tg.csv"]
    if isinstance(files, str):
        files = [files]

seen = set()
for item in files:
    name = Path(str(item).strip()).stem
    if name and name not in seen:
        seen.add(name)
        print(name)
PY
)

if [[ "${#PROPERTIES[@]}" -eq 0 ]]; then
  echo "ERROR: no properties resolved from ${CONFIG_PATH}" >&2
  exit 2
fi

SBATCH_ARGS=(
  --account=nawimem
  --mem=256G
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --time=8-00:00:00
  --gres=gpu:1
  --chdir="${REPO_DIR}"
)
if [[ -n "${PARTITION}" ]]; then
  SBATCH_ARGS+=(--partition "${PARTITION}")
fi
if [[ -n "${QOS}" ]]; then
  SBATCH_ARGS+=(--qos "${QOS}")
fi

declare -a PROPERTY_JIDS=()
for prop in "${PROPERTIES[@]}"; do
  prop_slug="$(printf '%s' "${prop}" | tr -cs '[:alnum:]_-' '_' | sed 's/^_*//; s/_*$//')"
  if [[ -z "${prop_slug}" ]]; then
    prop_slug="prop"
  fi

  export_parts=(
    "ALL"
    "MVF_REPO_DIR=${REPO_DIR}"
    "MVF_PROPERTY_FILES=${prop}.csv"
    "MVF_TMP_CONFIG_SUFFIX=property_regression_${prop_slug}"
  )

  jid="$(
    sbatch \
      --parsable \
      "${SBATCH_ARGS[@]}" \
      --job-name "MVF_PR_${prop}_${MODEL_SIZE}" \
      --output "${LOG_DIR}/%x_%j.out" \
      --error "${LOG_DIR}/%x_%j.err" \
      --export "$(IFS=,; echo "${export_parts[*]}")" \
      "${SCRIPT_DIR}/submit_nrel.sh" \
      "${MODEL_SIZE}" \
      "${PROPERTY_VIEWS}"
  )"
  PROPERTY_JIDS+=("${jid}")
  echo "Submitted property_regression job (${prop}): ${jid}"
done

echo "Submitted property_regression jobs: ${PROPERTY_JIDS[*]}"
