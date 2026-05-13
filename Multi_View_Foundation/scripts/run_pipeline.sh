#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODEL_SIZE="${1:-}"
CONFIG_PATH="${MVF_CONFIG_PATH:-configs/config.yaml}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_step() {
  local label="$1"
  shift
  local start_ts
  local end_ts
  start_ts=$(date +%s)
  echo "[$(timestamp)] Starting ${label}"
  "$@"
  end_ts=$(date +%s)
  echo "[$(timestamp)] Finished ${label} (duration: $((end_ts - start_ts))s)"
}

if [[ -n "${MODEL_SIZE}" ]]; then
  TMP_SUFFIX_RAW="${MVF_TMP_CONFIG_SUFFIX:-}"
  if [[ -n "${TMP_SUFFIX_RAW}" ]]; then
    TMP_SUFFIX="$(printf '%s' "${TMP_SUFFIX_RAW}" | tr -cs '[:alnum:]_-' '_' | sed 's/^_*//; s/_*$//')"
    if [[ -n "${TMP_SUFFIX}" ]]; then
      TMP_CONFIG="configs/config_${MODEL_SIZE}_${TMP_SUFFIX}.yaml"
    else
      TMP_CONFIG="configs/config_${MODEL_SIZE}.yaml"
    fi
  else
    TMP_CONFIG="configs/config_${MODEL_SIZE}.yaml"
  fi

  python - "${CONFIG_PATH}" "${TMP_CONFIG}" "${MODEL_SIZE}" <<'PY'
import os
import sys
from pathlib import Path

import yaml


def _csv_list(value):
    if value is None:
        return None
    return [x.strip() for x in str(value).split(",") if x.strip()]


in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
model_size = sys.argv[3].lower().strip()

with in_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

paths_cfg = cfg.get("paths", {}) or {}
results_dir = f"results_{model_size}"
paths_cfg["results_dir"] = results_dir
cfg["paths"] = paths_cfg

for key in ("smiles_encoder", "smiles_bpe_encoder", "selfies_encoder", "group_selfies_encoder", "graph_encoder"):
    if isinstance(cfg.get(key), dict):
        cfg[key]["model_size"] = model_size

prop_cfg = cfg.get("property", {}) or {}
prop_cfg["val_ratio"] = float(prop_cfg.get("val_ratio", 0.1))
prop_cfg["test_ratio"] = float(prop_cfg.get("test_ratio", 0.1))

forced_prop_files = _csv_list(os.environ.get("MVF_PROPERTY_FILES"))
if forced_prop_files:
    prop_cfg["files"] = forced_prop_files

if os.environ.get("MVF_PROPERTY_DEVICE"):
    prop_cfg["device"] = os.environ["MVF_PROPERTY_DEVICE"].strip()

hpo_cfg = prop_cfg.get("hyperparameter_tuning", {}) or {}
hpo_cfg["validation_strategy"] = "holdout_8_1_1"

hpo_cfg["n_trials"] = 50
hpo_cfg["final_training_epochs"] = 200

if os.environ.get("MVF_PROPERTY_TUNING_EPOCHS"):
    hpo_cfg["tuning_epochs"] = int(os.environ["MVF_PROPERTY_TUNING_EPOCHS"])

if os.environ.get("MVF_PROPERTY_TUNING_PATIENCE"):
    hpo_cfg["tuning_patience"] = int(os.environ["MVF_PROPERTY_TUNING_PATIENCE"])

prop_cfg["hyperparameter_tuning"] = hpo_cfg
cfg["property"] = prop_cfg

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(
    f"Generated {out_path} for property_regression: "
    f"results_dir={results_dir} "
    f"property.files={prop_cfg.get('files')} "
    f"property.views={prop_cfg.get('views')} "
    f"split=train/val/test={1.0 - prop_cfg['val_ratio'] - prop_cfg['test_ratio']:.1f}/"
    f"{prop_cfg['val_ratio']:.1f}/{prop_cfg['test_ratio']:.1f} "
    f"n_trials={hpo_cfg.get('n_trials')} "
    f"final_training_epochs={hpo_cfg.get('final_training_epochs')}"
)
PY
  CONFIG_PATH="${TMP_CONFIG}"
fi

python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

prop_cfg = cfg.get("property", {}) or {}
print(
    "property_regression config:",
    f"results_dir={cfg.get('paths', {}).get('results_dir')}",
    f"files={prop_cfg.get('files')}",
    f"views={prop_cfg.get('views')}",
    f"val_ratio={prop_cfg.get('val_ratio', 0.1)}",
    f"test_ratio={prop_cfg.get('test_ratio', 0.1)}",
)
PY

PROPERTY_CMD=(python scripts/step1_property_regression.py --config "${CONFIG_PATH}")

case "${MVF_PROPERTY_TUNE:-}" in
  1|true|TRUE|yes|YES|on|ON) PROPERTY_CMD+=(--tune) ;;
  0|false|FALSE|no|NO|off|OFF) PROPERTY_CMD+=(--no_tune) ;;
esac

case "${MVF_PROPERTY_FIGURES:-}" in
  1|true|TRUE|yes|YES|on|ON) PROPERTY_CMD+=(--generate_figures) ;;
  0|false|FALSE|no|NO|off|OFF) PROPERTY_CMD+=(--no_figures) ;;
esac

run_step "property_regression" "${PROPERTY_CMD[@]}"
