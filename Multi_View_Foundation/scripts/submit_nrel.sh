#!/bin/bash
#SBATCH --account=nawimem
#SBATCH --time=8-00:00:00
#SBATCH --job-name=MVF
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# Preferred: bash scripts/submit_nrel.sh [small|medium|large|xl] [smiles|smiles_bpe|selfies|group_selfies|graph]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${MVF_REPO_DIR:-}" && -d "${MVF_REPO_DIR}" ]]; then
  REPO_DIR="${MVF_REPO_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
else
  REPO_DIR="${DEFAULT_REPO_DIR}"
fi

if [[ ! -f "${REPO_DIR}/scripts/run_pipeline.sh" ]]; then
  REPO_DIR="${DEFAULT_REPO_DIR}"
fi

LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  MODEL_SIZE="${1:-small}"
  PROPERTY_VIEWS="${2:-}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitting MVF NREL job (model_size=${MODEL_SIZE}, property_views=${PROPERTY_VIEWS:-config/default})"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] SLURM stdout/err: ${LOG_DIR}/MVF_<jobid>.out/.err"
  sbatch \
    --account=nawimem \
    --time=8-00:00:00 \
    --job-name=MVF \
    --output="${LOG_DIR}/%x_%j.out" \
    --error="${LOG_DIR}/%x_%j.err" \
    --mem=256G \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gres=gpu:1 \
    --chdir="${REPO_DIR}" \
    --export=ALL,MVF_REPO_DIR="${REPO_DIR}" \
    "${DEFAULT_REPO_DIR}/scripts/submit_nrel.sh" "${MODEL_SIZE}" "${PROPERTY_VIEWS}"
  exit $?
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Host: $(hostname)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Repo: ${REPO_DIR}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submit dir: ${SLURM_SUBMIT_DIR:-unknown}"

cd "${REPO_DIR}"

CONDA_CANDIDATES=(
  "${CONDA_DIR:-/home/htang/anaconda3}"
  "/home/htang228/anaconda3"
  "/srv/home/htang228/anaconda3"
)
CONDA_DIR_RESOLVED=""
for candidate in "${CONDA_CANDIDATES[@]}"; do
  if [[ -x "${candidate}/bin/conda" ]]; then
    CONDA_DIR_RESOLVED="${candidate}"
    break
  fi
done

if [[ -z "${CONDA_DIR_RESOLVED}" ]]; then
  echo "ERROR: conda executable not found in expected paths." >&2
  exit 2
fi

eval "$(${CONDA_DIR_RESOLVED}/bin/conda shell.bash hook)"
conda activate kl_active_learning

python - <<'PY'
import os
import sys
import torch

required = os.environ.get("MVF_REQUIRE_CUDA", "1").strip().lower() not in {"0", "false", "no"}
available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
print(f"CUDA available={available} device_count={count} required={required}")
if required and not available:
    print("ERROR: CUDA is unavailable. Check conda env / module setup on NREL.")
    sys.exit(3)
PY

nvidia-smi -L || true

MODEL_SIZE="${1:-small}"
PROPERTY_VIEWS="${2:-}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching MVF pipeline with model_size=${MODEL_SIZE}, property_views=${PROPERTY_VIEWS:-config/default}"
bash scripts/run_pipeline.sh "${MODEL_SIZE}" "${PROPERTY_VIEWS}"
