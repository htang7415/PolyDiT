#!/bin/bash
#SBATCH --job-name=mvf
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=164G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00

# Usage:
#   bash scripts/submit_euler.sh <small|medium|large|xl>
#
# This submits one Euler job for the requested model size. The job runs all
# configured properties across all five property_regression views.

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

usage() {
  echo "Usage: bash scripts/submit_euler.sh <small|medium|large|xl>" >&2
}

if [[ "$#" -ne 1 ]]; then
  usage
  exit 2
fi

MODEL_SIZE="$1"
case "${MODEL_SIZE}" in
  small|medium|large|xl) ;;
  *)
    echo "ERROR: unknown model size '${MODEL_SIZE}'. Expected one of: small, medium, large, xl." >&2
    exit 2
    ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitting MVF Euler job (model_size=${MODEL_SIZE}, properties=all, views=all)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] SLURM stdout/err: ${LOG_DIR}/mvf_${MODEL_SIZE}_<jobid>.out/.err"
  sbatch \
    --job-name="mvf_${MODEL_SIZE}" \
    --output="${LOG_DIR}/%x_%j.out" \
    --error="${LOG_DIR}/%x_%j.err" \
    --nodes=1 \
    --ntasks=1 \
    --mem=164G \
    --cpus-per-task=16 \
    --ntasks-per-node=1 \
    --partition=research \
    --gres=gpu:1 \
    --time=8-00:00:00 \
    --chdir="${REPO_DIR}" \
    --export=ALL,MVF_REPO_DIR="${REPO_DIR}" \
    "${DEFAULT_REPO_DIR}/scripts/submit_euler.sh" "${MODEL_SIZE}"
  exit $?
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Host: $(hostname)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Repo: ${REPO_DIR}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submit dir: ${SLURM_SUBMIT_DIR:-unknown}"

cd "${REPO_DIR}"

CONDA_CANDIDATES=(
  "${CONDA_DIR:-/srv/home/htang228/anaconda3}"
  "/home/htang228/anaconda3"
  "/home/htang/anaconda3"
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
conda activate euler_active_learning

python - <<'PY'
import os
import sys
import torch

required = os.environ.get("MVF_REQUIRE_CUDA", "1").strip().lower() not in {"0", "false", "no"}
available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
print(f"CUDA available={available} device_count={count} required={required}")
if required and not available:
    print("ERROR: CUDA is unavailable. Check conda env / module setup on Euler.")
    sys.exit(3)
PY

nvidia-smi -L || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching MVF pipeline with model_size=${MODEL_SIZE}, properties=all, views=all"
bash scripts/run_pipeline.sh "${MODEL_SIZE}"
