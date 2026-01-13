#!/bin/bash
# Option B submit wrapper (Euler): submit Step 1 DDP, then Steps 2-6 after success.
# Usage: bash scripts/submit_all_euler.sh <model_size> [property] [targets] [epsilon]
#        [num_samples] [num_candidates] [polymer_class] [class_target] [class_epsilon]
#        bash scripts/submit_all_nrel.sh small
#        bash scripts/submit_all_nrel.sh medium
#        bash scripts/submit_all_nrel.sh large
#        bash scripts/submit_all_nrel.sh xl

set -e

PREFIX="gsel"

MODEL_SIZE=${1:-medium}
PROPERTY=${2:-Tg}
TARGETS=${3:-"300"}
EPSILON=${4:-10.0}
NUM_SAMPLES=${5:-50000}
NUM_CANDIDATES=${6:-10000}
POLYMER_CLASS=${7:-polyimide}
CLASS_TARGET=${8:-300}
CLASS_EPSILON=${9:-10.0}
SIZE_TAG="m"
case "$MODEL_SIZE" in
  small|s) SIZE_TAG="s" ;;
  medium|m) SIZE_TAG="m" ;;
  large|l) SIZE_TAG="l" ;;
  xl|x) SIZE_TAG="xl" ;;
  *) SIZE_TAG="$MODEL_SIZE" ;;
esac

JOB1="${PREFIX}_${SIZE_TAG}_1"
JOB2="${PREFIX}_${SIZE_TAG}_2_6"

mkdir -p logs


jid=$(sbatch --parsable --job-name "$JOB1" --output "logs/%x_%j.out" --error "logs/%x_%j.err" scripts/submit_step1_ddp_euler.sh "$MODEL_SIZE")
sbatch --job-name "$JOB2" --output "logs/%x_%j.out" --error "logs/%x_%j.err" --dependency=afterok:$jid scripts/submit_steps2_6_euler.sh   "$MODEL_SIZE" "$PROPERTY" "$TARGETS" "$EPSILON" "$NUM_SAMPLES"   "$NUM_CANDIDATES" "$POLYMER_CLASS" "$CLASS_TARGET" "$CLASS_EPSILON"
