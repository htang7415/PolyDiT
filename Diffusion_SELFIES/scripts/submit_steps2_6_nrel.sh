#!/bin/bash
#SBATCH --account=nawimem
#SBATCH --time=2-00:00:00
#SBATCH --job-name=sel_2_6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# Steps 2-6 (NREL, single GPU)
# Usage: sbatch scripts/submit_steps2_6_nrel.sh <model_size> [property] [targets] [epsilon]
#        [num_samples] [num_candidates] [polymer_class] [class_target] [class_epsilon]
# Tip: pass --partition/--qos at submit time for your GPU type.

set -e

# Conda setup
CONDA_DIR="/home/htang/anaconda3"
eval "$("$CONDA_DIR"/bin/conda shell.bash hook)"
conda activate kl_active_learning

# Work directory
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p logs

MODEL_SIZE=${1:-medium}
PROPERTY=${2:-Tg}
TARGETS=${3:-"300"}
EPSILON=${4:-10.0}
NUM_SAMPLES=${5:-50000}
NUM_CANDIDATES=${6:-10000}
POLYMER_CLASS=${7:-polyimide}
CLASS_TARGET=${8:-300}
CLASS_EPSILON=${9:-10.0}

python scripts/step2_sample_and_evaluate.py --config configs/config.yaml --model_size "$MODEL_SIZE" --num_samples "$NUM_SAMPLES"
python scripts/step3_train_property_head.py --config configs/config.yaml --model_size "$MODEL_SIZE" --property "$PROPERTY"
python scripts/step4_inverse_design.py --config configs/config.yaml --model_size "$MODEL_SIZE" --property "$PROPERTY" --targets "$TARGETS" --epsilon "$EPSILON" --num_candidates "$NUM_CANDIDATES"
python scripts/step5_class_design.py --config configs/config.yaml --model_size "$MODEL_SIZE" --polymer_class "$POLYMER_CLASS" --num_candidates "$NUM_CANDIDATES"
python scripts/step5_class_design.py --config configs/config.yaml --model_size "$MODEL_SIZE" --polymer_class "$POLYMER_CLASS" --property "$PROPERTY" --target_value "$CLASS_TARGET" --epsilon "$CLASS_EPSILON" --num_candidates "$NUM_CANDIDATES"
