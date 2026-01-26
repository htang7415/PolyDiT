#!/bin/bash
#SBATCH --job-name=mvf
#SBATCH --output=logs/mvf_%x_%j.out
#SBATCH --error=logs/mvf_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00

set -e

CONDA_DIR="/home/htang/anaconda3"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate kl_active_learning

mkdir -p logs
bash scripts/run_pipeline.sh
