#!/bin/bash
set -e

python scripts/step0_build_paired_dataset.py --config configs/config.yaml
python scripts/step1_train_alignment.py --config configs/config.yaml
python scripts/step2_evaluate_retrieval.py --config configs/config.yaml
python scripts/step3_train_property_heads.py --config configs/config.yaml
python scripts/step4_ood_analysis.py --config configs/config.yaml
echo "F5 requires candidates CSV and is not run by default."
