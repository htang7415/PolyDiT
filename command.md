# Runbook

Run diffusion commands inside the chosen `Bi_Diffusion_*` method. Run MVF commands inside `Multi_View_Foundation`. Run water-miscibility commands from the repository root.

## One-Time Data Split

```bash
python Data/Polymer/split_unlabeled_full_columns.py --overwrite
```

## Diffusion Baselines

Run a full scaled pipeline from a `Bi_Diffusion_*` method:

```bash
python scripts/run_scaling_pipeline.py --model_size small --property Tg --target 300
```

Run individual steps:

```bash
python scripts/step0_prepare_data.py --config configs/config.yaml
python scripts/step1_train_backbone.py --config configs/config.yaml --model_size small
python scripts/step2_sample_and_evaluate.py --config configs/config.yaml --model_size small --num_samples 1000
python scripts/step3_train_property_head.py --config configs/config.yaml --model_size small --property Tg
python scripts/step4_inverse_design.py --config configs/config.yaml --model_size small --property Tg --targets 300 --epsilon 10
python scripts/step5_class_design.py --config configs/config.yaml --model_size small --polymer_class polyimide
python scripts/step5_class_design.py --config configs/config.yaml --model_size small --polymer_class polyimide --property Tg --target_value 300 --epsilon 10
python scripts/step6_hyperparameter_tuning.py --config configs/config.yaml --model_size small --mode property --property Tg
```

Use `--skip_step*` flags in `run_scaling_pipeline.py` when reusing trained artifacts. In that wrapper, Step6 is joint class-property design; hyperparameter tuning is run separately with `step6_hyperparameter_tuning.py`.

## Multi-View Foundation

Run the config-driven local pipeline:

```bash
cd Multi_View_Foundation
bash scripts/run_pipeline.sh
```

Run a staged NREL workflow for one model size:

```bash
cd Multi_View_Foundation
bash scripts/submit_property_workflow_nrel.sh small
```

Run all model sizes and then F8:

```bash
cd Multi_View_Foundation
bash scripts/submit_multisize_with_f8_nrel.sh small,medium,large,xl
```

Rerun only selected MVF stages:

```bash
cd Multi_View_Foundation
MVF_STEP_START=5 MVF_STEP_END=7 bash scripts/run_pipeline.sh
```

Useful overrides:

- `MVF_PROPERTY_LIST=Tg,Tm,Td,Eg`
- `MVF_F5_RUN_ALL_PROPERTIES=1`
- `MVF_F6_RUN_ALL_PROPERTIES=1`
- `MVF_F5_PROPOSAL_VIEWS=smiles,selfies,graph`
- `MVF_EXPORT_PAPER_PACKAGE=0`

## Water Miscibility

Run a fast local smoke pass:

```bash
python water_miscible/scripts/run_five_view_tasks.py --config water_miscible/configs/config_water.yaml --views smiles --no_tune --max_rows 200
```

Run local submit wrappers:

```bash
bash water_miscible/scripts/submit_local_chi.sh small
bash water_miscible/scripts/submit_local_water.sh small
```

Use `--no_tune` and `--max_rows` for debugging; full HPO is expensive.
