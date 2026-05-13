# Pipeline Guide: Multi-View Foundation

The active MVF workflow is `property_regression`.

It compares property prediction quality across five polymer representations:
`smiles`, `smiles_bpe`, `selfies`, `group_selfies`, and `graph`.

## property_regression

- Script: `python scripts/step1_property_regression.py --config configs/config.yaml`
- Wrapper: `bash scripts/run_property_regression.sh <size>`
- Euler submit: `bash scripts/submit_euler.sh <size>`
- Output root: `results_<size>/step1_property_regression/`
- Default model sizes: `small`, `medium`, `large`, `xl`
- Default properties: `Tg`, `Tm`, `Td`, `Eg`, `Ced`, `Ea`, `Eib`, `In`

The split policy is fixed:

- train: 80%
- validation: 10%
- test: 10%
- seed: `42`

The validation split is used for Optuna HPO. The test split is held out until final evaluation, so the test metrics are the numbers to compare across the five views and four MVF sizes.

## Commands

Run all configured property/view heads for one model size:

```bash
bash scripts/run_property_regression.sh medium
```

Submit one Euler job for all configured properties and all five views:

```bash
bash scripts/submit_euler.sh medium
```

Allowed model sizes are `small`, `medium`, `large`, and `xl`.

Advanced local run for one property across all five views:

```bash
MVF_REQUIRE_CUDA=1 MVF_PROPERTY_FILES=Tg.csv bash scripts/run_property_regression.sh medium
```

## Outputs

Each run writes model, metadata, HPO, split, metric, and figure artifacts under `step1_property_regression`. HPO is fixed to 50 Optuna trials, final training is fixed to 200 epochs, and only the best final checkpoint is saved for each property/view head. Existing metrics for other property/view pairs are preserved when rerunning one head at a time.
