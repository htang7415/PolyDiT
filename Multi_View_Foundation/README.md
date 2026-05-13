# Multi-View Foundation

MVF is currently scoped to `property_regression`: property prediction from frozen representation backbones.

The active views are:

- `smiles`
- `smiles_bpe`
- `selfies`
- `group_selfies`
- `graph`

`property_regression` uses a fixed split for every property CSV:

- train: 80%
- validation: 10%
- test: 10%
- random seed: `42`

Optuna uses only the validation split for hyperparameter selection. The final model is trained on train plus validation, and the test split is used only for final comparison across views and model sizes.

## Run

Run all configured properties and views for one size:

```bash
bash scripts/run_property_regression.sh medium
```

Submit one Euler job for all configured properties and all five views:

```bash
bash scripts/submit_euler.sh medium
```

Advanced direct script form for a view subset:

```bash
python scripts/step1_property_regression.py --config configs/config.yaml --views smiles_bpe
```

HPO is fixed to `50` Optuna trials. Final training with the selected hyperparameters is fixed to `200` epochs. Property-regression checkpoints are not saved; the workflow keeps metrics, splits, HPO diagnostics, metadata, and figures. `finetune_last_layers` is included in HPO as an integer range from `0` to the configured backbone layer count for each view.

See:

- [Pipeline.md](Pipeline.md)
- [technical_guide.md](technical_guide.md)
- [results.md](results.md)
