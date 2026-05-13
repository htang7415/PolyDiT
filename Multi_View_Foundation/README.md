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

Run one property and one view:

```bash
MVF_REQUIRE_CUDA=1 MVF_PROPERTY_FILES=Tg.csv MVF_PROPERTY_VIEWS=smiles_bpe bash scripts/run_property_regression.sh medium
```

Direct script form:

```bash
python scripts/step1_property_regression.py --config configs/config.yaml --views smiles_bpe
```

Default HPO settings are `50` Optuna trials and `200` final-training epochs.

See:

- [Pipeline.md](Pipeline.md)
- [technical_guide.md](technical_guide.md)
- [results.md](results.md)
