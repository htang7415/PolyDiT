# Water Miscibility

This workflow compares five trained representation backbones on:

- `chi` regression with temperature and volume fraction inputs.
- `water_miscible` classification from polymer embeddings.

The script task keys are `chi_regression` and `water_classification`; aliases such as `chi` and `water_miscible` are accepted.

The default chi split groups rows by polymer SMILES so the same structure does not appear in multiple splits.

## Run

Fast smoke pass:

```bash
python water_miscible/scripts/run_five_view_tasks.py --config water_miscible/configs/config_water.yaml --views smiles --no_tune --max_rows 200
```

Full local run:

```bash
python water_miscible/scripts/run_five_view_tasks.py --config water_miscible/configs/config_water.yaml
```

Submit wrappers:

```bash
bash water_miscible/scripts/submit_local_chi.sh small
bash water_miscible/scripts/submit_local_water.sh small
```

Use `--fresh_embeddings` after changing data or moving from a capped debug run to a full run. Full HPO is expensive, so prefer `--no_tune` for smoke tests.
