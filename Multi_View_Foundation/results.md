# Results Template: Multi-View Foundation

Expected artifacts for `property_regression`:

- `results_<size>/step1_property_regression/metrics/metrics_property.csv`
- `results_<size>/step1_property_regression/<PROPERTY>/metrics/metrics_property.csv`
- `results_<size>/step1_property_regression/<PROPERTY>/files/<PROPERTY>_<VIEW>_mlp.pt`
- `results_<size>/step1_property_regression/<PROPERTY>/files/<PROPERTY>_<VIEW>_meta.json`
- `results_<size>/step1_property_regression/<PROPERTY>/files/<PROPERTY>_<VIEW>_mlp_hpo_trials.csv`
- `results_<size>/step1_property_regression/<PROPERTY>/files/<PROPERTY>_split_seed42.csv`
- `results_<size>/step1_property_regression/<PROPERTY>/files/<PROPERTY>_<VIEW>_optuna_summary.json`
- `results_<size>/step1_property_regression/figures/`
- `results_<size>/step1_property_regression/<PROPERTY>/figures/`

Metric rows include:

- `stage=property_regression`
- `view`
- `representation`
- `model_size`
- `property`
- `split=train|val|test`
- `mae`
- `rmse`
- `r2`

Use `split=test` for final performance comparison across the five views and four model sizes. Use `split=val` only to inspect HPO behavior.
