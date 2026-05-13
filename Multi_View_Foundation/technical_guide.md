# Technical Guide: Multi-View Foundation

MVF is currently a `property_regression` workflow.

## Objective

Train and compare property predictors on top of five frozen polymer representation backbones:

- `smiles`
- `smiles_bpe`
- `selfies`
- `group_selfies`
- `graph`

The comparison target is test-set property prediction performance for each property, view, and model size.

## Split Policy

Each property CSV is split once using `data.random_seed: 42`:

- 80% train
- 10% validation
- 10% test

The validation split is used by Optuna. The test split is not used for model selection.

## Training Policy

For each property/view pair:

1. Load the frozen backbone checkpoint for the configured model size.
2. Embed valid polymers for that view.
3. Optimize MLP-head hyperparameters with validation R2.
4. Retrain the selected head on train plus validation.
5. Evaluate final metrics on the held-out test split.

Default HPO settings:

- `n_trials: 50`
- `tuning_epochs: 50`
- `tuning_patience: 10`
- `final_training_epochs: 200`

Metric files preserve previous property/view rows when one head is rerun, so incomplete size matrices can be filled one command at a time.

## Comparison Rule

Use only `split=test` rows in `metrics_property.csv` for paper tables and cross-method comparison. Validation rows are HPO diagnostics, not final performance numbers.
