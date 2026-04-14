# water_miscible

Standalone five-view experiments for:

- `chi` regression: `chi = f(embedding, temperature, phi)`
- `water_miscible` classification: `water_miscible = f(embedding)`

The project consumes already trained Step1 backbones from:

- `Bi_Diffusion_SMILES`
- `Bi_Diffusion_SMILES_BPE`
- `Bi_Diffusion_SELFIES`
- `Bi_Diffusion_Group_SELFIES`
- `Bi_Diffusion_graph`

For chi regression, the default split is grouped by `SMILES`, so all rows for the same polymer structure across different `temperature` and `phi` values stay in the same train/validation/test partition. This avoids relying on incomplete `Polymer` names.

Run from the repository root:

```bash
python water_miscible/scripts/run_five_view_tasks.py --config water_miscible/configs/config_water.yaml
```

For a fast smoke pass:

```bash
python water_miscible/scripts/run_five_view_tasks.py --config water_miscible/configs/config_water.yaml --views smiles --no_tune --max_rows 200
```

Use `--fresh_embeddings` after changing datasets or moving from a capped debug run to a full run.

HPO writes trial logs and best-parameter JSON, but not per-trial model checkpoints. Each completed task/view writes one final `checkpoint.pt`; the script does not load previous task-head checkpoints during HPO or final training.

Outputs are written under `water_miscible/results/` by default.
