# Multi-View Foundation

MVF compares polymer representations through shared alignment, retrieval, property prediction, inverse design, interpretability, and chemistry analysis.

The enabled views are SMILES, SMILES-BPE, SELFIES, Group SELFIES, and Graph. F5 compares proposal views under a shared downstream scorer by default; `property_model_mode: all` can instead export committee predictions from all available F3 heads.

## Run

```bash
bash scripts/run_pipeline.sh
```

Run selected stages:

```bash
MVF_STEP_START=5 MVF_STEP_END=7 bash scripts/run_pipeline.sh
```

Submit one staged NREL workflow:

```bash
bash scripts/submit_property_workflow_nrel.sh small
```

Configure properties, proposal views, targets, and paper export in `configs/config.yaml`.
