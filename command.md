# command.md

Concise runbook for `PolyDiT`.

## 1) Repo root

```bash
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiT
```

## 2) One-time data split

```bash
python Data/Polymer/split_unlabeled_full_columns.py --overwrite
```

## 3) Baseline methods (Step0-2)

```bash
sbatch scripts/submit_step0_all_nrel.sh
bash scripts/submit_all_5_methods_nrel.sh all
```

## 4) MVF single-scale run (NREL, staged CPU/GPU workflow)

Edit `Multi_View_Foundation/configs/config.yaml` first. In MVF, the recommended workflow is:

- pass only `model_size` on the command line
- keep encoder view, property list, F5/F6/F7 settings, and F8 paper settings in `config.yaml`
- submit all four sizes first
- run `F8` once after the four size workflows finish

```bash
cd Multi_View_Foundation
bash scripts/submit_property_workflow_nrel.sh small
bash scripts/submit_property_workflow_nrel.sh medium
bash scripts/submit_property_workflow_nrel.sh large
bash scripts/submit_property_workflow_nrel.sh xl
cd ..
```

Per scale this submits: `F0,F1,F2` -> `F3 (array over properties)` -> `F4` -> `F5,F6 (per config)` -> `F7 (per config)`.
The current NREL wrapper stages work as:

- `F0` on CPU
- `F1,F2` on GPU
- `F3` as a GPU array job
- `F4` on CPU
- `F5,F6` on GPU
- `F7` on CPU

The default NREL wrapper uses `f3_parallel=4`. If you need a different array cap, pass it as the fifth argument without changing the config-driven MVF settings.
F8 is intentionally not included here.

## 5) MVF all four sizes + dependent F8 (NREL)

```bash
cd Multi_View_Foundation
bash scripts/submit_multisize_with_f8_nrel.sh small,medium,large,xl
cd ..
```

This submits the staged workflow for all four sizes, then submits one CPU `F8` job that waits for the four final `F7` jobs.

## 6) MVF F8 only (run after all four sizes finish)

```bash
cd Multi_View_Foundation
MVF_PAPER_CLEAN=1 bash scripts/submit_f8_nrel.sh
cd ..
```

Run this only after `small`, `medium`, `large`, and `xl` MVF workflows have all finished successfully.
`F8` reads its paper-package settings from `config.yaml`.

## 7) MVF F5-F7 only (NREL, optional partial rerun)

```bash
cd Multi_View_Foundation
MVF_STEP_START=5 MVF_STEP_END=7 MVF_EXPORT_PAPER_PACKAGE=0 bash scripts/submit_nrel.sh small
cd ..
```

Compatibility shortcut (auto-routes to the same CPU F8 submit path):

```bash
cd Multi_View_Foundation
MVF_STEP_START=8 MVF_STEP_END=8 MVF_PAPER_CLEAN=1 bash scripts/submit_nrel.sh
cd ..
```

Optional local fallback:

```bash
cd Multi_View_Foundation
python scripts/step8_build_paper_package.py --config configs/config.yaml --clean
cd ..
```

## 8) Optional MVF overrides

- `MVF_F5_RUN_ALL_PROPERTIES=1`
- `MVF_F6_RUN_ALL_PROPERTIES=1`
- `MVF_PROPERTY_LIST=Tg,Tm,Td,Eg`
- `MVF_F5_PROPOSAL_VIEWS=smiles`
- `MVF_EXPORT_PAPER_PACKAGE=0`

Use these only when you intentionally want to override `config.yaml`.

## 9) water_miscible five-view run

One argument: `model_size` (`small`, `medium`, `large`, or `xl`). Requires matching trained backbones under `Bi_Diffusion_*/results_<model_size>/`.

Local separate submissions:

```bash
bash water_miscible/scripts/submit_local_chi.sh small
bash water_miscible/scripts/submit_local_water.sh small
```

Show local output in terminal:

```bash
WM_LOCAL_FOREGROUND=1 bash water_miscible/scripts/submit_local_chi.sh small
WM_LOCAL_FOREGROUND=1 bash water_miscible/scripts/submit_local_water.sh small
```

Local debug:

```bash
bash water_miscible/scripts/submit_local_chi.sh small --no_tune --max_rows 800
bash water_miscible/scripts/submit_local_water.sh small --no_tune --max_rows 800
```

Euler/NREL:

```bash
bash water_miscible/scripts/submit_euler.sh small
bash water_miscible/scripts/submit_nrel.sh small
```

Local outputs/logs go to `water_miscible/results_<model_size>/` and `water_miscible/logs/local_*`.
Euler/NREL submit 17 jobs: 5 shared embedding precompute jobs, `5 views x 2 tasks` train jobs, then one chi postprocess job and one water postprocess job.
NREL walltime: precompute/train jobs use 24h; postprocess jobs use 2h.

## 10) Key outputs

```text
Multi_View_Foundation/results_*/step3_property/
Multi_View_Foundation/results_*/step4_embedding_research/
Multi_View_Foundation/results_*/step5_foundation_inverse/files/candidate_scores_<PROP>.csv
Multi_View_Foundation/results_*/step6_dit_interpretability/files/dit_token_summary_<PROP>.csv
Multi_View_Foundation/results_*/paper_package/manuscript/figures/Figure_1.png ... Figure_6.png
Multi_View_Foundation/results_*/paper_package/supporting_information/figures/Figure_S1.png ... Figure_S9.png
Multi_View_Foundation/results_*/paper_package/*/captions/figure_captions.txt
```
