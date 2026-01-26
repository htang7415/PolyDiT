# Discrete-Diffusion-Model-for-inverse-polymer-design

This repo implements inverse polymer design across multiple representations with two model families:

- Bi_Diffusion_*: bidirectional discrete masking diffusion
- AR_Transformer_*: causal autoregressive baselines

Representations supported: p-SMILES, SELFIES, Group-SELFIES, Graph. Targets are polymer p-SMILES with exactly two "*" attachment points (SELFIES uses [I+3]).

Project goal (from Outline.md): build a multi-view polymer foundation model that aligns representations, then run a controlled generation benchmark (diffusion vs AR) and inverse design with consistent metrics.

Docs to use first:
- Outline.md (paper-level research plan)
- technical_report.md (current pipeline details + roadmap)
- PROJECT_PLAN.md (repo modifications, step outputs, standardized metrics)
- command.md (Euler and NREL submission commands)

Repo layout:
- Data/: shared raw datasets
- Bi_Diffusion_* and AR_Transformer_*: per-representation pipelines with configs/, scripts/, src/, results/
- Multi_View_Foundation/: multi-view alignment, retrieval, OOD, and reranking (kept independent)
- shared/: cross-method docs and utilities (canonicalization policy, metrics helpers)
- scripts/aggregate_metrics.py: repo-level metrics merger

Standard pipeline steps (per subproject):
Step0 data prep and tokenizer, Step1 backbone training, Step2 sampling and eval, Step3 property head, Step4 inverse design, Step5 class design, Step6 class+property, Step7 tuning.

For consistent cross-method comparisons, follow PROJECT_PLAN.md and shared/canonicalization_policy.md.
