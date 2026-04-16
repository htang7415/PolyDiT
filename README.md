# PolyDiT

PolyDiT studies inverse polymer design with discrete diffusion models and multi-view foundation analysis.

The core target is polymer p-SMILES with exactly two `*` attachment points. SELFIES-based views use `[I+3]` only as an internal placeholder and convert back to `*` before evaluation.

Main workflows:

- Train and sample one representation-specific diffusion model.
- Train property heads for `Tg`, `Tm`, `Td`, or `Eg`.
- Generate candidates by property target, polymer class, or both.
- Compare representations with the multi-view foundation workflow.

Use `command.md` for run commands and the local pipeline docs for stage-level notes.
