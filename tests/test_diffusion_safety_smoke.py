"""Smoke tests for diffusion schedule/loss safety guards."""

from pathlib import Path


DISCRETE_DIFFUSION_FILES = [
    Path("Bi_Diffusion_SMILES/src/model/diffusion.py"),
    Path("Bi_Diffusion_SELFIES/src/model/diffusion.py"),
    Path("Bi_Diffusion_Group_SELFIES/src/model/diffusion.py"),
    Path("Bi_Diffusion_graph/src/model/diffusion.py"),
]

GRAPH_DIFFUSION_FILE = Path("Bi_Diffusion_graph/src/model/graph_diffusion.py")


def test_discrete_diffusion_has_clean_t0_gate_and_zero_loss_guard():
    for file_path in DISCRETE_DIFFUSION_FILES:
        source = file_path.read_text(encoding="utf-8")
        assert "force_clean_t0: bool = False" in source, f"Missing force_clean_t0 flag in {file_path}"
        assert "if self.force_clean_t0:" in source, f"Missing schedule t=0 gate in {file_path}"
        assert "if valid_count.item() == 0:" in source, f"Missing empty-mask guard in {file_path}"
        assert "return logits.sum() * 0.0" in source, f"Missing connected zero-loss return in {file_path}"


def test_graph_diffusion_has_clean_t0_gate_and_zero_loss_guard():
    source = GRAPH_DIFFUSION_FILE.read_text(encoding="utf-8")
    assert "force_clean_t0: bool = False" in source
    assert "if self.force_clean_t0:" in source
    assert source.count("return logits.sum() * 0.0") >= 2
