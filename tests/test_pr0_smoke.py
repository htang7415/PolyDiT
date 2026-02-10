"""Minimal smoke checks for early hotfix regressions."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_graph_step2_sampling_timer_initialized_before_use():
    source = _read("Bi_Diffusion_graph/scripts/step2_sample_and_evaluate.py")
    init_stmt = "sampling_start = time.time()"
    use_stmt = "sampling_time_sec = time.time() - sampling_start"

    assert init_stmt in source
    assert use_stmt in source
    assert source.index(init_stmt) < source.index(use_stmt)


def test_graph_step0_roundtrip_match_is_strict():
    source = _read("Bi_Diffusion_graph/scripts/step0_prepare_data.py")

    assert "or reconstructed is not None" not in source
    assert "'match': smiles == reconstructed" in source
