"""Smoke tests for backbone save_pretrained config completeness."""

from pathlib import Path


AR_BACKBONES = [
    Path("AR_Transformer_SMILES/src/model/backbone.py"),
    Path("AR_Transformer_SELFIES/src/model/backbone.py"),
    Path("AR_Transformer_Group_SELFIES/src/model/backbone.py"),
]

DIFFUSION_BACKBONES = [
    Path("Bi_Diffusion_SMILES/src/model/backbone.py"),
    Path("Bi_Diffusion_SELFIES/src/model/backbone.py"),
    Path("Bi_Diffusion_Group_SELFIES/src/model/backbone.py"),
    Path("Bi_Diffusion_graph/src/model/backbone.py"),
]

COMMON_KEYS = [
    "'vocab_size': self.vocab_size",
    "'hidden_size': self.hidden_size",
    "'num_layers': self.num_layers",
    "'num_heads': self.num_heads",
    "'ffn_hidden_size': self.ffn_hidden_size",
    "'max_position_embeddings': self.max_position_embeddings",
    "'dropout': self.dropout",
    "'pad_token_id': self.pad_token_id",
]


def test_ar_backbones_save_full_config():
    for file_path in AR_BACKBONES:
        source = file_path.read_text(encoding="utf-8")
        for key in COMMON_KEYS:
            assert key in source, f"Missing {key} in {file_path}"
        assert "'num_diffusion_steps': self.num_diffusion_steps" in source
        assert "'causal': self.causal" in source


def test_diffusion_backbones_save_full_config():
    for file_path in DIFFUSION_BACKBONES:
        source = file_path.read_text(encoding="utf-8")
        for key in COMMON_KEYS:
            assert key in source, f"Missing {key} in {file_path}"
        assert "'num_diffusion_steps': self.num_diffusion_steps" in source
