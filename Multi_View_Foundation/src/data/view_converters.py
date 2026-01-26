"""View conversion utilities for SMILES/SELFIES/Group-SELFIES/Graph."""

from typing import Optional

import selfies as sf

PLACEHOLDER_SMILES = "[I+3]"


def smiles_to_selfies(psmiles: str) -> Optional[str]:
    if psmiles is None or not isinstance(psmiles, str):
        return None
    try:
        smiles_with_placeholder = psmiles.replace("*", PLACEHOLDER_SMILES)
        selfies = sf.encoder(smiles_with_placeholder)
        return selfies
    except Exception:
        return None


def smiles_to_group_selfies(psmiles: str, tokenizer=None) -> Optional[str]:
    if tokenizer is None:
        return None
    if psmiles is None or not isinstance(psmiles, str):
        return None
    try:
        tokens = tokenizer.tokenize(psmiles)
        if not tokens or tokens == ['[UNK]']:
            return None
        if hasattr(tokenizer, "_tokens_to_gsf_string"):
            return tokenizer._tokens_to_gsf_string(tokens)
        return "".join(tokens)
    except Exception:
        return None


def smiles_to_graph(psmiles: str, graph_tokenizer=None):
    if graph_tokenizer is None:
        return None
    try:
        return graph_tokenizer.encode(psmiles)
    except Exception:
        return None
