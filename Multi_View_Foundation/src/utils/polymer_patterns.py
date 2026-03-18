"""Shared polymer SMARTS patterns used across MVF analysis steps."""

from __future__ import annotations


POLYMER_SMARTS_PATTERNS = {
    "polyimide": "[#6][CX3](=[OX1])[NX3][CX3](=[OX1])[#6]",
    "polyester": "[#6][CX3](=[OX1])[OX2][#6]",
    "polyamide": "[#6][CX3](=[OX1])[NX3;!$([N]([C](=O))[C](=O))][#6;!$([CX3](=[OX1]))]",
    "polyurethane": "[#6][OX2][CX3](=[OX1])[NX3][#6]",
    "polyether": "[#6;!$([CX3](=[OX1]))][OX2][#6;!$([CX3](=[OX1]))]",
    "polysiloxane": "[Si][OX2][Si]",
    "polycarbonate": "[#6][OX2][CX3](=[OX1])[OX2][#6]",
    "polysulfone": "[#6][SX4](=[OX1])(=[OX1])[#6]",
    "polyacrylate": "[#6]-[#6](=O)-[#8]",
    "polystyrene": "[#6]-[#6](c1ccccc1)-[#6]",
}

# Backward-compatible aliases for older step-local names.
POLYMER_CLASS_PATTERNS = dict(POLYMER_SMARTS_PATTERNS)
POLYMER_MOTIF_SMARTS = dict(POLYMER_SMARTS_PATTERNS)
