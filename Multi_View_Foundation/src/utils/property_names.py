"""Shared property-name helpers for MVF.

Canonical internal property tokens use the short names:
  Tg, Tm, Td, Eg, Ced, Ea, Eib, In

The helpers below keep backward compatibility with older long-form names
such as cohesive_energy_density and ionization_energy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


PROPERTY_ORDER = ["Tg", "Tm", "Td", "Eg", "Ced", "Ea", "Eib", "In"]

_PROPERTY_ALIAS_TABLE = {
    "Tg": ["Tg"],
    "Tm": ["Tm"],
    "Td": ["Td"],
    "Eg": ["Eg"],
    "Ced": ["Ced", "CED", "cohesive_energy_density"],
    "Ea": ["Ea", "EA", "electron_affinity"],
    "Eib": ["Eib", "EIB", "electron_injection_barrier"],
    "In": ["In", "IE", "ionization_energy"],
}

_ALIAS_TO_CANONICAL = {
    alias.strip().lower(): canonical
    for canonical, aliases in _PROPERTY_ALIAS_TABLE.items()
    for alias in aliases
}

_DISPLAY_LABELS = {
    "Tg": "Tg",
    "Tm": "Tm",
    "Td": "Td",
    "Eg": "Eg",
    "Ced": "CED",
    "Ea": "EA",
    "Eib": "EIB",
    "In": "IE",
}


def normalize_property_name(value) -> str:
    text = str(value).strip()
    if not text:
        return ""
    path = Path(text)
    if path.suffix.lower() == ".csv":
        text = path.stem
    text = text.strip()
    canonical = _ALIAS_TO_CANONICAL.get(text.lower())
    return canonical if canonical is not None else text


def property_display_name(value) -> str:
    canonical = normalize_property_name(value)
    if not canonical:
        return ""
    return _DISPLAY_LABELS.get(canonical, canonical)


def property_aliases(value) -> List[str]:
    canonical = normalize_property_name(value)
    if not canonical:
        return []
    aliases = list(_PROPERTY_ALIAS_TABLE.get(canonical, [canonical]))
    if canonical not in aliases:
        aliases.insert(0, canonical)
    return aliases


def property_file_candidates(value) -> List[str]:
    names: List[str] = []
    for alias in property_aliases(value):
        candidate = f"{alias}.csv"
        if candidate not in names:
            names.append(candidate)
    return names


def property_column_candidates(value) -> List[str]:
    names: List[str] = []
    for alias in property_aliases(value):
        if alias not in names:
            names.append(alias)
    return names


def ordered_properties(values: Iterable[str]) -> List[str]:
    normalized = [normalize_property_name(v) for v in values]
    seen = set()
    ordered: List[str] = []
    for prop in PROPERTY_ORDER:
        if prop in normalized and prop not in seen:
            ordered.append(prop)
            seen.add(prop)
    for prop in normalized:
        if prop and prop not in seen:
            ordered.append(prop)
            seen.add(prop)
    return ordered
