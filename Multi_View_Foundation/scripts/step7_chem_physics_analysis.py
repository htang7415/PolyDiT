#!/usr/bin/env python
"""F7: Chemistry + physics analysis for OOD-aware inverse design results.

This step summarizes science-facing evidence across configured properties:
- descriptor shifts (reference vs F5 candidates vs F6 top-k)
- polymer motif enrichment
- physics-consistency checks from predefined heuristic rules
- nearest-neighbor explanations against property reference datasets
- paper-style figures (per-property + cross-property summary)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.utils.output_layout import ensure_step_dirs, save_csv, save_json
from src.utils.visualization import (
    COLOR_MUTED as SHARED_COLOR_MUTED,
    COLOR_TEXT as SHARED_COLOR_TEXT,
    NATURE_PALETTE as SHARED_NATURE_PALETTE,
    PUBLICATION_STYLE as SHARED_PUBLICATION_STYLE,
    normalize_view_name,
    ordered_views,
    save_figure_png as shared_save_figure_png,
    view_color,
    view_label,
)

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None

try:  # pragma: no cover
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    rdBase.DisableLog("rdApp.*")
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None
    Descriptors = None
    rdMolDescriptors = None


DEFAULT_PROPERTIES = ["Tg"]

DESCRIPTOR_COLUMNS = [
    "mol_wt",
    "heavy_atoms",
    "ring_count",
    "aromatic_ring_count",
    "aromatic_atom_fraction",
    "fraction_csp3",
    "rotatable_bonds",
    "tpsa",
    "logp",
    "hba",
    "hbd",
    "hetero_atom_fraction",
    "bertz_ct",
    "sa_score",
    "star_count",
]

POLYMER_MOTIF_SMARTS = {
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

# Heuristic direction is defined for the property value itself:
# +1 means the feature tends to increase the property, -1 means decrease.
# For target_mode='le' objectives, the expected sign is flipped automatically.
PHYSICS_RULES = {
    "Tg": [
        ("aromatic_ring_count", +1, "Higher chain rigidity tends to increase Tg."),
        ("ring_count", +1, "More cyclic content can reduce segmental mobility."),
        ("rotatable_bonds", -1, "Fewer rotatable bonds generally increase Tg."),
        ("fraction_csp3", -1, "Lower aliphatic flexibility often correlates with higher Tg."),
        ("polyimide", +1, "Imide-like motifs are commonly associated with high thermal performance."),
    ],
    "Tm": [
        ("ring_count", +1, "More rigid repeat units can support higher melting transitions."),
        ("rotatable_bonds", -1, "Reduced flexibility can improve ordered packing."),
        ("aromatic_ring_count", +1, "Aromatic/rigid motifs often raise thermal transitions."),
        ("polyamide", +1, "Hydrogen-bonding motifs may support higher Tm."),
    ],
    "Td": [
        ("aromatic_ring_count", +1, "Aromatic stabilization is often linked to higher Td."),
        ("ring_count", +1, "More rigid structures can improve thermal decomposition resistance."),
        ("rotatable_bonds", -1, "Excess flexibility can reduce thermal robustness."),
        ("polysulfone", +1, "Sulfone-containing motifs are often thermally stable."),
    ],
    "Eg": [
        ("aromatic_ring_count", -1, "For bandgap-like Eg, stronger conjugation often lowers Eg."),
        ("fraction_csp3", +1, "Less conjugated/aliphatic character can increase Eg."),
        ("ring_count", -1, "Extensive aromatic cyclic systems can reduce Eg."),
        ("polystyrene", -1, "Aromatic-rich motifs may correspond to lower Eg."),
    ],
    "cohesive_energy_density": [
        ("aromatic_ring_count", +1, "Aromatic packing interactions can increase cohesive energy density."),
        ("ring_count", +1, "Rigid cyclic units can improve chain packing and intermolecular cohesion."),
        ("tpsa", +1, "Higher polarity can strengthen intermolecular attraction."),
        ("rotatable_bonds", -1, "Reduced flexibility can support denser packing."),
    ],
    "electron_affinity": [
        ("aromatic_ring_count", +1, "Conjugated aromatic systems can stabilize an added electron."),
        ("hetero_atom_fraction", +1, "Hetero atoms can increase electron-withdrawing character."),
        ("fraction_csp3", -1, "Higher saturation generally weakens electron-accepting character."),
        ("tpsa", +1, "Polar functionality can correlate with stronger electron-acceptor behavior."),
    ],
    "electron_injection_barrier": [
        ("aromatic_ring_count", -1, "Greater conjugation can reduce electron injection barriers."),
        ("fraction_csp3", +1, "More saturated backbones can increase injection barriers."),
        ("hetero_atom_fraction", -1, "Electron-withdrawing hetero atoms can lower injection barriers."),
    ],
    "ionization_energy": [
        ("aromatic_ring_count", -1, "Extended conjugation can reduce ionization energy."),
        ("fraction_csp3", +1, "More saturated backbones can increase ionization energy."),
        ("hetero_atom_fraction", +1, "Polar/hetero-atom-rich motifs can increase ionization energy."),
    ],
}

PUBLICATION_STYLE = {
    **SHARED_PUBLICATION_STYLE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": SHARED_PUBLICATION_STYLE.get("axes.linewidth", 0.9),
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
}

NATURE_PALETTE = list(SHARED_NATURE_PALETTE)
COLOR_PRIMARY = NATURE_PALETTE[3]
COLOR_SECONDARY = NATURE_PALETTE[0]
COLOR_ACCENT = NATURE_PALETTE[2]
COLOR_TERTIARY = NATURE_PALETTE[1]
COLOR_QUATERNARY = NATURE_PALETTE[5]
COLOR_MUTED = SHARED_COLOR_MUTED
COLOR_TEXT = SHARED_COLOR_TEXT

if plt is not None:  # pragma: no branch
    PUBLICATION_STYLE["axes.prop_cycle"] = matplotlib.cycler(color=NATURE_PALETTE)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid integer value.")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _to_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid float value.")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _save_figure_png(fig, output_base: Path) -> None:
    shared_save_figure_png(fig, output_base, font_size=16, dpi=600, legend_loc="best")


def _nature_sequential_cmap():
    if plt is None:
        return None
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "mvf_nature_seq",
        ["#F7FBFF", COLOR_TERTIARY, COLOR_PRIMARY],
    )


def _nature_diverging_cmap():
    if plt is None:
        return None
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "mvf_nature_div",
        [COLOR_PRIMARY, "#F7F7F7", COLOR_SECONDARY],
    )


def _make_panel_figure(
    nrows: int,
    ncols: int,
    *,
    panel_width: float = 6.6,
    panel_height: float = 5.6,
):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * max(ncols, 1), panel_height * max(nrows, 1)),
        constrained_layout=True,
        squeeze=False,
    )
    return fig, axes


def _wrap_ticklabels(ax, axis: str = "x", width: int = 16, rotation: int = 32) -> None:
    if axis == "x":
        ticks = ax.get_xticklabels()
    else:
        ticks = ax.get_yticklabels()
    updated = []
    needs_rotation = False
    for tick in ticks:
        text = str(tick.get_text())
        if not text:
            updated.append(text)
            continue
        words = text.replace("_", " ").split()
        lines: list[str] = []
        current = ""
        for word in words:
            proposal = word if not current else f"{current} {word}"
            if len(proposal) <= width:
                current = proposal
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        wrapped = "\n".join(lines) if lines else text
        if "\n" in wrapped or len(text) > width:
            needs_rotation = True
        updated.append(wrapped)
    if axis == "x":
        ax.set_xticklabels(updated, rotation=rotation if needs_rotation else 0, ha="right" if needs_rotation else "center")
    else:
        ax.set_yticklabels(updated)


def _annotate_scatter_subset(
    ax,
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    label_col: str,
    max_labels: int = 6,
) -> None:
    if df.empty:
        return
    label_df = df.copy().head(max_labels)
    offsets = [(8, 6), (8, -10), (-10, 8), (-10, -12), (10, 12), (-12, 12)]
    for idx, (_, row) in enumerate(label_df.iterrows()):
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(
            str(row[label_col]),
            (float(row[x_col]), float(row[y_col])),
            xytext=(dx, dy),
            textcoords="offset points",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 1.5},
            color=COLOR_TEXT,
        )


def _numeric_array(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float32)
    return arr[np.isfinite(arr)]


def _normalize_property_name(value) -> str:
    text = str(value).strip()
    if not text:
        return ""
    p = Path(text)
    if p.suffix.lower() == ".csv":
        text = p.stem
    return text.strip()


def _parse_properties(args, cfg: dict) -> List[str]:
    raw = args.properties if args.properties else cfg.get("properties", DEFAULT_PROPERTIES)
    if isinstance(raw, str):
        props = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        props = [str(x).strip() for x in raw if str(x).strip()]
    if not props:
        return list(DEFAULT_PROPERTIES)
    names = []
    for prop in props:
        name = _normalize_property_name(prop)
        if name and name not in names:
            names.append(name)
    return names or list(DEFAULT_PROPERTIES)


def _lookup_property_setting(mapping: dict, property_name: str):
    if not isinstance(mapping, dict):
        return None
    key = _normalize_property_name(property_name)
    if key in mapping:
        return mapping[key]
    target = key.lower()
    for k, v in mapping.items():
        if _normalize_property_name(k).lower() == target:
            return v
    return None

def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sa_score_fn():
    chem_path = REPO_ROOT / "Bi_Diffusion_SMILES" / "src" / "utils" / "chemistry.py"
    if not chem_path.exists():
        return None
    try:
        mod = _load_module("mvf_chemistry_utils", chem_path)
    except Exception:
        return None
    fn = getattr(mod, "compute_sa_score", None)
    return fn if callable(fn) else None


def _compile_motifs() -> Dict[str, object]:
    if Chem is None:
        return {}
    compiled = {}
    for name, smarts in POLYMER_MOTIF_SMARTS.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                compiled[name] = patt
        except Exception:
            continue
    return compiled


def _smiles_to_mol(smiles: str):
    if Chem is None:
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text.replace("*", "[H]"))
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    y_std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    pooled = np.sqrt(max((x_std * x_std + y_std * y_std) / 2.0, 1e-12))
    return float((x_mean - y_mean) / pooled)


def _prediction_uncertainty_series(df: pd.DataFrame, property_name: str) -> pd.Series:
    prop = _normalize_property_name(property_name)
    candidates = [
        "prediction_uncertainty",
        f"pred_{prop}_std",
        "prediction_std",
    ]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _resolve_property_columns(df: pd.DataFrame, property_name: str) -> Tuple[str, str]:
    smiles_candidates = ["SMILES", "smiles", "p_smiles", "psmiles"]
    smiles_col = next((c for c in smiles_candidates if c in df.columns), None)
    if smiles_col is None:
        raise ValueError("Property CSV must contain a SMILES column.")

    prop_lower = str(property_name).strip().lower()
    value_col = None
    for col in df.columns:
        if col.lower() == prop_lower:
            value_col = col
            break
    if value_col is None:
        for col in df.columns:
            c = col.lower()
            if c in {"smiles", "p_smiles", "psmiles", "pid", "polymer_id", "id"}:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break
    if value_col is None:
        raise ValueError(f"Could not determine property value column for {property_name}.")
    return smiles_col, value_col


def _load_property_reference(property_dir: Path, property_name: str, max_samples: Optional[int]) -> pd.DataFrame:
    path = property_dir / f"{property_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing property file: {path}")
    df = pd.read_csv(path)
    smiles_col, value_col = _resolve_property_columns(df, property_name)
    out = df[[smiles_col, value_col]].copy()
    out = out.rename(columns={smiles_col: "smiles", value_col: "property_value"})
    out["property_value"] = pd.to_numeric(out["property_value"], errors="coerce")
    out = out.dropna(subset=["smiles", "property_value"])
    out["smiles"] = out["smiles"].astype(str)
    if max_samples is not None and max_samples > 0:
        out = out.head(int(max_samples)).copy()
    out["property"] = property_name
    return out


def _load_json_dict(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_neighbor_run_meta(artifact_path: Optional[Path], property_name: str) -> Tuple[dict, str]:
    if artifact_path is None:
        return {}, ""
    prop = _normalize_property_name(property_name)
    candidates = [
        artifact_path.parent / f"run_meta_{prop}.json",
        artifact_path.parent / "run_meta.json",
    ]
    for candidate in candidates:
        payload = _load_json_dict(candidate)
        if payload:
            return payload, str(candidate)
    return {}, ""


def _stringify_list_like(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        items = [str(x).strip() for x in value if str(x).strip()]
        return ",".join(items)
    text = str(value).strip()
    return text


def _stringify_jsonish(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True)
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple, set)):
        try:
            return json.dumps([str(x) for x in value])
        except Exception:
            return _stringify_list_like(value)
    return str(value)


def _boolish_rate(values) -> float:
    series = pd.Series(values)
    if series.empty:
        return np.nan
    if pd.api.types.is_bool_dtype(series):
        numeric = series.astype(float)
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        if not numeric.notna().any():
            text = series.astype(str).str.strip().str.lower()
            numeric = pd.to_numeric(
                text.map(
                    {
                        "true": 1.0,
                        "false": 0.0,
                        "yes": 1.0,
                        "no": 0.0,
                        "1": 1.0,
                        "0": 0.0,
                    }
                ),
                errors="coerce",
            )
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def _series_mean(values) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.mean()) if series.notna().any() else np.nan


def _smiles_unique_rate(df: pd.DataFrame) -> float:
    if df.empty or "smiles" not in df.columns:
        return np.nan
    smiles = df["smiles"].astype(str).str.strip()
    if smiles.empty:
        return np.nan
    return float(smiles.nunique(dropna=True) / max(len(smiles), 1))


def _smiles_two_star_rate(df: pd.DataFrame) -> float:
    if df.empty or "smiles" not in df.columns:
        return np.nan
    return float(df["smiles"].astype(str).map(lambda x: float(str(x).count("*") == 2)).mean())


def _validity_rate(df: pd.DataFrame, desc_df: pd.DataFrame) -> float:
    if "is_valid" in df.columns:
        return _boolish_rate(df["is_valid"])
    if df.empty:
        return np.nan
    return float(len(desc_df) / max(len(df), 1))


def _sa_series(df: pd.DataFrame, desc_df: pd.DataFrame) -> pd.Series:
    if "sa_score" in df.columns:
        series = pd.to_numeric(df["sa_score"], errors="coerce")
        if series.notna().any():
            return series
    if "sa_score" in desc_df.columns:
        series = pd.to_numeric(desc_df["sa_score"], errors="coerce")
        if series.notna().any():
            return series
    return pd.Series(dtype=float)


def _sa_lt_rate(df: pd.DataFrame, desc_df: pd.DataFrame, max_sa: Optional[float]) -> float:
    if max_sa is None:
        return np.nan
    series = _sa_series(df, desc_df)
    if not series.notna().any():
        return np.nan
    return float((series < float(max_sa)).mean())


def _build_design_filter_audit_row(
    *,
    property_name: str,
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    cand_desc: pd.DataFrame,
    top_desc: pd.DataFrame,
    cand_path: Optional[Path],
    topk_path: Optional[Path],
    f5_run_meta: dict,
    f5_run_meta_path: str,
    f6_run_meta: dict,
    f6_run_meta_path: str,
    cfg_f5: dict,
    cfg_f6: dict,
    reference_class_filter_requested: bool,
    reference_class_filter_applied: bool,
    reference_class_filter: str,
) -> dict:
    prop_name = _normalize_property_name(property_name)
    f5_cfg_property = _normalize_property_name(cfg_f5.get("property", ""))
    f6_cfg_property = _normalize_property_name(cfg_f6.get("property", ""))
    f5_cfg_active = bool(f5_cfg_property) and f5_cfg_property.lower() == prop_name.lower()
    f6_cfg_active = bool(f6_cfg_property) and f6_cfg_property.lower() == prop_name.lower()
    f5_meta_property = _normalize_property_name(f5_run_meta.get("property", ""))
    f6_meta_property = _normalize_property_name(f6_run_meta.get("property", ""))
    f5_meta_active = bool(f5_meta_property) and f5_meta_property.lower() == prop_name.lower()
    f6_meta_active = bool(f6_meta_property) and f6_meta_property.lower() == prop_name.lower()

    f5_source = "run_meta" if f5_meta_active else ("config" if f5_cfg_active else "")
    f6_source = "run_meta" if f6_meta_active else ("config" if f6_cfg_active else "")

    if f5_meta_active:
        f5_target_classes = f5_run_meta.get("target_classes", [])
        f5_target_class = _stringify_list_like(f5_target_classes)
        f5_proposal_views = _stringify_list_like(f5_run_meta.get("proposal_views"))
        f5_encoder_view = str(f5_run_meta.get("encoder_view", "")).strip()
        f5_require_validity = _to_bool(f5_run_meta.get("require_validity"), True)
        f5_require_two_stars = _to_bool(f5_run_meta.get("require_two_stars"), True)
        f5_require_novel = _to_bool(f5_run_meta.get("require_novel"), True)
        f5_require_unique = _to_bool(f5_run_meta.get("require_unique"), True)
        f5_max_sa = _to_float_or_none(f5_run_meta.get("max_sa"))
        f5_rerank_strategy = str(f5_run_meta.get("rerank_strategy", "")).strip()
        f5_property_model_mode = str(f5_run_meta.get("property_model_mode", "")).strip()
    elif f5_cfg_active:
        f5_target_class = str(cfg_f5.get("target_class", "")).strip()
        f5_proposal_views = _stringify_list_like(cfg_f5.get("proposal_views"))
        f5_encoder_view = str(cfg_f5.get("encoder_view", "")).strip()
        f5_require_validity = _to_bool(cfg_f5.get("require_validity", True), True)
        f5_require_two_stars = _to_bool(cfg_f5.get("require_two_stars", True), True)
        f5_require_novel = _to_bool(cfg_f5.get("require_novel", True), True)
        f5_require_unique = _to_bool(cfg_f5.get("require_unique", True), True)
        f5_max_sa = _to_float_or_none(cfg_f5.get("max_sa"))
        f5_rerank_strategy = str(cfg_f5.get("rerank_strategy", "")).strip()
        f5_property_model_mode = str(cfg_f5.get("property_model_mode", "")).strip()
    else:
        f5_target_class = ""
        f5_proposal_views = ""
        f5_encoder_view = ""
        f5_require_validity = None
        f5_require_two_stars = None
        f5_require_novel = None
        f5_require_unique = None
        f5_max_sa = np.nan
        f5_rerank_strategy = ""
        f5_property_model_mode = ""

    if f6_meta_active:
        f6_encoder_view = str(f6_run_meta.get("encoder_view", "")).strip()
        f6_ood_views_requested = _stringify_list_like(f6_run_meta.get("ood_views_requested"))
        f6_property_weight = _to_float_or_none(f6_run_meta.get("objective_property_weight"))
        f6_ood_weight = _to_float_or_none(f6_run_meta.get("objective_ood_weight"))
        f6_uncertainty_weight = _to_float_or_none(f6_run_meta.get("objective_uncertainty_weight"))
        f6_constraint_weight = _to_float_or_none(f6_run_meta.get("objective_constraint_weight"))
        f6_sa_weight = _to_float_or_none(f6_run_meta.get("objective_sa_weight"))
        f6_descriptor_weight = _to_float_or_none(f6_run_meta.get("objective_descriptor_weight"))
        f6_constraint_properties = _stringify_list_like(f6_run_meta.get("constraint_properties"))
        f6_descriptor_constraints = _stringify_list_like(f6_run_meta.get("descriptor_constraints"))
        f6_d2_distance_source = str(f6_run_meta.get("d2_distance_source", "")).strip()
    elif f6_cfg_active:
        f6_encoder_view = str(cfg_f6.get("encoder_view", "")).strip()
        f6_ood_views_requested = _stringify_list_like(cfg_f6.get("ood_views"))
        f6_property_weight = _to_float_or_none(cfg_f6.get("property_weight"))
        f6_ood_weight = _to_float_or_none(cfg_f6.get("ood_weight"))
        f6_uncertainty_weight = _to_float_or_none(cfg_f6.get("uncertainty_weight"))
        f6_constraint_weight = _to_float_or_none(cfg_f6.get("constraint_weight"))
        f6_sa_weight = _to_float_or_none(cfg_f6.get("sa_weight"))
        f6_descriptor_weight = _to_float_or_none(cfg_f6.get("descriptor_weight"))
        f6_constraint_properties = _stringify_list_like(cfg_f6.get("constraint_properties"))
        f6_descriptor_constraints = _stringify_jsonish(cfg_f6.get("descriptor_constraints"))
        f6_d2_distance_source = ""
    else:
        f6_encoder_view = ""
        f6_ood_views_requested = ""
        f6_property_weight = np.nan
        f6_ood_weight = np.nan
        f6_uncertainty_weight = np.nan
        f6_constraint_weight = np.nan
        f6_sa_weight = np.nan
        f6_descriptor_weight = np.nan
        f6_constraint_properties = ""
        f6_descriptor_constraints = ""
        f6_d2_distance_source = ""

    cand_mean_sa = _series_mean(_sa_series(candidate_df, cand_desc))
    top_mean_sa = _series_mean(_sa_series(topk_df, top_desc))
    cand_valid_rate = _validity_rate(candidate_df, cand_desc)
    top_valid_rate = _validity_rate(topk_df, top_desc)
    cand_two_star_rate = _boolish_rate(candidate_df["is_two_star"]) if "is_two_star" in candidate_df.columns else _smiles_two_star_rate(candidate_df)
    top_two_star_rate = _boolish_rate(topk_df["is_two_star"]) if "is_two_star" in topk_df.columns else _smiles_two_star_rate(topk_df)
    cand_unique_rate = _smiles_unique_rate(candidate_df)
    top_unique_rate = _smiles_unique_rate(topk_df)
    cand_novel_rate = _boolish_rate(candidate_df["is_novel"]) if "is_novel" in candidate_df.columns else np.nan
    top_novel_rate = _boolish_rate(topk_df["is_novel"]) if "is_novel" in topk_df.columns else np.nan
    cand_sa_lt_rate = _sa_lt_rate(candidate_df, cand_desc, f5_max_sa)
    top_sa_lt_rate = _sa_lt_rate(topk_df, top_desc, f5_max_sa)

    return {
        "property": property_name,
        "candidate_scores_path": str(cand_path) if cand_path else "",
        "topk_scores_path": str(topk_path) if topk_path else "",
        "f5_run_meta_path": f5_run_meta_path,
        "f6_run_meta_path": f6_run_meta_path,
        "f5_policy_source": f5_source,
        "f6_policy_source": f6_source,
        "reference_class_filter_requested": bool(reference_class_filter_requested),
        "reference_class_filter_applied": bool(reference_class_filter_applied),
        "reference_class_filter": reference_class_filter or "",
        "n_candidates": int(len(candidate_df)),
        "n_topk": int(len(topk_df)),
        "candidate_valid_rate": round(cand_valid_rate, 4) if np.isfinite(cand_valid_rate) else np.nan,
        "topk_valid_rate": round(top_valid_rate, 4) if np.isfinite(top_valid_rate) else np.nan,
        "candidate_two_star_rate": round(cand_two_star_rate, 4) if np.isfinite(cand_two_star_rate) else np.nan,
        "topk_two_star_rate": round(top_two_star_rate, 4) if np.isfinite(top_two_star_rate) else np.nan,
        "candidate_unique_smiles_rate": round(cand_unique_rate, 4) if np.isfinite(cand_unique_rate) else np.nan,
        "topk_unique_smiles_rate": round(top_unique_rate, 4) if np.isfinite(top_unique_rate) else np.nan,
        "candidate_novel_rate": round(cand_novel_rate, 4) if np.isfinite(cand_novel_rate) else np.nan,
        "topk_novel_rate": round(top_novel_rate, 4) if np.isfinite(top_novel_rate) else np.nan,
        "candidate_mean_sa": round(cand_mean_sa, 4) if np.isfinite(cand_mean_sa) else np.nan,
        "topk_mean_sa": round(top_mean_sa, 4) if np.isfinite(top_mean_sa) else np.nan,
        "candidate_sa_lt_max_rate": round(cand_sa_lt_rate, 4) if np.isfinite(cand_sa_lt_rate) else np.nan,
        "topk_sa_lt_max_rate": round(top_sa_lt_rate, 4) if np.isfinite(top_sa_lt_rate) else np.nan,
        "f5_encoder_view": f5_encoder_view,
        "f5_proposal_views": f5_proposal_views,
        "f5_require_validity": f5_require_validity,
        "f5_require_two_stars": f5_require_two_stars,
        "f5_require_novel": f5_require_novel,
        "f5_require_unique": f5_require_unique,
        "f5_target_class": f5_target_class,
        "f5_max_sa": f5_max_sa,
        "f5_rerank_strategy": f5_rerank_strategy,
        "f5_property_model_mode": f5_property_model_mode,
        "f6_encoder_view": f6_encoder_view,
        "f6_ood_views_requested": f6_ood_views_requested,
        "f6_property_weight": f6_property_weight,
        "f6_ood_weight": f6_ood_weight,
        "f6_uncertainty_weight": f6_uncertainty_weight,
        "f6_constraint_weight": f6_constraint_weight,
        "f6_sa_weight": f6_sa_weight,
        "f6_descriptor_weight": f6_descriptor_weight,
        "f6_constraint_properties": f6_constraint_properties,
        "f6_descriptor_constraints": f6_descriptor_constraints,
        "f6_d2_distance_source": f6_d2_distance_source,
    }


def _resolve_template_path(template: Optional[str], property_name: str) -> Optional[Path]:
    if template is None:
        return None
    text = str(template).strip()
    if not text:
        return None
    try:
        text = text.format(property=property_name)
    except Exception:
        pass
    return _resolve_path(text)


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _default_candidate_paths(results_dir: Path, property_name: str) -> List[Path]:
    return [
        results_dir / "step5_foundation_inverse" / property_name / "files" / f"candidate_scores_{property_name}.csv",
        results_dir / "step5_foundation_inverse" / property_name / "files" / "candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / property_name / f"candidate_scores_{property_name}.csv",
        results_dir / "step5_foundation_inverse" / property_name / "candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / "files" / f"candidate_scores_{property_name}.csv",
        results_dir / "step5_foundation_inverse" / "files" / f"{property_name}_candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / "files" / "candidate_scores.csv",
        results_dir / "step5_foundation_inverse" / f"candidate_scores_{property_name}.csv",
        results_dir / "step5_foundation_inverse" / "candidate_scores.csv",
    ]


def _default_topk_paths(results_dir: Path, property_name: str) -> List[Path]:
    return [
        results_dir / "step6_ood_aware_inverse" / property_name / "files" / f"ood_objective_topk_{property_name}.csv",
        results_dir / "step6_ood_aware_inverse" / property_name / "files" / "ood_objective_topk.csv",
        results_dir / "step6_ood_aware_inverse" / property_name / f"ood_objective_topk_{property_name}.csv",
        results_dir / "step6_ood_aware_inverse" / property_name / "ood_objective_topk.csv",
        results_dir / "step6_ood_aware_inverse" / "files" / f"ood_objective_topk_{property_name}.csv",
        results_dir / "step6_ood_aware_inverse" / "files" / f"{property_name}_ood_objective_topk.csv",
        results_dir / "step6_ood_aware_inverse" / "files" / "ood_objective_topk.csv",
        results_dir / "step6_ood_aware_inverse" / f"ood_objective_topk_{property_name}.csv",
        results_dir / "step6_ood_aware_inverse" / "ood_objective_topk.csv",
    ]


def _resolve_property_file_inputs(
    *,
    property_name: str,
    results_dir: Path,
    candidate_template: Optional[str],
    topk_template: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    candidate_path = _resolve_template_path(candidate_template, property_name)
    if candidate_path is not None and not candidate_path.exists():
        candidate_path = None
    if candidate_path is None:
        candidate_path = _find_first_existing(_default_candidate_paths(results_dir, property_name))

    topk_path = _resolve_template_path(topk_template, property_name)
    if topk_path is not None and not topk_path.exists():
        topk_path = None
    if topk_path is None:
        topk_path = _find_first_existing(_default_topk_paths(results_dir, property_name))
    return candidate_path, topk_path


def _is_generic_scores_file(path: Optional[Path]) -> bool:
    if path is None:
        return False
    return path.name in {"candidate_scores.csv", "ood_objective_scores.csv", "ood_objective_topk.csv"}


def _filter_property_rows_if_available(
    df: pd.DataFrame,
    *,
    property_name: str,
    source_path: Path,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if "property" not in df.columns:
        return df, None
    prop_series = df["property"].astype(str).str.strip()
    match_mask = prop_series == property_name
    if bool(match_mask.any()):
        return df.loc[match_mask].copy(), None
    seen = [x for x in sorted(prop_series.unique().tolist()) if x]
    seen_preview = ",".join(seen[:6]) if seen else "(empty)"
    msg = (
        f"{property_name}: property column mismatch in {source_path}. "
        f"Found properties={seen_preview}."
    )
    return pd.DataFrame(columns=df.columns), msg


def _descriptor_and_motif_rows(
    smiles_list: List[str],
    property_name: str,
    split: str,
    sa_score_fn,
    compiled_motifs: Dict[str, object],
) -> pd.DataFrame:
    rows: List[dict] = []
    for smiles in smiles_list:
        mol = _smiles_to_mol(smiles)
        if mol is None:
            continue
        heavy_atoms = float(mol.GetNumHeavyAtoms())
        aromatic_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()))
        hetero_atoms = float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)))
        row = {
            "property": property_name,
            "split": split,
            "smiles": str(smiles),
            "mol_wt": float(Descriptors.MolWt(mol)) if Descriptors is not None else np.nan,
            "heavy_atoms": heavy_atoms,
            "ring_count": float(Descriptors.RingCount(mol)) if Descriptors is not None else np.nan,
            "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(mol)) if rdMolDescriptors is not None else np.nan,
            "aromatic_atom_fraction": float(aromatic_atoms / max(heavy_atoms, 1.0)),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)) if rdMolDescriptors is not None else np.nan,
            "rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)) if Descriptors is not None else np.nan,
            "tpsa": float(Descriptors.TPSA(mol)) if Descriptors is not None else np.nan,
            "logp": float(Descriptors.MolLogP(mol)) if Descriptors is not None else np.nan,
            "hba": float(Descriptors.NumHAcceptors(mol)) if Descriptors is not None else np.nan,
            "hbd": float(Descriptors.NumHDonors(mol)) if Descriptors is not None else np.nan,
            "hetero_atom_fraction": float(hetero_atoms / max(heavy_atoms, 1.0)),
            "bertz_ct": float(Descriptors.BertzCT(mol)) if Descriptors is not None else np.nan,
            "star_count": float(str(smiles).count("*")),
            "sa_score": np.nan,
        }
        if sa_score_fn is not None:
            try:
                score = sa_score_fn(str(smiles))
                if score is not None:
                    row["sa_score"] = float(score)
            except Exception:
                pass
        for motif_name, patt in compiled_motifs.items():
            key = f"motif_{motif_name}"
            try:
                row[key] = bool(mol.HasSubstructMatch(patt))
            except Exception:
                row[key] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_stats(values: np.ndarray) -> Tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(values)), float(np.std(values)), float(np.median(values))


def _descriptor_shift_table(
    property_name: str,
    ref_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for desc in DESCRIPTOR_COLUMNS:
        x_ref = ref_df[desc].dropna().to_numpy(dtype=np.float32) if desc in ref_df.columns else np.zeros((0,), dtype=np.float32)
        x_cand = cand_df[desc].dropna().to_numpy(dtype=np.float32) if desc in cand_df.columns else np.zeros((0,), dtype=np.float32)
        x_top = topk_df[desc].dropna().to_numpy(dtype=np.float32) if desc in topk_df.columns else np.zeros((0,), dtype=np.float32)

        ref_mean, ref_std, ref_med = _summary_stats(x_ref)
        cand_mean, cand_std, cand_med = _summary_stats(x_cand)
        top_mean, top_std, top_med = _summary_stats(x_top)
        rows.append(
            {
                "property": property_name,
                "descriptor": desc,
                "ref_n": int(x_ref.size),
                "cand_n": int(x_cand.size),
                "topk_n": int(x_top.size),
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "ref_median": ref_med,
                "candidate_mean": cand_mean,
                "candidate_std": cand_std,
                "candidate_median": cand_med,
                "topk_mean": top_mean,
                "topk_std": top_std,
                "topk_median": top_med,
                "delta_topk_vs_ref": (top_mean - ref_mean) if np.isfinite(top_mean) and np.isfinite(ref_mean) else np.nan,
                "delta_topk_vs_candidate": (top_mean - cand_mean) if np.isfinite(top_mean) and np.isfinite(cand_mean) else np.nan,
                "cohens_d_topk_vs_ref": _cohens_d(x_top, x_ref),
                "cohens_d_topk_vs_candidate": _cohens_d(x_top, x_cand),
            }
        )
    return pd.DataFrame(rows)


def _motif_enrichment_table(
    property_name: str,
    ref_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> pd.DataFrame:
    motif_cols = [c for c in ref_df.columns if c.startswith("motif_")]
    rows = []
    eps = 1e-6
    for col in motif_cols:
        ref = ref_df[col].astype(float).to_numpy(dtype=np.float32) if col in ref_df.columns else np.zeros((0,), dtype=np.float32)
        cand = cand_df[col].astype(float).to_numpy(dtype=np.float32) if col in cand_df.columns else np.zeros((0,), dtype=np.float32)
        top = topk_df[col].astype(float).to_numpy(dtype=np.float32) if col in topk_df.columns else np.zeros((0,), dtype=np.float32)
        ref_freq = float(np.mean(ref)) if ref.size else 0.0
        cand_freq = float(np.mean(cand)) if cand.size else 0.0
        top_freq = float(np.mean(top)) if top.size else 0.0
        ratio = (top_freq + eps) / (ref_freq + eps)
        rows.append(
            {
                "property": property_name,
                "motif": col.replace("motif_", ""),
                "ref_freq": ref_freq,
                "candidate_freq": cand_freq,
                "topk_freq": top_freq,
                "delta_freq_topk_vs_ref": top_freq - ref_freq,
                "delta_freq_topk_vs_candidate": top_freq - cand_freq,
                "enrichment_ratio_topk_vs_ref": ratio,
                "log2_enrichment_topk_vs_ref": float(np.log2(ratio)),
            }
        )
    return pd.DataFrame(rows)


def _physics_consistency_table(
    property_name: str,
    descriptor_shift_df: pd.DataFrame,
    motif_enrich_df: pd.DataFrame,
    target_mode: str = "window",
) -> pd.DataFrame:
    rules = PHYSICS_RULES.get(property_name, [])
    rows = []
    mode = str(target_mode).strip().lower()
    direction_multiplier = -1 if mode == "le" else 1
    for feature, expected, rationale in rules:
        if feature in set(descriptor_shift_df["descriptor"].astype(str)):
            row = descriptor_shift_df[descriptor_shift_df["descriptor"] == feature].iloc[0]
            observed = float(row["delta_topk_vs_ref"]) if pd.notna(row["delta_topk_vs_ref"]) else np.nan
            source = "descriptor"
        elif feature in set(motif_enrich_df["motif"].astype(str)):
            row = motif_enrich_df[motif_enrich_df["motif"] == feature].iloc[0]
            observed = float(row["delta_freq_topk_vs_ref"]) if pd.notna(row["delta_freq_topk_vs_ref"]) else np.nan
            source = "motif"
        else:
            observed = np.nan
            source = "missing"
        expected_for_objective = int(expected) * int(direction_multiplier)
        sign_match = bool(observed * expected_for_objective > 0) if np.isfinite(observed) else False
        rows.append(
            {
                "property": property_name,
                "feature": feature,
                "source": source,
                "target_mode": mode,
                "expected_direction_property": int(expected),
                "expected_direction": int(expected_for_objective),
                "observed_delta_topk_vs_ref": observed,
                "sign_match": sign_match,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def _fingerprint(mol):
    if AllChem is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None


def _nearest_neighbor_explanations(
    property_name: str,
    ref_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    ref_desc: pd.DataFrame,
    top_desc: pd.DataFrame,
) -> pd.DataFrame:
    if Chem is None or DataStructs is None:
        return pd.DataFrame()
    if ref_df.empty or topk_df.empty:
        return pd.DataFrame()

    ref_smiles = ref_df["smiles"].astype(str).tolist()
    ref_values = ref_df["property_value"].to_numpy(dtype=np.float32)
    ref_mols = [_smiles_to_mol(s) for s in ref_smiles]
    ref_fps = [_fingerprint(m) if m is not None else None for m in ref_mols]

    valid_ref_idx = [i for i, fp in enumerate(ref_fps) if fp is not None]
    if not valid_ref_idx:
        return pd.DataFrame()
    ref_fps_valid = [ref_fps[i] for i in valid_ref_idx]

    ref_desc_map = {str(r["smiles"]): r for _, r in ref_desc.iterrows()}
    top_desc_map = {str(r["smiles"]): r for _, r in top_desc.iterrows()}

    rows = []
    for _, top_row in topk_df.iterrows():
        top_smiles = str(top_row.get("smiles", ""))
        mol = _smiles_to_mol(top_smiles)
        if mol is None:
            continue
        fp = _fingerprint(mol)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps_valid)
        if not sims:
            continue
        local_idx = int(np.argmax(np.asarray(sims, dtype=np.float32)))
        best_ref_global_idx = valid_ref_idx[local_idx]
        best_ref_smiles = ref_smiles[best_ref_global_idx]
        best_sim = float(sims[local_idx])
        best_ref_value = float(ref_values[best_ref_global_idx])

        top_desc_row = top_desc_map.get(top_smiles)
        ref_desc_row = ref_desc_map.get(best_ref_smiles)
        delta_aromatic = np.nan
        delta_ring = np.nan
        delta_rotatable = np.nan
        delta_fraction_csp3 = np.nan
        delta_mw = np.nan
        delta_tpsa = np.nan
        delta_logp = np.nan
        if top_desc_row is not None and ref_desc_row is not None:
            delta_aromatic = float(top_desc_row.get("aromatic_ring_count", np.nan) - ref_desc_row.get("aromatic_ring_count", np.nan))
            delta_ring = float(top_desc_row.get("ring_count", np.nan) - ref_desc_row.get("ring_count", np.nan))
            delta_rotatable = float(top_desc_row.get("rotatable_bonds", np.nan) - ref_desc_row.get("rotatable_bonds", np.nan))
            delta_fraction_csp3 = float(top_desc_row.get("fraction_csp3", np.nan) - ref_desc_row.get("fraction_csp3", np.nan))
            delta_mw = float(top_desc_row.get("mol_wt", np.nan) - ref_desc_row.get("mol_wt", np.nan))
            delta_tpsa = float(top_desc_row.get("tpsa", np.nan) - ref_desc_row.get("tpsa", np.nan))
            delta_logp = float(top_desc_row.get("logp", np.nan) - ref_desc_row.get("logp", np.nan))

        rows.append(
            {
                "property": property_name,
                "topk_smiles": top_smiles,
                "topk_prediction": pd.to_numeric(top_row.get("prediction"), errors="coerce"),
                "topk_d2_distance": pd.to_numeric(top_row.get("d2_distance"), errors="coerce"),
                "nearest_reference_smiles": best_ref_smiles,
                "nearest_reference_value": best_ref_value,
                "nearest_tanimoto": best_sim,
                "delta_aromatic_ring_count": delta_aromatic,
                "delta_ring_count": delta_ring,
                "delta_rotatable_bonds": delta_rotatable,
                "delta_fraction_csp3": delta_fraction_csp3,
                "delta_mol_wt": delta_mw,
                "delta_tpsa": delta_tpsa,
                "delta_logp": delta_logp,
            }
        )
    return pd.DataFrame(rows)


def _property_error_from_predictions(
    pred: np.ndarray,
    target: Optional[float],
    target_mode: str,
) -> np.ndarray:
    if target is None:
        return np.full_like(pred, np.nan, dtype=np.float32)
    target = float(target)
    mode = str(target_mode).strip().lower()
    if mode == "window":
        return np.abs(pred - target).astype(np.float32, copy=False)
    if mode == "ge":
        return np.maximum(0.0, target - pred).astype(np.float32, copy=False)
    if mode == "le":
        return np.maximum(0.0, pred - target).astype(np.float32, copy=False)
    return np.abs(pred - target).astype(np.float32, copy=False)


def _target_excess_from_predictions(
    pred: np.ndarray,
    target: Optional[float],
    target_mode: str,
) -> np.ndarray:
    if target is None:
        return np.full_like(pred, np.nan, dtype=np.float32)
    target = float(target)
    mode = str(target_mode).strip().lower()
    if mode == "le":
        return (target - pred).astype(np.float32, copy=False)
    return (pred - target).astype(np.float32, copy=False)


def _target_excess_axis_label(property_name: str, target_mode: str) -> str:
    mode = str(target_mode).strip().lower()
    if mode == "ge":
        return f"Target excess (predicted {property_name} - target)"
    if mode == "le":
        return f"Target excess (target - predicted {property_name})"
    return "Signed error (prediction - target)"


def _resolve_target_config(
    property_name: str,
    cfg_step7: dict,
    cfg_f5: dict,
    cfg_f6: Optional[dict] = None,
    f5_run_meta: Optional[dict] = None,
    f6_run_meta: Optional[dict] = None,
) -> Tuple[Optional[float], str, Optional[float]]:
    targets = cfg_step7.get("targets", {}) or {}
    target_modes = cfg_step7.get("target_modes", {}) or {}
    epsilons = cfg_step7.get("epsilons", {}) or {}
    cfg_f6 = cfg_f6 or {}
    f5_run_meta = f5_run_meta or {}
    f6_run_meta = f6_run_meta or {}

    prop_name = _normalize_property_name(property_name)
    prop_name_lower = prop_name.lower()
    f5_property = _normalize_property_name(cfg_f5.get("property", ""))
    f5_property_is_active = f5_property.lower() == prop_name_lower
    f6_property = _normalize_property_name(cfg_f6.get("property", ""))
    f6_property_is_active = f6_property.lower() == prop_name_lower
    f5_run_meta_is_active = _normalize_property_name(f5_run_meta.get("property", "")).lower() == prop_name_lower
    f6_run_meta_is_active = _normalize_property_name(f6_run_meta.get("property", "")).lower() == prop_name_lower

    target = _to_float_or_none(_lookup_property_setting(targets, prop_name))
    if target is None and f6_run_meta_is_active:
        target = _to_float_or_none(f6_run_meta.get("target_value"))
    if target is None and f6_property_is_active:
        target = _to_float_or_none(cfg_f6.get("target"))
    if target is None and f5_run_meta_is_active:
        target = _to_float_or_none(f5_run_meta.get("target_value"))
    if target is None:
        target = _to_float_or_none(_lookup_property_setting(cfg_f5.get("targets", {}) or {}, prop_name))
    if target is None and f5_property_is_active:
        target = _to_float_or_none(cfg_f5.get("target"))

    target_mode = str(_lookup_property_setting(target_modes, prop_name) or "").strip()
    if not target_mode and f6_run_meta_is_active:
        target_mode = str(f6_run_meta.get("target_mode", "")).strip()
    if not target_mode and f6_property_is_active:
        target_mode = str(cfg_f6.get("target_mode", "")).strip()
    if not target_mode and f5_run_meta_is_active:
        target_mode = str(f5_run_meta.get("target_mode", "")).strip()
    if not target_mode:
        target_mode = str(_lookup_property_setting(cfg_f5.get("target_modes", {}) or {}, prop_name) or "").strip()
    if not target_mode and f5_property_is_active:
        target_mode = str(cfg_f5.get("target_mode", "window")).strip()
    if not target_mode:
        target_mode = "window"
    target_mode = target_mode.lower()
    if target_mode not in {"window", "ge", "le"}:
        target_mode = "window"

    epsilon = _to_float_or_none(_lookup_property_setting(epsilons, prop_name))
    if epsilon is None and f6_run_meta_is_active:
        epsilon = _to_float_or_none(f6_run_meta.get("epsilon"))
    if epsilon is None and f6_property_is_active:
        epsilon = _to_float_or_none(cfg_f6.get("epsilon"))
    if epsilon is None and f5_run_meta_is_active:
        epsilon = _to_float_or_none(f5_run_meta.get("epsilon"))
    if epsilon is None:
        epsilon = _to_float_or_none(_lookup_property_setting(cfg_f5.get("epsilons", {}) or {}, prop_name))
    if epsilon is None and f5_property_is_active:
        epsilon = _to_float_or_none(cfg_f5.get("epsilon"))
    return target, target_mode, epsilon


def _resolve_reference_class(
    *,
    property_name: str,
    cfg_step7: dict,
    cfg_f5: dict,
    f5_run_meta: Optional[dict],
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    override_class: Optional[str] = None,
) -> str:
    text = str(override_class or "").strip().lower()
    if text:
        return text

    class_map = cfg_step7.get("reference_classes", {}) or {}
    if isinstance(class_map, dict):
        text = str(class_map.get(property_name, "")).strip().lower()
        if text:
            return text

    text = str(cfg_step7.get("reference_class", "")).strip().lower()
    if text:
        return text

    f5_run_meta = f5_run_meta or {}
    f5_run_meta_property = _normalize_property_name(f5_run_meta.get("property", ""))
    if f5_run_meta_property.lower() == _normalize_property_name(property_name).lower():
        target_classes = f5_run_meta.get("target_classes")
        if isinstance(target_classes, str):
            text = str(target_classes).strip().lower()
            if text:
                return text
        if isinstance(target_classes, (list, tuple, set)):
            uniq = [str(x).strip().lower() for x in target_classes if str(x).strip()]
            uniq = sorted(set(uniq))
            if len(uniq) == 1:
                return uniq[0]

    # Use F5 target_class when this property is the active F5 target.
    f5_prop = str(cfg_f5.get("property", "")).strip()
    if f5_prop == property_name:
        text = str(cfg_f5.get("target_class", "")).strip().lower()
        if text:
            return text

    # Optional per-property class map from F5 configs.
    f5_class_map = cfg_f5.get("target_classes", {}) or {}
    if isinstance(f5_class_map, dict):
        text = str(f5_class_map.get(property_name, "")).strip().lower()
        if text:
            return text

    # Last fallback: infer single matched_class from candidate/top-k tables.
    for df in (topk_df, candidate_df):
        if "matched_class" not in df.columns:
            continue
        series = df["matched_class"].astype(str).str.strip().str.lower()
        uniq = sorted({x for x in series.tolist() if x})
        if len(uniq) == 1:
            return uniq[0]
    return ""


def _apply_reference_class_filter(
    *,
    property_name: str,
    ref_df: pd.DataFrame,
    ref_desc: pd.DataFrame,
    reference_class: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    cls = str(reference_class).strip().lower()
    if not cls:
        return ref_df, ref_desc, None
    motif_col = f"motif_{cls}"
    if motif_col not in ref_desc.columns:
        return ref_df, ref_desc, f"{property_name}: reference class filter '{cls}' not found in motif columns; using unfiltered reference."

    mask = ref_desc[motif_col].fillna(False).astype(bool)
    n_match = int(mask.sum())
    if n_match <= 0:
        return ref_df, ref_desc, f"{property_name}: reference class filter '{cls}' matched 0 rows; using unfiltered reference."

    filtered_desc = ref_desc.loc[mask].copy()
    allowed_smiles = set(filtered_desc["smiles"].astype(str).tolist())
    filtered_ref = ref_df[ref_df["smiles"].astype(str).isin(allowed_smiles)].copy()
    if filtered_ref.empty:
        return ref_df, ref_desc, f"{property_name}: reference class filter '{cls}' removed all property rows; using unfiltered reference."
    return filtered_ref, filtered_desc, None


def _prepare_candidate_error_df(candidate_df: pd.DataFrame, target: Optional[float], target_mode: str) -> pd.DataFrame:
    cdf = candidate_df.copy()
    if "prediction" in cdf.columns:
        cdf["prediction"] = pd.to_numeric(cdf["prediction"], errors="coerce")
    if "d2_distance" in cdf.columns:
        cdf["d2_distance"] = pd.to_numeric(cdf["d2_distance"], errors="coerce")

    if "target_excess" in cdf.columns:
        cdf["target_excess"] = pd.to_numeric(cdf["target_excess"], errors="coerce")
    elif "prediction" in cdf.columns and target is not None:
        cdf["target_excess"] = _target_excess_from_predictions(
            cdf["prediction"].to_numpy(dtype=np.float32),
            target=target,
            target_mode=target_mode,
        )
    else:
        cdf["target_excess"] = np.nan

    if "target_violation" in cdf.columns:
        cdf["target_violation"] = pd.to_numeric(cdf["target_violation"], errors="coerce")
    elif "prediction" in cdf.columns and target is not None:
        cdf["target_violation"] = _property_error_from_predictions(
            cdf["prediction"].to_numpy(dtype=np.float32),
            target=target,
            target_mode=target_mode,
        )
    elif "abs_error" in cdf.columns:
        cdf["target_violation"] = pd.to_numeric(cdf["abs_error"], errors="coerce")
    elif "property_error_raw" in cdf.columns:
        cdf["target_violation"] = pd.to_numeric(cdf["property_error_raw"], errors="coerce")
    elif "property_error_normed" in cdf.columns:
        cdf["target_violation"] = pd.to_numeric(cdf["property_error_normed"], errors="coerce")
    else:
        cdf["target_violation"] = np.nan

    cdf["property_error"] = cdf["target_violation"]
    return cdf


def _plot_descriptor_detail_figure(property_name: str, descriptor_shift_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or descriptor_shift_df.empty:
        return
    ds = descriptor_shift_df.copy().replace([np.inf, -np.inf], np.nan)
    ds["cohens_d_topk_vs_ref"] = pd.to_numeric(ds.get("cohens_d_topk_vs_ref"), errors="coerce")
    ds["ref_median"] = pd.to_numeric(ds.get("ref_median"), errors="coerce")
    ds["topk_median"] = pd.to_numeric(ds.get("topk_median"), errors="coerce")
    ds = ds.dropna(subset=["descriptor", "cohens_d_topk_vs_ref"])
    if ds.empty:
        return
    ds["abs_d"] = np.abs(ds["cohens_d_topk_vs_ref"])
    ds = ds.sort_values("abs_d", ascending=False).head(8).iloc[::-1]

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = _make_panel_figure(1, 2)
        ax0, ax1 = axes[0]

        colors = [COLOR_ACCENT if v >= 0 else COLOR_SECONDARY for v in ds["cohens_d_topk_vs_ref"].to_numpy(dtype=np.float32)]
        ax0.barh(ds["descriptor"].astype(str), ds["cohens_d_topk_vs_ref"], color=colors, alpha=0.9)
        ax0.axvline(0.0, color=COLOR_TEXT, linewidth=1.0)
        ax0.set_xlabel("Cohen's d (top-k vs reference)")
        ax0.grid(axis="x", alpha=0.25)
        _wrap_ticklabels(ax0, axis="y", width=18)

        pair_df = ds.dropna(subset=["ref_median", "topk_median"]).copy()
        if not pair_df.empty:
            ax1.scatter(pair_df["ref_median"], pair_df["topk_median"], s=62, color=COLOR_PRIMARY, alpha=0.9)
            lo = float(np.nanmin(np.concatenate([pair_df["ref_median"].to_numpy(dtype=np.float32), pair_df["topk_median"].to_numpy(dtype=np.float32)])))
            hi = float(np.nanmax(np.concatenate([pair_df["ref_median"].to_numpy(dtype=np.float32), pair_df["topk_median"].to_numpy(dtype=np.float32)])))
            if np.isfinite(lo) and np.isfinite(hi):
                ax1.plot([lo, hi], [lo, hi], linestyle="--", color=COLOR_MUTED, linewidth=1.1)
            label_df = pair_df.assign(distance=np.abs(pair_df["topk_median"] - pair_df["ref_median"])).sort_values("distance", ascending=False)
            _annotate_scatter_subset(ax1, label_df, x_col="ref_median", y_col="topk_median", label_col="descriptor", max_labels=6)
            ax1.set_xlabel("Reference median")
            ax1.set_ylabel("Top-k median")
            ax1.grid(alpha=0.25)
        else:
            ax1.text(0.5, 0.5, "No median values", ha="center", va="center")
            ax1.set_axis_off()

        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_motif_detail_figure(property_name: str, motif_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or motif_df.empty:
        return
    mf = motif_df.copy().replace([np.inf, -np.inf], np.nan)
    mf["log2_enrichment_topk_vs_ref"] = pd.to_numeric(mf.get("log2_enrichment_topk_vs_ref"), errors="coerce")
    mf["ref_freq"] = pd.to_numeric(mf.get("ref_freq"), errors="coerce")
    mf["candidate_freq"] = pd.to_numeric(mf.get("candidate_freq"), errors="coerce")
    mf["topk_freq"] = pd.to_numeric(mf.get("topk_freq"), errors="coerce")
    mf = mf.dropna(subset=["motif", "log2_enrichment_topk_vs_ref"])
    if mf.empty:
        return
    mf["abs_log2"] = np.abs(mf["log2_enrichment_topk_vs_ref"])
    mf = mf.sort_values("abs_log2", ascending=False).head(8).iloc[::-1]

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = _make_panel_figure(1, 2)
        ax0, ax1 = axes[0]

        colors = [COLOR_ACCENT if v >= 0 else COLOR_SECONDARY for v in mf["log2_enrichment_topk_vs_ref"].to_numpy(dtype=np.float32)]
        ax0.barh(mf["motif"].astype(str), mf["log2_enrichment_topk_vs_ref"], color=colors, alpha=0.9)
        ax0.axvline(0.0, color=COLOR_TEXT, linewidth=1.0)
        ax0.set_xlabel("log2(top-k / reference)")
        ax0.grid(axis="x", alpha=0.25)
        _wrap_ticklabels(ax0, axis="y", width=18)

        x = np.arange(len(mf), dtype=np.float32)
        width = 0.25
        ax1.bar(x - width, mf["ref_freq"].fillna(0.0).to_numpy(dtype=np.float32), width=width, color=COLOR_MUTED, alpha=0.8, label="Reference")
        ax1.bar(x, mf["candidate_freq"].fillna(0.0).to_numpy(dtype=np.float32), width=width, color=COLOR_PRIMARY, alpha=0.8, label="Candidate")
        ax1.bar(x + width, mf["topk_freq"].fillna(0.0).to_numpy(dtype=np.float32), width=width, color=COLOR_ACCENT, alpha=0.9, label="Top-k")
        ax1.set_xticks(x)
        ax1.set_xticklabels(mf["motif"].astype(str), rotation=30, ha="right")
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Frequency")
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend(loc="best")
        _wrap_ticklabels(ax1, axis="x", width=16, rotation=35)

        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_physics_rule_detail_figure(property_name: str, physics_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or physics_df.empty:
        return
    pf = physics_df.copy().replace([np.inf, -np.inf], np.nan)
    pf["observed_delta_topk_vs_ref"] = pd.to_numeric(pf.get("observed_delta_topk_vs_ref"), errors="coerce")
    pf = pf.dropna(subset=["feature", "observed_delta_topk_vs_ref"])
    if pf.empty:
        return

    pf["abs_delta"] = np.abs(pf["observed_delta_topk_vs_ref"])
    pf = pf.sort_values("abs_delta", ascending=False).head(8).copy()

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = _make_panel_figure(1, 2)
        ax0, ax1 = axes[0]

        labels = [f"{r['feature']} ({r['source']})" for _, r in pf.iterrows()]
        vals = pf["observed_delta_topk_vs_ref"].to_numpy(dtype=np.float32)
        ok = pf["sign_match"].astype(bool).to_numpy(dtype=bool)
        colors = [COLOR_ACCENT if m else COLOR_SECONDARY for m in ok]
        y = np.arange(len(labels), dtype=np.float32)
        ax0.barh(y, vals, color=colors, alpha=0.9)
        ax0.set_yticks(y)
        ax0.set_yticklabels(labels)
        ax0.axvline(0.0, color=COLOR_TEXT, linewidth=1.0)
        ax0.set_xlabel("Observed delta (top-k vs reference)")
        ax0.grid(axis="x", alpha=0.25)
        _wrap_ticklabels(ax0, axis="y", width=20)

        n_match = int(np.sum(ok))
        n_total = int(len(ok))
        n_mismatch = max(n_total - n_match, 0)
        ax1.bar(["Match", "Mismatch"], [n_match, n_mismatch], color=[COLOR_ACCENT, COLOR_SECONDARY], alpha=0.9)
        ax1.set_ylabel("Count")
        ax1.grid(axis="y", alpha=0.25)
        if n_total > 0:
            ax1.text(0.5, max(n_match, n_mismatch, 1) * 1.02, f"Consistency: {n_match / n_total:.2f}", ha="center", va="bottom")

        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_property_ood_landscape_figure(
    property_name: str,
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    target: Optional[float],
    target_mode: str,
    output_base: Path,
) -> None:
    if plt is None or candidate_df.empty:
        return
    cdf = _prepare_candidate_error_df(candidate_df, target=target, target_mode=target_mode)
    use_target_excess = str(target_mode).strip().lower() in {"ge", "le"} and cdf["target_excess"].notna().any()
    y_col = "target_excess" if use_target_excess else "property_error"
    y_label = (
        f"Target excess (≥0 = hit)"
        if use_target_excess
        else "Property error"
    )
    if {"d2_distance", y_col, "smiles"}.issubset(set(cdf.columns)) is False:
        return

    plot_df = cdf.dropna(subset=["d2_distance", y_col]).copy()
    if plot_df.empty:
        return
    if len(plot_df) > 12000:
        plot_df = plot_df.sample(n=12000, random_state=42)
    topk_set = set(topk_df["smiles"].astype(str).tolist()) if "smiles" in topk_df.columns else set()
    plot_df["is_topk"] = plot_df["smiles"].astype(str).isin(topk_set)
    base = plot_df[~plot_df["is_topk"]]
    tk = plot_df[plot_df["is_topk"]]

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = _make_panel_figure(1, 2)
        ax0, ax1 = axes[0]

        if not base.empty:
            ax0.scatter(base["d2_distance"], base[y_col], s=12, alpha=0.24, color=COLOR_TERTIARY, label="Candidates")
        if not tk.empty:
            ax0.scatter(tk["d2_distance"], tk[y_col], s=30, alpha=0.9, color=COLOR_SECONDARY, label="F6 top-k")
        if use_target_excess:
            ax0.axhline(0.0, color=COLOR_TEXT, linewidth=1.0, linestyle="--")
        ax0.set_xlabel("D2 distance (lower is better)")
        ax0.set_ylabel(y_label)
        ax0.grid(alpha=0.25)
        ax0.legend(loc="best")

        err_cand = _numeric_array(base[y_col] if not base.empty else [])
        err_top = _numeric_array(tk[y_col] if not tk.empty else [])
        if err_cand.size:
            ax1.hist(err_cand, bins=40, color=COLOR_PRIMARY, alpha=0.5, label="Candidates")
        if err_top.size:
            ax1.hist(err_top, bins=25, color=COLOR_SECONDARY, alpha=0.75, label="F6 top-k")
        if err_cand.size or err_top.size:
            ax1.set_xlabel(y_label)
            ax1.set_ylabel("Count")
            ax1.grid(alpha=0.25)
            ax1.legend(loc="best")
        else:
            ax1.text(0.5, 0.5, "No finite error values", ha="center", va="center")
            ax1.set_axis_off()

        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_nearest_neighbor_detail_figure(property_name: str, nn_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or nn_df.empty:
        return
    ndf = nn_df.copy()
    ndf["nearest_tanimoto"] = pd.to_numeric(ndf.get("nearest_tanimoto"), errors="coerce")
    ndf["topk_prediction"] = pd.to_numeric(ndf.get("topk_prediction"), errors="coerce")
    ndf["nearest_reference_value"] = pd.to_numeric(ndf.get("nearest_reference_value"), errors="coerce")

    sims = _numeric_array(ndf["nearest_tanimoto"])
    if sims.size == 0:
        return

    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = _make_panel_figure(1, 3, panel_width=5.8, panel_height=5.8)
        ax0, ax1, ax2 = axes[0]

        ax0.hist(sims, bins=18, color=COLOR_QUATERNARY, alpha=0.85)
        ax0.axvline(float(np.mean(sims)), color=COLOR_TEXT, linestyle="--", linewidth=1.1, label="Mean")
        ax0.set_xlabel("Tanimoto")
        ax0.set_ylabel("Count")
        ax0.grid(alpha=0.25)
        ax0.legend(loc="best")

        pair_df = ndf.dropna(subset=["topk_prediction", "nearest_reference_value"]).copy()
        if not pair_df.empty:
            ax1.scatter(pair_df["nearest_reference_value"], pair_df["topk_prediction"], s=24, alpha=0.75, color=COLOR_PRIMARY)
            lo = float(np.nanmin(np.concatenate([pair_df["nearest_reference_value"].to_numpy(dtype=np.float32), pair_df["topk_prediction"].to_numpy(dtype=np.float32)])))
            hi = float(np.nanmax(np.concatenate([pair_df["nearest_reference_value"].to_numpy(dtype=np.float32), pair_df["topk_prediction"].to_numpy(dtype=np.float32)])))
            if np.isfinite(lo) and np.isfinite(hi):
                ax1.plot([lo, hi], [lo, hi], linestyle="--", color=COLOR_MUTED, linewidth=1.1)
            ax1.set_xlabel("Nearest reference value")
            ax1.set_ylabel("Top-k prediction")
            ax1.grid(alpha=0.25)
        else:
            ax1.text(0.5, 0.5, "No paired values", ha="center", va="center")
            ax1.set_axis_off()

        # Show structural/count descriptors only (comparable scales);
        # large-magnitude descriptors (mol_wt, tpsa, logp) displayed as text annotations.
        bar_cols = [
            ("delta_aromatic_ring_count", "Δ aromatic rings"),
            ("delta_ring_count", "Δ ring count"),
            ("delta_rotatable_bonds", "Δ rotatable bonds"),
            ("delta_fraction_csp3", "Δ frac CSP3"),
        ]
        note_cols = [
            ("delta_mol_wt", "Δ mol wt"),
            ("delta_tpsa", "Δ TPSA"),
            ("delta_logp", "Δ logP"),
        ]
        labels, vals = [], []
        for col, label in bar_cols:
            if col in ndf.columns:
                arr = _numeric_array(ndf[col])
                if arr.size:
                    labels.append(label)
                    vals.append(float(np.mean(arr)))
        note_lines = []
        for col, label in note_cols:
            if col in ndf.columns:
                arr = _numeric_array(ndf[col])
                if arr.size:
                    note_lines.append(f"{label}: {float(np.mean(arr)):+.2f}")
        if vals:
            colors = [COLOR_ACCENT if v >= 0 else COLOR_SECONDARY for v in vals]
            y_pos = np.arange(len(labels), dtype=np.float32)
            ax2.barh(y_pos, vals, color=colors, alpha=0.85)
            ax2.axvline(0.0, color=COLOR_TEXT, linewidth=1.0)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels)
            ax2.set_xlabel("Mean Δ (top-k vs nearest reference)")
            ax2.set_title("Descriptor shifts")
            ax2.grid(axis="x", alpha=0.25)
            _wrap_ticklabels(ax2, axis="y", width=18)
            if note_lines:
                note_text = "\n".join(note_lines)
                ax2.text(
                    0.02,
                    0.02,
                    note_text,
                    transform=ax2.transAxes,
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 1.5},
                    color=COLOR_TEXT,
                    style="italic",
                )
        else:
            ax2.text(0.5, 0.5, "No descriptor delta columns", ha="center", va="center")
            ax2.set_axis_off()

        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_property_figure_suite(
    *,
    property_name: str,
    descriptor_shift_df: pd.DataFrame,
    motif_df: pd.DataFrame,
    physics_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    target: Optional[float],
    target_mode: str,
    figures_dir: Path,
) -> None:
    _plot_property_figure(
        property_name=property_name,
        descriptor_shift_df=descriptor_shift_df,
        motif_df=motif_df,
        candidate_df=candidate_df,
        topk_df=topk_df,
        nn_df=nn_df,
        target=target,
        target_mode=target_mode,
        output_base=figures_dir / f"figure_f7_chem_physics_{property_name}",
    )
    _plot_descriptor_detail_figure(
        property_name=property_name,
        descriptor_shift_df=descriptor_shift_df,
        output_base=figures_dir / f"figure_f7_descriptors_{property_name}",
    )
    _plot_motif_detail_figure(
        property_name=property_name,
        motif_df=motif_df,
        output_base=figures_dir / f"figure_f7_motif_enrichment_{property_name}",
    )
    _plot_physics_rule_detail_figure(
        property_name=property_name,
        physics_df=physics_df,
        output_base=figures_dir / f"figure_f7_physics_rules_{property_name}",
    )
    _plot_property_ood_landscape_figure(
        property_name=property_name,
        candidate_df=candidate_df,
        topk_df=topk_df,
        target=target,
        target_mode=target_mode,
        output_base=figures_dir / f"figure_f7_property_ood_landscape_{property_name}",
    )
    _plot_nearest_neighbor_detail_figure(
        property_name=property_name,
        nn_df=nn_df,
        output_base=figures_dir / f"figure_f7_nearest_neighbor_{property_name}",
    )


def _plot_property_figure(
    property_name: str,
    descriptor_shift_df: pd.DataFrame,
    motif_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    target: Optional[float],
    target_mode: str,
    output_base: Path,
) -> None:
    if plt is None:
        return
    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = _make_panel_figure(2, 2, panel_width=6.7, panel_height=5.7)
        ax1, ax2, ax3, ax4 = axes.ravel()

        ds = descriptor_shift_df.copy()
        ds = ds.replace([np.inf, -np.inf], np.nan).dropna(subset=["cohens_d_topk_vs_ref"])
        if not ds.empty:
            ds = ds.assign(abs_d=np.abs(ds["cohens_d_topk_vs_ref"]))
            ds = ds.sort_values("abs_d", ascending=False).head(8).iloc[::-1]
            bar_colors = [COLOR_ACCENT if v >= 0 else COLOR_SECONDARY for v in ds["cohens_d_topk_vs_ref"].to_numpy(dtype=np.float32)]
            ax1.barh(ds["descriptor"], ds["cohens_d_topk_vs_ref"], color=bar_colors)
            ax1.axvline(0.0, color=COLOR_TEXT, linewidth=1.0)
            ax1.set_xlabel("Effect size")
            ax1.grid(axis="x", alpha=0.25)
            _wrap_ticklabels(ax1, axis="y", width=18)
        else:
            ax1.text(0.5, 0.5, "No descriptor shift data", ha="center", va="center")
            ax1.set_axis_off()

        mf = motif_df.copy()
        mf = mf.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2_enrichment_topk_vs_ref"])
        if not mf.empty:
            mf = mf.assign(abs_log2=np.abs(mf["log2_enrichment_topk_vs_ref"]))
            mf = mf.sort_values("abs_log2", ascending=False).head(8).iloc[::-1]
            bar_colors = [COLOR_ACCENT if v >= 0 else COLOR_SECONDARY for v in mf["log2_enrichment_topk_vs_ref"].to_numpy(dtype=np.float32)]
            ax2.barh(mf["motif"], mf["log2_enrichment_topk_vs_ref"], color=bar_colors)
            ax2.axvline(0.0, color=COLOR_TEXT, linewidth=1.0)
            ax2.set_xlabel("log2 enrichment")
            ax2.grid(axis="x", alpha=0.25)
            _wrap_ticklabels(ax2, axis="y", width=18)
        else:
            ax2.text(0.5, 0.5, "No motif data", ha="center", va="center")
            ax2.set_axis_off()

        cdf = _prepare_candidate_error_df(candidate_df, target=target, target_mode=target_mode)
        use_target_excess = str(target_mode).strip().lower() in {"ge", "le"} and cdf["target_excess"].notna().any()
        y_col = "target_excess" if use_target_excess else "property_error"
        y_label = (
            f"{_target_excess_axis_label(property_name, target_mode)} (>=0 is hit)"
            if use_target_excess
            else "Property error/proxy"
        )

        if {"d2_distance", y_col, "smiles"}.issubset(set(cdf.columns)):
            plot_df = cdf.dropna(subset=["d2_distance", y_col]).copy()
            if not plot_df.empty:
                topk_set = set(topk_df["smiles"].astype(str).tolist()) if "smiles" in topk_df.columns else set()
                plot_df["is_topk"] = plot_df["smiles"].astype(str).isin(topk_set)
                base = plot_df[~plot_df["is_topk"]]
                tk = plot_df[plot_df["is_topk"]]
                if not base.empty:
                    ax3.scatter(base["d2_distance"], base[y_col], s=12, alpha=0.24, label="Candidates", color=COLOR_TERTIARY)
                if not tk.empty:
                    ax3.scatter(tk["d2_distance"], tk[y_col], s=28, alpha=0.9, label="F6 top-k", color=COLOR_SECONDARY)
                if use_target_excess:
                    ax3.axhline(0.0, color=COLOR_TEXT, linewidth=1.0, linestyle="--")
                ax3.set_xlabel("D2 distance (lower is closer)")
                ax3.set_ylabel(y_label)
                ax3.grid(alpha=0.25)
                ax3.legend(frameon=False, loc="best")
            else:
                ax3.text(0.5, 0.5, "No valid scatter points", ha="center", va="center")
                ax3.set_axis_off()
        else:
            ax3.text(0.5, 0.5, "Need prediction + d2_distance", ha="center", va="center")
            ax3.set_axis_off()

        if not nn_df.empty and "nearest_tanimoto" in nn_df.columns:
            vals = pd.to_numeric(nn_df["nearest_tanimoto"], errors="coerce").dropna().to_numpy(dtype=np.float32)
            if vals.size:
                ax4.hist(vals, bins=15, color=COLOR_QUATERNARY, alpha=0.85)
                ax4.set_xlabel("Tanimoto similarity")
                ax4.set_ylabel("Count")
                ax4.grid(axis="y", alpha=0.25)
            else:
                ax4.text(0.5, 0.5, "No NN similarity values", ha="center", va="center")
                ax4.set_axis_off()
        else:
            ax4.text(0.5, 0.5, "No NN explanation data", ha="center", va="center")
            ax4.set_axis_off()

        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_summary_figure(metrics_df: pd.DataFrame, output_base: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = _make_panel_figure(1, 3, panel_width=5.8, panel_height=5.2)
        axes = axes[0]

        df = metrics_df.copy().sort_values("property")
        props = df["property"].astype(str).tolist()

        axes[0].bar(props, pd.to_numeric(df["descriptor_consistency_rate"], errors="coerce").fillna(0.0), color=COLOR_PRIMARY)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Rate")
        axes[0].grid(axis="y", alpha=0.25)
        _wrap_ticklabels(axes[0], axis="x", width=14, rotation=35)

        axes[1].bar(props, pd.to_numeric(df["topk_hit_rate"], errors="coerce").fillna(0.0), color=COLOR_ACCENT)
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis="y", alpha=0.25)
        _wrap_ticklabels(axes[1], axis="x", width=14, rotation=35)

        axes[2].bar(props, pd.to_numeric(df["mean_nn_similarity"], errors="coerce").fillna(0.0), color=COLOR_QUATERNARY)
        axes[2].set_ylim(0, 1)
        axes[2].grid(axis="y", alpha=0.25)
        _wrap_ticklabels(axes[2], axis="x", width=14, rotation=35)

        _save_figure_png(fig, output_base)
        plt.close(fig)


def _plot_view_summary_figure(view_summary_df: pd.DataFrame, output_base: Path, property_name: str) -> None:
    if plt is None or view_summary_df.empty:
        return
    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = _make_panel_figure(1, 3, panel_width=5.8, panel_height=5.0)
        axes = axes[0]
        df = view_summary_df.copy()
        df["proposal_view"] = df["proposal_view"].astype(str).map(normalize_view_name)
        views = ordered_views(df["proposal_view"].tolist())
        if not views:
            plt.close(fig)
            return
        df = df.set_index("proposal_view").reindex(views).reset_index()
        xpos = np.arange(len(df), dtype=np.int64)
        colors = [view_color(v) for v in df["proposal_view"].tolist()]
        labels = [view_label(v) for v in df["proposal_view"].tolist()]

        axes[0].bar(xpos, pd.to_numeric(df["topk_hit_rate"], errors="coerce").fillna(0.0), color=colors, alpha=0.82)
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Top-k hit rate")
        axes[0].set_xticks(xpos)
        axes[0].set_xticklabels(labels, rotation=20, ha="right")
        axes[0].grid(axis="y", alpha=0.25)

        fair_col = "topk_fair_hit_rate" if "topk_fair_hit_rate" in df.columns else "topk_hit_rate"
        axes[1].bar(xpos, pd.to_numeric(df[fair_col], errors="coerce").fillna(0.0), color=colors, alpha=0.82)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Top-k fair hit rate")
        axes[1].set_xticks(xpos)
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].grid(axis="y", alpha=0.25)

        axes[2].bar(xpos, pd.to_numeric(df["topk_mean_ood_prop"], errors="coerce").fillna(0.0), color=colors, alpha=0.82)
        axes[2].set_ylabel("Top-k mean OOD-prop")
        axes[2].set_xticks(xpos)
        axes[2].set_xticklabels(labels, rotation=20, ha="right")
        axes[2].grid(axis="y", alpha=0.25)

        fig.suptitle(f"F7 View Summary: {property_name}", fontsize=16, fontweight="bold")
        _save_figure_png(fig, output_base)
        plt.close(fig)


def main(args):
    if Chem is None:
        raise RuntimeError("RDKit is required for step7_chem_physics_analysis.py")

    config = load_config(args.config)
    cfg_step7 = config.get("chem_physics_analysis", {}) or {}
    cfg_f5 = config.get("foundation_inverse", {}) or {}
    cfg_f6 = config.get("ood_aware_inverse", {}) or {}

    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step7_chem_physics_analysis")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    properties = _parse_properties(args, cfg_step7)
    property_dir = _resolve_path(config["paths"]["property_dir"])

    candidate_template = args.candidate_scores_template
    if candidate_template is None:
        candidate_template = str(cfg_step7.get("candidate_scores_template", "")).strip() or None
    topk_template = args.topk_scores_template
    if topk_template is None:
        topk_template = str(cfg_step7.get("topk_scores_template", "")).strip() or None

    max_ref_samples = _to_int_or_none(args.max_reference_samples)
    if max_ref_samples is None:
        max_ref_samples = _to_int_or_none(cfg_step7.get("max_reference_samples"))

    top_k_override = _to_int_or_none(args.top_k)
    if top_k_override is None:
        top_k_override = _to_int_or_none(cfg_step7.get("top_k"))
    if top_k_override is None:
        top_k_override = 100

    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(cfg_step7.get("generate_figures", True), True)
    generate_figures = bool(generate_figures and plt is not None)

    skip_missing = args.skip_missing_property
    if skip_missing is None:
        skip_missing = True
    skip_missing = bool(skip_missing)

    reference_class_override = str(args.reference_class or "").strip().lower() or None
    reference_class_filter_enabled = args.reference_class_filter_enabled
    if reference_class_filter_enabled is None:
        reference_class_filter_enabled = _to_bool(cfg_step7.get("reference_class_filter_enabled", True), True)
    reference_class_filter_enabled = bool(reference_class_filter_enabled)

    sa_score_fn = _load_sa_score_fn()
    compiled_motifs = _compile_motifs()

    all_descriptor_shift = []
    all_motif = []
    all_physics = []
    all_nn = []
    metric_rows = []
    file_rows = []
    design_audit_rows = []
    skipped = []
    multi_property_run = len(properties) > 1

    for prop in properties:
        prop_step_dirs = ensure_step_dirs(results_dir, "step7_chem_physics_analysis", prop)
        save_config(config, prop_step_dirs["files_dir"] / "config_used.yaml")
        try:
            ref_df = _load_property_reference(property_dir, prop, max_samples=max_ref_samples)
        except Exception as exc:
            msg = f"{prop}: failed to load reference property data ({exc})"
            if skip_missing:
                skipped.append(msg)
                continue
            raise

        cand_path, topk_path = _resolve_property_file_inputs(
            property_name=prop,
            results_dir=results_dir,
            candidate_template=candidate_template,
            topk_template=topk_template,
        )
        f5_run_meta, f5_run_meta_path = _load_neighbor_run_meta(cand_path, prop)
        f6_run_meta, f6_run_meta_path = _load_neighbor_run_meta(topk_path, prop)
        prop_file_row = {
            "property": prop,
            "candidate_scores_path": str(cand_path) if cand_path else "",
            "topk_scores_path": str(topk_path) if topk_path else "",
            "f5_run_meta_path": f5_run_meta_path,
            "f6_run_meta_path": f6_run_meta_path,
        }
        file_rows.append(
            dict(prop_file_row)
        )
        save_csv(
            pd.DataFrame([prop_file_row]),
            prop_step_dirs["files_dir"] / "property_input_files.csv",
            index=False,
        )

        if topk_path is None:
            msg = f"{prop}: top-k file not found (set step6 output or topk_scores_template)."
            if skip_missing:
                skipped.append(msg)
                continue
            raise FileNotFoundError(msg)

        topk_df = pd.read_csv(topk_path)
        if topk_df.empty:
            msg = f"{prop}: top-k file is empty: {topk_path}"
            if skip_missing:
                skipped.append(msg)
                continue
            raise RuntimeError(msg)
        if multi_property_run and _is_generic_scores_file(topk_path) and "property" not in topk_df.columns:
            msg = (
                f"{prop}: only generic top-k file found ({topk_path}) without a property column. "
                "Use property-specific file naming (ood_objective_topk_<PROPERTY>.csv)."
            )
            if skip_missing:
                skipped.append(msg)
                continue
            raise RuntimeError(msg)

        topk_df, topk_property_msg = _filter_property_rows_if_available(
            topk_df,
            property_name=prop,
            source_path=topk_path,
        )
        if topk_property_msg is not None:
            if skip_missing:
                skipped.append(topk_property_msg)
                continue
            raise RuntimeError(topk_property_msg)
        if topk_df.empty:
            msg = f"{prop}: top-k file has no usable rows after property filtering: {topk_path}"
            if skip_missing:
                skipped.append(msg)
                continue
            raise RuntimeError(msg)
        if "smiles" not in topk_df.columns:
            raise ValueError(f"{prop}: top-k file must contain 'smiles': {topk_path}")

        if top_k_override is not None and top_k_override > 0 and len(topk_df) > top_k_override:
            if "ood_aware_rank" in topk_df.columns:
                topk_df = topk_df.sort_values("ood_aware_rank").head(int(top_k_override)).copy()
            else:
                topk_df = topk_df.head(int(top_k_override)).copy()

        if cand_path is not None and cand_path.exists():
            candidate_df = pd.read_csv(cand_path)
            if multi_property_run and _is_generic_scores_file(cand_path) and "property" not in candidate_df.columns:
                skipped.append(
                    f"{prop}: candidate file {cand_path} is generic without property column; using top-k rows instead."
                )
                candidate_df = topk_df.copy()
            else:
                candidate_df, cand_property_msg = _filter_property_rows_if_available(
                    candidate_df,
                    property_name=prop,
                    source_path=cand_path,
                )
                if cand_property_msg is not None:
                    skipped.append(f"{cand_property_msg} Falling back to top-k rows for candidate analysis.")
                    candidate_df = topk_df.copy()
                elif candidate_df.empty:
                    skipped.append(f"{prop}: candidate file has no rows after property filtering; using top-k rows.")
                    candidate_df = topk_df.copy()
        else:
            candidate_df = topk_df.copy()

        target, target_mode, epsilon = _resolve_target_config(
            prop,
            cfg_step7,
            cfg_f5,
            cfg_f6=cfg_f6,
            f5_run_meta=f5_run_meta,
            f6_run_meta=f6_run_meta,
        )

        resolved_reference_class = _resolve_reference_class(
            property_name=prop,
            cfg_step7=cfg_step7,
            cfg_f5=cfg_f5,
            f5_run_meta=f5_run_meta,
            candidate_df=candidate_df,
            topk_df=topk_df,
            override_class=reference_class_override,
        )

        ref_desc = _descriptor_and_motif_rows(
            ref_df["smiles"].astype(str).tolist(),
            property_name=prop,
            split="reference",
            sa_score_fn=sa_score_fn,
            compiled_motifs=compiled_motifs,
        )
        ref_df_used = ref_df
        ref_desc_used = ref_desc
        if reference_class_filter_enabled and resolved_reference_class:
            ref_df_used, ref_desc_used, ref_filter_msg = _apply_reference_class_filter(
                property_name=prop,
                ref_df=ref_df,
                ref_desc=ref_desc,
                reference_class=resolved_reference_class,
            )
            if ref_filter_msg:
                skipped.append(ref_filter_msg)
            else:
                print(
                    f"[F7] Applied reference class filter '{resolved_reference_class}' "
                    f"for {prop}: {len(ref_df_used)}/{len(ref_df)} reference rows kept."
                )
        cand_desc = _descriptor_and_motif_rows(
            candidate_df["smiles"].astype(str).tolist(),
            property_name=prop,
            split="candidate",
            sa_score_fn=sa_score_fn,
            compiled_motifs=compiled_motifs,
        )
        top_desc = _descriptor_and_motif_rows(
            topk_df["smiles"].astype(str).tolist(),
            property_name=prop,
            split="topk",
            sa_score_fn=sa_score_fn,
            compiled_motifs=compiled_motifs,
        )

        if ref_desc_used.empty or top_desc.empty:
            msg = f"{prop}: insufficient descriptor rows after RDKit parsing."
            if skip_missing:
                skipped.append(msg)
                continue
            raise RuntimeError(msg)

        shift_df = _descriptor_shift_table(prop, ref_desc_used, cand_desc, top_desc)
        motif_df = _motif_enrichment_table(prop, ref_desc_used, cand_desc, top_desc)
        physics_df = _physics_consistency_table(prop, shift_df, motif_df, target_mode=target_mode)
        nn_df = _nearest_neighbor_explanations(prop, ref_df_used, topk_df, ref_desc_used, top_desc)
        reference_class_filter_applied = bool(reference_class_filter_enabled and bool(resolved_reference_class))
        design_audit_row = _build_design_filter_audit_row(
            property_name=prop,
            candidate_df=candidate_df,
            topk_df=topk_df,
            cand_desc=cand_desc,
            top_desc=top_desc,
            cand_path=cand_path,
            topk_path=topk_path,
            f5_run_meta=f5_run_meta,
            f5_run_meta_path=f5_run_meta_path,
            f6_run_meta=f6_run_meta,
            f6_run_meta_path=f6_run_meta_path,
            cfg_f5=cfg_f5,
            cfg_f6=cfg_f6,
            reference_class_filter_requested=bool(reference_class_filter_enabled),
            reference_class_filter_applied=reference_class_filter_applied,
            reference_class_filter=resolved_reference_class,
        )

        all_descriptor_shift.append(shift_df)
        all_motif.append(motif_df)
        all_physics.append(physics_df)
        if not nn_df.empty:
            all_nn.append(nn_df)
        design_audit_rows.append(design_audit_row)

        save_csv(shift_df, prop_step_dirs["files_dir"] / "descriptor_shifts.csv", index=False)
        save_csv(motif_df, prop_step_dirs["files_dir"] / "motif_enrichment.csv", index=False)
        save_csv(physics_df, prop_step_dirs["files_dir"] / "physics_consistency.csv", index=False)
        save_csv(nn_df, prop_step_dirs["files_dir"] / "nearest_neighbor_explanations.csv", index=False)
        save_csv(pd.DataFrame([design_audit_row]), prop_step_dirs["files_dir"] / "design_filter_audit.csv", index=False)

        if "property_hit" in topk_df.columns:
            hit_vals = pd.to_numeric(topk_df["property_hit"], errors="coerce")
            topk_hit_rate = float(np.nanmean(hit_vals)) if hit_vals.notna().any() else np.nan
        elif "prediction" in topk_df.columns and target is not None:
            pred_vals = pd.to_numeric(topk_df["prediction"], errors="coerce").to_numpy(dtype=np.float32)
            epsilon_eval = 0.0 if epsilon is None else float(epsilon)
            if target_mode == "window":
                hits = np.abs(pred_vals - float(target)) <= epsilon_eval
            elif target_mode == "ge":
                hits = pred_vals >= float(target)
            elif target_mode == "le":
                hits = pred_vals <= float(target)
            else:
                hits = np.abs(pred_vals - float(target)) <= epsilon_eval
            topk_hit_rate = float(np.mean(hits)) if hits.size else np.nan
        else:
            topk_hit_rate = np.nan

        physics_valid = physics_df["sign_match"].astype(bool)
        consistency_rate = float(physics_valid.mean()) if len(physics_valid) else np.nan
        mean_nn = float(pd.to_numeric(nn_df.get("nearest_tanimoto", pd.Series(dtype=float)), errors="coerce").mean()) if not nn_df.empty else np.nan
        mean_d2 = float(pd.to_numeric(topk_df.get("d2_distance", pd.Series(dtype=float)), errors="coerce").mean()) if "d2_distance" in topk_df.columns else np.nan
        unc_series = _prediction_uncertainty_series(topk_df, prop)
        mean_unc = float(unc_series.mean()) if unc_series.notna().any() else np.nan
        if "conservative_objective" in topk_df.columns:
            mean_obj = float(pd.to_numeric(topk_df["conservative_objective"], errors="coerce").mean())
        else:
            mean_obj = float(pd.to_numeric(topk_df.get("ood_aware_objective", pd.Series(dtype=float)), errors="coerce").mean()) if "ood_aware_objective" in topk_df.columns else np.nan
        mean_constraint = float(pd.to_numeric(topk_df.get("constraint_violation_total", pd.Series(dtype=float)), errors="coerce").mean()) if "constraint_violation_total" in topk_df.columns else np.nan
        metric_row = {
            "method": "Multi_View_Foundation",
            "property": prop,
            "n_reference": int(len(ref_df_used)),
            "n_candidates": int(len(candidate_df)),
            "n_topk": int(len(topk_df)),
            "reference_class_filter_enabled": reference_class_filter_applied,
            "reference_class_filter": resolved_reference_class if resolved_reference_class else "",
            "descriptor_consistency_rate": round(consistency_rate, 4) if np.isfinite(consistency_rate) else np.nan,
            "mean_nn_similarity": round(mean_nn, 4) if np.isfinite(mean_nn) else np.nan,
            "topk_hit_rate": round(topk_hit_rate, 4) if np.isfinite(topk_hit_rate) else np.nan,
            "topk_mean_d2_distance": round(mean_d2, 6) if np.isfinite(mean_d2) else np.nan,
            "topk_mean_prediction_uncertainty": round(mean_unc, 6) if np.isfinite(mean_unc) else np.nan,
            "topk_mean_conservative_objective": round(mean_obj, 6) if np.isfinite(mean_obj) else np.nan,
            "topk_mean_constraint_violation": round(mean_constraint, 6) if np.isfinite(mean_constraint) else np.nan,
            "target_value": target,
            "target_mode": target_mode,
            "epsilon": epsilon,
            "f5_require_novel": design_audit_row["f5_require_novel"],
            "f5_max_sa": design_audit_row["f5_max_sa"],
            "f5_target_class": design_audit_row["f5_target_class"],
            "candidate_novel_rate": design_audit_row["candidate_novel_rate"],
            "topk_novel_rate": design_audit_row["topk_novel_rate"],
            "candidate_sa_lt_max_rate": design_audit_row["candidate_sa_lt_max_rate"],
            "topk_sa_lt_max_rate": design_audit_row["topk_sa_lt_max_rate"],
            "f6_property_weight": design_audit_row["f6_property_weight"],
            "f6_ood_weight": design_audit_row["f6_ood_weight"],
            "f6_uncertainty_weight": design_audit_row["f6_uncertainty_weight"],
            "f6_constraint_weight": design_audit_row["f6_constraint_weight"],
            "f6_sa_weight": design_audit_row["f6_sa_weight"],
            "f6_descriptor_weight": design_audit_row["f6_descriptor_weight"],
        }
        metric_rows.append(metric_row)
        save_csv(pd.DataFrame([metric_row]), prop_step_dirs["metrics_dir"] / "metrics_chem_physics.csv", index=False)

        view_summary_df = pd.DataFrame()
        if "proposal_view" in candidate_df.columns and "proposal_view" in topk_df.columns:
            view_rows = []
            candidate_view_series = candidate_df["proposal_view"].astype(str).map(normalize_view_name)
            topk_view_series = topk_df["proposal_view"].astype(str).map(normalize_view_name)
            all_views = ordered_views(
                pd.concat(
                    [
                        candidate_view_series,
                        topk_view_series,
                    ],
                    ignore_index=True,
                ).tolist()
            )
            for proposal_view in all_views:
                cand_view = candidate_df.loc[candidate_view_series == proposal_view].copy()
                topk_view = topk_df.loc[topk_view_series == proposal_view].copy()
                if cand_view.empty or topk_view.empty:
                    continue
                if "property_hit" in topk_view.columns:
                    hit_rate_view = float(
                        pd.to_numeric(topk_view["property_hit"], errors="coerce").fillna(0.0).mean()
                    )
                elif "prediction" in topk_view.columns and target is not None:
                    pred_view = pd.to_numeric(topk_view["prediction"], errors="coerce").to_numpy(dtype=np.float32)
                    epsilon_eval = 0.0 if epsilon is None else float(epsilon)
                    if target_mode == "window":
                        hit_mask_view = np.abs(pred_view - float(target)) <= epsilon_eval
                    elif target_mode == "ge":
                        hit_mask_view = pred_view >= float(target)
                    elif target_mode == "le":
                        hit_mask_view = pred_view <= float(target)
                    else:
                        hit_mask_view = np.abs(pred_view - float(target)) <= epsilon_eval
                    hit_rate_view = float(np.mean(hit_mask_view)) if hit_mask_view.size else np.nan
                else:
                    hit_rate_view = np.nan
                fair_hit_rate_view = float(
                    pd.to_numeric(topk_view.get("fair_hit", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()
                ) if "fair_hit" in topk_view.columns else hit_rate_view
                mean_ood_prop_view = float(
                    pd.to_numeric(topk_view.get("ood_prop", pd.Series(dtype=float)), errors="coerce").mean()
                ) if "ood_prop" in topk_view.columns else np.nan
                mean_obj_view = float(
                    pd.to_numeric(topk_view.get("conservative_objective", pd.Series(dtype=float)), errors="coerce").mean()
                ) if "conservative_objective" in topk_view.columns else np.nan
                view_rows.append(
                    {
                        "property": prop,
                        "proposal_view": proposal_view,
                        "n_candidates": int(len(cand_view)),
                        "n_topk": int(len(topk_view)),
                        "topk_hit_rate": round(hit_rate_view, 4) if np.isfinite(hit_rate_view) else np.nan,
                        "topk_fair_hit_rate": round(fair_hit_rate_view, 4) if np.isfinite(fair_hit_rate_view) else np.nan,
                        "topk_mean_ood_prop": round(mean_ood_prop_view, 6) if np.isfinite(mean_ood_prop_view) else np.nan,
                        "topk_mean_conservative_objective": round(mean_obj_view, 6) if np.isfinite(mean_obj_view) else np.nan,
                    }
                )
            if view_rows:
                view_summary_df = pd.DataFrame(view_rows)
                save_csv(view_summary_df, prop_step_dirs["files_dir"] / "view_summary.csv", index=False)

        if generate_figures:
            _plot_property_figure_suite(
                property_name=prop,
                descriptor_shift_df=shift_df,
                motif_df=motif_df,
                physics_df=physics_df,
                candidate_df=candidate_df,
                topk_df=topk_df,
                nn_df=nn_df,
                target=target,
                target_mode=target_mode,
                figures_dir=prop_step_dirs["figures_dir"],
            )
            if not view_summary_df.empty:
                _plot_view_summary_figure(
                    view_summary_df=view_summary_df,
                    output_base=prop_step_dirs["figures_dir"] / f"figure_f7_view_summary_{prop}",
                    property_name=prop,
                )

        save_json(
            {
                "property": prop,
                "candidate_scores_path": str(cand_path) if cand_path else "",
                "topk_scores_path": str(topk_path) if topk_path else "",
                "f5_run_meta_path": f5_run_meta_path,
                "f6_run_meta_path": f6_run_meta_path,
                "reference_class_filter_requested": bool(reference_class_filter_enabled),
                "reference_class_filter_enabled": reference_class_filter_applied,
                "reference_class_filter": resolved_reference_class if resolved_reference_class else "",
                "target_value": target,
                "target_mode": target_mode,
                "epsilon": epsilon,
                "generate_figures": bool(generate_figures),
                "design_filter_audit": design_audit_row,
            },
            prop_step_dirs["files_dir"] / "run_meta.json",
        )

    if not metric_rows:
        raise RuntimeError(
            "Step7 found no analyzable properties. "
            "Provide valid F5/F6 outputs per property or set templates."
        )

    descriptor_shift_all = pd.concat(all_descriptor_shift, ignore_index=True) if all_descriptor_shift else pd.DataFrame()
    motif_all = pd.concat(all_motif, ignore_index=True) if all_motif else pd.DataFrame()
    physics_all = pd.concat(all_physics, ignore_index=True) if all_physics else pd.DataFrame()
    nn_all = pd.concat(all_nn, ignore_index=True) if all_nn else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    files_df = pd.DataFrame(file_rows)
    design_audit_df = pd.DataFrame(design_audit_rows)

    save_csv(
        descriptor_shift_all,
        step_dirs["files_dir"] / "descriptor_shifts.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "descriptor_shifts.csv"],
        index=False,
    )
    save_csv(
        motif_all,
        step_dirs["files_dir"] / "motif_enrichment.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "motif_enrichment.csv"],
        index=False,
    )
    save_csv(
        physics_all,
        step_dirs["files_dir"] / "physics_consistency.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "physics_consistency.csv"],
        index=False,
    )
    save_csv(
        nn_all,
        step_dirs["files_dir"] / "nearest_neighbor_explanations.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "nearest_neighbor_explanations.csv"],
        index=False,
    )
    save_csv(
        files_df,
        step_dirs["files_dir"] / "property_input_files.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "property_input_files.csv"],
        index=False,
    )
    save_csv(
        design_audit_df,
        step_dirs["files_dir"] / "design_filter_audit.csv",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "design_filter_audit.csv"],
        index=False,
    )
    save_csv(
        metrics_df,
        step_dirs["metrics_dir"] / "metrics_chem_physics.csv",
        legacy_paths=[results_dir / "metrics_chem_physics.csv"],
        index=False,
    )

    if generate_figures:
        _plot_summary_figure(
            metrics_df=metrics_df,
            output_base=step_dirs["figures_dir"] / "figure_f7_chem_physics_all_properties",
        )

    save_json(
        {
            "properties_requested": properties,
            "properties_processed": metrics_df["property"].astype(str).tolist(),
            "skipped": skipped,
            "candidate_scores_template": candidate_template,
            "topk_scores_template": topk_template,
            "reference_class_filter_requested": bool(reference_class_filter_enabled),
            "reference_class_filter_enabled": (
                bool(
                    design_audit_df["reference_class_filter_applied"]
                    .fillna(False)
                    .astype(bool)
                    .any()
                )
                if not design_audit_df.empty and "reference_class_filter_applied" in design_audit_df.columns
                else False
            ),
            "reference_class_override": reference_class_override or "",
            "generate_figures": bool(generate_figures),
            "design_filter_audit_path": str(step_dirs["files_dir"] / "design_filter_audit.csv"),
        },
        step_dirs["files_dir"] / "run_meta.json",
        legacy_paths=[results_dir / "step7_chem_physics_analysis" / "run_meta.json"],
    )

    if skipped:
        print("Step7 warnings:")
        for msg in skipped:
            print(f"  - {msg}")
    print(f"Saved metrics_chem_physics.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--properties", type=str, default=None, help="Comma-separated property list, e.g. Tg,Tm,Td,Eg")
    parser.add_argument("--candidate_scores_template", type=str, default=None, help="Path template for F5 candidate CSV; supports {property}")
    parser.add_argument("--topk_scores_template", type=str, default=None, help="Path template for F6 top-k CSV; supports {property}")
    parser.add_argument("--max_reference_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--reference_class", type=str, default=None, help="Reference class filter (e.g., polyamide)")
    parser.add_argument("--enable_reference_class_filter", dest="reference_class_filter_enabled", action="store_true")
    parser.add_argument("--disable_reference_class_filter", dest="reference_class_filter_enabled", action="store_false")
    parser.set_defaults(reference_class_filter_enabled=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    parser.add_argument("--skip_missing_property", dest="skip_missing_property", action="store_true")
    parser.add_argument("--strict", dest="skip_missing_property", action="store_false")
    parser.set_defaults(skip_missing_property=None)
    main(parser.parse_args())
