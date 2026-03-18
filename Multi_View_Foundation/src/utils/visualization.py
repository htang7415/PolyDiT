"""Shared publication-style visualization helpers for MVF steps."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:  # pragma: no cover
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None


NATURE_PALETTE = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]

COLOR_TEXT = "#1F2937"
COLOR_MUTED = "#B8BFC9"

VIEW_ORDER = ["smiles", "smiles_bpe", "selfies", "group_selfies", "graph", "all"]
VIEW_LABELS = {
    "smiles": "SMILES",
    "smiles_bpe": "SMILES-BPE",
    "selfies": "SELFIES",
    "group_selfies": "Group-SELFIES",
    "graph": "Graph",
    "all": "All Views",
}
VIEW_COLORS = {
    "smiles": "#3C5488",
    "smiles_bpe": "#4DBBD5",
    "selfies": "#00A087",
    "group_selfies": "#E64B35",
    "graph": "#8491B4",
    "all": "#7E6148",
}

PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 600,
}


def normalize_view_name(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return ""
    aliases = {
        "groupselfies": "group_selfies",
        "group_selfie": "group_selfies",
        "groupselfies": "group_selfies",
        "smilesbpe": "smiles_bpe",
        "multi_view": "all",
        "all_views": "all",
    }
    return aliases.get(text, text)


def view_label(value: object) -> str:
    key = normalize_view_name(value)
    return VIEW_LABELS.get(key, str(value).strip() or "Unknown")


def view_color(value: object) -> str:
    key = normalize_view_name(value)
    return VIEW_COLORS.get(key, COLOR_MUTED)


def ordered_views(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        key = normalize_view_name(value)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    ordered = [view for view in VIEW_ORDER if view in seen]
    for view in normalized:
        if view not in ordered:
            ordered.append(view)
    return ordered


def set_publication_style() -> None:
    if plt is None or matplotlib is None:  # pragma: no cover
        return
    style = dict(PUBLICATION_STYLE)
    style["axes.prop_cycle"] = matplotlib.cycler(color=NATURE_PALETTE)
    plt.rcParams.update(style)


def standardize_figure_text_and_legend(
    fig,
    font_size: int = 16,
    legend_loc: str = "best",
    strip_titles: bool = True,
) -> None:
    for text_obj in fig.findobj(match=lambda artist: hasattr(artist, "set_fontsize")):
        try:
            text_obj.set_fontsize(font_size)
        except Exception:
            continue
    for ax in fig.axes:
        if strip_titles:
            try:
                ax.set_title("")
                ax.set_title("", loc="left")
                ax.set_title("", loc="right")
            except Exception:
                pass
        legend = ax.get_legend()
        if legend is not None:
            set_legend_location(legend, legend_loc)
    if strip_titles:
        try:
            suptitle = getattr(fig, "_suptitle", None)
            if suptitle is not None:
                suptitle.set_text("")
        except Exception:
            pass


def set_legend_location(legend, legend_loc: str = "best") -> None:
    setter = getattr(legend, "set_loc", None)
    if callable(setter):
        try:
            setter(legend_loc)
            return
        except Exception:
            pass
    private_setter = getattr(legend, "_set_loc", None)
    if callable(private_setter):
        try:
            private_setter(legend_loc)
            return
        except Exception:
            pass
    try:
        legend._loc = legend_loc
    except Exception:
        pass


def save_figure_png(
    fig,
    output_base: Path,
    *,
    font_size: int = 16,
    dpi: int = 600,
    legend_loc: str = "best",
    strip_titles: bool = True,
) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    standardize_figure_text_and_legend(
        fig,
        font_size=font_size,
        legend_loc=legend_loc,
        strip_titles=strip_titles,
    )
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
