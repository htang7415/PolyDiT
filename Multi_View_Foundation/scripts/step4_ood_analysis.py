#!/usr/bin/env python
"""F4: OOD analysis (initial implementation)."""

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from shared.ood_metrics import compute_ood_metrics_from_files, knn_distances
from src.utils.output_layout import ensure_step_dirs, save_csv, save_numpy

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
}

if plt is not None:
    plt.rcParams.update(PUBLICATION_STYLE)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (BASE_DIR / path)


def _resolve_view_embedding_path(results_dir: Path, view: str, dataset: str) -> Path:
    """Resolve embedding file path for a given view and dataset (d1 or d2).

    Checks in order:
      1. results_dir/embeddings_{view}_{dataset}.npy
      2. results_dir/step1_alignment_embeddings/files/embeddings_{view}_{dataset}.npy
      3. Legacy results_dir/embeddings_{dataset}.npy (smiles view only)

    Raises FileNotFoundError if none exist.
    """
    candidate1 = results_dir / f"embeddings_{view}_{dataset}.npy"
    if candidate1.exists():
        return candidate1

    candidate2 = results_dir / "step1_alignment_embeddings" / "files" / f"embeddings_{view}_{dataset}.npy"
    if candidate2.exists():
        return candidate2

    if view == "smiles":
        legacy = results_dir / f"embeddings_{dataset}.npy"
        if legacy.exists():
            return legacy

    raise FileNotFoundError(
        f"Embedding file not found for view={view}, dataset={dataset}. "
        f"Searched: {candidate1}, {candidate2}"
        + (f", {results_dir / f'embeddings_{dataset}.npy'}" if view == "smiles" else "")
    )


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _standardize_figure_text_and_legend(fig, font_size: int = 16, legend_loc: str = "best") -> None:
    for text_obj in fig.findobj(match=lambda artist: hasattr(artist, "set_fontsize")):
        try:
            text_obj.set_fontsize(font_size)
        except Exception:
            continue
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_loc(legend_loc)


def _save_figure_png(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    _standardize_figure_text_and_legend(fig, font_size=16, legend_loc="best")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def _plot_f4_ood_diagnostics(
    *,
    d1_to_d2_dist: np.ndarray,
    gen_to_d2_dist: Optional[np.ndarray],
    gen_to_d1_dist: Optional[np.ndarray] = None,
    metrics_row: dict,
    figures_dir: Path,
) -> None:
    """4-panel OOD diagnostics figure (2×2):
    (A) Distance PDFs for all three distributions.
    (B) Empirical CDFs for visual stochastic-dominance comparison.
    (C) Box-plot comparison of Gen→D2 vs Gen→D1 signals.
    (D) Summary statistics table.
    """
    if plt is None:
        return
    d1_vals = np.asarray(d1_to_d2_dist, dtype=np.float32).reshape(-1)
    d1_vals = d1_vals[np.isfinite(d1_vals)]
    if d1_vals.size == 0:
        return

    gen_vals = None
    if gen_to_d2_dist is not None:
        g = np.asarray(gen_to_d2_dist, dtype=np.float32).reshape(-1)
        g = g[np.isfinite(g)]
        if g.size:
            gen_vals = g

    gen_d1_vals = None
    if gen_to_d1_dist is not None:
        gd1 = np.asarray(gen_to_d1_dist, dtype=np.float32).reshape(-1)
        gd1 = gd1[np.isfinite(gd1)]
        if gd1.size:
            gen_d1_vals = gd1

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax0, ax1, ax2, ax3 = axes.reshape(-1)

    # (A) Distance PDFs
    n_bins = 50
    ax0.hist(d1_vals, bins=n_bins, density=True, alpha=0.65, color="#4E79A7",
             label=f"D1→D2 (n={d1_vals.size:,})")
    if gen_vals is not None:
        ax0.hist(gen_vals, bins=n_bins, density=True, alpha=0.6, color="#E15759",
                 label=f"Gen→D2 / OOD-prop (n={gen_vals.size:,})")
    if gen_d1_vals is not None:
        ax0.hist(gen_d1_vals, bins=n_bins, density=True, alpha=0.5, color="#59A14F",
                 label=f"Gen→D1 / OOD-gen (n={gen_d1_vals.size:,})")
    ax0.set_xlabel("Cosine distance")
    ax0.set_ylabel("Density")
    ax0.set_title("(A) Distance distributions (PDF)")
    ax0.grid(alpha=0.25)
    ax0.legend()

    # (B) Empirical CDFs
    for arr, color, label in [
        (d1_vals, "#4E79A7", "D1→D2"),
        (gen_vals, "#E15759", "Gen→D2 (OOD-prop)"),
        (gen_d1_vals, "#59A14F", "Gen→D1 (OOD-gen)"),
    ]:
        if arr is None or arr.size == 0:
            continue
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax1.plot(sorted_arr, cdf, color=color, linewidth=2.0, label=label)
        # Mark median
        med = float(np.median(sorted_arr))
        ax1.axvline(med, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.set_xlabel("Cosine distance")
    ax1.set_ylabel("Cumulative probability")
    ax1.set_title("(B) Empirical CDFs (dotted lines = medians)")
    ax1.grid(alpha=0.25)
    ax1.legend()

    # (C) Box-plot comparing OOD-prop vs OOD-gen signals
    box_data = []
    box_labels = []
    box_colors = []
    if gen_vals is not None and gen_vals.size:
        box_data.append(gen_vals)
        box_labels.append("OOD-prop\n(Gen→D2)")
        box_colors.append("#E15759")
    if gen_d1_vals is not None and gen_d1_vals.size:
        box_data.append(gen_d1_vals)
        box_labels.append("OOD-gen\n(Gen→D1)")
        box_colors.append("#59A14F")
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                         notch=False, showfliers=True,
                         flierprops=dict(marker=".", markersize=2, alpha=0.3))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median_line in bp["medians"]:
            median_line.set_color("black")
            median_line.set_linewidth(2)
        ax2.set_ylabel("Cosine distance")
        ax2.set_title("(C) OOD signal comparison (box plot)")
        ax2.grid(axis="y", alpha=0.25)
        # Annotate means
        for i, arr in enumerate(box_data, start=1):
            mean_val = float(np.mean(arr))
            ax2.scatter([i], [mean_val], color="black", zorder=5, s=40, marker="D")
            ax2.text(i + 0.08, mean_val, f"μ={mean_val:.3f}", va="center", fontsize=12)
    else:
        ax2.text(0.5, 0.5, "No OOD signal data", ha="center", va="center")
        ax2.set_axis_off()

    # (D) Summary stats table
    summary_keys = [
        ("d1_to_d2_mean_dist", "D1→D2 mean dist"),
        ("ood_prop_mean", "OOD-prop mean (Gen→D2)"),
        ("ood_gen_mean", "OOD-gen mean (Gen→D1)"),
        ("frac_generated_near_d2", "Frac gen near D2"),
        ("frac_generated_near_d1", "Frac gen near D1"),
    ]
    stat_lines = []
    for key, label in summary_keys:
        val = metrics_row.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except Exception:
            continue
        if np.isfinite(val):
            fmt = f"{val:.4f}" if key.startswith("frac") else f"{val:.4f}"
            stat_lines.append((label, fmt))
    if d1_vals.size:
        stat_lines.append(("D1 samples", f"{d1_vals.size:,}"))
    if gen_vals is not None:
        stat_lines.append(("Gen samples (OOD-prop)", f"{gen_vals.size:,}"))
    if gen_d1_vals is not None:
        stat_lines.append(("Gen samples (OOD-gen)", f"{gen_d1_vals.size:,}"))
    ax3.set_axis_off()
    if stat_lines:
        col_labels = ["Metric", "Value"]
        table_data = [[lbl, v] for lbl, v in stat_lines]
        tbl = ax3.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.0, 1.9)
        for col_idx in range(len(col_labels)):
            tbl[(0, col_idx)].set_facecolor("#4E79A7")
            tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")
        ax3.set_title("(D) OOD Summary Statistics", pad=4)
    else:
        ax3.text(0.5, 0.5, "No finite summary metrics", ha="center", va="center", fontsize=13)

    fig.suptitle("F4: Out-of-Distribution Analysis", fontsize=18, fontweight="bold")
    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f4_ood_diagnostics")
    plt.close(fig)


def _load_model_size(results_dir: Path, config: dict) -> str:
    for meta_path in [
        results_dir / "embedding_meta.json",
        results_dir / "step1_alignment_embeddings" / "files" / "embedding_meta.json",
    ]:
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return meta.get("model_size", "base")
    return (
        config.get("smiles_encoder", {}).get("model_size")
        or config.get("smiles_bpe_encoder", {}).get("model_size")
        or "base"
    )


def _load_primary_view(results_dir: Path, config: dict) -> str:
    for meta_path in [
        results_dir / "embedding_meta.json",
        results_dir / "step1_alignment_embeddings" / "files" / "embedding_meta.json",
    ]:
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            view = meta.get("view")
            if isinstance(view, str) and view:
                return view
    if (results_dir / "embeddings_smiles_bpe_d1.npy").exists():
        return "smiles_bpe"
    if config.get("smiles_bpe_encoder", {}).get("method_dir") and not config.get("smiles_encoder", {}).get("method_dir"):
        return "smiles_bpe"
    return "smiles"


def _view_to_representation(view: str) -> str:
    return "SMILES_BPE" if view == "smiles_bpe" else "SMILES"


def main(args):
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step4_ood")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")
    primary_view = _load_primary_view(results_dir, config)

    d1_path = _resolve_view_embedding_path(results_dir, primary_view, "d1")
    d2_path = _resolve_view_embedding_path(results_dir, primary_view, "d2")

    gen_path = Path(args.generated_embeddings) if args.generated_embeddings else None

    k = args.k or int(config.get("ood", {}).get("nn_k", 1))
    if args.use_faiss is None:
        use_faiss = bool(config.get("ood", {}).get("use_faiss", True))
    else:
        use_faiss = bool(args.use_faiss)
    print(f"OOD settings: k={k}, use_faiss={use_faiss}")
    sig = inspect.signature(compute_ood_metrics_from_files)
    if "use_faiss" in sig.parameters:
        ood_metrics = compute_ood_metrics_from_files(d1_path, d2_path, gen_path, k=k, use_faiss=use_faiss)
    else:
        print("Warning: loaded compute_ood_metrics_from_files() does not support use_faiss; ignoring setting.")
        ood_metrics = compute_ood_metrics_from_files(d1_path, d2_path, gen_path, k=k)

    row = {
        "method": "Multi_View_Foundation",
        "representation": _view_to_representation(primary_view),
        "model_size": _load_model_size(results_dir, config),
        **ood_metrics,
    }

    out_path = step_dirs["metrics_dir"] / "metrics_ood.csv"
    import pandas as pd
    save_csv(pd.DataFrame([row]), out_path, legacy_paths=[results_dir / "metrics_ood.csv"], index=False)

    ood_cfg = config.get("ood", {}) or {}
    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(ood_cfg.get("generate_figures", True), True)
    if generate_figures and plt is None:
        print("Warning: matplotlib unavailable; skipping F4 figures.")
        generate_figures = False
    if generate_figures:
        d1 = np.load(d1_path)
        d2 = np.load(d2_path)
        d1_to_d2_dist = knn_distances(d1, d2, k=k, use_faiss=use_faiss).mean(axis=1)
        gen_to_d2_dist = None
        gen_to_d1_dist = None
        if gen_path is not None and gen_path.exists():
            try:
                gen = np.load(gen_path)
                if gen.size:
                    gen_to_d2_dist = knn_distances(gen, d2, k=k, use_faiss=use_faiss).mean(axis=1)
                    gen_to_d1_dist = knn_distances(gen, d1, k=k, use_faiss=use_faiss).mean(axis=1)
            except Exception:
                gen_to_d2_dist = None
                gen_to_d1_dist = None

        save_csv(
            pd.DataFrame({"d1_to_d2_distance": d1_to_d2_dist}),
            step_dirs["files_dir"] / "d1_to_d2_distances.csv",
            legacy_paths=[results_dir / "step4_ood" / "d1_to_d2_distances.csv"],
            index=False,
        )
        if gen_to_d2_dist is not None:
            save_csv(
                pd.DataFrame({"generated_to_d2_distance": gen_to_d2_dist}),
                step_dirs["files_dir"] / "generated_to_d2_distances.csv",
                legacy_paths=[results_dir / "step4_ood" / "generated_to_d2_distances.csv"],
                index=False,
            )
        if gen_to_d1_dist is not None:
            save_csv(
                pd.DataFrame({"generated_to_d1_distance": gen_to_d1_dist}),
                step_dirs["files_dir"] / "generated_to_d1_distances.csv",
                legacy_paths=[results_dir / "step4_ood" / "generated_to_d1_distances.csv"],
                index=False,
            )

        _plot_f4_ood_diagnostics(
            d1_to_d2_dist=d1_to_d2_dist,
            gen_to_d2_dist=gen_to_d2_dist,
            gen_to_d1_dist=gen_to_d1_dist,
            metrics_row=row,
            figures_dir=step_dirs["figures_dir"],
        )

    print(f"Saved metrics_ood.csv to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--generated_embeddings", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--use_faiss", dest="use_faiss", action="store_true")
    parser.add_argument("--no_faiss", dest="use_faiss", action="store_false")
    parser.set_defaults(use_faiss=None)
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    main(parser.parse_args())
