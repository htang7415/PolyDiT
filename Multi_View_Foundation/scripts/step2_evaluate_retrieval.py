#!/usr/bin/env python
"""F2: Retrieval evaluation across views."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.utils.config import load_config, save_config
from src.evaluation.retrieval_metrics import compute_recall_at_k
from src.utils.embedding_artifacts import load_view_embeddings, load_view_index
from src.utils.output_layout import ensure_step_dirs, save_csv
from src.utils.runtime import resolve_path as _shared_resolve_path, to_bool as _to_bool, to_int_or_none as _to_int_or_none
from src.utils.visualization import VIEW_ORDER, ordered_views, save_figure_png as shared_save_figure_png, set_publication_style

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

if plt is not None:
    set_publication_style()

def _resolve_path(path_str: str) -> Path:
    return _shared_resolve_path(path_str, BASE_DIR)


def _save_figure_png(fig, output_base: Path) -> None:
    shared_save_figure_png(fig, output_base, font_size=16, legend_loc="best")


def _metric_label(metric: str) -> str:
    if metric.startswith("recall_at_"):
        suffix = str(metric).split("_")[-1]
        return f"Recall@{suffix}"
    labels = {
        "mean_cosine_sim_matched": "Matched cosine similarity",
        "match_rate": "Matched ID rate",
    }
    return labels.get(metric, str(metric))


def _draw_heatmap(
    fig,
    ax,
    matrix: np.ndarray,
    row_labels,
    col_labels,
    *,
    colorbar_label: str,
    vmin=0.0,
    vmax=1.0,
) -> None:
    arr = np.asarray(matrix, dtype=np.float32)
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    n_rows, n_cols = arr.shape[0], arr.shape[1]
    # Scale tick/cell fontsize to matrix size so text fits in cells
    cell_fs = max(7, min(11, 80 // max(n_rows, n_cols, 1)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=cell_fs + 1)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=cell_fs + 1)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            val = arr[r, c]
            if np.isfinite(val):
                text_color = "white" if float(val) > 0.65 * (vmax - vmin) + vmin else "black"
                ax.text(c, r, f"{float(val):.2f}", ha="center", va="center", fontsize=cell_fs, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)


def _split_dataset_and_view(token: str):
    raw = str(token).strip()
    for view in sorted(VIEW_ORDER, key=len, reverse=True):
        suffix = f"_{view}"
        if raw.endswith(suffix):
            dataset = raw[: -len(suffix)]
            if dataset:
                return dataset, view
    return None


def _parse_view_pair(text: str):
    raw = str(text)
    if "->" not in raw:
        return None
    left, right = raw.split("->", 1)
    left_parts = _split_dataset_and_view(left)
    right_parts = _split_dataset_and_view(right)
    if left_parts is None or right_parts is None:
        return None
    l_dataset, l_view = left_parts
    r_dataset, r_view = right_parts
    return l_dataset, l_view, r_dataset, r_view


def _align_embeddings_with_index(
    *,
    idx_df: pd.DataFrame,
    embeddings: np.ndarray,
    dataset: str,
    view: str,
):
    subset = idx_df[idx_df["dataset"] == dataset].copy()
    if subset.empty:
        return None, None
    if "row_index" not in subset.columns or "polymer_id" not in subset.columns:
        raise ValueError(f"embedding_index_{view}.csv must contain row_index and polymer_id columns.")

    row_idx = pd.to_numeric(subset["row_index"], errors="coerce")
    if row_idx.isna().any():
        raise ValueError(f"Non-numeric row_index detected for view={view} dataset={dataset}.")
    subset["_row_index"] = row_idx.astype(np.int64)

    if subset["_row_index"].duplicated().any():
        raise ValueError(f"Duplicate row_index detected for view={view} dataset={dataset}.")
    if int(np.min(subset["_row_index"])) < 0:
        raise ValueError(f"Negative row_index detected for view={view} dataset={dataset}.")

    n_emb = int(embeddings.shape[0])
    n_idx = int(len(subset))
    if n_idx != n_emb:
        raise ValueError(
            f"Embedding/index length mismatch for view={view} dataset={dataset}: "
            f"index_rows={n_idx} embedding_rows={n_emb}."
        )

    subset = subset.sort_values("_row_index")
    expected = np.arange(n_emb, dtype=np.int64)
    observed = subset["_row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"row_index is not a 0..N-1 sequence for view={view} dataset={dataset}. "
            "Refusing to evaluate with potentially misaligned IDs/embeddings."
        )

    ids = subset["polymer_id"].astype(str).tolist()
    aligned = embeddings[observed]
    return ids, aligned


def _plot_f2_retrieval_heatmaps(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None or metrics_df.empty:
        return
    df = metrics_df.copy()
    parsed = df["view_pair"].astype(str).map(_parse_view_pair)
    df = df.assign(_parsed=parsed)
    df = df[df["_parsed"].notna()].copy()
    if df.empty:
        return
    df[["src_dataset", "src_view", "tgt_dataset", "tgt_view"]] = pd.DataFrame(df["_parsed"].tolist(), index=df.index)
    df = df[df["src_dataset"] == df["tgt_dataset"]].copy()
    if df.empty:
        return

    datasets = sorted(df["src_dataset"].astype(str).unique().tolist())
    views = ordered_views(sorted(set(df["src_view"].astype(str)).union(set(df["tgt_view"].astype(str)))))
    if not datasets or not views:
        return

    recall_cols = [c for c in df.columns if str(c).startswith("recall_at_")]
    recall_cols = sorted(recall_cols, key=lambda x: int(str(x).split("_")[-1]) if str(x).split("_")[-1].isdigit() else 0)
    cos_sim_cols = [c for c in df.columns if c == "mean_cosine_sim_matched"]
    metric_cols = recall_cols + cos_sim_cols + ["match_rate"]

    # Scale panel size so 5×5 cell text remains legible
    n_views = len(views)
    panel_w = max(5.5, 1.1 * n_views)
    panel_h = max(4.5, 1.0 * n_views)
    fig, axes = plt.subplots(
        len(datasets),
        len(metric_cols),
        figsize=(panel_w * len(metric_cols), panel_h * len(datasets)),
        squeeze=False,
    )

    for row_idx, dataset in enumerate(datasets):
        sub = df[df["src_dataset"] == dataset]
        for col_idx, metric in enumerate(metric_cols):
            ax = axes[row_idx, col_idx]
            matrix = np.full((len(views), len(views)), np.nan, dtype=np.float32)
            for i, src in enumerate(views):
                for j, tgt in enumerate(views):
                    r = sub[(sub["src_view"] == src) & (sub["tgt_view"] == tgt)]
                    if r.empty or metric not in r.columns:
                        continue
                    val = pd.to_numeric(r[metric], errors="coerce").mean()
                    if np.isfinite(val):
                        matrix[i, j] = float(val)
            finite = matrix[np.isfinite(matrix)]
            vmax = 1.0
            if finite.size and metric == "match_rate":
                vmax = min(1.0, max(0.05, float(np.nanmax(finite))))
            _draw_heatmap(
                fig,
                ax,
                matrix,
                views,
                views,
                colorbar_label=_metric_label(metric),
                vmin=0.0,
                vmax=vmax,
            )
            if row_idx == 0:
                ax.set_title(_metric_label(metric))
            if row_idx == len(datasets) - 1:
                ax.set_xlabel("Target view")
            if col_idx == 0:
                ax.set_ylabel(f"{dataset}\nSource view")

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f2_retrieval_heatmaps")
    plt.close(fig)


def _plot_f2_cosine_sim_bars(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    """Bar chart of mean cosine similarity per view pair, grouped by dataset."""
    if plt is None or metrics_df.empty or "mean_cosine_sim_matched" not in metrics_df.columns:
        return
    df = metrics_df.copy()
    parsed = df["view_pair"].astype(str).map(_parse_view_pair)
    df = df.assign(_parsed=parsed)
    df = df[df["_parsed"].notna()].copy()
    if df.empty:
        return
    df[["src_dataset", "src_view", "tgt_dataset", "tgt_view"]] = pd.DataFrame(df["_parsed"].tolist(), index=df.index)
    df = df[df["src_dataset"] == df["tgt_dataset"]].copy()
    if df.empty:
        return
    df["pair_label"] = df["src_view"] + u"\u2192" + df["tgt_view"]

    datasets = sorted(df["src_dataset"].astype(str).unique().tolist())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    palette = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2", "#FF9DA7", "#9C755F"]
    fig, axes = plt.subplots(1, n_datasets, figsize=(max(6, 3.5 * n_datasets), 5.5), squeeze=False)

    for col_idx, dataset in enumerate(datasets):
        ax = axes[0, col_idx]
        sub = df[df["src_dataset"] == dataset].sort_values("pair_label")
        pairs = sub["pair_label"].tolist()
        sims = pd.to_numeric(sub["mean_cosine_sim_matched"], errors="coerce").tolist()
        bar_colors = [palette[i % len(palette)] for i in range(len(pairs))]
        bars = ax.bar(range(len(pairs)), sims, color=bar_colors, alpha=0.88, width=0.6)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, rotation=40, ha="right", fontsize=11)
        finite_sims = [float(val) for val in sims if np.isfinite(val)]
        upper = max(finite_sims) + 0.08 if finite_sims else 1.0
        ax.set_ylim(0.0, max(1.05, upper))
        ax.set_ylabel("Mean Cosine Similarity (matched pairs)", fontsize=13)
        ax.set_title(f"Dataset: {dataset}", fontsize=14)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, sims):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    float(val) + 0.012,
                    f"{float(val):.3f}",
                    ha="center", va="bottom", fontsize=10,
                )

    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f2_cosine_sim_bars")
    plt.close(fig)


def _plot_f2_recall_bars(metrics_df: pd.DataFrame, figures_dir: Path) -> None:
    """Grouped bar chart of recall@k per source-view per dataset, one panel per dataset.

    For each dataset, shows how well each view retrieves across all target views (mean recall@k).
    This makes cross-view retrieval strength immediately comparable.
    """
    if plt is None or metrics_df.empty:
        return
    df = metrics_df.copy()
    parsed = df["view_pair"].astype(str).map(_parse_view_pair)
    df = df.assign(_parsed=parsed)
    df = df[df["_parsed"].notna()].copy()
    if df.empty:
        return
    df[["src_dataset", "src_view", "tgt_dataset", "tgt_view"]] = pd.DataFrame(df["_parsed"].tolist(), index=df.index)
    df = df[df["src_dataset"] == df["tgt_dataset"]].copy()
    if df.empty:
        return

    recall_cols = sorted(
        [c for c in df.columns if str(c).startswith("recall_at_")],
        key=lambda s: int(str(s).split("_")[-1]) if str(s).split("_")[-1].isdigit() else 0,
    )
    if not recall_cols:
        return

    k_labels = [str(c).replace("recall_at_", "Recall@") for c in recall_cols]
    datasets = sorted(df["src_dataset"].astype(str).unique().tolist())
    if not datasets:
        return

    palette = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]
    n_datasets = len(datasets)
    n_k = len(recall_cols)

    fig, axes = plt.subplots(1, n_datasets, figsize=(max(8, 5 * n_datasets), 5.5), squeeze=False)
    for col_idx, dataset in enumerate(datasets):
        ax = axes[0, col_idx]
        sub = df[df["src_dataset"] == dataset]
        src_views = ordered_views(sub["src_view"].astype(str).unique().tolist())
        if not src_views:
            ax.set_axis_off()
            continue
        x = np.arange(len(src_views), dtype=np.float32)
        width = 0.7 / max(n_k, 1)
        offsets = np.linspace(-0.35 + width / 2, 0.35 - width / 2, n_k)
        max_mean = 0.0
        for k_idx, (recall_col, k_label) in enumerate(zip(recall_cols, k_labels)):
            means = []
            for sv in src_views:
                vals = pd.to_numeric(sub[sub["src_view"] == sv][recall_col], errors="coerce").dropna()
                means.append(float(vals.mean()) if len(vals) else 0.0)
            if means:
                max_mean = max(max_mean, max(means))
            color = palette[k_idx % len(palette)]
            bars = ax.bar(x + offsets[k_idx], means, width=width, color=color, alpha=0.88, label=k_label)
            for bar, val in zip(bars, means):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, float(val) + 0.008,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=9, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(src_views, rotation=35, ha="right")
        ax.set_ylim(0.0, max(1.05, float(max_mean) + 0.08))
        ax.set_ylabel("Mean recall (avg over target views)")
        ax.set_title(f"Dataset: {dataset}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")

    fig.suptitle("F2: Retrieval Recall@k per Source View", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_figure_png(fig, figures_dir / "figure_f2_recall_bars")
    plt.close(fig)


def _load_existing_f2_metrics(results_dir: Path, step_dirs: dict) -> pd.DataFrame:
    candidates = [
        step_dirs["metrics_dir"] / "metrics_alignment.csv",
        step_dirs["step_dir"] / "metrics_alignment.csv",
        results_dir / "metrics_alignment.csv",
    ]
    metrics_path = next((p for p in candidates if p.exists()), None)
    if metrics_path is None:
        raise FileNotFoundError(
            "No metrics_alignment.csv found. Run F2 first or provide the expected metrics file."
        )
    try:
        metrics_df = pd.read_csv(metrics_path)
    except Exception as exc:
        raise RuntimeError(
            f"metrics_alignment.csv is unreadable or empty: {metrics_path}. "
            "Run F1/F2 first to generate retrieval metrics before regenerating figures."
        ) from exc
    if metrics_df.empty:
        raise RuntimeError(
            f"metrics_alignment.csv is empty: {metrics_path}. "
            "Run F1/F2 first to generate retrieval metrics before regenerating figures."
        )
    return metrics_df


def main(args):
    config = load_config(args.config)
    results_dir = _resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = ensure_step_dirs(results_dir, "step2_retrieval")
    save_config(config, results_dir / "config_used.yaml")
    save_config(config, step_dirs["files_dir"] / "config_used.yaml")

    eval_cfg = config.get("evaluation", {}) or {}
    generate_figures = args.generate_figures
    if generate_figures is None:
        generate_figures = _to_bool(eval_cfg.get("generate_figures", True), True)
    if generate_figures and plt is None:
        print("Warning: matplotlib unavailable; skipping F2 figures.")
        generate_figures = False
    if args.figures_only:
        metrics_df = _load_existing_f2_metrics(results_dir, step_dirs)
        if generate_figures:
            _plot_f2_retrieval_heatmaps(metrics_df, step_dirs["figures_dir"])
            _plot_f2_cosine_sim_bars(metrics_df, step_dirs["figures_dir"])
            _plot_f2_recall_bars(metrics_df, step_dirs["figures_dir"])
        print(f"Saved F2 figures to {step_dirs['figures_dir']}")
        return

    views = config.get("alignment_views", ["smiles"])
    ks = config.get("evaluation", {}).get("recall_ks", [1, 5, 10])
    max_eval_d1 = _to_int_or_none(eval_cfg.get("max_samples_d1"))
    max_eval_d2 = _to_int_or_none(eval_cfg.get("max_samples_d2"))

    view_data = {}
    view_dims = {}
    for view in views:
        idx_df = _load_view_index(results_dir, view)
        if idx_df is None:
            continue
        view_data[view] = {}
        for dataset in ["d1", "d2"]:
            emb = _load_view_embeddings(results_dir, view, dataset)
            if emb is None:
                continue
            ids, emb_aligned = _align_embeddings_with_index(
                idx_df=idx_df,
                embeddings=emb,
                dataset=dataset,
                view=view,
            )
            if ids is None or emb_aligned is None:
                continue
            dataset_cap = max_eval_d1 if dataset == "d1" else max_eval_d2
            if dataset_cap is not None and len(ids) > dataset_cap:
                ids = ids[:dataset_cap]
                emb_aligned = emb_aligned[:dataset_cap]
            view_data[view][dataset] = {"embeddings": emb_aligned, "ids": ids}
            view_dims[view] = emb_aligned.shape[1]

    rows = []
    for dataset in ["d1", "d2"]:
        available_views = [v for v in views if dataset in view_data.get(v, {})]
        if available_views:
            sample_sizes = {v: len(view_data[v][dataset]["ids"]) for v in available_views}
            print(f"Retrieval dataset={dataset} views={available_views} sample_sizes={sample_sizes}")
        for src_view in available_views:
            for tgt_view in available_views:
                src = view_data[src_view][dataset]
                tgt = view_data[tgt_view][dataset]
                metrics = compute_recall_at_k(
                    src["embeddings"],
                    tgt["embeddings"],
                    src["ids"],
                    tgt["ids"],
                    ks,
                    exclude_self=(src_view == tgt_view),
                )
                rows.append({
                    "view_pair": f"{dataset}_{src_view}->{dataset}_{tgt_view}",
                    "view_dropout_mode": "none",
                    "self_excluded": int(src_view == tgt_view),
                    **metrics,
                })

    metrics_df = pd.DataFrame(rows)
    save_csv(
        metrics_df,
        step_dirs["metrics_dir"] / "metrics_alignment.csv",
        legacy_paths=[results_dir / "metrics_alignment.csv"],
        index=False,
    )
    if generate_figures:
        _plot_f2_retrieval_heatmaps(metrics_df, step_dirs["figures_dir"])
        _plot_f2_cosine_sim_bars(metrics_df, step_dirs["figures_dir"])
        _plot_f2_recall_bars(metrics_df, step_dirs["figures_dir"])

    print(f"Saved metrics_alignment.csv to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--generate_figures", dest="generate_figures", action="store_true")
    parser.add_argument("--no_figures", dest="generate_figures", action="store_false")
    parser.set_defaults(generate_figures=None)
    parser.add_argument("--figures_only", action="store_true", help="Regenerate F2 figures from an existing metrics_alignment.csv.")
    main(parser.parse_args())
