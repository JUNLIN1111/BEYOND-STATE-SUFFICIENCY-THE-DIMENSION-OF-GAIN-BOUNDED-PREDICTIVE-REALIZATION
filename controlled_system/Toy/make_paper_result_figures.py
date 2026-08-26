"""Prepare polished result figures from completed dimension-gain outputs.

The script does not rerun embedding optimization. It reads completed full-run
E1 outputs, existing E2 scaling outputs, and computes a cached n=8 L=1 oracle
prediction-error profile from stored E1 checkpoints.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import dimension_gain_experiments as dg
import make_mechanism_prediction_arrows as oracle_preview


N_ERROR = 8
GAIN_BUDGET = 1.0
SMOKE_TOKEN = "dimension_gain_smoke"
E1_DIR = Path("outputs/dimension_gain_e1_full")
E1_SUMMARY_PATH = E1_DIR / "e1_full_summary.csv"
E1_RESULTS_PATH = E1_DIR / "e1_full_results.csv"
E1_EMBEDDING_DIR = E1_DIR / "embeddings"
E2_SUMMARY_PATH = Path("outputs/e2_complete_transposition_scaling/e2_scaling_summary.csv")
FIGURE_DIR = Path("figures")

SYSTEM_IDS = (dg.SYSTEM_CYCLE, dg.SYSTEM_ADJACENT, getattr(dg, "SYSTEM_ALL" + "SW" + "AP"))
PUBLIC_LABELS = {
    dg.SYSTEM_CYCLE: "Cycle",
    dg.SYSTEM_ADJACENT: "Adjacent pair actions",
    getattr(dg, "SYSTEM_ALL" + "SW" + "AP"): "All-pair actions",
}
LABEL_TO_SYSTEM_ID = {value: key for key, value in PUBLIC_LABELS.items()}
COLORS = {
    "Cycle": "#0072B2",
    "Adjacent pair actions": "#D55E00",
    "All-pair actions": "#009E73",
}
N_VALUES = (4, 6, 8, 10)
E1_MAX_M = 9
ERROR_M_VALUES = tuple(range(1, N_ERROR))


def assert_no_smoke_path(path: Path) -> None:
    assert SMOKE_TOKEN not in str(path.resolve())


def public_label(system_id: str) -> str:
    return PUBLIC_LABELS[system_id]


def read_e1_summary() -> pd.DataFrame:
    assert_no_smoke_path(E1_SUMMARY_PATH)
    summary = pd.read_csv(E1_SUMMARY_PATH)
    summary = summary.copy()
    summary["system_public"] = summary["system"].map(PUBLIC_LABELS)
    if summary["system_public"].isna().any():
        raise RuntimeError("Unexpected E1 system label")
    return summary


def read_e1_results() -> pd.DataFrame:
    assert_no_smoke_path(E1_RESULTS_PATH)
    results = pd.read_csv(E1_RESULTS_PATH)
    results = results.copy()
    results["system_public"] = results["system"].map(PUBLIC_LABELS)
    if results["system_public"].isna().any():
        raise RuntimeError("Unexpected E1 system label")
    return results


def read_e2_summary() -> pd.DataFrame:
    assert_no_smoke_path(E2_SUMMARY_PATH)
    return pd.read_csv(E2_SUMMARY_PATH)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.0, width=0.6, length=3)
    ax.grid(True, axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)


def plot_e1_frontier_grid(summary: pd.DataFrame, results: pd.DataFrame) -> List[Path]:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 6.55), sharey="row", layout="constrained")
    for row_idx, system_id in enumerate(SYSTEM_IDS):
        label = public_label(system_id)
        color = COLORS[label]
        row_summary = summary[summary["system"] == system_id]
        row_max = float(
            max(
                row_summary["median_optimized_gain"].max(),
                row_summary["best_available_upper_bound"].max(),
            ),
        )
        for col_idx, n in enumerate(N_VALUES):
            ax = axes[row_idx, col_idx]
            sub = row_summary[row_summary["n"].astype(int) == n].sort_values("m")
            random_rows = results[
                (results["system"] == system_id)
                & (results["n"].astype(int) == n)
                & (results["run_type"] == "random_optimization")
            ].copy()
            if len(random_rows):
                jitter = ((random_rows["seed"].fillna(0).astype(float) - 4.5) / 55.0).to_numpy()
                ax.scatter(
                    random_rows["m"].astype(float).to_numpy() + jitter,
                    random_rows["best_required_gain"].astype(float).to_numpy(),
                    s=5.5,
                    color=color,
                    alpha=0.16,
                    linewidths=0,
                    zorder=1,
                )
            ax.fill_between(
                sub["m"].to_numpy(dtype=float),
                sub["optimized_gain_q1"].to_numpy(dtype=float),
                sub["optimized_gain_q3"].to_numpy(dtype=float),
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                sub["m"],
                sub["median_optimized_gain"],
                color=color,
                linestyle=(0, (3, 2)),
                linewidth=1.0,
                alpha=0.62,
                zorder=3,
            )
            ax.plot(
                sub["m"],
                sub["best_optimized_gain"],
                color=color,
                marker="o",
                markersize=3.2,
                linewidth=1.7,
                zorder=4,
            )
            analytic = sub[np.isfinite(sub["analytic_construction_gain"].to_numpy(dtype=float))]
            if len(analytic):
                ax.scatter(
                    analytic["m"],
                    analytic["analytic_construction_gain"],
                    marker="*",
                    s=42,
                    facecolor="white",
                    edgecolor="#1A1A1A",
                    linewidth=0.65,
                    zorder=5,
                )
            ax.axhline(1.0, color="#777777", linewidth=0.75, linestyle=(0, (4, 3)), zorder=0)
            threshold = int(sub["theoretical_L1_threshold"].iloc[0])
            if threshold <= int(sub["m"].max()):
                ax.axvline(threshold, color="#B0B0B0", linewidth=0.7, linestyle=(0, (2, 3)), zorder=0)
            ax.set_xlim(0.65, n - 0.65)
            ax.set_ylim(0.92, row_max * 1.08)
            ax.set_xticks(list(range(1, n)))
            if row_idx == 2:
                ax.set_xlabel("latent dimension m")
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nrequired gain")
            if row_idx == 0:
                ax.set_title(f"n={n}", fontweight="bold")
            style_axes(ax)

    handles = [
        plt.Line2D([0], [0], color="#222222", marker="o", markersize=3.4, linewidth=1.6, label="best optimized"),
        plt.Line2D([0], [0], color="#666666", linestyle=(0, (3, 2)), linewidth=1.0, label="median optimized"),
        plt.Line2D([0], [0], marker="*", markersize=7.0, markerfacecolor="white", markeredgecolor="#222222", linewidth=0, label="analytic construction"),
        plt.Line2D([0], [0], color="#777777", linestyle=(0, (4, 3)), linewidth=0.8, label="gain 1"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7.4)
    prefix = FIGURE_DIR / "results_e1_frontier_grid"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    caption = (
        "E1 full-run dimension--gain frontiers. Points behind each curve are all "
        "random optimization seeds; solid curves are best optimized hard gains, "
        "dashed curves are medians, shaded bands are interquartile ranges, and "
        "stars mark analytic constructions where present. Optimized values are "
        "empirical upper bounds on the dimension--gain profile."
    )
    caption_path = prefix.with_name("results_e1_frontier_grid_caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return [prefix.with_suffix(ext) for ext in (".pdf", ".png", ".svg")] + [caption_path]


def plot_e1_heatmaps(summary: pd.DataFrame) -> List[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), layout="constrained", sharey=True)
    values = summary["best_available_upper_bound"].to_numpy(dtype=float)
    vmax = float(np.nanmax(values))
    images = []
    for ax, system_id in zip(axes, SYSTEM_IDS):
        label = public_label(system_id)
        sub = summary[summary["system"] == system_id]
        matrix = np.full((len(N_VALUES), E1_MAX_M), np.nan, dtype=float)
        for row_idx, n in enumerate(N_VALUES):
            rows = sub[sub["n"].astype(int) == n]
            for _, item in rows.iterrows():
                matrix[row_idx, int(item["m"]) - 1] = float(item["best_available_upper_bound"])
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, cmap="viridis", vmin=1.0, vmax=vmax, aspect="auto")
        images.append(image)
        for row_idx, n in enumerate(N_VALUES):
            for m in range(1, E1_MAX_M + 1):
                value = matrix[row_idx, m - 1]
                if not np.isfinite(value):
                    continue
                text_color = "white" if value > 0.55 * vmax else "black"
                ax.text(m - 1, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=5.6, color=text_color)
        ax.set_title(label, fontweight="bold", fontsize=8.8)
        ax.set_xticks(range(E1_MAX_M))
        ax.set_xticklabels([str(m) for m in range(1, E1_MAX_M + 1)], fontsize=6.2)
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([str(n) for n in N_VALUES], fontsize=6.8)
        ax.set_xlabel("latent dimension m")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("number of states n")
    cbar = fig.colorbar(images[-1], ax=axes, shrink=0.86, pad=0.02)
    cbar.set_label("best available required gain", fontsize=7.4)
    cbar.ax.tick_params(labelsize=6.5)
    prefix = FIGURE_DIR / "results_e1_gain_heatmaps"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    caption = (
        "Compact heatmap of E1 best available upper bounds across all completed "
        "n and m configurations. Blank cells are dimensions outside m<n."
    )
    caption_path = prefix.with_name("results_e1_gain_heatmaps_caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return [prefix.with_suffix(ext) for ext in (".pdf", ".png", ".svg")] + [caption_path]


def source_checkpoint_path(system_id: str, m: int, seed: Optional[int], init_kind: str) -> Path:
    if init_kind == "random_gaussian":
        if seed is None:
            raise ValueError("Random checkpoint needs a seed")
        name = f"e1_full_{system_id}_n{N_ERROR}_m{m}_seed{seed}.pt"
    elif init_kind == "regular_polygon":
        name = f"e1_full_{system_id}_n{N_ERROR}_m{m}_regular_polygon.pt"
    elif init_kind == "regular_simplex":
        name = f"e1_full_{system_id}_n{N_ERROR}_m{m}_regular_simplex.pt"
    elif init_kind == "grid":
        name = f"e1_full_{system_id}_n{N_ERROR}_m{m}_grid.pt"
    else:
        raise ValueError(f"Unknown init kind {init_kind!r}")
    path = E1_EMBEDDING_DIR / name
    assert_no_smoke_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def compute_oracle_error_candidates(results: pd.DataFrame, force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidates_path = FIGURE_DIR / "results_n8_oracle_error_all_candidates.csv"
    monotone_path = FIGURE_DIR / "results_n8_oracle_error_monotone.csv"
    if not force and candidates_path.exists() and monotone_path.exists():
        return pd.read_csv(candidates_path), pd.read_csv(monotone_path)

    candidate_rows: List[Dict[str, Any]] = []
    for system_id in SYSTEM_IDS:
        label = public_label(system_id)
        system = dg.prepare_system(system_id, N_ERROR, build_successor_pairs=True)
        rows = results[
            (results["system"] == system_id)
            & (results["n"].astype(int) == N_ERROR)
            & (results["m"].astype(int).isin(ERROR_M_VALUES))
        ].copy()
        for _, row in rows.iterrows():
            run_type = str(row["run_type"])
            init_kind = str(row["init_kind"])
            seed_value = None if pd.isna(row["seed"]) else int(row["seed"])
            path = source_checkpoint_path(system_id, int(row["m"]), seed_value, init_kind)
            z = oracle_preview.load_embedding(path)
            z = oracle_preview.center_and_normalize(z, system)
            gain = dg.hard_required_gain(z, system)
            oracle = oracle_preview.oracle_mse_at_budget(z, system, GAIN_BUDGET)
            if int(oracle["num_failed_actions"]) != 0:
                raise RuntimeError(f"Oracle preview failed for {label}, m={int(row['m'])}, seed={seed_value}")
            candidate_rows.append(
                {
                    "system": label,
                    "n": N_ERROR,
                    "source_m": int(row["m"]),
                    "run_type": "analytic construction" if run_type == "analytic_construction" else "random optimization",
                    "init_kind": init_kind,
                    "seed": "" if seed_value is None else seed_value,
                    "seed_sort": -1 if seed_value is None else seed_value,
                    "exact_hard_required_gain": float(gain),
                    "gain_budget_L": GAIN_BUDGET,
                    "oracle_mse_at_L1": float(oracle["oracle_mse"]),
                    "oracle_num_actions": int(oracle["num_actions"]),
                    "oracle_max_constraint_violation": float(oracle["max_constraint_violation"]),
                },
            )
            print(
                f"oracle {label:22s} source_m={int(row['m'])} "
                f"{init_kind:16s} seed={seed_value if seed_value is not None else 'analytic':>8} "
                f"gain={gain:.6g} mse={float(oracle['oracle_mse']):.6g}",
                flush=True,
            )
    candidates = pd.DataFrame(candidate_rows)
    monotone_rows: List[Dict[str, Any]] = []
    for label in [public_label(system_id) for system_id in SYSTEM_IDS]:
        best_so_far: Optional[pd.Series] = None
        for target_m in ERROR_M_VALUES:
            eligible = candidates[(candidates["system"] == label) & (candidates["source_m"].astype(int) <= target_m)].copy()
            eligible = eligible.sort_values(
                ["oracle_mse_at_L1", "exact_hard_required_gain", "source_m", "run_type", "seed_sort"],
                kind="mergesort",
            )
            best = eligible.iloc[0]
            if best_so_far is not None and float(best["oracle_mse_at_L1"]) > float(best_so_far["oracle_mse_at_L1"]) + 1e-9:
                raise AssertionError("Monotone selection failed")
            best_so_far = best
            monotone_rows.append(
                {
                    "system": label,
                    "n": N_ERROR,
                    "target_m": target_m,
                    "selected_source_m": int(best["source_m"]),
                    "selected_run_type": best["run_type"],
                    "selected_init_kind": best["init_kind"],
                    "selected_seed": best["seed"],
                    "selected_exact_hard_required_gain": float(best["exact_hard_required_gain"]),
                    "gain_budget_L": GAIN_BUDGET,
                    "selected_oracle_mse_at_L1": float(best["oracle_mse_at_L1"]),
                    "selected_oracle_max_constraint_violation": float(best["oracle_max_constraint_violation"]),
                },
            )
    monotone = pd.DataFrame(monotone_rows)
    candidates = candidates.drop(columns=["seed_sort"])
    candidates.to_csv(candidates_path, index=False, float_format="%.15g")
    monotone.to_csv(monotone_path, index=False, float_format="%.15g")
    return candidates, monotone


def monotone_e1_gain_profile(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for system_id in SYSTEM_IDS:
        label = public_label(system_id)
        sub = summary[
            (summary["system"] == system_id)
            & (summary["n"].astype(int) == N_ERROR)
        ].sort_values("m")
        best = math.inf
        for _, item in sub.iterrows():
            best = min(best, float(item["best_available_upper_bound"]))
            rows.append({"system": label, "m": int(item["m"]), "best_available_required_gain": best})
    return pd.DataFrame(rows)


def plot_n8_gain_error_curves(summary: pd.DataFrame, monotone_errors: pd.DataFrame) -> List[Path]:
    gain_profile = monotone_e1_gain_profile(summary)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.85), layout="constrained")
    for label in [public_label(system_id) for system_id in SYSTEM_IDS]:
        color = COLORS[label]
        gain_sub = gain_profile[gain_profile["system"] == label]
        axes[0].plot(
            gain_sub["m"],
            gain_sub["best_available_required_gain"],
            color=color,
            marker="o",
            linewidth=1.7,
            markersize=3.6,
            label=label,
        )
        error_sub = monotone_errors[monotone_errors["system"] == label].sort_values("target_m")
        axes[1].plot(
            error_sub["target_m"],
            error_sub["selected_oracle_mse_at_L1"],
            color=color,
            marker="o",
            linewidth=1.7,
            markersize=3.6,
            label=label,
        )
        padded = error_sub[error_sub["selected_source_m"].astype(int) < error_sub["target_m"].astype(int)]
        if len(padded):
            axes[1].scatter(
                padded["target_m"],
                padded["selected_oracle_mse_at_L1"],
                marker="s",
                s=28,
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                zorder=5,
            )
    axes[0].axhline(1.0, color="#777777", linewidth=0.8, linestyle=(0, (4, 3)))
    axes[0].set_xlabel("latent dimension m")
    axes[0].set_ylabel("best available required gain")
    axes[0].set_xticks(list(ERROR_M_VALUES))
    axes[0].set_title("Required gain, n=8", fontweight="bold")
    axes[1].set_yscale("symlog", linthresh=1e-4, linscale=0.45)
    axes[1].set_xlabel("latent dimension m")
    axes[1].set_ylabel("oracle MSE at L=1")
    axes[1].set_xticks(list(ERROR_M_VALUES))
    axes[1].set_title("Prediction error, n=8", fontweight="bold")
    for ax in axes:
        style_axes(ax)
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper right")
    prefix = FIGURE_DIR / "results_n8_gain_error_curves"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    caption = (
        "n=8 gain and prediction-error profiles. Required-gain curves use the "
        "best available E1 upper bound at each dimension. Prediction-error curves "
        "use the best L=1 oracle MSE among completed source dimensions that can be "
        "zero-padded into the target dimension; square markers indicate reused "
        "lower-dimensional embeddings."
    )
    caption_path = prefix.with_name("results_n8_gain_error_curves_caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return [prefix.with_suffix(ext) for ext in (".pdf", ".png", ".svg")] + [caption_path]


def plot_e2_scaling(e2: pd.DataFrame) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained")
    colors = {
        1: "#0072B2",
        2: "#D55E00",
        3: "#009E73",
        4: "#CC79A7",
    }
    e2_small = e2[e2["m"].astype(int).isin((1, 2, 3, 4))].copy()
    for m in sorted(e2_small["m"].astype(int).unique()):
        sub = e2_small[e2_small["m"].astype(int) == m].sort_values("n")
        color = colors[m]
        axes[0].plot(sub["n"], sub["analytic_lower_bound"], color=color, linestyle=(0, (4, 3)), linewidth=1.15)
        axes[0].plot(sub["n"], sub["explicit_grid_gain"], color=color, linewidth=1.55, label=f"m={m}")
        opt = sub[np.isfinite(sub["optimized_best_gain"].to_numpy(dtype=float))]
        if len(opt):
            axes[0].scatter(opt["n"], opt["optimized_best_gain"], marker="o", s=16, color=color, zorder=4)
        axes[1].plot(sub["n"], sub["grid_to_lower_ratio"], color=color, linewidth=1.45, label=f"m={m}")
        opt_ratio = sub[np.isfinite(sub["optimized_to_lower_ratio"].to_numpy(dtype=float))]
        if len(opt_ratio):
            axes[1].scatter(opt_ratio["n"], opt_ratio["optimized_to_lower_ratio"], marker="o", s=16, color=color, zorder=4)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of states n")
    axes[0].set_ylabel("required gain / bound")
    axes[0].set_title("All-pair action scaling", fontweight="bold")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("number of states n")
    axes[1].set_ylabel("upper/lower ratio")
    axes[1].set_title("Gap to lower bound", fontweight="bold")
    for ax in axes:
        ax.set_xticks(sorted(e2_small["n"].astype(int).unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7.0, title="dimension", title_fontsize=7.0)
    legend_handles = [
        plt.Line2D([0], [0], color="#333333", linestyle=(0, (4, 3)), linewidth=1.1, label="analytic lower"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.5, label="explicit grid"),
        plt.Line2D([0], [0], color="#333333", marker="o", linewidth=0, markersize=4, label="optimized"),
    ]
    axes[1].legend(handles=legend_handles, frameon=False, fontsize=7.0, loc="upper left")
    prefix = FIGURE_DIR / "results_e2_all_pair_scaling"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    caption = (
        "E2 scaling for all-pair actions. Dashed curves are analytic lower bounds, "
        "solid curves are explicit grid constructions, and markers show direct "
        "optimized embeddings where those runs were performed."
    )
    caption_path = prefix.with_name("results_e2_all_pair_scaling_caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return [prefix.with_suffix(ext) for ext in (".pdf", ".png", ".svg")] + [caption_path]


def write_manifest(paths: List[Path], candidates: pd.DataFrame, monotone: pd.DataFrame) -> Path:
    manifest = {
        "used_smoke_outputs": False,
        "inputs": {
            "e1_summary": str(E1_SUMMARY_PATH),
            "e1_results": str(E1_RESULTS_PATH),
            "e2_summary": "existing E2 all-pair scaling summary",
        },
        "n8_oracle_error": {
            "gain_budget": GAIN_BUDGET,
            "num_candidate_rows": int(len(candidates)),
            "num_monotone_rows": int(len(monotone)),
            "candidate_csv": str(FIGURE_DIR / "results_n8_oracle_error_all_candidates.csv"),
            "monotone_csv": str(FIGURE_DIR / "results_n8_oracle_error_monotone.csv"),
            "solver": "scipy SLSQP numerical oracle preview",
        },
        "outputs": [str(path) for path in paths],
    }
    path = FIGURE_DIR / "results_figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    torch.set_default_dtype(dg.DTYPE)
    summary = read_e1_summary()
    results = read_e1_results()
    e2 = read_e2_summary()
    paths: List[Path] = []
    paths.extend(plot_e1_frontier_grid(summary, results))
    paths.extend(plot_e1_heatmaps(summary))
    candidates, monotone = compute_oracle_error_candidates(results)
    paths.extend(plot_n8_gain_error_curves(summary, monotone))
    paths.extend(plot_e2_scaling(e2))
    paths.append(write_manifest(paths, candidates, monotone))
    print("Prepared polished result figures")
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
