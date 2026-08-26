from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not path:
        return []
    with Path(path).open() as file:
        return list(csv.DictReader(file))


def _float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _method_label(row: Dict[str, str]) -> str:
    method = row.get("projection_method", "")
    if method == "none":
        return "original"
    if method == "random":
        return "random projection"
    if method == "pca":
        return "PCA"
    return method or row.get("source", "unknown")


def _filtered(rows: Sequence[Dict[str, str]], gamma: float | None = None) -> List[Dict[str, str]]:
    if gamma is None:
        return list(rows)
    return [row for row in rows if abs(_float(row, "gamma") - gamma) < 1e-8]


def _with_goal_radius_ratios(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        if "pairwise_median_over_goal_radius" not in copied or copied.get("pairwise_median_over_goal_radius", "") == "":
            pairwise_median = _float(copied, "pairwise_dist_median")
            goal_radius = _float(copied, "M_goal_radius")
            if math.isfinite(pairwise_median) and math.isfinite(goal_radius) and goal_radius > 0:
                copied["pairwise_median_over_goal_radius"] = str(pairwise_median / goal_radius)
        out.append(copied)
    return out


def _aggregate(rows: Iterable[Dict[str, str]], y_key: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        y = _float(row, y_key)
        dim = int(round(_float(row, "effective_dim")))
        if math.isfinite(y) and dim > 0:
            grouped[(_method_label(row), dim)].append(y)
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        out[key] = {
            "mean": float(np.nanmean(array)),
            "stderr": float(np.nanstd(array) / math.sqrt(max(np.sum(~np.isnan(array)), 1))),
        }
    return out


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def _plot_dim_curve(rows: Sequence[Dict[str, str]], y_key: str, ylabel: str, title: str, output_path: Path) -> None:
    stats = _aggregate(rows, y_key)
    styles = {
        "random projection": {"marker": "o", "linewidth": 2.5, "alpha": 1.0},
        "PCA": {"marker": "s", "linewidth": 1.5, "alpha": 0.45},
        "original": {"marker": "*", "linewidth": 0.0, "alpha": 1.0},
    }
    plt.figure(figsize=(7.2, 4.8))
    for label in ["random projection", "PCA", "original"]:
        points = sorted((dim, values["mean"], values["stderr"]) for (method, dim), values in stats.items() if method == label)
        if not points:
            continue
        xs = np.asarray([point[0] for point in points], dtype=np.float64)
        ys = np.asarray([point[1] for point in points], dtype=np.float64)
        es = np.asarray([point[2] for point in points], dtype=np.float64)
        style = styles[label]
        if label == "original":
            plt.scatter(xs, ys, marker=style["marker"], s=150, label="baseline192 original", zorder=5)
        else:
            plt.errorbar(xs, ys, yerr=es, label=label, capsize=3, **style)
    plt.xscale("log", base=2)
    plt.xlabel("effective dimension")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_path)


def _plot_rescale(rows: Sequence[Dict[str, str]], score_rho: float, output_path: Path) -> None:
    if not rows:
        return
    keys = [
        ("score_alias_tau_0.1", "fixed tau score alias"),
        (f"norm_score_alias_score_rho_{score_rho:g}", f"normalized score alias {score_rho:g}"),
        ("pairwise_rank_acc", "pairwise rank acc"),
        ("spearman", "Spearman"),
    ]
    means_before = []
    means_after = []
    labels = []
    for key, label in keys:
        before = np.asarray([_float(row, f"{key}_before") for row in rows], dtype=np.float64)
        after = np.asarray([_float(row, f"{key}_after") for row in rows], dtype=np.float64)
        if np.all(~np.isfinite(before)) or np.all(~np.isfinite(after)):
            continue
        labels.append(label)
        means_before.append(float(np.nanmean(before)))
        means_after.append(float(np.nanmean(after)))
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - 0.18, means_before, width=0.36, label="before R rescale")
    plt.bar(x + 0.18, means_after, width=0.36, label="after R rescale")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("metric value")
    plt.title("Compression rescale sanity")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean compression plots for gamma=0.02 main figures.")
    parser.add_argument("--metrics_csv", required=True)
    parser.add_argument("--scale_csv", default="")
    parser.add_argument("--rescale_csv", default="")
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--score_rho", type=float, default=0.005)
    parser.add_argument("--output_dir", default="rollout_results/compression_clean/plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics = _filtered(_read_csv(args.metrics_csv), args.gamma)
    scale_rows = _with_goal_radius_ratios(_read_csv(args.scale_csv))
    rescale_rows = _filtered(_read_csv(args.rescale_csv), args.gamma)

    _plot_dim_curve(
        metrics,
        f"norm_geom_alias_rho_{args.rho:g}",
        f"normalized geometric aliasing, rho={args.rho:g}",
        f"Compression aliasing, gamma={args.gamma:g}",
        output_dir / "compression_clean_norm_alias_gamma002_rho01",
    )
    _plot_dim_curve(
        metrics,
        "pairwise_rank_acc",
        "pairwise rank accuracy",
        f"Compression ranking, gamma={args.gamma:g}",
        output_dir / "compression_clean_pairwise_rank_acc_gamma002",
    )
    _plot_dim_curve(
        metrics,
        f"norm_score_alias_score_rho_{args.score_rho:g}",
        f"normalized score alias, score_rho={args.score_rho:g}",
        f"Compression score aliasing, gamma={args.gamma:g}",
        output_dir / "compression_clean_norm_score_alias_gamma002",
    )
    if scale_rows:
        _plot_dim_curve(
            scale_rows,
            "R_max",
            "R_max",
            "Compression scale by dimension",
            output_dir / "compression_scale_Rmax_vs_dim",
        )
        _plot_dim_curve(
            scale_rows,
            "R_max_over_sqrt_dim",
            "R_max / sqrt(dim)",
            "Compression normalized scale by dimension",
            output_dir / "compression_scale_Rmax_over_sqrt_dim_vs_dim",
        )
        _plot_dim_curve(
            scale_rows,
            "pairwise_median_over_Rmax",
            "pairwise median / R_max",
            "Compression pairwise scale by dimension",
            output_dir / "compression_pairwise_median_over_R_vs_dim",
        )
        _plot_dim_curve(
            scale_rows,
            "M_goal_radius",
            "M_goal_radius = max ||z_i - z_goal||",
            "Compression planner-centered radius by dimension",
            output_dir / "compression_goal_radius_vs_dim",
        )
        _plot_dim_curve(
            scale_rows,
            "pairwise_median_over_goal_radius",
            "pairwise median / M_goal_radius",
            "Compression relative spread around goal",
            output_dir / "compression_pairwise_median_over_goal_radius_vs_dim",
        )
    _plot_rescale(
        rescale_rows,
        args.score_rho,
        output_dir / "compression_rescale_sanity_score_alias",
    )
    print(f"[plot_compression] wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
