from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _read_csv(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return []
    with Path(path).open() as file:
        return list(csv.DictReader(file))


def _float(row: Dict[str, str], key: str, default=float("nan")) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def _label(row: Dict[str, str]) -> str:
    if "model" in row and row["model"]:
        return f"{row['model']}_D{row.get('effective_dim', '')}"
    return f"{row.get('source', '')}_{row.get('projection_method', '')}_D{row.get('effective_dim', '')}"


def _plot_lines(rows, x_key, y_key, title, xlabel, ylabel, output_path):
    groups = {}
    for row in rows:
        groups.setdefault(_label(row), []).append(row)
    plt.figure(figsize=(8, 5))
    for label, group in sorted(groups.items()):
        points = sorted((_float(row, x_key), _float(row, y_key)) for row in group)
        points = [(x, y) for x, y in points if x == x and y == y]
        if not points:
            continue
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_by_dim(rows, y_key, title, ylabel, output_path):
    groups = {}
    for row in rows:
        groups.setdefault(f"{row.get('source', '')}_{row.get('projection_method', '')}", []).append(row)
    plt.figure(figsize=(8, 5))
    for label, group in sorted(groups.items()):
        points = sorted((_float(row, "effective_dim"), _float(row, y_key)) for row in group)
        points = [(x, y) for x, y in points if x == x and y == y]
        if not points:
            continue
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", label=label)
    plt.title(title)
    plt.xlabel("effective_dim")
    plt.ylabel(ylabel)
    plt.xscale("log", base=2)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot aliasing diagnostic aggregate CSVs.")
    parser.add_argument("--vary_k_aggregate_csv", default=None)
    parser.add_argument("--compression_aggregate_csv", default=None)
    parser.add_argument("--output_dir", default="rollout_results/aliasing_plots")
    parser.add_argument("--rho", default="0.1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vary_rows = _read_csv(args.vary_k_aggregate_csv)
    compression_rows = _read_csv(args.compression_aggregate_csv)
    alias_key = f"norm_geom_alias_rho_{args.rho}_mean"

    if vary_rows:
        _plot_lines(
            vary_rows,
            "K_gamma_mean",
            alias_key,
            "Vary-K normalized aliasing vs K_gamma",
            "K_gamma mean",
            f"norm geom alias rho={args.rho}",
            output_dir / "vary_K_norm_alias_vs_Kgamma.png",
        )
        _plot_lines(
            vary_rows,
            "K_gamma_mean",
            "pairwise_rank_acc_mean",
            "Vary-K pairwise accuracy vs K_gamma",
            "K_gamma mean",
            "pairwise rank accuracy",
            output_dir / "vary_K_pairwise_acc_vs_Kgamma.png",
        )
        _plot_lines(
            vary_rows,
            "K_gamma_mean",
            "regret_mean",
            "Vary-K regret vs K_gamma",
            "K_gamma mean",
            "regret",
            output_dir / "vary_K_regret_vs_Kgamma.png",
        )
    if compression_rows:
        _plot_by_dim(
            compression_rows,
            alias_key,
            "Compression normalized aliasing by dimension",
            f"norm geom alias rho={args.rho}",
            output_dir / "compression_norm_alias_by_dim.png",
        )
        y_key = "pairwise_rank_acc_mean" if "pairwise_rank_acc_mean" in compression_rows[0] else "topk_recall_30_mean"
        _plot_by_dim(
            compression_rows,
            y_key,
            "Compression ranking by dimension",
            y_key.replace("_mean", ""),
            output_dir / "compression_ranking_by_dim.png",
        )
    print(f"[plot] wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
