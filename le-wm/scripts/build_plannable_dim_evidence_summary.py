from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_SR = {
    "state8": 36,
    "state16": 72,
    "state32": 78,
    "state64": 90,
    "baseline192": 96,
    "state302": "",
    "state502": "",
}

DEFAULT_DIMS = {
    "state8": 8,
    "state16": 16,
    "state32": 32,
    "state64": 64,
    "baseline192": 192,
    "state302": 302,
    "state502": 502,
}


def _read_csv_dict(path: Path, key: str = "model") -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open() as file:
        return {row[key]: row for row in csv.DictReader(file) if key in row}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _build_table(root: Path, models: List[str]) -> List[Dict[str, object]]:
    trained = _read_csv_dict(root / "trained_latent_metric" / "summary.csv")
    candidate = _read_csv_dict(root / "trained_latent_metric" / "candidate_future_metric_summary.csv")
    prediction = _read_csv_dict(root / "prediction_loss_control" / "prediction_loss_summary.csv")
    alias = _read_csv_dict(root / "summaries" / "old_aliasing_summary.csv")
    rows = []
    for model in models:
        trained_row = trained.get(model, {})
        candidate_row = candidate.get(model, {})
        prediction_row = prediction.get(model, {})
        alias_row = alias.get(model, {})
        rows.append(
            {
                "model": model,
                "latent_dim": DEFAULT_DIMS.get(model, ""),
                "SR_100": DEFAULT_SR.get(model, ""),
                "prediction_loss": prediction_row.get("prediction_loss", prediction_row.get("one_step_mse", "")),
                "graph_distance_spearman": trained_row.get("latent_graph_spearman", ""),
                "false_shortcut_rate": trained_row.get("false_shortcut_rate", ""),
                "goal_relative_spearman": trained_row.get("goal_relative_spearman", ""),
                "candidate_goal_metric_spearman": candidate_row.get("candidate_goal_metric_spearman", ""),
                "candidate_false_shortcut_rate": candidate_row.get("candidate_false_shortcut_rate", ""),
                "candidate_pairwise_rank_acc": candidate_row.get("candidate_pairwise_rank_acc", ""),
                "score_aliasing": alias_row.get("score_aliasing", alias_row.get("score_alias", "")),
                "pairwise_rank_acc": alias_row.get("pairwise_rank_acc", ""),
                "regret": alias_row.get("regret", ""),
            }
        )
    return rows


def _plot_evidence_chain(root: Path, rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dims = np.asarray([_as_float(row["latent_dim"]) for row in rows], dtype=float)
    order = np.argsort(dims)
    rows = [rows[i] for i in order]
    dims = dims[order]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("false_shortcut_rate", "lower is better", axes[0, 0]),
        ("candidate_pairwise_rank_acc", "higher is better", axes[0, 1]),
        ("SR_100", "success rate", axes[1, 0]),
        ("graph_distance_spearman", "higher is better", axes[1, 1]),
    ]
    for metric, ylabel, ax in panels:
        values = np.asarray([_as_float(row.get(metric, "")) for row in rows], dtype=float)
        ax.plot(dims, values, marker="o")
        ax.set_xscale("log")
        ax.set_xlabel("latent dimension")
        ax.set_ylabel(metric)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)
        for x in [96, 163, 302]:
            ax.axvline(x, linestyle="--", color="0.7", linewidth=1)
    fig.tight_layout()
    fig.savefig(plots_dir / "evidence_chain_by_latent_dim.png", dpi=220)
    plt.close(fig)


def _write_markdown(root: Path, table_path: Path) -> None:
    text = f"""# Plannable Latent Dimension Evidence

## Main Claim

In reward-free / goal-conditioned latent planning, latent distance is the planner cost. The representation therefore needs more than decodable future structure: transition/reachability structure must be encoded in the latent metric itself.

## How To Read This Package

- `spectra/`: graph-distance MDS spectra and `d_plan(q)` estimates.
- `false_shortcuts/`: low-dimensional MDS embeddings and graph-far/latent-near shortcut rates.
- `trained_latent_metric/`: learned latent Euclidean distances compared against graph/reachability distances.
- `compression_metric/`: post-hoc projection of baseline latents as a geometry-loss control.
- `prediction_loss_control/`: prediction-loss summaries when available.
- `toy/`: decodable-but-not-plannable toy examples.

## Cautious Interpretation

`d_plan(q)` is a geometry-retention curve, not a magic exact latent dimension. Average stress can be low in small dimension while planner-facing false shortcuts remain high. The key diagnostic is whether graph-far futures contract into latent-near pairs, because those are exactly the false shortcuts that a distance-based planner can confuse.

## Outputs

- Model evidence table: `{table_path}`
- Main evidence-chain plot: `plots/evidence_chain_by_latent_dim.png`

## What Remains Weak

This package estimates metric embeddability and compares it against cached model diagnostics. It does not prove optimal representation width, and downstream closed-loop success still depends on policy/CEM details, action distribution, and model rollout quality.
"""
    out = root / "summaries" / "evidence_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper-facing evidence summary table for plannable latent dimension diagnostics.")
    parser.add_argument("--root", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--models", default="state8,state16,state32,state64,baseline192,state302,state502")
    args = parser.parse_args()
    root = Path(args.root)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    rows = _build_table(root, models)
    table_path = root / "summaries" / "model_evidence_table.csv"
    _write_csv(table_path, rows)
    _plot_evidence_chain(root, rows)
    _write_markdown(root, table_path)
    print(f"[evidence_summary] wrote {table_path}")
    print(f"[evidence_summary] wrote {root / 'summaries' / 'evidence_summary.md'}")


if __name__ == "__main__":
    main()
