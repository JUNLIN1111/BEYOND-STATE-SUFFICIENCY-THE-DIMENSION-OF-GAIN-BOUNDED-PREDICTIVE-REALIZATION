from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


FALLBACK_SR = {
    "state8": 36.0,
    "state16": 72.0,
    "state32": 78.0,
    "state64": 90.0,
    "baseline192": 96.0,
    "local_k32": 88.0,
    "global_k32": 42.0,
}

FALLBACK_ALIAS = {
    "state8": {"norm_alias": 0.131, "score_alias": 0.117},
    "state16": {"norm_alias": 0.091, "score_alias": 0.078},
    "state32": {"norm_alias": 0.058, "score_alias": 0.032},
    "state64": {"norm_alias": 0.040, "score_alias": 0.009},
    "baseline192": {"norm_alias": 0.032, "score_alias": 0.003},
}


def _parse_specs(specs: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected spec as model=path, got: {spec}")
        model, path = spec.split("=", 1)
        out[model.strip()] = Path(path)
    return out


def _parse_rates(spec: str) -> Dict[str, float]:
    rates = dict(FALLBACK_SR)
    for item in str(spec).split(","):
        if not item.strip():
            continue
        model, value = item.split("=", 1)
        rates[model.strip()] = float(value)
    return rates


def _load_dataset_span(specs: Sequence[str]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for model, path in _parse_specs(specs).items():
        with path.open() as file:
            data = json.load(file)
        rows[model] = {
            "dataset_rank90": float(data.get("rank90", data.get("rank_90", float("nan")))),
            "dataset_rank99": float(data.get("rank99", data.get("rank_99", float("nan")))),
            "dataset_effective_rank": float(data.get("effective_rank", float("nan"))),
            "local_dim": float(data.get("local_dim", 2)),
            "latent_dim": float(data.get("latent_dim", float("nan"))),
        }
    return rows


def _load_candidate_span(specs: Sequence[str]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for model, path in _parse_specs(specs).items():
        with path.open() as file:
            data = json.load(file)
        summary = data.get("summary", {})
        rows[model] = {
            "candidate_rank90": float(summary.get("tangent_span_rank90_mean", float("nan"))),
            "candidate_rank99": float(summary.get("tangent_span_rank99_mean", float("nan"))),
            "candidate_effective_rank": float(summary.get("tangent_span_effective_rank_mean", float("nan"))),
            "candidate_global_rank90": float(summary.get("global_rank90_mean", float("nan"))),
            "candidate_global_rank99": float(summary.get("global_rank99_mean", float("nan"))),
            "latent_dim": float(data.get("latent_dim", float("nan"))),
        }
    return rows


def _load_aliasing(specs: Sequence[str], gamma: float, rho: float, tau: float) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for model, path in _parse_specs(specs).items():
        with path.open() as file:
            data = json.load(file)
        aggregate = data["aggregate"]
        rows[model] = {
            "norm_alias": float(aggregate["alias_norm"][f"gamma={gamma},rho={rho}"]["mean"]),
            "score_alias": float(aggregate["alias_score"][f"gamma={gamma},tau={tau}"]["mean"]),
            "spearman": float(aggregate["auxiliary"]["mean_spearman"]),
            "alias_source": "file",
        }
    return rows


def _merge_rows(
    dataset_rows: Dict[str, Dict[str, float]],
    candidate_rows: Dict[str, Dict[str, float]],
    alias_rows: Dict[str, Dict[str, float]],
    success_rates: Dict[str, float],
) -> List[Dict[str, float | str]]:
    models = sorted(set(dataset_rows) | set(candidate_rows) | set(alias_rows) | set(FALLBACK_ALIAS))
    merged: List[Dict[str, float | str]] = []
    for model in models:
        row: Dict[str, float | str] = {"model": model}
        row.update(dataset_rows.get(model, {}))
        row.update(candidate_rows.get(model, {}))
        alias = dict(FALLBACK_ALIAS.get(model, {}))
        alias["alias_source"] = "fallback" if alias else "missing"
        alias.update(alias_rows.get(model, {}))
        row.update(alias)
        row["success_rate"] = success_rates.get(model, float("nan"))
        merged.append(row)
    return merged


def _float(row: Dict[str, float | str], key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def _model_order(model: str) -> tuple[int, str]:
    order = {"state8": 0, "state16": 1, "state32": 2, "state64": 3, "baseline192": 4, "local_k32": 5, "global_k32": 6}
    return order.get(model, 100), model


def _plot_rank90_by_model(rows: List[Dict[str, float | str]], output_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: _model_order(str(row["model"])))
    models = [str(row["model"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    dataset = [_float(row, "dataset_rank90") for row in rows]
    candidate = [_float(row, "candidate_rank90") for row in rows]
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, dataset, width=width, label="dataset observations")
    plt.bar(x + width / 2, candidate, width=width, label="candidate futures")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("tangent span rank90")
    plt.title("Tangent span rank90 by model")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_dir / "tangent_span_rank90_by_model")


def _scatter(rows: List[Dict[str, float | str]], x_key: str, y_key: str, xlabel: str, ylabel: str, title: str, path: Path) -> None:
    plt.figure(figsize=(6.5, 5))
    for row in rows:
        x = _float(row, x_key)
        y = _float(row, y_key)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        plt.scatter([x], [y], s=75)
        plt.annotate(str(row["model"]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    _savefig(path)


def _plot_local_dim(rows: List[Dict[str, float | str]], output_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: _model_order(str(row["model"])))
    models = [str(row["model"]) for row in rows]
    x = np.arange(len(rows))
    local = [_float(row, "local_dim") for row in rows]
    rank90 = [_float(row, "dataset_rank90") for row in rows]
    plt.figure(figsize=(9, 5))
    plt.plot(x, local, marker="o", label="local_dim")
    plt.plot(x, rank90, marker="s", label="dataset tangent span rank90")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("dimension / rank")
    plt.title("Low local dimension, high tangent span")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_dir / "local_dim_vs_tangent_span")


def _write_summary_csv(rows: List[Dict[str, float | str]], path: Path) -> None:
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict[str, float | str]], zero_padding_csv: str, path: Path) -> None:
    lines = [
        "# Tangent Span Interpretation",
        "",
        "Local intrinsic dimension can be low while tangent-span dimension is high.",
        "High ambient dimension alone does not reduce aliasing; zero-padding is the control for nominal dimension.",
        "",
        "## Summary Table",
        "",
        "| model | dataset rank90 | candidate rank90 | score alias | norm alias | SR | alias source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: _model_order(str(item["model"]))):
        lines.append(
            f"| {row['model']} | {_float(row, 'dataset_rank90'):.3g} | {_float(row, 'candidate_rank90'):.3g} | "
            f"{_float(row, 'score_alias'):.4g} | {_float(row, 'norm_alias'):.4g} | {_float(row, 'success_rate'):.3g} | "
            f"{row.get('alias_source', '')} |"
        )
    lines.extend(
        [
            "",
            "## Zero-Padding Control",
            "",
            f"Zero-padding CSV: `{zero_padding_csv}`" if zero_padding_csv else "Zero-padding CSV not provided.",
            "",
            "Expected interpretation: zero-padding changes nominal ambient dimension but leaves distances, scores, aliasing, and ranking unchanged.",
            "",
            "## Cautious Interpretation",
            "",
            "- Do not claim that high ambient dimension automatically reduces aliasing.",
            "- The evidence supports that learned high-D latents can use extra coordinates to unfold many local low-D future charts.",
            "- Low-D models may alias because they cannot realize enough distinct tangent/offset directions.",
            "- Although local intrinsic dimension is low, the collection of local tangent planes spans many ambient directions.",
            "",
            "Suggested sentence:",
            "",
            "> Although local intrinsic dimension is low, the collection of local tangent planes spans many ambient directions. This indicates that the planner latent is not a single global low-dimensional plane, but a multi-chart geometry: locally low-dimensional futures are unfolded across a wider ambient space. This helps explain why low local intrinsic dimension does not imply low plannable latent dimension.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tangent-span summaries against aliasing/ranking/SR.")
    parser.add_argument("--dataset_jsons", nargs="*", default=[], help="model=tangent_span_observation.json")
    parser.add_argument("--candidate_jsons", nargs="*", default=[], help="model=tangent_span_candidate_summary.json")
    parser.add_argument("--aliasing_jsons", nargs="*", default=[], help="model=fixedN_aliasing.json")
    parser.add_argument("--success_rates", default="")
    parser.add_argument("--zero_padding_csv", default="")
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--output_dir", default="rollout_results/tangent_span_plots")
    parser.add_argument("--markdown_output", default="rollout_results/tangent_span_interpretation.md")
    args = parser.parse_args()

    rows = _merge_rows(
        _load_dataset_span(args.dataset_jsons),
        _load_candidate_span(args.candidate_jsons),
        _load_aliasing(args.aliasing_jsons, args.gamma, args.rho, args.tau),
        _parse_rates(args.success_rates),
    )
    output_dir = Path(args.output_dir)
    _write_summary_csv(rows, output_dir / "tangent_span_summary_table.csv")
    _plot_rank90_by_model(rows, output_dir)
    _scatter(
        rows,
        "candidate_rank90",
        "score_alias",
        "candidate-future tangent span rank90",
        "ScoreAlias",
        "Candidate tangent span vs score aliasing",
        output_dir / "tangent_span_vs_score_alias",
    )
    _scatter(
        rows,
        "candidate_rank90",
        "success_rate",
        "candidate-future tangent span rank90",
        "closed-loop SR",
        "Candidate tangent span vs SR",
        output_dir / "tangent_span_vs_sr",
    )
    _plot_local_dim(rows, output_dir)
    _write_markdown(rows, args.zero_padding_csv, Path(args.markdown_output))
    print(f"[tangent_plot] wrote plots to {output_dir}")
    print(f"[tangent_plot] wrote markdown to {args.markdown_output}")


if __name__ == "__main__":
    main()
