from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_specs(specs: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected spec as model=path, got: {spec}")
        name, path = spec.split("=", 1)
        out[name.strip()] = Path(path)
    return out


def _parse_rates(spec: str) -> Dict[str, float]:
    rates = {}
    for item in str(spec).split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        rates[name.strip()] = float(value)
    return rates


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _float(row: Dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        corr = spearmanr(x_arr, y_arr).correlation
        return float(corr) if corr is not None else float("nan")
    except ImportError:
        xr = np.argsort(np.argsort(x_arr)).astype(np.float64)
        yr = np.argsort(np.argsort(y_arr)).astype(np.float64)
        return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _alias_json_metrics(path: Path, gamma: float, rho: float, tau: float) -> Dict[str, float]:
    with path.open() as file:
        data = json.load(file)
    aggregate = data["aggregate"]
    gamma_key = str(gamma)
    alias_norm_key = f"gamma={gamma},rho={rho}"
    score_alias_key = f"gamma={gamma},tau={tau}"
    out = {
        "latent_dim": float(data.get("model", {}).get("latent_dim", float("nan"))),
        "norm_geom_alias": float(aggregate["alias_norm"][alias_norm_key]["mean"]),
        "score_alias": float(aggregate["alias_score"][score_alias_key]["mean"]),
        "spearman_ranking": float(aggregate["auxiliary"]["mean_spearman"]),
        "candidate_regret_mean": float(aggregate["auxiliary"]["candidate_regret_mean"]),
        "elite_win_rate": float(aggregate.get("elite", {}).get("mean_elite_progress_win_rate", float("nan"))),
        "top30_overlap": float(aggregate.get("elite", {}).get("mean_elite_overlap", float("nan"))),
        "expert_in_latent_topk": float(aggregate.get("expert", {}).get("expert_in_latent_topk_rate", float("nan"))),
    }
    return out


def _prediction_rows(path: Path) -> Dict[str, Dict[str, str]]:
    rows = _read_csv(path)
    return {row["model"]: row for row in rows}


def _plot_scatter(rows: List[Dict[str, object]], x_key: str, y_key: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(6.5, 5.0))
    for row in rows:
        x = _float(row, x_key)
        y = _float(row, y_key)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        plt.scatter([x], [y], s=70)
        plt.annotate(row["model"], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def _plot_corr_bar(rows: List[Dict[str, object]], corr_rows: List[Dict[str, object]], target: str, path: Path) -> None:
    selected = [row for row in corr_rows if row["target"] == target]
    selected = sorted(selected, key=lambda row: abs(float(row["spearman_corr"])), reverse=True)
    labels = [row["predictor"] for row in selected]
    values = [abs(float(row["spearman_corr"])) for row in selected]
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel(f"|Spearman corr with {target}|")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def _write_markdown_summary(output_dir: Path, rows: List[Dict[str, object]], corr_rows: List[Dict[str, object]]) -> None:
    def best_for(target: str):
        candidates = [row for row in corr_rows if row["target"] == target and math.isfinite(float(row["spearman_corr"]))]
        candidates = sorted(candidates, key=lambda row: abs(float(row["spearman_corr"])), reverse=True)
        return candidates[:5]

    lines = [
        "# Reviewer Response Summary: Prediction Loss vs Future Aliasing",
        "",
        "This analysis compares generic multi-step latent prediction loss against planner-facing aliasing diagnostics.",
        "",
        "Cautious interpretation:",
        "- Prediction loss is necessary but not sufficient for planning.",
        "- Aliasing captures pairwise separability of task-distinct candidate futures in the planner score space.",
        "- Offline aliasing diagnostics can help flag insufficient future-separability capacity before full closed-loop evaluation.",
        "",
        "## Model Table",
        "",
        "| model | SR | normalized mean L2 | norm geom alias | score alias | Spearman ranking | regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {_float(row, 'success_rate'):.3g} | "
            f"{_float(row, 'normalized_mean_multistep_l2_mean'):.4g} | "
            f"{_float(row, 'norm_geom_alias'):.4g} | "
            f"{_float(row, 'score_alias'):.4g} | "
            f"{_float(row, 'spearman_ranking'):.4g} | "
            f"{_float(row, 'candidate_regret_mean'):.4g} |"
        )
    lines.extend(["", "## Strongest Spearman Correlations", ""])
    for target in ["success_rate", "spearman_ranking", "candidate_regret_mean"]:
        lines.append(f"### Target: {target}")
        for row in best_for(target):
            lines.append(f"- `{row['predictor']}`: Spearman={float(row['spearman_corr']):.4f}")
        lines.append("")
    lines.append("Avoid overclaiming: these results test whether aliasing provides planner-facing information complementary to prediction loss.")
    (output_dir / "reviewer_response_summary.md").write_text("\n".join(lines))


def _compression_control(
    compression_csv: Path,
    baseline_pred_row: Dict[str, str],
    output_dir: Path,
    gamma: float,
    rho: float,
    score_rho: float,
) -> None:
    if not compression_csv:
        return
    rows = [row for row in _read_csv(compression_csv) if abs(_float(row, "gamma") - gamma) < 1e-8]
    out_rows: List[Dict[str, object]] = []
    pred_loss = _float(baseline_pred_row, "normalized_mean_multistep_l2_mean")
    for row in rows:
        out_rows.append(
            {
                "source": row.get("source", ""),
                "projection_method": row.get("projection_method", ""),
                "effective_dim": int(round(_float(row, "effective_dim"))),
                "projection_seed": int(round(_float(row, "projection_seed", -1))),
                "original_model_prediction_loss": pred_loss,
                "norm_score_alias": _float(row, f"norm_score_alias_score_rho_{score_rho:g}"),
                "norm_geom_alias": _float(row, f"norm_geom_alias_rho_{rho:g}"),
                "pairwise_rank_acc": _float(row, "pairwise_rank_acc"),
                "regret": _float(row, "regret"),
                "pairwise_median_over_Rmax": _float(row, "pairwise_median_over_Rmax"),
                "pairwise_median_over_goal_radius": _float(row, "pairwise_median_over_goal_radius"),
            }
        )
    _write_csv(output_dir / "compression_same_prediction_loss_control.csv", out_rows)
    grouped = defaultdict(list)
    for row in out_rows:
        grouped[(row["projection_method"], row["effective_dim"])].append(row)
    plt.figure(figsize=(7.5, 5.0))
    for method in ["random", "pca", "none"]:
        points = []
        for (row_method, dim), group in grouped.items():
            if row_method != method:
                continue
            values = np.asarray([_float(row, "norm_score_alias") for row in group], dtype=np.float64)
            points.append((dim, float(np.nanmean(values))))
        points = sorted(points)
        if points:
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o" if method != "none" else "*", label=method)
    plt.xscale("log", base=2)
    plt.xlabel("effective dimension")
    plt.ylabel(f"normalized score alias, score_rho={score_rho:g}")
    plt.title("Same trained prediction loss; different planner-space aliasing")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "compression_same_prediction_loss_different_aliasing.png", dpi=220)
    plt.savefig(output_dir / "compression_same_prediction_loss_different_aliasing.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aliasing diagnostics against multi-step prediction loss.")
    parser.add_argument("--aliasing_jsons", nargs="+", required=True, help="List of model=aliasing.json")
    parser.add_argument("--prediction_loss_summary", required=True)
    parser.add_argument("--success_rates", required=True, help="Comma list, e.g. state8=36,state16=72")
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--score_rho", type=float, default=0.005)
    parser.add_argument("--compression_metrics_csv", default="")
    parser.add_argument("--baseline_model", default="baseline192")
    parser.add_argument("--output_dir", default="rollout_results/prediction_loss")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_by_model = _prediction_rows(Path(args.prediction_loss_summary))
    sr = _parse_rates(args.success_rates)
    rows: List[Dict[str, object]] = []
    for model, path in _parse_specs(args.aliasing_jsons).items():
        if model not in pred_by_model:
            print(f"[compare] warning: missing prediction loss row for {model}; skipping")
            continue
        pred = pred_by_model[model]
        row: Dict[str, object] = {"model": model, "success_rate": sr.get(model, float("nan"))}
        row.update(_alias_json_metrics(path, args.gamma, args.rho, args.tau))
        for key in [
            "normalized_terminal_l2_mean",
            "normalized_mean_multistep_l2_mean",
            "terminal_latent_mse_mean",
            "mean_multistep_latent_mse_mean",
            "per_dim_terminal_mse_mean",
            "per_dim_mean_multistep_mse_mean",
            "variance_normalized_mean_mse_mean",
        ]:
            row[key] = _float(pred, key)
        rows.append(row)

    predictors = [
        "normalized_terminal_l2_mean",
        "normalized_mean_multistep_l2_mean",
        "terminal_latent_mse_mean",
        "mean_multistep_latent_mse_mean",
        "per_dim_terminal_mse_mean",
        "per_dim_mean_multistep_mse_mean",
        "variance_normalized_mean_mse_mean",
        "norm_geom_alias",
        "score_alias",
    ]
    targets = ["success_rate", "spearman_ranking", "candidate_regret_mean", "elite_win_rate", "top30_overlap"]
    corr_rows: List[Dict[str, object]] = []
    for predictor in predictors:
        x = [_float(row, predictor) for row in rows]
        for target in targets:
            y = [_float(row, target) for row in rows]
            corr_rows.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "pearson_corr": _pearson(x, y),
                    "spearman_corr": _spearman(x, y),
                    "n_models": int(np.sum(np.isfinite(x) & np.isfinite(y))),
                }
            )

    _write_csv(output_dir / "alias_vs_prediction_summary.csv", rows)
    _write_csv(output_dir / "correlation_table.csv", corr_rows)
    _write_markdown_summary(output_dir, rows, corr_rows)
    _plot_scatter(rows, "normalized_mean_multistep_l2_mean", "success_rate", "normalized mean multistep latent L2", "closed-loop SR", output_dir / "prediction_loss_vs_sr")
    _plot_scatter(rows, "norm_geom_alias", "success_rate", "NormAlias gamma=.02,rho=.1", "closed-loop SR", output_dir / "aliasing_vs_sr")
    _plot_scatter(rows, "normalized_mean_multistep_l2_mean", "spearman_ranking", "normalized prediction loss", "Spearman ranking", output_dir / "prediction_loss_vs_pairwise_rank_acc")
    _plot_scatter(rows, "norm_geom_alias", "spearman_ranking", "NormAlias gamma=.02,rho=.1", "Spearman ranking", output_dir / "aliasing_vs_pairwise_rank_acc")
    _plot_scatter(rows, "normalized_mean_multistep_l2_mean", "candidate_regret_mean", "normalized prediction loss", "candidate regret", output_dir / "prediction_loss_vs_regret")
    _plot_scatter(rows, "norm_geom_alias", "candidate_regret_mean", "NormAlias gamma=.02,rho=.1", "candidate regret", output_dir / "aliasing_vs_regret")
    _plot_corr_bar(rows, corr_rows, "success_rate", output_dir / "predictor_correlation_bar")

    if args.compression_metrics_csv and args.baseline_model in pred_by_model:
        _compression_control(
            Path(args.compression_metrics_csv),
            pred_by_model[args.baseline_model],
            output_dir,
            args.gamma,
            args.rho,
            args.score_rho,
        )
    print(f"[compare] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
