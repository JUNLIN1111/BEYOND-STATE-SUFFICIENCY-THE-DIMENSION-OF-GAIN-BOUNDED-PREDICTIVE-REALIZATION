from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _parse_specs(specs: Sequence[str]) -> Dict[str, Path]:
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected model=path, got: {spec}")
        name, path = spec.split("=", 1)
        out[name.strip()] = Path(path)
    return out


def _parse_rates(spec: str) -> Dict[str, float]:
    out = {}
    for item in str(spec).split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        out[name.strip()] = float(value)
    return out


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


def _float(row: Dict[str, object], key: str, default=float("nan")) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _model_dim(model: str, fallback: float) -> int:
    if model == "baseline192":
        return 192
    digits = "".join(ch for ch in model if ch.isdigit())
    return int(digits) if digits else int(round(fallback))


def _alias_data(path: Path, gamma: float, rho: float, tau: float) -> Dict[str, object]:
    with path.open() as file:
        data = json.load(file)
    aggregate = data["aggregate"]
    gamma_key = str(gamma)
    norm_key = f"gamma={gamma},rho={rho}"
    score_key = f"gamma={gamma},tau={tau}"
    contexts = data.get("contexts", [])
    return {
        "latent_dim": int(data.get("model", {}).get("latent_dim", -1)),
        "norm_geom_alias": float(aggregate["alias_norm"][norm_key]["mean"]),
        "score_alias": float(aggregate["alias_score"][score_key]["mean"]),
        "spearman_ranking": float(aggregate["auxiliary"]["mean_spearman"]),
        "candidate_regret_mean": float(aggregate["auxiliary"]["candidate_regret_mean"]),
        "elite_win_rate": float(aggregate.get("elite", {}).get("mean_elite_progress_win_rate", float("nan"))),
        "top30_overlap": float(aggregate.get("elite", {}).get("mean_elite_overlap", float("nan"))),
        "contexts_norm_geom_alias": [float(ctx["alias_norm"][norm_key]["alias_rate"]) for ctx in contexts],
        "contexts_score_alias": [float(ctx["alias_score"][score_key]["alias_rate"]) for ctx in contexts],
        "contexts_spearman": [float(ctx["auxiliary"]["spearman"]) for ctx in contexts],
    }


def _prediction_by_model(path: Path) -> Dict[str, Dict[str, str]]:
    return {row["model"]: row for row in _read_csv(path)}


def _select_smallest(rows: List[Dict[str, object]], predicate) -> Dict[str, object] | None:
    for row in sorted(rows, key=lambda item: int(item["D"])):
        if predicate(row):
            return row
    return None


def _selection_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    best_pred = min(_float(row, "normalized_mean_multistep_l2") for row in rows)
    baseline = next(row for row in rows if row["model"] == "baseline192")
    for delta in [0.05, 0.10, 0.20]:
        selected = _select_smallest(rows, lambda row, d=delta: _float(row, "normalized_mean_multistep_l2") <= best_pred * (1.0 + d))
        out.append(_selection_record("prediction_within_best", delta, selected))
    selected = min(rows, key=lambda row: _float(row, "normalized_mean_multistep_l2"))
    out.append(_selection_record("prediction_min_loss", 0.0, selected))
    for metric in ["norm_geom_alias", "score_alias"]:
        base_value = _float(baseline, metric)
        for delta in [0.25, 0.50, 1.00]:
            selected = _select_smallest(rows, lambda row, m=metric, b=base_value, d=delta: _float(row, m) <= b * (1.0 + d))
            out.append(_selection_record(f"{metric}_within_baseline", delta, selected))
    base_rank = _float(baseline, "spearman_ranking")
    for delta in [0.01, 0.02, 0.05]:
        selected = _select_smallest(rows, lambda row, b=base_rank, d=delta: _float(row, "spearman_ranking") >= b - d)
        out.append(_selection_record("spearman_within_baseline", delta, selected))
    return out


def _selection_record(rule: str, delta: float, selected: Dict[str, object] | None) -> Dict[str, object]:
    if selected is None:
        return {"rule": rule, "delta": delta, "selected_model": "none", "selected_D": float("nan"), "selected_SR": float("nan")}
    return {
        "rule": rule,
        "delta": delta,
        "selected_model": selected["model"],
        "selected_D": selected["D"],
        "selected_SR": selected["success_rate"],
    }


def _budget_rows(model_rows: Dict[str, Dict[str, object]], budget_windows: Sequence[int], repeat_seeds: int) -> List[Dict[str, object]]:
    out = []
    for budget in budget_windows:
        for seed in range(repeat_seeds):
            selected_indices = {}
            for model, row in model_rows.items():
                n = len(row["contexts_norm_geom_alias"])
                stable_model_hash = sum((idx + 1) * ord(ch) for idx, ch in enumerate(model))
                local_rng = np.random.default_rng(seed + 1000 * budget + stable_model_hash)
                selected_indices[model] = local_rng.choice(n, size=min(budget, n), replace=False)
            sampled = []
            for model, row in model_rows.items():
                idx = selected_indices[model]
                copied = dict(row)
                copied["norm_geom_alias"] = float(np.nanmean(np.asarray(row["contexts_norm_geom_alias"])[idx]))
                copied["score_alias"] = float(np.nanmean(np.asarray(row["contexts_score_alias"])[idx]))
                copied["spearman_ranking"] = float(np.nanmean(np.asarray(row["contexts_spearman"])[idx]))
                sampled.append(copied)
            base_alias = next(row for row in sampled if row["model"] == "baseline192")["norm_geom_alias"]
            selected = _select_smallest(sampled, lambda row, b=base_alias: _float(row, "norm_geom_alias") <= b * 1.5)
            out.append(
                {
                    "budget_windows": int(budget),
                    "seed": int(seed),
                    "rule": "budget_norm_geom_alias_within_1.5x_baseline",
                    "selected_model": selected["model"] if selected else "none",
                    "selected_D": selected["D"] if selected else float("nan"),
                    "selected_SR": selected["success_rate"] if selected else float("nan"),
                }
            )
    return out


def _plot_metric_vs_d(rows: List[Dict[str, object]], output_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: int(row["D"]))
    xs = [row["D"] for row in rows]
    plt.figure(figsize=(8, 5))
    for key, label in [
        ("normalized_mean_multistep_l2", "prediction loss"),
        ("norm_geom_alias", "norm geom alias"),
        ("score_alias", "score alias"),
    ]:
        vals = np.asarray([_float(row, key) for row in rows], dtype=np.float64)
        vals = vals / np.nanmax(vals)
        plt.plot(xs, vals, marker="o", label=label)
    sr = np.asarray([_float(row, "success_rate") for row in rows], dtype=np.float64)
    plt.plot(xs, sr / np.nanmax(sr), marker="*", label="SR normalized")
    plt.xscale("log", base=2)
    plt.xlabel("latent dimension")
    plt.ylabel("normalized metric")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "dimension_selection_metric_vs_D.png", dpi=220)
    plt.savefig(output_dir / "dimension_selection_metric_vs_D.pdf")
    plt.close()


def _plot_budget(rows: List[Dict[str, object]], output_dir: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["budget_windows"])].append(_float(row, "selected_D"))
    labels = sorted(grouped)
    data = [grouped[label] for label in labels]
    plt.figure(figsize=(7, 4.8))
    plt.boxplot(data, labels=[str(label) for label in labels])
    plt.xlabel("budget windows")
    plt.ylabel("selected D")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "selected_D_by_budget_aliasing.png", dpi=220)
    plt.savefig(output_dir / "selected_D_by_budget_aliasing.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline latent dimension selection diagnostics.")
    parser.add_argument("--aliasing_jsons", nargs="+", required=True)
    parser.add_argument("--prediction_loss_summary", required=True)
    parser.add_argument("--success_rates", required=True)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--budget_windows", default="3,5,10,20")
    parser.add_argument("--repeat_seeds", type=int, default=20)
    parser.add_argument("--output_dir", default="rollout_results/dimension_selection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = _prediction_by_model(Path(args.prediction_loss_summary))
    sr = _parse_rates(args.success_rates)
    rows: List[Dict[str, object]] = []
    model_rows: Dict[str, Dict[str, object]] = {}
    for model, path in _parse_specs(args.aliasing_jsons).items():
        if model not in pred:
            print(f"[dim_select] warning: missing prediction row for {model}; skipping")
            continue
        alias = _alias_data(path, args.gamma, args.rho, args.tau)
        row = {
            "model": model,
            "D": _model_dim(model, alias["latent_dim"]),
            "success_rate": sr.get(model, float("nan")),
            "normalized_mean_multistep_l2": _float(pred[model], "normalized_mean_multistep_l2_mean"),
            "normalized_terminal_l2": _float(pred[model], "normalized_terminal_l2_mean"),
        }
        row.update(alias)
        rows.append(row)
        model_rows[model] = row

    selection = _selection_rows(rows)
    budgets = _budget_rows(model_rows, [int(x) for x in str(args.budget_windows).split(",") if x], args.repeat_seeds)
    _write_csv(output_dir / "dimension_selection_summary.csv", rows + selection)
    _write_csv(output_dir / "dimension_selection_by_budget.csv", budgets)
    _plot_metric_vs_d(rows, output_dir)
    _plot_budget(budgets, output_dir)
    print(f"[dim_select] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
