from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-8


def _parse_list(value: str, caster=float):
    if isinstance(value, (list, tuple)):
        return [caster(item) for item in value]
    return [caster(item) for item in str(value).split(",") if str(item).strip()]


def _parse_raw_pools(specs: Sequence[str]) -> Dict[str, Path]:
    pools: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected raw pool spec as model=path, got: {spec}")
        name, path = spec.split("=", 1)
        pools[name.strip()] = Path(path)
    return pools


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _stderr(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.std(valid) / math.sqrt(valid.size))


def _aggregate(rows: List[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out_rows: List[Dict[str, object]] = []
    for key_values, group in grouped.items():
        out = {key: value for key, value in zip(group_keys, key_values)}
        metric_keys = sorted(set().union(*(row.keys() for row in group)) - set(group_keys))
        for metric_key in metric_keys:
            values = []
            for row in group:
                value = row.get(metric_key)
                if isinstance(value, (int, float, np.integer, np.floating, bool)):
                    values.append(float(value))
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            out[f"{metric_key}_mean"] = float(np.nanmean(array))
            out[f"{metric_key}_stderr"] = _stderr(array)
            out[f"{metric_key}_count"] = int(np.sum(np.isfinite(array)))
        out_rows.append(out)
    return out_rows


def _pairwise_window_arrays(Z: np.ndarray, z_goal: np.ndarray, progress: np.ndarray, latent_scores: np.ndarray, gamma: float):
    Z = np.asarray(Z, dtype=np.float64)
    progress = np.asarray(progress, dtype=np.float64).reshape(-1)
    latent_scores = np.asarray(latent_scores, dtype=np.float64).reshape(-1)
    left, right = np.triu_indices(Z.shape[0], k=1)
    progress_diff = progress[left] - progress[right]
    valid = np.abs(progress_diff) >= gamma
    left = left[valid]
    right = right[valid]
    progress_diff = progress_diff[valid]
    score_diff = latent_scores[left] - latent_scores[right]

    sq_norm = np.sum(Z * Z, axis=1)
    gram = Z @ Z.T
    dist2 = np.maximum(sq_norm[left] + sq_norm[right] - 2.0 * gram[left, right], 0.0)
    geom_dist = np.sqrt(dist2)
    R_max = float(max(np.max(np.sqrt(sq_norm)), np.linalg.norm(np.asarray(z_goal, dtype=np.float64)), EPS))
    norm_geom = geom_dist / max(R_max, EPS)
    norm_score_gap = np.abs(score_diff) / max(R_max * R_max, EPS)

    ties = np.abs(score_diff) < EPS
    wrong = progress_diff * score_diff > 0.0
    pair_error = wrong.astype(np.float64)
    pair_error[ties] = 0.5
    return {
        "num_valid_pairs": int(valid.sum()),
        "pair_error": pair_error,
        "ties": ties,
        "norm_geom": norm_geom,
        "score_gap": np.abs(score_diff),
        "norm_score_gap": norm_score_gap,
    }


def _odds_ratio(p_alias: float, p_non: float) -> float:
    if not math.isfinite(p_alias) or not math.isfinite(p_non):
        return float("nan")
    return float((p_alias / max(1.0 - p_alias, EPS)) / max(p_non / max(1.0 - p_non, EPS), EPS))


def _alias_stats(arrays: Dict[str, np.ndarray], alias: np.ndarray) -> Dict[str, object]:
    pair_error = arrays["pair_error"]
    ties = arrays["ties"]
    if pair_error.size == 0:
        return {
            "alias_pair_frac": float("nan"),
            "p_error_all": float("nan"),
            "p_error_given_alias": float("nan"),
            "p_error_given_non_alias": float("nan"),
            "risk_ratio": float("nan"),
            "odds_ratio": float("nan"),
            "error_share_in_alias_pairs": float("nan"),
            "alias_pair_count": 0,
            "non_alias_pair_count": 0,
            "tie_rate_alias": float("nan"),
            "tie_rate_non_alias": float("nan"),
        }
    non_alias = ~alias
    p_all = float(np.mean(pair_error))
    p_alias = float(np.mean(pair_error[alias])) if np.any(alias) else float("nan")
    p_non = float(np.mean(pair_error[non_alias])) if np.any(non_alias) else float("nan")
    error_sum = float(np.sum(pair_error))
    return {
        "alias_pair_frac": float(np.mean(alias)),
        "p_error_all": p_all,
        "p_error_given_alias": p_alias,
        "p_error_given_non_alias": p_non,
        "risk_ratio": float(p_alias / max(p_non, EPS)) if math.isfinite(p_alias) and math.isfinite(p_non) else float("nan"),
        "odds_ratio": _odds_ratio(p_alias, p_non),
        "error_share_in_alias_pairs": float(np.sum(pair_error[alias]) / max(error_sum, EPS)),
        "alias_pair_count": int(np.sum(alias)),
        "non_alias_pair_count": int(np.sum(non_alias)),
        "tie_rate_alias": float(np.mean(ties[alias])) if np.any(alias) else float("nan"),
        "tie_rate_non_alias": float(np.mean(ties[non_alias])) if np.any(non_alias) else float("nan"),
    }


def _load_pool(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    pool = {key: data[key] for key in data.files}
    required = ["terminal_latents", "goal_latents", "progress"]
    missing = [key for key in required if key not in pool]
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}. Available keys: {sorted(pool)}")
    if "latent_scores" not in pool:
        Z = np.asarray(pool["terminal_latents"], dtype=np.float64)
        goals = np.asarray(pool["goal_latents"], dtype=np.float64)
        pool["latent_scores"] = np.sum((Z - goals[:, None, :]) ** 2, axis=-1)
    return pool


def _analyze_model(
    model: str,
    pool: Dict[str, np.ndarray],
    gamma: float,
    rho_values: Sequence[float],
    score_rho_values: Sequence[float],
    tau_values: Sequence[float],
    score_gap_bins: np.ndarray,
    geom_bins: np.ndarray,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    Z_all = np.asarray(pool["terminal_latents"], dtype=np.float64)
    goals_all = np.asarray(pool["goal_latents"], dtype=np.float64)
    progress_all = np.asarray(pool["progress"], dtype=np.float64)
    scores_all = np.asarray(pool["latent_scores"], dtype=np.float64)
    per_window: List[Dict[str, object]] = []
    score_bin_rows: List[Dict[str, object]] = []
    geom_bin_rows: List[Dict[str, object]] = []
    cdf_rows: List[Dict[str, object]] = []
    for window_idx in range(Z_all.shape[0]):
        arrays = _pairwise_window_arrays(
            Z_all[window_idx],
            goals_all[window_idx],
            progress_all[window_idx],
            scores_all[window_idx],
            gamma,
        )
        base = {
            "model": model,
            "window_idx": int(window_idx),
            "gamma": float(gamma),
            "effective_dim": int(Z_all.shape[-1]),
            "num_valid_pairs": int(arrays["num_valid_pairs"]),
        }
        for rho in rho_values:
            row = dict(base)
            row.update({"alias_type": "geom", "threshold": float(rho)})
            row.update(_alias_stats(arrays, arrays["norm_geom"] <= float(rho)))
            per_window.append(row)
        for score_rho in score_rho_values:
            row = dict(base)
            row.update({"alias_type": "norm_score", "threshold": float(score_rho)})
            row.update(_alias_stats(arrays, arrays["norm_score_gap"] <= float(score_rho)))
            per_window.append(row)
            cdf_rows.append(
                {
                    **base,
                    "threshold": float(score_rho),
                    "frac_pairs_le_threshold": float(np.mean(arrays["norm_score_gap"] <= float(score_rho)))
                    if arrays["num_valid_pairs"]
                    else float("nan"),
                }
            )
        for tau in tau_values:
            row = dict(base)
            row.update({"alias_type": "fixed_score", "threshold": float(tau)})
            row.update(_alias_stats(arrays, arrays["score_gap"] <= 2.0 * float(tau)))
            per_window.append(row)

        score_bin_rows.extend(_bin_error_rows(model, window_idx, gamma, arrays["norm_score_gap"], arrays["pair_error"], score_gap_bins, "score_gap"))
        geom_bin_rows.extend(_bin_error_rows(model, window_idx, gamma, arrays["norm_geom"], arrays["pair_error"], geom_bins, "geom_dist"))
    return per_window, score_bin_rows, geom_bin_rows, cdf_rows


def _bin_error_rows(
    model: str,
    window_idx: int,
    gamma: float,
    values: np.ndarray,
    pair_error: np.ndarray,
    bins: np.ndarray,
    bin_type: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for bin_idx in range(len(bins) - 1):
        lo = bins[bin_idx]
        hi = bins[bin_idx + 1]
        mask = (values >= lo) & (values < hi)
        rows.append(
            {
                "model": model,
                "window_idx": int(window_idx),
                "gamma": float(gamma),
                "bin_type": bin_type,
                "bin_idx": int(bin_idx),
                "bin_left": float(lo),
                "bin_right": float(hi),
                "pair_count": int(np.sum(mask)),
                "error_rate": float(np.mean(pair_error[mask])) if np.any(mask) else float("nan"),
            }
        )
    return rows


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def _float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _plot_error_bars(summary_rows: List[Dict[str, str]], alias_type: str, threshold: float, output_path: Path, title: str) -> None:
    rows = [
        row for row in summary_rows
        if row.get("alias_type") == alias_type and abs(_float(row, "threshold") - threshold) < 1e-12
    ]
    rows = sorted(rows, key=lambda row: row.get("model", ""))
    labels = [row["model"] for row in rows]
    alias_values = [_float(row, "p_error_given_alias_mean") for row in rows]
    non_values = [_float(row, "p_error_given_non_alias_mean") for row in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x - 0.18, alias_values, width=0.36, label="alias")
    plt.bar(x + 0.18, non_values, width=0.36, label="non-alias")
    for idx, row in enumerate(rows):
        ratio = _float(row, "risk_ratio_mean")
        if math.isfinite(ratio):
            plt.text(idx, max(alias_values[idx], non_values[idx]) + 0.02, f"{ratio:.1f}x", ha="center", fontsize=8)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("P(ranking error)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_path)


def _plot_threshold_lines(summary_rows: List[Dict[str, str]], alias_type: str, y_key: str, output_path: Path, title: str, ylabel: str) -> None:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in summary_rows:
        if alias_type and row.get("alias_type") != alias_type:
            continue
        grouped[row["model"]].append((_float(row, "threshold"), _float(row, y_key)))
    plt.figure(figsize=(8.2, 4.8))
    for model, points in sorted(grouped.items()):
        points = sorted((x, y) for x, y in points if math.isfinite(x) and math.isfinite(y))
        if not points:
            continue
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", linewidth=2, label=model)
    plt.xscale("log")
    plt.xlabel("threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_path)


def _plot_bin_lines(summary_rows: List[Dict[str, str]], output_path: Path, title: str, xlabel: str) -> None:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in summary_rows:
        left = _float(row, "bin_left")
        right = _float(row, "bin_right")
        if not math.isfinite(left) or not math.isfinite(right):
            continue
        x = math.sqrt(max(left, EPS) * right) if math.isfinite(right) else left
        grouped[row["model"]].append((x, _float(row, "error_rate_mean")))
    plt.figure(figsize=(8.2, 4.8))
    for model, points in sorted(grouped.items()):
        points = sorted((x, y) for x, y in points if math.isfinite(x) and math.isfinite(y))
        if not points:
            continue
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", linewidth=2, label=model)
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("ranking error rate")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_path)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise aliasing-conditioned ranking error analysis.")
    parser.add_argument("--raw_pools", nargs="+", required=True, help="List of model=raw_pool.npz")
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--rho_values", default="0.05,0.1,0.2")
    parser.add_argument("--score_rho_values", default="0.001,0.002,0.005,0.01,0.02")
    parser.add_argument("--tau_values", default="0.1")
    parser.add_argument("--selected_rho", type=float, default=0.1)
    parser.add_argument("--selected_score_rho", type=float, default=0.005)
    parser.add_argument("--output_dir", default="rollout_results/pairwise_aliasing_error")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rho_values = _parse_list(args.rho_values, float)
    score_rho_values = _parse_list(args.score_rho_values, float)
    tau_values = _parse_list(args.tau_values, float)
    score_gap_bins = np.asarray([0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, np.inf])
    geom_bins = np.asarray([0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf])

    per_window: List[Dict[str, object]] = []
    score_bin_rows: List[Dict[str, object]] = []
    geom_bin_rows: List[Dict[str, object]] = []
    cdf_rows: List[Dict[str, object]] = []
    for model, path in _parse_raw_pools(args.raw_pools).items():
        print(f"[pairwise] analyzing {model}: {path}", flush=True)
        pool = _load_pool(path)
        model_rows, model_score_bins, model_geom_bins, model_cdf = _analyze_model(
            model,
            pool,
            args.gamma,
            rho_values,
            score_rho_values,
            tau_values,
            score_gap_bins,
            geom_bins,
        )
        per_window.extend(model_rows)
        score_bin_rows.extend(model_score_bins)
        geom_bin_rows.extend(model_geom_bins)
        cdf_rows.extend(model_cdf)

    gamma_tag = f"{int(round(args.gamma * 100)):03d}"
    per_window_path = output_dir / f"pairwise_error_per_window_gamma{gamma_tag}.csv"
    summary_path = output_dir / f"pairwise_error_summary_gamma{gamma_tag}.csv"
    score_bins_path = output_dir / f"ranking_error_norm_score_gap_bins_gamma{gamma_tag}.csv"
    geom_bins_path = output_dir / f"ranking_error_norm_geom_distance_bins_gamma{gamma_tag}.csv"
    cdf_path = output_dir / f"score_gap_cdf_task_distinct_pairs_gamma{gamma_tag}.csv"
    _write_csv(per_window_path, per_window)
    summary = _aggregate(per_window, ["model", "alias_type", "threshold", "gamma", "effective_dim"])
    _write_csv(summary_path, summary)
    score_bin_summary = _aggregate(score_bin_rows, ["model", "bin_type", "bin_idx", "bin_left", "bin_right", "gamma"])
    geom_bin_summary = _aggregate(geom_bin_rows, ["model", "bin_type", "bin_idx", "bin_left", "bin_right", "gamma"])
    cdf_summary = _aggregate(cdf_rows, ["model", "threshold", "gamma"])
    _write_csv(score_bins_path, score_bin_summary)
    _write_csv(geom_bins_path, geom_bin_summary)
    _write_csv(cdf_path, cdf_summary)

    summary_str = [{key: str(value) for key, value in row.items()} for row in summary]
    score_bins_str = [{key: str(value) for key, value in row.items()} for row in score_bin_summary]
    geom_bins_str = [{key: str(value) for key, value in row.items()} for row in geom_bin_summary]
    cdf_str = [{key: str(value) for key, value in row.items()} for row in cdf_summary]
    _plot_error_bars(
        summary_str,
        "geom",
        args.selected_rho,
        output_dir / f"pairwise_error_given_geom_alias_gamma{gamma_tag}_rho01",
        f"Ranking errors concentrate in geom aliases, gamma={args.gamma:g}, rho={args.selected_rho:g}",
    )
    _plot_error_bars(
        summary_str,
        "norm_score",
        args.selected_score_rho,
        output_dir / f"pairwise_error_given_score_alias_gamma{gamma_tag}",
        f"Ranking errors concentrate in score aliases, gamma={args.gamma:g}, score_rho={args.selected_score_rho:g}",
    )
    _plot_threshold_lines(
        summary_str,
        "norm_score",
        "risk_ratio_mean",
        output_dir / f"score_alias_risk_ratio_vs_threshold_gamma{gamma_tag}",
        f"Score alias risk ratio vs threshold, gamma={args.gamma:g}",
        "risk ratio",
    )
    _plot_bin_lines(
        score_bins_str,
        output_dir / f"ranking_error_vs_norm_score_gap_bins_gamma{gamma_tag}",
        f"Ranking error vs normalized score gap, gamma={args.gamma:g}",
        "normalized score gap bin center",
    )
    _plot_threshold_lines(
        cdf_str,
        "",
        "frac_pairs_le_threshold_mean",
        output_dir / f"score_gap_cdf_task_distinct_pairs_gamma{gamma_tag}",
        f"Task-distinct pair score-gap CDF, gamma={args.gamma:g}",
        "fraction of task-distinct pairs",
    )
    _plot_bin_lines(
        geom_bins_str,
        output_dir / f"ranking_error_vs_norm_geom_distance_bins_gamma{gamma_tag}",
        f"Ranking error vs normalized geometric distance, gamma={args.gamma:g}",
        "normalized geometric distance bin center",
    )
    print(f"[pairwise] wrote {per_window_path}")
    print(f"[pairwise] wrote {summary_path}")
    print("[pairwise] ranking errors concentrate on aliased task-distinct pairs if alias conditional error exceeds non-alias error.")


if __name__ == "__main__":
    main()
