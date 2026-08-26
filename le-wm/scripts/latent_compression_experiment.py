from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aliasing_metrics import compute_aliasing_and_ranking_metrics  # noqa: E402


def _parse_list(value: str, caster=float):
    if isinstance(value, (list, tuple)):
        return [caster(item) for item in value]
    return [caster(item) for item in str(value).split(",") if str(item).strip()]


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _aggregate_rows(rows: List[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    aggregate = []
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
            count = int(np.sum(~np.isnan(array)))
            out[f"{metric_key}_mean"] = float(np.nanmean(array))
            out[f"{metric_key}_std"] = float(np.nanstd(array))
            out[f"{metric_key}_stderr"] = float(np.nanstd(array) / math.sqrt(max(count, 1)))
            out[f"{metric_key}_count"] = count
        aggregate.append(out)
    return aggregate


def _fit_pca_basis(Z_all: np.ndarray, target_dim: int):
    mu = np.mean(Z_all, axis=0)
    _, _, vh = np.linalg.svd(Z_all - mu[None], full_matrices=False)
    basis = vh[:target_dim].T
    return mu, basis


def _scale_metrics(Z_window: np.ndarray, z_goal: np.ndarray, effective_dim: int, original_R_max: float) -> Dict[str, float]:
    norms = np.linalg.norm(Z_window, axis=1)
    goal_norm = float(np.linalg.norm(z_goal))
    R_max = float(max(np.max(norms), goal_norm)) if Z_window.shape[0] else goal_norm
    M_goal_radius = float(np.max(np.linalg.norm(Z_window - z_goal[None], axis=1))) if Z_window.shape[0] else 0.0
    pairwise_dist = _pairwise_distances_from_gram(Z_window)
    pairwise_median = float(np.percentile(pairwise_dist, 50)) if pairwise_dist.size else float("nan")
    return {
        "R_max": R_max,
        "original_R_max": float(original_R_max),
        "R_max_over_original_R_max": float(R_max / max(original_R_max, 1e-8)),
        "R_max_over_sqrt_dim": float(R_max / math.sqrt(max(effective_dim, 1))),
        "pairwise_dist_median": pairwise_median,
        "pairwise_dist_p10": float(np.percentile(pairwise_dist, 10)) if pairwise_dist.size else float("nan"),
        "pairwise_dist_p90": float(np.percentile(pairwise_dist, 90)) if pairwise_dist.size else float("nan"),
        "pairwise_median_over_Rmax": float(pairwise_median / max(R_max, 1e-8)),
        "M_goal_radius": M_goal_radius,
        "pairwise_median_over_goal_radius": float(pairwise_median / max(M_goal_radius, 1e-8)),
    }


def _pairwise_distances_from_gram(Z_window: np.ndarray) -> np.ndarray:
    if Z_window.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    gram = Z_window @ Z_window.T
    sq_norm = np.sum(Z_window * Z_window, axis=1)
    dist2 = np.maximum(sq_norm[:, None] + sq_norm[None, :] - 2.0 * gram, 0.0)
    left, right = np.triu_indices(Z_window.shape[0], k=1)
    return np.sqrt(dist2[left, right])


def _selected_metric_subset(metrics: Dict[str, float], rho_values: Sequence[float], score_tau_values: Sequence[float], score_rho_values: Sequence[float]) -> Dict[str, float]:
    selected = {
        "spearman": metrics.get("spearman", float("nan")),
        "pairwise_rank_acc": metrics.get("pairwise_rank_acc", float("nan")),
        "regret": metrics.get("regret", float("nan")),
    }
    for rho in rho_values:
        key = f"norm_geom_alias_rho_{rho:g}"
        selected[key] = metrics.get(key, float("nan"))
    for tau in score_tau_values:
        key = f"score_alias_tau_{tau:g}"
        selected[key] = metrics.get(key, float("nan"))
    for score_rho in score_rho_values:
        key = f"norm_score_alias_score_rho_{score_rho:g}"
        selected[key] = metrics.get(key, float("nan"))
    return selected


def _compute_rows_for_projection(
    Z: np.ndarray,
    goals: np.ndarray,
    progress: np.ndarray,
    terminal_cost: np.ndarray,
    original_R_max_by_window: np.ndarray,
    args,
    source: str,
    projection_method: str,
    projection_seed: int,
    effective_dim: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    gamma_values = _parse_list(args.gamma_values, float)
    rho_values = _parse_list(args.rho_values, float)
    score_tau_values = _parse_list(args.score_tau_values, float)
    score_rho_values = _parse_list(args.score_rho_values, float)
    topk_values = _parse_list(args.topk_values, int)
    rows = []
    scale_rows = []
    rescale_rows = []
    for window_idx in range(Z.shape[0]):
        latent_scores = np.sum((Z[window_idx] - goals[window_idx][None]) ** 2, axis=1)
        scale = _scale_metrics(Z[window_idx], goals[window_idx], effective_dim, original_R_max_by_window[window_idx])
        scale_row = {
            "source": source,
            "original_dim": int(args.original_dim),
            "effective_dim": int(effective_dim),
            "projection_method": projection_method,
            "projection_seed": int(projection_seed),
            "window_idx": int(window_idx),
        }
        scale_row.update(scale)
        scale_rows.append(scale_row)
        for gamma in gamma_values:
            metrics = compute_aliasing_and_ranking_metrics(
                Z=Z[window_idx],
                z_goal=goals[window_idx],
                progress=progress[window_idx],
                terminal_cost=terminal_cost[window_idx],
                latent_scores=latent_scores,
                true_metric=args.true_metric,
                gamma_values=[gamma],
                rho_values=rho_values,
                score_tau_values=score_tau_values,
                score_rho_values=score_rho_values,
                topk_values=topk_values,
                effective_dim=effective_dim,
            )
            row = {
                "source": source,
                "original_dim": int(args.original_dim),
                "effective_dim": int(effective_dim),
                "projection_method": projection_method,
                "projection_seed": int(projection_seed),
                "window_idx": int(window_idx),
                "gamma": float(gamma),
            }
            row.update(scale)
            row.update(metrics)
            row["pairwise_median_over_Rmax"] = scale["pairwise_median_over_Rmax"]
            rows.append(row)
            if args.enable_rescale_sanity:
                alpha = float(original_R_max_by_window[window_idx] / max(scale["R_max"], 1e-8))
                Z_rescaled = alpha * Z[window_idx]
                goal_rescaled = alpha * goals[window_idx]
                latent_scores_rescaled = np.sum((Z_rescaled - goal_rescaled[None]) ** 2, axis=1)
                metrics_rescaled = compute_aliasing_and_ranking_metrics(
                    Z=Z_rescaled,
                    z_goal=goal_rescaled,
                    progress=progress[window_idx],
                    terminal_cost=terminal_cost[window_idx],
                    latent_scores=latent_scores_rescaled,
                    true_metric=args.true_metric,
                    gamma_values=[gamma],
                    rho_values=rho_values,
                    score_tau_values=score_tau_values,
                    score_rho_values=score_rho_values,
                    topk_values=topk_values,
                    effective_dim=effective_dim,
                )
                before = _selected_metric_subset(metrics, rho_values, score_tau_values, score_rho_values)
                after = _selected_metric_subset(metrics_rescaled, rho_values, score_tau_values, score_rho_values)
                rescale_row = {
                    "source": source,
                    "original_dim": int(args.original_dim),
                    "effective_dim": int(effective_dim),
                    "projection_method": projection_method,
                    "projection_seed": int(projection_seed),
                    "window_idx": int(window_idx),
                    "gamma": float(gamma),
                    "alpha": alpha,
                    "R_max_before": scale["R_max"],
                    "R_max_after": _scale_metrics(Z_rescaled, goal_rescaled, effective_dim, original_R_max_by_window[window_idx])["R_max"],
                    "original_R_max": float(original_R_max_by_window[window_idx]),
                }
                for key, value in before.items():
                    rescale_row[f"{key}_before"] = value
                    rescale_row[f"{key}_after"] = after.get(key, float("nan"))
                    rescale_row[f"{key}_abs_diff"] = abs(float(value) - float(after.get(key, float("nan"))))
                rescale_rows.append(rescale_row)
    return rows, scale_rows, rescale_rows


def main():
    parser = argparse.ArgumentParser(description="Artificial compression diagnostics for cached baseline latents.")
    parser.add_argument("--input_raw_pool_npz", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_aggregate_csv", default=None)
    parser.add_argument("--output_scale_csv", default=None)
    parser.add_argument("--output_rescale_csv", default=None)
    parser.add_argument("--target_dims", default="8,16,32,64")
    parser.add_argument("--projection_methods", default="random,pca")
    parser.add_argument("--num_random_projection_seeds", type=int, default=20)
    parser.add_argument("--random_projection_seed", type=int, default=0)
    parser.add_argument("--pca_fit_scope", choices=["all_windows"], default="all_windows")
    parser.add_argument("--gamma_values", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--rho_values", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--score_tau_values", default="0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--score_rho_values", default="0.001,0.002,0.005,0.01,0.02")
    parser.add_argument("--topk_values", default="10,30")
    parser.add_argument("--true_metric", choices=["progress", "terminal_cost"], default="progress")
    parser.add_argument("--enable_rescale_sanity", action="store_true")
    args = parser.parse_args()

    pool = np.load(args.input_raw_pool_npz, allow_pickle=False)
    Z = np.asarray(pool["terminal_latents"], dtype=np.float64)
    goals = np.asarray(pool["goal_latents"], dtype=np.float64)
    progress = np.asarray(pool["progress"], dtype=np.float64)
    terminal_cost = np.asarray(pool["terminal_cost"], dtype=np.float64)
    saved_scores = np.asarray(pool["latent_scores"], dtype=np.float64)
    args.original_dim = Z.shape[-1]

    recomputed = np.sum((Z - goals[:, None, :]) ** 2, axis=-1)
    max_score_diff = float(np.max(np.abs(recomputed - saved_scores)))
    print(f"[compression] max |recomputed original score - saved score| = {max_score_diff:.8f}")

    rows = []
    scale_rows = []
    rescale_rows = []
    original_R_max_by_window = np.asarray(
        [_scale_metrics(Z[window_idx], goals[window_idx], Z.shape[-1], 1.0)["R_max"] for window_idx in range(Z.shape[0])],
        dtype=np.float64,
    )
    metric_part, scale_part, rescale_part = _compute_rows_for_projection(
            Z,
            goals,
            progress,
            terminal_cost,
            original_R_max_by_window,
            args,
            source="baseline192_original",
            projection_method="none",
            projection_seed=-1,
            effective_dim=Z.shape[-1],
        )
    rows.extend(metric_part)
    scale_rows.extend(scale_part)
    rescale_rows.extend(rescale_part)

    target_dims = _parse_list(args.target_dims, int)
    methods = set(_parse_list(args.projection_methods, str))
    Z_all = Z.reshape(-1, Z.shape[-1])
    for target_dim in target_dims:
        if target_dim <= 0 or target_dim > Z.shape[-1]:
            raise ValueError(f"Invalid target dim {target_dim} for original dim {Z.shape[-1]}")
        if "random" in methods:
            for seed_idx in range(args.num_random_projection_seeds):
                seed = args.random_projection_seed + seed_idx
                rng = np.random.default_rng(seed)
                A = rng.normal(0.0, 1.0 / math.sqrt(target_dim), size=(target_dim, Z.shape[-1]))
                Z_proj = Z @ A.T
                goals_proj = goals @ A.T
                metric_part, scale_part, rescale_part = _compute_rows_for_projection(
                        Z_proj,
                        goals_proj,
                        progress,
                        terminal_cost,
                        original_R_max_by_window,
                        args,
                        source="projected_random",
                        projection_method="random",
                        projection_seed=seed,
                        effective_dim=target_dim,
                    )
                rows.extend(metric_part)
                scale_rows.extend(scale_part)
                rescale_rows.extend(rescale_part)
        if "pca" in methods:
            mu, basis = _fit_pca_basis(Z_all, target_dim)
            Z_proj = (Z - mu[None, None, :]) @ basis
            goals_proj = (goals - mu[None, :]) @ basis
            metric_part, scale_part, rescale_part = _compute_rows_for_projection(
                    Z_proj,
                    goals_proj,
                    progress,
                    terminal_cost,
                    original_R_max_by_window,
                    args,
                    source="projected_pca",
                    projection_method="pca",
                    projection_seed=-1,
                    effective_dim=target_dim,
                )
            rows.extend(metric_part)
            scale_rows.extend(scale_part)
            rescale_rows.extend(rescale_part)

    output_csv = Path(args.output_csv)
    _write_csv(output_csv, rows)
    aggregate_rows = _aggregate_rows(rows, ["source", "projection_method", "effective_dim", "gamma"])
    aggregate_path = Path(args.output_aggregate_csv) if args.output_aggregate_csv else output_csv.with_name(output_csv.stem + "_aggregate.csv")
    _write_csv(aggregate_path, aggregate_rows)
    if args.output_scale_csv:
        scale_path = Path(args.output_scale_csv)
        _write_csv(scale_path, scale_rows)
        _write_csv(scale_path.with_name(scale_path.stem + "_aggregate.csv"), _aggregate_rows(scale_rows, ["source", "projection_method", "effective_dim"]))
    if args.output_rescale_csv:
        _write_csv(Path(args.output_rescale_csv), rescale_rows)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_json).open("w") as file:
            json.dump(
                _jsonify(
                    {
                        "max_original_score_diff": max_score_diff,
                        "rows": rows,
                        "scale_rows": scale_rows,
                        "rescale_rows": rescale_rows,
                        "aggregate": aggregate_rows,
                    }
                ),
                file,
                indent=2,
            )
    print(f"[compression] wrote {len(rows)} rows to {output_csv}")
    print(f"[compression] wrote aggregate rows to {aggregate_path}")
    if args.output_scale_csv:
        print(f"[compression] wrote scale rows to {args.output_scale_csv}")
    if args.output_rescale_csv:
        print(f"[compression] wrote rescale rows to {args.output_rescale_csv}")


if __name__ == "__main__":
    main()
