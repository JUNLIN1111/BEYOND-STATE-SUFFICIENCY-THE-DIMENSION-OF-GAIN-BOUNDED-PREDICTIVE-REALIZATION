from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aliasing_metrics import compute_aliasing_and_ranking_metrics  # noqa: E402


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _pairwise_distances(Z: np.ndarray) -> np.ndarray:
    sq = np.sum(Z * Z, axis=1)
    dist2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    left, right = np.triu_indices(Z.shape[0], k=1)
    return np.sqrt(dist2[left, right])


def _pad(Z: np.ndarray, target_dim: int) -> np.ndarray:
    if target_dim < Z.shape[-1]:
        raise ValueError(f"target_dim={target_dim} must be >= current dim={Z.shape[-1]}")
    if target_dim == Z.shape[-1]:
        return Z.copy()
    zeros = np.zeros((*Z.shape[:-1], target_dim - Z.shape[-1]), dtype=Z.dtype)
    return np.concatenate([Z, zeros], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-padding control: nominal ambient dimension alone changes nothing.")
    parser.add_argument("--raw_npz", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--target_dim", type=int, default=192)
    parser.add_argument("--output_csv", default="rollout_results/zero_padding_control.csv")
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--score_tau", type=float, default=0.1)
    parser.add_argument("--score_rho", type=float, default=0.005)
    parser.add_argument("--true_metric", choices=["progress", "terminal_cost"], default="progress")
    parser.add_argument("--assert_tol", type=float, default=1e-5)
    args = parser.parse_args()

    data = np.load(args.raw_npz, allow_pickle=False)
    print(f"[zero_padding] available keys: {sorted(data.files)}")
    Z_all = np.asarray(data["terminal_latents"], dtype=np.float64)
    if Z_all.ndim == 2:
        Z_all = Z_all[None]
    goals = np.asarray(data["goal_latents"], dtype=np.float64)
    if goals.ndim == 1:
        goals = goals[None]
    progress = np.asarray(data["progress"], dtype=np.float64)
    if progress.ndim == 1:
        progress = progress[None]
    terminal_cost = np.asarray(data["terminal_cost"], dtype=np.float64) if "terminal_cost" in data else -progress
    if terminal_cost.ndim == 1:
        terminal_cost = terminal_cost[None]

    rows: List[Dict[str, object]] = []
    max_score_diff_all = 0.0
    max_pairwise_diff_all = 0.0
    for window_idx in range(Z_all.shape[0]):
        Z = Z_all[window_idx]
        goal = goals[window_idx]
        Z_pad = _pad(Z, args.target_dim)
        goal_pad = _pad(goal[None], args.target_dim)[0]
        scores = np.sum((Z - goal[None]) ** 2, axis=1)
        scores_pad = np.sum((Z_pad - goal_pad[None]) ** 2, axis=1)
        score_diff = float(np.max(np.abs(scores - scores_pad)))
        pairwise_diff = float(np.max(np.abs(_pairwise_distances(Z) - _pairwise_distances(Z_pad))))
        max_score_diff_all = max(max_score_diff_all, score_diff)
        max_pairwise_diff_all = max(max_pairwise_diff_all, pairwise_diff)
        metrics = compute_aliasing_and_ranking_metrics(
            Z=Z,
            z_goal=goal,
            progress=progress[window_idx],
            terminal_cost=terminal_cost[window_idx],
            latent_scores=scores,
            true_metric=args.true_metric,
            gamma_values=[args.gamma],
            rho_values=[args.rho],
            score_tau_values=[args.score_tau],
            score_rho_values=[args.score_rho],
            topk_values=[10, 30],
            effective_dim=Z.shape[-1],
        )
        metrics_pad = compute_aliasing_and_ranking_metrics(
            Z=Z_pad,
            z_goal=goal_pad,
            progress=progress[window_idx],
            terminal_cost=terminal_cost[window_idx],
            latent_scores=scores_pad,
            true_metric=args.true_metric,
            gamma_values=[args.gamma],
            rho_values=[args.rho],
            score_tau_values=[args.score_tau],
            score_rho_values=[args.score_rho],
            topk_values=[10, 30],
            effective_dim=args.target_dim,
        )
        row: Dict[str, object] = {
            "model": args.model_name,
            "window_idx": int(window_idx),
            "original_dim": int(Z.shape[-1]),
            "target_dim": int(args.target_dim),
            "max_score_diff": score_diff,
            "max_pairwise_distance_diff": pairwise_diff,
        }
        for key in [
            "spearman",
            "pairwise_rank_acc",
            f"norm_geom_alias_rho_{args.rho:g}",
            f"score_alias_tau_{args.score_tau:g}",
            f"norm_score_alias_score_rho_{args.score_rho:g}",
            "regret",
            "topk_recall_10",
            "topk_recall_30",
        ]:
            row[f"{key}_original"] = metrics.get(key, float("nan"))
            row[f"{key}_padded"] = metrics_pad.get(key, float("nan"))
            row[f"{key}_abs_diff"] = abs(float(row[f"{key}_original"]) - float(row[f"{key}_padded"]))
        rows.append(row)

    _write_csv(Path(args.output_csv), rows)
    print(f"[zero_padding] wrote {args.output_csv}")
    print(f"[zero_padding] max score diff: {max_score_diff_all:.8g}")
    print(f"[zero_padding] max pairwise distance diff: {max_pairwise_diff_all:.8g}")
    if max_score_diff_all > args.assert_tol or max_pairwise_diff_all > args.assert_tol:
        raise AssertionError(
            f"Zero-padding changed scores/distances beyond tol={args.assert_tol}: "
            f"score={max_score_diff_all}, pairwise={max_pairwise_diff_all}"
        )


if __name__ == "__main__":
    main()
