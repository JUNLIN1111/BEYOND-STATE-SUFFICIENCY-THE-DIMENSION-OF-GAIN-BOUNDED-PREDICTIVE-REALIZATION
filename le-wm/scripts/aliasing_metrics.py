from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-8


def _key_float(value: float) -> str:
    return f"{float(value):g}"


def compute_K_gamma(values: np.ndarray, gamma: float) -> int:
    vals = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if vals.size == 0:
        return 0
    count = 0
    last = -np.inf
    tol = 1e-12
    for value in vals:
        if value + tol >= last + gamma:
            count += 1
            last = value
    return int(count)


def pairwise_valid_mask(true_values: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(true_values, dtype=np.float64).reshape(-1)
    left, right = np.triu_indices(values.size, k=1)
    valid = np.abs(values[left] - values[right]) >= gamma
    return left, right, valid


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2 or np.nanstd(x) < EPS or np.nanstd(y) < EPS:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        corr = spearmanr(x, y, nan_policy="omit").correlation
        return float(corr) if corr is not None else float("nan")
    except ImportError:
        x_rank = np.argsort(np.argsort(x)).astype(np.float64)
        y_rank = np.argsort(np.argsort(y)).astype(np.float64)
        if np.std(x_rank) < EPS or np.std(y_rank) < EPS:
            return float("nan")
        return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _pairwise_distances(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(Z.shape[0], k=1)
    distances = np.linalg.norm(Z[left] - Z[right], axis=1)
    return left, right, distances


def _pairwise_rank_accuracy(
    latent_scores: np.ndarray,
    true_values: np.ndarray,
    true_metric: str,
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
    eps: float,
) -> float:
    if not np.any(valid):
        return float("nan")
    score_diff = latent_scores[left[valid]] - latent_scores[right[valid]]
    true_diff = true_values[left[valid]] - true_values[right[valid]]
    score_sign = np.sign(score_diff)
    true_sign = np.sign(true_diff)
    ties = np.abs(score_diff) < eps
    if true_metric == "progress":
        correct = score_sign == -true_sign
    elif true_metric == "terminal_cost":
        correct = score_sign == true_sign
    else:
        raise ValueError(f"Unknown true_metric: {true_metric}")
    values = correct.astype(np.float64)
    values[ties] = 0.5
    return float(np.mean(values))


def _topk_recall(latent_scores: np.ndarray, target_values: np.ndarray, true_metric: str, k: int) -> float:
    k_eff = min(int(k), len(latent_scores))
    if k_eff <= 0:
        return float("nan")
    latent_top = set(np.argsort(latent_scores)[:k_eff].tolist())
    if true_metric == "progress":
        true_top = set(np.argsort(-target_values)[:k_eff].tolist())
    elif true_metric == "terminal_cost":
        true_top = set(np.argsort(target_values)[:k_eff].tolist())
    else:
        raise ValueError(f"Unknown true_metric: {true_metric}")
    return float(len(latent_top & true_top) / k_eff)


def _regret(latent_scores: np.ndarray, progress: np.ndarray, terminal_cost: np.ndarray, true_metric: str) -> float:
    chosen = int(np.argmin(latent_scores))
    if true_metric == "progress":
        return float(np.max(progress) - progress[chosen])
    if true_metric == "terminal_cost":
        return float(terminal_cost[chosen] - np.min(terminal_cost))
    raise ValueError(f"Unknown true_metric: {true_metric}")


def _pressure_metrics(
    out: Dict[str, float],
    K_gamma: int,
    effective_dim: int,
    R_max: float,
    score_tau_values: Sequence[float],
    rho_values: Sequence[float],
) -> None:
    if K_gamma <= 1:
        for tau in score_tau_values:
            key = _key_float(tau)
            out[f"capacity_ratio_tau_{key}"] = float("nan")
            out[f"demand_pressure_tau_{key}"] = float("nan")
        for rho in rho_values:
            key = _key_float(rho)
            out[f"capacity_ratio_rho_{key}"] = float("nan")
            out[f"demand_pressure_rho_{key}"] = float("nan")
        return
    log_k = math.log(K_gamma)
    for tau in score_tau_values:
        key = _key_float(tau)
        denom = effective_dim * math.log1p(4.0 * R_max * R_max / max(float(tau), EPS))
        out[f"capacity_ratio_tau_{key}"] = float(denom / log_k)
        out[f"demand_pressure_tau_{key}"] = float(log_k / denom)
    for rho in rho_values:
        key = _key_float(rho)
        denom = effective_dim * math.log1p(2.0 / max(float(rho), EPS))
        out[f"capacity_ratio_rho_{key}"] = float(denom / log_k)
        out[f"demand_pressure_rho_{key}"] = float(log_k / denom)


def compute_aliasing_and_ranking_metrics(
    Z: np.ndarray,
    z_goal: np.ndarray,
    progress: np.ndarray,
    terminal_cost: np.ndarray,
    latent_scores: Optional[np.ndarray] = None,
    true_metric: str = "progress",
    gamma_values: Sequence[float] = (0.02,),
    rho_values: Sequence[float] = (0.1,),
    score_tau_values: Sequence[float] = (0.02,),
    score_rho_values: Sequence[float] = (0.005,),
    topk_values: Sequence[int] = (10, 30),
    effective_dim: Optional[int] = None,
    eps: float = EPS,
) -> Dict[str, float]:
    Z = np.asarray(Z, dtype=np.float64)
    z_goal = np.asarray(z_goal, dtype=np.float64).reshape(-1)
    progress = np.asarray(progress, dtype=np.float64).reshape(-1)
    terminal_cost = np.asarray(terminal_cost, dtype=np.float64).reshape(-1)
    if Z.ndim != 2:
        raise ValueError(f"Expected Z shape [N,D], got {Z.shape}")
    if z_goal.shape[0] != Z.shape[1]:
        raise ValueError(f"z_goal dim {z_goal.shape[0]} does not match Z dim {Z.shape[1]}")
    if progress.shape[0] != Z.shape[0] or terminal_cost.shape[0] != Z.shape[0]:
        raise ValueError("progress and terminal_cost must have length N")
    if latent_scores is None:
        latent_scores = np.sum((Z - z_goal[None, :]) ** 2, axis=1)
    else:
        latent_scores = np.asarray(latent_scores, dtype=np.float64).reshape(-1)
    if latent_scores.shape[0] != Z.shape[0]:
        raise ValueError("latent_scores must have length N")

    effective_dim = int(effective_dim or Z.shape[1])
    true_values = progress if true_metric == "progress" else terminal_cost
    ranking_target = progress if true_metric == "progress" else -terminal_cost
    latent_utility = -latent_scores

    norms = np.linalg.norm(Z, axis=1)
    goal_norm = float(np.linalg.norm(z_goal))
    R_max = float(max(np.max(norms), goal_norm)) if Z.shape[0] else goal_norm
    M_goal_radius = float(np.max(np.linalg.norm(Z - z_goal[None, :], axis=1))) if Z.shape[0] else 0.0
    left_all, right_all, pairwise_dist = _pairwise_distances(Z)
    pairwise_median = _percentile(pairwise_dist, 50)

    out: Dict[str, float] = {
        "N": int(Z.shape[0]),
        "effective_dim": effective_dim,
        "R_max": R_max,
        "M_goal_radius": M_goal_radius,
        "pairwise_dist_p10": _percentile(pairwise_dist, 10),
        "pairwise_dist_median": pairwise_median,
        "pairwise_dist_p90": _percentile(pairwise_dist, 90),
        "R_max_over_sqrt_dim": float(R_max / math.sqrt(max(effective_dim, 1))),
        "R_max_over_pairwise_median": float(R_max / max(pairwise_median, eps)),
        "spearman": _spearman(ranking_target, latent_utility),
        "regret": _regret(latent_scores, progress, terminal_cost, true_metric),
    }
    for topk in topk_values:
        out[f"topk_recall_{int(topk)}"] = _topk_recall(latent_scores, true_values, true_metric, int(topk))

    single_gamma = len(gamma_values) == 1
    for gamma in gamma_values:
        gamma_key = _key_float(gamma)
        left, right, valid = pairwise_valid_mask(true_values, float(gamma))
        num_valid = int(np.sum(valid))
        K_gamma = compute_K_gamma(true_values, float(gamma))
        prefix = "" if single_gamma else f"gamma_{gamma_key}_"
        out[f"{prefix}gamma"] = float(gamma)
        out[f"{prefix}K_gamma"] = int(K_gamma)
        out[f"{prefix}num_valid_pairs"] = num_valid
        out[f"{prefix}pairwise_rank_acc"] = _pairwise_rank_accuracy(
            latent_scores, true_values, true_metric, left, right, valid, eps
        )

        valid_dist = np.linalg.norm(Z[left[valid]] - Z[right[valid]], axis=1) if num_valid else np.asarray([])
        score_gap = np.abs(latent_scores[left[valid]] - latent_scores[right[valid]]) if num_valid else np.asarray([])
        for rho in rho_values:
            key = _key_float(rho)
            out[f"{prefix}norm_geom_alias_rho_{key}"] = (
                float(np.mean(valid_dist / max(R_max, eps) <= float(rho))) if num_valid else float("nan")
            )
        for tau in score_tau_values:
            key = _key_float(tau)
            out[f"{prefix}score_alias_tau_{key}"] = (
                float(np.mean(score_gap <= 2.0 * float(tau))) if num_valid else float("nan")
            )
        for score_rho in score_rho_values:
            key = _key_float(score_rho)
            out[f"{prefix}norm_score_alias_score_rho_{key}"] = (
                float(np.mean(score_gap / max(R_max * R_max, eps) <= float(score_rho))) if num_valid else float("nan")
            )
        gamma_pressure = {}
        _pressure_metrics(gamma_pressure, K_gamma, effective_dim, R_max, score_tau_values, rho_values)
        out.update({f"{prefix}{key}": value for key, value in gamma_pressure.items()})
    return out


def _assert_close_random_projection_norm(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(256, 128))
    ratios = []
    for idx in range(10):
        A = rng.normal(0.0, 1.0 / math.sqrt(64), size=(64, 128))
        projected = X @ A.T
        ratios.append(np.mean(np.sum(projected * projected, axis=1) / np.sum(X * X, axis=1)))
    ratio = float(np.mean(ratios))
    assert 0.75 < ratio < 1.25, ratio


def run_sanity_checks() -> None:
    assert compute_K_gamma(np.asarray([0.0, 0.01, 0.03]), 0.02) == 2
    assert compute_K_gamma(np.asarray([0.0, 0.02, 0.04]), 0.02) == 3
    assert compute_K_gamma(np.asarray([]), 0.02) == 0
    assert compute_K_gamma(np.asarray([-0.03, -0.01, 0.02]), 0.02) == 3
    metrics = compute_aliasing_and_ranking_metrics(
        Z=np.zeros((3, 2)),
        z_goal=np.zeros(2),
        progress=np.asarray([0.0, 0.001, 0.002]),
        terminal_cost=np.asarray([1.0, 0.999, 0.998]),
        true_metric="progress",
        gamma_values=[1.0],
    )
    assert metrics["num_valid_pairs"] == 0
    assert np.isnan(metrics["pairwise_rank_acc"])
    _assert_close_random_projection_norm()


if __name__ == "__main__":
    run_sanity_checks()
    print("aliasing_metrics sanity checks passed")
