from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import hdf5plugin  # noqa: F401

    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aliasing_experiment import (  # noqa: E402
    _action_normalization,
    _get_env_state,
    _make_env,
    _reset_env_to_state,
    _step_env,
)
from planner_success_reference_diagnostic import (  # noqa: E402
    _encode_pixels,
    _extract_pixels_from_obs,
    _find_key,
    _load_model,
    _preprocess_pixels,
    _read_h5_rows,
)
from task_cost import task_cost  # noqa: E402


EPS = 1e-8
DEFAULT_STEPS = (0, 5, 10, 15, 20, 25)
REAL_GAP_NORM_MIN_FOR_RATIOS = 1e-4
MODEL_DIMS = {
    "state8": 8,
    "state16": 16,
    "state32": 32,
    "state64": 64,
    "baseline192": 192,
    "state302": 302,
    "state502": 502,
    "global_k32": 32,
    "local_k32": 32,
}


def _parse_name_paths(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path)
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
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


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_trace_attempts(trace_dir: Path) -> Dict[int, Dict[str, object]]:
    summary_path = trace_dir / "trace_run_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text())
    attempts = payload.get("metrics", {}).get("attempts", [])
    out = {}
    for item in attempts:
        try:
            out[int(item["trace_episode_id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _group_rows(rows: Sequence[Dict[str, str]], keys: Sequence[str]) -> Dict[Tuple[int, ...], List[Dict[str, str]]]:
    grouped: Dict[Tuple[int, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(_safe_int(row.get(key)) for key in keys)].append(row)
    return grouped


def _pool_paths(trace_dir: Path) -> Dict[Tuple[int, int], Path]:
    paths = {}
    for path in sorted((trace_dir / "candidate_pools").glob("candidate_pool_ep*_step*.npz")):
        stem = path.stem
        try:
            left, step = stem.split("_step")
            episode = int(left.split("_ep")[-1])
            paths[(episode, int(step))] = path
        except (ValueError, IndexError):
            continue
    return paths


def _candidate_pool_path(row_group: Sequence[Dict[str, str]], fallback_paths: Dict[Tuple[int, int], Path], key: Tuple[int, int]) -> Optional[Path]:
    for row in row_group:
        value = row.get("candidate_pool_npz", "")
        if value:
            path = Path(value)
            if path.exists():
                return path
    return fallback_paths.get(key)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def _sqdist(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sum(delta * delta))


def _pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    delta = x[:, None, :] - x[None, :, :]
    dist = np.sum(delta * delta, axis=-1)
    tri = np.triu_indices(x.shape[0], k=1)
    return dist[tri]


def _effective_rank(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return float("nan")
    centered = x - np.mean(x, axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, x.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = eig[eig > 1e-12]
    if eig.size == 0:
        return 0.0
    p = eig / np.sum(eig)
    return float(np.exp(-np.sum(p * np.log(p + EPS))))


def _mean_pairwise_cosine(displacements: np.ndarray) -> Tuple[float, float, float]:
    if displacements.shape[0] < 2:
        return float("nan"), float("nan"), float("nan")
    norm = np.linalg.norm(displacements, axis=-1, keepdims=True)
    unit = displacements / (norm + EPS)
    cos = unit @ unit.T
    tri = np.triu_indices(unit.shape[0], k=1)
    values = cos[tri]
    return float(np.mean(values)), float(np.max(values)), float(np.mean(values > 0.9))


def _future_index(num_rollout_points: int, k: int, planning_horizon_raw: int) -> int:
    if num_rollout_points <= 1:
        return 0
    ratio = min(max(float(k) / float(max(1, planning_horizon_raw)), 0.0), 1.0)
    return int(round(ratio * (num_rollout_points - 1)))


def _model_latent_dim(model_name: str, latent: np.ndarray) -> int:
    return MODEL_DIMS.get(model_name, int(latent.shape[-1]))


def _rollout_predicted_latents(
    model,
    context_pixels: np.ndarray,
    action_sequence: np.ndarray,
    candidate_indices: np.ndarray,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_actions = np.asarray(action_sequence[candidate_indices], dtype=np.float32)
    context = np.repeat(context_pixels[None], selected_actions.shape[0], axis=0)
    predicted_chunks = []
    with torch.no_grad():
        for start in range(0, selected_actions.shape[0], batch_size):
            action_batch = torch.from_numpy(selected_actions[start:start + batch_size]).float().to(device)
            pixel_batch = _preprocess_pixels(context[start:start + batch_size], img_size, device)
            history_size = int(getattr(model.predictor, "pos_embedding").shape[1])
            rollout = model.rollout({"pixels": pixel_batch.unsqueeze(0)}, action_batch.unsqueeze(0), history_size=history_size)
            predicted_chunks.append(rollout["predicted_emb"][0].detach().float().cpu().numpy())
        start_latent = model.encode({"pixels": _preprocess_pixels(context_pixels[None], img_size, device)})["emb"][0, -1]
    predicted = np.concatenate(predicted_chunks, axis=0)
    return predicted, start_latent.detach().float().cpu().numpy(), selected_actions


def _action_sequence_to_raw_future(
    action_sequence: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    planning_horizon_raw: int,
    candidate_actions_are_normalized: bool,
) -> np.ndarray:
    seq = np.asarray(action_sequence, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq[None]
    raw_dim = int(action_mean.reshape(-1).shape[0])
    flat = seq.reshape(-1, seq.shape[-1])
    if candidate_actions_are_normalized:
        mean = action_mean.reshape(-1)
        std = action_std.reshape(-1)
        if flat.shape[-1] != raw_dim and flat.shape[-1] % raw_dim == 0:
            repeat = flat.shape[-1] // raw_dim
            mean = np.tile(mean, repeat)
            std = np.tile(std, repeat)
        if flat.shape[-1] == mean.size:
            flat = flat * std.reshape(1, -1) + mean.reshape(1, -1)
    if flat.shape[-1] == raw_dim:
        raw = flat.reshape(-1, raw_dim)
    elif flat.shape[-1] % raw_dim == 0:
        raw = flat.reshape(-1, raw_dim)
    else:
        raise ValueError(f"Cannot convert action_sequence last dim {flat.shape[-1]} to raw action dim {raw_dim}.")
    if raw.shape[0] >= planning_horizon_raw:
        raw = raw[-planning_horizon_raw:]
    return raw.astype(np.float32)


def _replay_candidate_observations(
    env,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions: np.ndarray,
    steps: Sequence[int],
    reset_state_tol: float,
) -> Tuple[Dict[int, np.ndarray], Dict[str, object]]:
    max_step = max(steps)
    obs, _info = _reset_env_to_state(env, start_state, goal_state, reset_state_tol)
    pixels_by_step = {0: _extract_pixels_from_obs(env, obs)}
    state0 = np.asarray(_get_env_state(env, obs=obs, info=_info), dtype=np.float64)[: goal_state.size]
    states_by_step = {0: state0}
    task_cost_by_step = {0: float(task_cost(state0, goal_state))}
    done = False
    info = {}
    executed = 0
    for action in raw_actions[:max_step]:
        obs, _reward, done, info = _step_env(env, action)
        executed += 1
        if executed in steps:
            pixels_by_step[executed] = _extract_pixels_from_obs(env, obs)
            state = np.asarray(_get_env_state(env, obs=obs, info=info), dtype=np.float64)[: goal_state.size]
            states_by_step[executed] = state
            task_cost_by_step[executed] = float(task_cost(state, goal_state))
        if done:
            break
    if executed < max_step:
        last_pixels = _extract_pixels_from_obs(env, obs)
        last_state = np.asarray(_get_env_state(env, obs=obs, info=info), dtype=np.float64)[: goal_state.size]
        last_cost = float(task_cost(last_state, goal_state))
        for step in steps:
            if step not in pixels_by_step:
                pixels_by_step[step] = last_pixels
            if step not in states_by_step:
                states_by_step[step] = last_state
            if step not in task_cost_by_step:
                task_cost_by_step[step] = last_cost
    terminal_state = np.asarray(_get_env_state(env, obs=obs, info=info), dtype=np.float64)[: goal_state.size]
    return pixels_by_step, {
        "executed_raw_steps": int(executed),
        "done": int(bool(done)),
        "terminal_state": terminal_state,
        "states_by_step": states_by_step,
        "task_cost_by_step": task_cost_by_step,
    }


def _encode_replay_latents(
    model,
    pixels_by_candidate: Dict[int, Dict[int, np.ndarray]],
    candidate_indices: Sequence[int],
    steps: Sequence[int],
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[int, Dict[int, np.ndarray]]:
    ordered_pixels = []
    keys = []
    for candidate_idx in candidate_indices:
        for step in steps:
            ordered_pixels.append(pixels_by_candidate[int(candidate_idx)][int(step)])
            keys.append((int(candidate_idx), int(step)))
    if not ordered_pixels:
        return {}
    emb = _encode_pixels(model, np.asarray(ordered_pixels), img_size, batch_size, device)[:, 0]
    out: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    for key, value in zip(keys, emb):
        out[key[0]][key[1]] = value.astype(np.float32)
    return out


def _select_candidates_for_call(
    pool: np.lib.npyio.NpzFile,
    replay_rows: Sequence[Dict[str, str]],
    topk: int,
) -> Tuple[List[Tuple[str, int, int]], np.ndarray]:
    costs = np.asarray(pool["terminal_cost_c25"], dtype=np.float64)
    replay_by_idx = {}
    for row in replay_rows:
        idx = _safe_int(row.get("candidate_idx"))
        if idx >= 0:
            replay_by_idx[idx] = row
    if not replay_by_idx:
        return [], np.asarray([], dtype=np.int64)

    true_best_idx = min(
        replay_by_idx,
        key=lambda idx: (
            _safe_float(replay_by_idx[idx].get("true_terminal_task_cost"), float("inf")),
            -_safe_float(replay_by_idx[idx].get("true_terminal_progress"), -float("inf")),
        ),
    )
    pred_best_idx = int(np.nanargmin(costs))
    progress_values = np.asarray(
        [_safe_float(row.get("true_terminal_progress")) for row in replay_by_idx.values()],
        dtype=np.float64,
    )
    median_progress = float(np.nanmedian(progress_values)) if progress_values.size else float("nan")
    poor = [
        idx
        for idx, row in replay_by_idx.items()
        if math.isfinite(median_progress) and _safe_float(row.get("true_terminal_progress")) <= median_progress
    ]
    bad_low_cost_idx = min(poor, key=lambda idx: costs[idx]) if poor else pred_best_idx

    pairs: List[Tuple[str, int, int]] = []
    for pair_type, bad_idx in (
        ("true_best_vs_bad_low_cost", bad_low_cost_idx),
        ("true_best_vs_pred_best", pred_best_idx),
    ):
        if bad_idx != true_best_idx:
            pairs.append((pair_type, int(true_best_idx), int(bad_idx)))

    top_indices = np.argsort(costs)[: min(topk, costs.shape[0])]
    needed = {true_best_idx, pred_best_idx, bad_low_cost_idx, *top_indices.tolist()}
    return pairs, np.asarray(sorted(needed), dtype=np.int64)


def _true_metrics_for_candidate(replay_rows: Sequence[Dict[str, str]]) -> Dict[int, Dict[str, float]]:
    out = {}
    for row in replay_rows:
        idx = _safe_int(row.get("candidate_idx"))
        if idx >= 0:
            out[idx] = {
                "true_progress": _safe_float(row.get("true_terminal_progress")),
                "true_terminal_task_cost": _safe_float(row.get("true_terminal_task_cost")),
                "pred_cost": _safe_float(row.get("predicted_terminal_cost_c25")),
            }
    return out


def _append_pair_rows(
    rows: List[Dict[str, object]],
    episode_id: int,
    mpc_step_idx: int,
    raw_env_step: int,
    model_name: str,
    pair_type: str,
    good_idx: int,
    bad_idx: int,
    steps: Sequence[int],
    predicted: np.ndarray,
    selected_indices: np.ndarray,
    real_latents: Dict[int, Dict[int, np.ndarray]],
    z0: np.ndarray,
    zg: np.ndarray,
    planning_horizon_raw: int,
    true_metrics: Dict[int, Dict[str, float]],
    pool_costs: np.ndarray,
    failure_labels: Dict[str, object],
) -> None:
    selected_pos = {int(idx): pos for pos, idx in enumerate(selected_indices)}
    if good_idx not in selected_pos or bad_idx not in selected_pos:
        return
    good_pos = selected_pos[good_idx]
    bad_pos = selected_pos[bad_idx]
    q = zg - z0
    q_norm = _sqdist(zg, z0)
    pred_len = predicted.shape[1]
    for step in steps:
        if int(step) == 0:
            pred_step = -1
            zhat_good = z0
            zhat_bad = z0
            zreal_good = z0
            zreal_bad = z0
        else:
            pred_step = _future_index(pred_len, step, planning_horizon_raw)
            zhat_good = predicted[good_pos, pred_step]
            zhat_bad = predicted[bad_pos, pred_step]
            zreal_good = real_latents.get(good_idx, {}).get(step)
            zreal_bad = real_latents.get(bad_idx, {}).get(step)
        if zreal_good is None or zreal_bad is None:
            continue
        pred_gap_sq = _sqdist(zhat_good, zhat_bad)
        real_gap_sq = _sqdist(zreal_good, zreal_bad)
        pred_err_good = _sqdist(zhat_good, zreal_good)
        pred_err_bad = _sqdist(zhat_bad, zreal_bad)
        d_good = zhat_good - z0
        d_bad = zhat_bad - z0
        real_d_good = zreal_good - z0
        real_d_bad = zreal_bad - z0
        c_good = _sqdist(zhat_good, zg)
        c_bad = _sqdist(zhat_bad, zg)
        real_c_good = _sqdist(zreal_good, zg)
        real_c_bad = _sqdist(zreal_bad, zg)
        row = {
                "episode_id": episode_id,
                "mpc_step_idx": mpc_step_idx,
                "raw_env_step": raw_env_step,
                "model": model_name,
                "latent_dim": _model_latent_dim(model_name, z0),
                "pair_type": pair_type,
                "good_candidate_idx": int(good_idx),
                "bad_candidate_idx": int(bad_idx),
                "k": int(step),
                "model_rollout_index": int(pred_step),
                "pred_gap_norm": pred_gap_sq / (q_norm + EPS),
                "real_gap_norm": real_gap_sq / (q_norm + EPS),
                "opening_ratio": pred_gap_sq / (real_gap_sq + EPS),
                "cos_good": _cosine(d_good, q),
                "cos_bad": _cosine(d_bad, q),
                "cos_gap": _cosine(d_bad, q) - _cosine(d_good, q),
                "real_cos_good": _cosine(real_d_good, q),
                "real_cos_bad": _cosine(real_d_bad, q),
                "real_cos_gap": _cosine(real_d_bad, q) - _cosine(real_d_good, q),
                "pred_err_good_norm": pred_err_good / (q_norm + EPS),
                "pred_err_bad_norm": pred_err_bad / (q_norm + EPS),
                "noise_ratio": (math.sqrt(pred_err_good) + math.sqrt(pred_err_bad)) / (math.sqrt(real_gap_sq) + EPS),
                "noise_ratio_pred_gap": (math.sqrt(pred_err_good) + math.sqrt(pred_err_bad)) / (math.sqrt(pred_gap_sq) + EPS),
                "c_good_norm": c_good / (q_norm + EPS),
                "c_bad_norm": c_bad / (q_norm + EPS),
                "c_gap_norm": (c_bad - c_good) / (q_norm + EPS),
                "real_c_good_norm": real_c_good / (q_norm + EPS),
                "real_c_bad_norm": real_c_bad / (q_norm + EPS),
                "real_c_gap_norm": (real_c_bad - real_c_good) / (q_norm + EPS),
                "c_good": c_good,
                "c_bad": c_bad,
                "c_gap": c_bad - c_good,
                "real_c_good": real_c_good,
                "real_c_bad": real_c_bad,
                "real_c_gap": real_c_bad - real_c_good,
                "goal_distance_norm_sq": q_norm,
                "true_progress_good": true_metrics.get(good_idx, {}).get("true_progress", float("nan")),
                "true_progress_bad": true_metrics.get(bad_idx, {}).get("true_progress", float("nan")),
                "true_terminal_task_cost_good": true_metrics.get(good_idx, {}).get("true_terminal_task_cost", float("nan")),
                "true_terminal_task_cost_bad": true_metrics.get(bad_idx, {}).get("true_terminal_task_cost", float("nan")),
                "pred_cost_good": float(pool_costs[good_idx]),
                "pred_cost_bad": float(pool_costs[bad_idx]),
            }
        row.update(failure_labels)
        rows.append(row)


def _append_topk_rows(
    rows: List[Dict[str, object]],
    episode_id: int,
    mpc_step_idx: int,
    raw_env_step: int,
    model_name: str,
    steps: Sequence[int],
    predicted: np.ndarray,
    selected_indices: np.ndarray,
    top_indices: np.ndarray,
    z0: np.ndarray,
    planning_horizon_raw: int,
) -> None:
    selected_pos = {int(idx): pos for pos, idx in enumerate(selected_indices)}
    positions = [selected_pos[int(idx)] for idx in top_indices if int(idx) in selected_pos]
    if len(positions) < 2:
        return
    pred_len = predicted.shape[1]
    for step in steps:
        latents = np.repeat(z0[None], len(positions), axis=0) if int(step) == 0 else predicted[positions, _future_index(pred_len, step, planning_horizon_raw)]
        dists = _pairwise_sq_dists(latents)
        displacements = latents - z0[None]
        mean_cos, max_cos, frac_cos = _mean_pairwise_cosine(displacements)
        rows.append(
            {
                "episode_id": episode_id,
                "mpc_step_idx": mpc_step_idx,
                "raw_env_step": raw_env_step,
                "model": model_name,
                "latent_dim": _model_latent_dim(model_name, z0),
                "k": int(step),
                "topk": int(len(positions)),
                "pairwise_pred_dist_mean": float(np.mean(dists)) if dists.size else float("nan"),
                "pairwise_pred_dist_min": float(np.min(dists)) if dists.size else float("nan"),
                "pairwise_pred_dist_median": float(np.median(dists)) if dists.size else float("nan"),
                "effective_rank": _effective_rank(latents),
                "mean_pairwise_cosine": mean_cos,
                "max_pairwise_cosine": max_cos,
                "fraction_pairwise_cosine_gt_0p9": frac_cos,
            }
        )


def _aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["k"]))].append(row)
    out = []
    for (model, step), group in sorted(grouped.items(), key=lambda item: (_safe_int(item[0][0].replace("state", "").replace("baseline", "192"), 999), item[0][1])):
        def arr(key: str) -> np.ndarray:
            return np.asarray([_safe_float(row.get(key)) for row in group], dtype=np.float64)

        cos_gap = arr("cos_gap")
        noise_ratio = arr("noise_ratio")
        c_gap = arr("c_gap_norm")
        pred_gap_norm = arr("pred_gap_norm")
        real_gap_norm = arr("real_gap_norm")
        opening_ratio = pred_gap_norm / (real_gap_norm + EPS)
        valid_ratio = np.isfinite(real_gap_norm) & (real_gap_norm > REAL_GAP_NORM_MIN_FOR_RATIOS)
        noise_ratio_valid = noise_ratio[valid_ratio]
        opening_ratio_valid = opening_ratio[valid_ratio]
        out.append(
            {
                "model": model,
                "latent_dim": _safe_int(group[0].get("latent_dim")),
                "k": step,
                "num_rows": len(group),
                "num_planning_calls": len({(row["episode_id"], row["mpc_step_idx"]) for row in group}),
                "mean_pred_gap_norm": float(np.nanmean(pred_gap_norm)),
                "median_pred_gap_norm": float(np.nanmedian(pred_gap_norm)),
                "mean_real_gap_norm": float(np.nanmean(real_gap_norm)),
                "median_real_gap_norm": float(np.nanmedian(real_gap_norm)),
                "ratio_valid_min_real_gap_norm": REAL_GAP_NORM_MIN_FOR_RATIOS,
                "ratio_valid_frac": float(np.nanmean(valid_ratio)),
                "mean_opening_ratio": float(np.nanmean(arr("opening_ratio"))),
                "median_opening_ratio": float(np.nanmedian(opening_ratio)),
                "mean_opening_ratio_valid": float(np.nanmean(opening_ratio_valid)) if noise_ratio_valid.size else float("nan"),
                "median_opening_ratio_valid": float(np.nanmedian(opening_ratio_valid)) if noise_ratio_valid.size else float("nan"),
                "frac_opening_ratio_lt_1_valid": float(np.nanmean(opening_ratio_valid < 1.0)) if opening_ratio_valid.size else float("nan"),
                "mean_cos_gap": float(np.nanmean(cos_gap)),
                "median_cos_gap": float(np.nanmedian(cos_gap)),
                "frac_cos_gap_gt_0": float(np.nanmean(cos_gap > 0)),
                "mean_noise_ratio": float(np.nanmean(noise_ratio)),
                "median_noise_ratio": float(np.nanmedian(noise_ratio)),
                "mean_noise_ratio_valid": float(np.nanmean(noise_ratio_valid)) if noise_ratio_valid.size else float("nan"),
                "median_noise_ratio_valid": float(np.nanmedian(noise_ratio_valid)) if noise_ratio_valid.size else float("nan"),
                "frac_noise_ratio_gt_1": float(np.nanmean(noise_ratio > 1)),
                "frac_noise_ratio_gt_1_valid": float(np.nanmean(noise_ratio_valid > 1.0)) if noise_ratio_valid.size else float("nan"),
                "mean_c_gap_norm": float(np.nanmean(c_gap)),
                "median_c_gap_norm": float(np.nanmedian(c_gap)),
                "frac_c_gap_norm_lt_0": float(np.nanmean(c_gap < 0)),
            }
        )
    return out


def _aggregate_rows_by_failure_label(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    label_keys = ("support_failure", "ranking_failure", "imagination_failure", "mean_sequence_failure", "first_block_failure")
    out = []
    for label_key in label_keys:
        valid_values = sorted({str(row.get(label_key, "")) for row in rows if str(row.get(label_key, "")) in {"0", "1"}})
        for label_value in valid_values:
            subset = [row for row in rows if str(row.get(label_key, "")) == label_value]
            for row in _aggregate_rows(subset):
                row["failure_label"] = label_key
                row["failure_label_value"] = int(label_value)
                out.append(row)
    return out


def _bad_path_cost_curve_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["k"]))].append(row)
    out = []
    for (model, step), group in sorted(grouped.items(), key=lambda item: (MODEL_DIMS.get(item[0][0], 999), item[0][1])):
        def values(key: str) -> np.ndarray:
            return np.asarray([_safe_float(row.get(key)) for row in group], dtype=np.float64)

        pred_bad = values("c_bad")
        real_bad = values("real_c_bad")
        pred_bad_norm = values("c_bad_norm")
        real_bad_norm = values("real_c_bad_norm")
        out.append(
            {
                "model": model,
                "latent_dim": _safe_int(group[0].get("latent_dim")),
                "k": step,
                "num_rows": len(group),
                "mean_pred_bad_c": float(np.nanmean(pred_bad)),
                "median_pred_bad_c": float(np.nanmedian(pred_bad)),
                "mean_real_bad_c": float(np.nanmean(real_bad)),
                "median_real_bad_c": float(np.nanmedian(real_bad)),
                "mean_pred_bad_c_norm": float(np.nanmean(pred_bad_norm)),
                "median_pred_bad_c_norm": float(np.nanmedian(pred_bad_norm)),
                "mean_real_bad_c_norm": float(np.nanmean(real_bad_norm)),
                "median_real_bad_c_norm": float(np.nanmedian(real_bad_norm)),
                "mean_pred_minus_real_bad_c_norm": float(np.nanmean(pred_bad_norm - real_bad_norm)),
                "median_pred_minus_real_bad_c_norm": float(np.nanmedian(pred_bad_norm - real_bad_norm)),
                "frac_pred_underestimates_real_bad_c": float(np.nanmean(pred_bad < real_bad)),
                "frac_pred_underestimates_real_bad_c_norm": float(np.nanmean(pred_bad_norm < real_bad_norm)),
            }
        )
    return out


def _terminal_cem_score_rows(rows: List[Dict[str, object]], planning_horizon_raw: int) -> List[Dict[str, object]]:
    terminal_rows = [row for row in rows if int(row.get("k", -1)) == int(planning_horizon_raw)]
    out = []
    for row in terminal_rows:
        out.append(
            {
                "episode_id": row.get("episode_id"),
                "mpc_step_idx": row.get("mpc_step_idx"),
                "raw_env_step": row.get("raw_env_step"),
                "model": row.get("model"),
                "latent_dim": row.get("latent_dim"),
                "pair_type": row.get("pair_type"),
                "good_candidate_idx": row.get("good_candidate_idx"),
                "bad_candidate_idx": row.get("bad_candidate_idx"),
                "terminal_horizon_raw": int(planning_horizon_raw),
                "pred_terminal_bad_c": row.get("c_bad"),
                "real_terminal_bad_c": row.get("real_c_bad"),
                "pred_terminal_good_c": row.get("c_good"),
                "real_terminal_good_c": row.get("real_c_good"),
                "pred_terminal_bad_c_norm": row.get("c_bad_norm"),
                "real_terminal_bad_c_norm": row.get("real_c_bad_norm"),
                "pred_terminal_good_c_norm": row.get("c_good_norm"),
                "real_terminal_good_c_norm": row.get("real_c_good_norm"),
                "pred_terminal_bad_minus_good_c": row.get("c_gap"),
                "real_terminal_bad_minus_good_c": row.get("real_c_gap"),
                "pred_terminal_bad_minus_good_c_norm": row.get("c_gap_norm"),
                "real_terminal_bad_minus_good_c_norm": row.get("real_c_gap_norm"),
                "pred_terminal_bad_underestimates_real": int(_safe_float(row.get("c_bad")) < _safe_float(row.get("real_c_bad"))),
                "pred_terminal_bad_wins_good": int(_safe_float(row.get("c_gap")) < 0),
                "true_terminal_task_cost_bad": row.get("true_terminal_task_cost_bad"),
                "true_terminal_task_cost_good": row.get("true_terminal_task_cost_good"),
                "true_progress_bad": row.get("true_progress_bad"),
                "true_progress_good": row.get("true_progress_good"),
            }
        )
    return out


def _terminal_cem_score_summary_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["mpc_step_idx"]))].append(row)
    out = []
    for (model, mpc_step_idx), group in sorted(grouped.items(), key=lambda item: (MODEL_DIMS.get(item[0][0], 999), item[0][1])):
        def values(key: str) -> np.ndarray:
            return np.asarray([_safe_float(row.get(key)) for row in group], dtype=np.float64)

        pred_bad = values("pred_terminal_bad_c_norm")
        real_bad = values("real_terminal_bad_c_norm")
        pred_gap = values("pred_terminal_bad_minus_good_c_norm")
        real_gap = values("real_terminal_bad_minus_good_c_norm")
        out.append(
            {
                "model": model,
                "latent_dim": _safe_int(group[0].get("latent_dim")),
                "mpc_step_idx": mpc_step_idx,
                "num_rows": len(group),
                "median_pred_terminal_bad_c_norm": float(np.nanmedian(pred_bad)),
                "median_real_terminal_bad_c_norm": float(np.nanmedian(real_bad)),
                "median_pred_minus_real_terminal_bad_c_norm": float(np.nanmedian(pred_bad - real_bad)),
                "frac_pred_terminal_bad_underestimates_real": float(np.nanmean(pred_bad < real_bad)),
                "median_pred_terminal_bad_minus_good_c_norm": float(np.nanmedian(pred_gap)),
                "median_real_terminal_bad_minus_good_c_norm": float(np.nanmedian(real_gap)),
                "frac_pred_terminal_bad_wins_good": float(np.nanmean(pred_gap < 0)),
                "frac_real_terminal_bad_wins_good": float(np.nanmean(real_gap < 0)),
            }
        )
    return out


def _plot_series(summary_rows: List[Dict[str, object]], key: str, ylabel: str, output_path: Path, hline: Optional[float] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[early_mode] matplotlib unavailable; skipping plots.", flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in summary_rows}, key=lambda m: MODEL_DIMS.get(m, 999))
    fig, ax = plt.subplots(figsize=(5.8, 3.8), facecolor="white")
    for model in models:
        rows = sorted([row for row in summary_rows if row["model"] == model], key=lambda row: int(row["k"]))
        ax.plot([int(row["k"]) for row in rows], [_safe_float(row.get(key)) for row in rows], marker="o", linewidth=2.0, label=model)
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("raw rollout step k")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=260)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_topk(topk_rows: List[Dict[str, object]], output_path: Path) -> None:
    if not topk_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in topk_rows:
        grouped[(str(row["model"]), int(row["k"]))].append(row)
    models = sorted({key[0] for key in grouped}, key=lambda m: MODEL_DIMS.get(m, 999))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), facecolor="white")
    for model in models:
        steps = sorted({key[1] for key in grouped if key[0] == model})
        med = []
        spread = []
        for step in steps:
            rows = grouped[(model, step)]
            med.append(float(np.nanmean([_safe_float(row["pairwise_pred_dist_median"]) for row in rows])))
            spread.append(float(np.nanmean([_safe_float(row["fraction_pairwise_cosine_gt_0p9"]) for row in rows])))
        axes[0].plot(steps, med, marker="o", label=model)
        axes[1].plot(steps, spread, marker="o", label=model)
    axes[0].set_title("TopK predicted distance")
    axes[0].set_ylabel("median pairwise ||z_i-z_j||^2")
    axes[1].set_title("TopK angular collapse")
    axes[1].set_ylabel("fraction pairwise cosine > 0.9")
    for ax in axes:
        ax.set_xlabel("raw rollout step k")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=260)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_case(rows: List[Dict[str, object]], episode_id: int, mpc_step_idx: int, output_path: Path) -> None:
    case = [row for row in rows if int(row["episode_id"]) == episode_id and int(row["mpc_step_idx"]) == mpc_step_idx]
    if not case:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in case}, key=lambda m: MODEL_DIMS.get(m, 999))
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.5), facecolor="white")
    for model in models:
        rows_m = sorted([row for row in case if row["model"] == model], key=lambda row: int(row["k"]))
        steps = [int(row["k"]) for row in rows_m]
        axes[0].plot(steps, [_safe_float(row["c_good_norm"]) for row in rows_m], marker="o", label=f"{model} good")
        axes[0].plot(steps, [_safe_float(row["c_bad_norm"]) for row in rows_m], marker="x", linestyle="--", label=f"{model} bad")
        axes[1].plot(steps, [_safe_float(row["real_c_good_norm"]) for row in rows_m], marker="o", label=f"{model} good")
        axes[1].plot(steps, [_safe_float(row["real_c_bad_norm"]) for row in rows_m], marker="x", linestyle="--", label=f"{model} bad")
        axes[2].plot(steps, [_safe_float(row["cos_gap"]) for row in rows_m], marker="o", label=model)
        axes[3].plot(steps, [_safe_float(row["noise_ratio"]) for row in rows_m], marker="o", label=model)
    axes[0].set_title("Predicted cost")
    axes[1].set_title("Real replay cost")
    axes[2].set_title("cos_bad - cos_good")
    axes[3].set_title("noise ratio")
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[3].axhline(1, color="black", linestyle="--", linewidth=1.0)
    for ax in axes:
        ax.set_xlabel("raw step k")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"episode {episode_id}, mpc step {mpc_step_idx}", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    steps = tuple(int(step) for step in args.steps)
    planning_horizon_raw = int(args.planning_horizon_raw)
    print(
        "[early_mode] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[early_mode] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    if not trace_dir.exists():
        raise FileNotFoundError(
            f"Trace directory does not exist: {trace_dir}. "
            "Run scripts/state8_cem_failure_audit.py first, or point --trace_dir to an existing CEM trace."
        )
    attempts = _load_trace_attempts(trace_dir)
    pool_rows = _read_csv(trace_dir / "candidate_pool_summary.csv")
    replay_rows_all = _read_csv(trace_dir / "candidate_true_replay_audit.csv")
    if not replay_rows_all:
        replay_rows_all = _read_csv(trace_dir / "true_replay_candidates.csv")
    missing_inputs = []
    if not pool_rows:
        missing_inputs.append(str(trace_dir / "candidate_pool_summary.csv"))
    if not replay_rows_all:
        missing_inputs.append(str(trace_dir / "candidate_true_replay_audit.csv"))
    if not attempts:
        missing_inputs.append(str(trace_dir / "trace_run_summary.json"))
    if missing_inputs:
        raise RuntimeError(
            "Trace directory exists but is missing required trace data:\n  - "
            + "\n  - ".join(missing_inputs)
            + "\nRe-run the state8 CEM failure audit with true replay/candidate pool tracing enabled."
        )
    failure_rows = _group_rows(_read_csv(trace_dir / "planning_call_failure_types.csv"), ["episode_id", "mpc_step_idx"])
    pool_rows_by_call = _group_rows(pool_rows, ["episode_id", "mpc_step_idx"])
    replay_rows_by_call = _group_rows(replay_rows_all, ["episode_id", "mpc_step_idx"])
    fallback_pool_paths = _pool_paths(trace_dir)
    models = _parse_name_paths(args.models)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    pair_rows: List[Dict[str, object]] = []
    topk_rows: List[Dict[str, object]] = []
    skip_rows: List[Dict[str, object]] = []
    processed_calls = 0
    env = _make_env(args.env_name)

    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        state_key = _find_key(h5, [args.state_key, "state"])
        action_key = _find_key(h5, [args.action_key, "action"])
        actions_dataset = np.asarray(h5[action_key]).astype(np.float32)
        action_mean, action_std = _action_normalization(actions_dataset)
        call_keys = sorted(replay_rows_by_call)
        if args.max_calls > 0:
            call_keys = call_keys[: args.max_calls]
        for episode_id, mpc_step_idx in call_keys:
            attempt = attempts.get(episode_id)
            if not attempt and mpc_step_idx == 0:
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "missing_trace_attempt_start_goal_rows"})
                continue
            start_row = _safe_int(attempt.get("start_row")) if attempt else -1
            goal_row = _safe_int(attempt.get("goal_row")) if attempt else -1
            row_group = pool_rows_by_call.get((episode_id, mpc_step_idx), [])
            pool_path = _candidate_pool_path(row_group, fallback_pool_paths, (episode_id, mpc_step_idx))
            if pool_path is None or not pool_path.exists():
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "missing_candidate_pool_npz"})
                continue
            pool = np.load(pool_path, allow_pickle=False)
            if "action_sequence" not in pool.files:
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "candidate_pool_missing_action_sequence"})
                continue
            has_saved_state = "current_state" in pool.files and "goal_state" in pool.files
            if args.step0_only and mpc_step_idx != 0 and not has_saved_state:
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "nonzero_mpc_step_without_saved_current_state"})
                continue
            if not has_saved_state and (start_row < 0 or goal_row < 0):
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "bad_start_goal_rows_and_no_saved_state"})
                continue
            replay_rows = replay_rows_by_call[(episode_id, mpc_step_idx)]
            pairs, selected_indices = _select_candidates_for_call(pool, replay_rows, args.topk)
            if not pairs or selected_indices.size == 0:
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "no_valid_true_best_bad_pairs"})
                continue
            costs = np.asarray(pool["terminal_cost_c25"], dtype=np.float64)
            top_indices = np.argsort(costs)[: min(args.topk, costs.shape[0])]
            all_indices = np.asarray(sorted(set(selected_indices.tolist()) | set(top_indices.tolist())), dtype=np.int64)
            if has_saved_state:
                start_state = np.asarray(pool["current_state"], dtype=np.float64).reshape(-1)
                goal_state = np.asarray(pool["goal_state"], dtype=np.float64).reshape(-1)
                obs, _info = _reset_env_to_state(env, start_state, goal_state, args.reset_state_tol)
                context_pixels = np.asarray([_extract_pixels_from_obs(env, obs)])
                goal_pixels = _read_h5_rows(h5[pixels_key], np.asarray([goal_row], dtype=np.int64)) if goal_row >= 0 else context_pixels
            else:
                start_state = np.asarray(h5[state_key][start_row], dtype=np.float64)
                goal_state = np.asarray(h5[state_key][goal_row], dtype=np.float64)
                context_pixels = _read_h5_rows(h5[pixels_key], np.asarray([start_row], dtype=np.int64))
                goal_pixels = _read_h5_rows(h5[pixels_key], np.asarray([goal_row], dtype=np.int64))
            action_sequence = np.asarray(pool["action_sequence"], dtype=np.float32)
            pixels_by_candidate: Dict[int, Dict[int, np.ndarray]] = {}
            replay_meta_by_candidate: Dict[int, Dict[str, object]] = {}
            for candidate_idx in all_indices:
                try:
                    raw_actions = _action_sequence_to_raw_future(
                        action_sequence[int(candidate_idx)],
                        action_mean,
                        action_std,
                        planning_horizon_raw,
                        args.candidate_actions_are_normalized,
                    )
                    pixels_by_step, replay_meta = _replay_candidate_observations(
                        env,
                        start_state,
                        goal_state,
                        raw_actions,
                        (0, *steps),
                        args.reset_state_tol,
                    )
                    pixels_by_candidate[int(candidate_idx)] = pixels_by_step
                    replay_meta_by_candidate[int(candidate_idx)] = replay_meta
                except Exception as exc:  # noqa: BLE001
                    skip_rows.append(
                        {
                            "episode_id": episode_id,
                            "mpc_step_idx": mpc_step_idx,
                            "candidate_idx": int(candidate_idx),
                            "skip_reason": f"true_replay_failed: {exc}",
                        }
                    )
            if not pixels_by_candidate:
                continue

            true_metrics = _true_metrics_for_candidate(replay_rows)
            failure_label_rows = failure_rows.get((episode_id, mpc_step_idx), [])
            failure_labels = {}
            if failure_label_rows:
                for key in ("support_failure", "ranking_failure", "imagination_failure", "mean_sequence_failure", "first_block_failure"):
                    if key in failure_label_rows[0]:
                        failure_labels[key] = failure_label_rows[0].get(key, "")
            for model_name, checkpoint in models.items():
                print(f"[early_mode] call ep={episode_id} step={mpc_step_idx}: model={model_name}", flush=True)
                model = _load_model(checkpoint, device)
                with torch.no_grad():
                    predicted, z0, _selected_actions = _rollout_predicted_latents(
                        model,
                        context_pixels,
                        action_sequence,
                        all_indices,
                        args.img_size,
                        args.batch_size,
                        device,
                    )
                    zg = _encode_pixels(model, goal_pixels, args.img_size, args.batch_size, device)[0, 0]
                    real_latents = _encode_replay_latents(
                        model,
                        pixels_by_candidate,
                        all_indices,
                        (0, *steps),
                        args.img_size,
                        args.batch_size,
                        device,
                    )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for pair_type, good_idx, bad_idx in pairs:
                    _append_pair_rows(
                        pair_rows,
                        episode_id,
                        mpc_step_idx,
                        _safe_int(row_group[0].get("raw_env_step")) if row_group else mpc_step_idx,
                        model_name,
                        pair_type,
                        good_idx,
                        bad_idx,
                        steps,
                        predicted,
                        all_indices,
                        real_latents,
                        z0,
                        zg,
                        planning_horizon_raw,
                        true_metrics,
                        costs,
                        failure_labels,
                    )
                _append_topk_rows(
                    topk_rows,
                    episode_id,
                    mpc_step_idx,
                    _safe_int(row_group[0].get("raw_env_step")) if row_group else mpc_step_idx,
                    model_name,
                    steps,
                    predicted,
                    all_indices,
                    top_indices,
                    z0,
                    planning_horizon_raw,
                )
            processed_calls += 1

    summary_rows = _aggregate_rows(pair_rows)
    summary_by_failure_rows = _aggregate_rows_by_failure_label(pair_rows)
    bad_curve_rows = _bad_path_cost_curve_rows(pair_rows)
    terminal_rows = _terminal_cem_score_rows(pair_rows, planning_horizon_raw)
    terminal_summary_rows = _terminal_cem_score_summary_rows(terminal_rows)
    _write_csv(output_dir / "early_mode_gap_rows.csv", pair_rows)
    _write_csv(output_dir / "early_mode_gap_summary.csv", summary_rows)
    _write_csv(output_dir / "early_mode_gap_summary_by_failure_type.csv", summary_by_failure_rows)
    _write_csv(output_dir / "bad_path_pred_vs_real_cost_curve.csv", bad_curve_rows)
    _write_csv(output_dir / "terminal_cem_score_rows.csv", terminal_rows)
    _write_csv(output_dir / "terminal_cem_score_summary.csv", terminal_summary_rows)
    _write_csv(output_dir / "topk_collapse_rows.csv", topk_rows)
    _write_csv(output_dir / "early_mode_gap_skipped_calls.csv", skip_rows)
    _write_json(
        output_dir / "early_mode_gap_metadata.json",
        {
            "args": vars(args),
            "processed_calls": processed_calls,
            "num_pair_rows": len(pair_rows),
            "num_topk_rows": len(topk_rows),
            "num_skip_rows": len(skip_rows),
            "notes": [
                "Nonzero MPC steps are skipped by default unless exact current planning state is saved.",
                "Model predicted rollout uses the action_sequence saved in the CEM candidate pool.",
                "Real replay latents are recomputed by resetting PushT to the dataset start state and replaying the candidate open-loop actions.",
                "k-to-rollout-index maps k=planning_horizon_raw to the final rollout latent to match terminal CEM cost.",
            ],
        },
    )
    if summary_rows:
        _plot_series(summary_rows, "mean_pred_gap_norm", "mean predicted mode gap / ||zg-z0||^2", figures_dir / "fig_pred_mode_gap_over_steps")
        _plot_series(summary_rows, "mean_real_gap_norm", "mean real mode gap / ||zg-z0||^2", figures_dir / "fig_real_mode_gap_over_steps")
        _plot_series(summary_rows, "mean_opening_ratio", "predicted gap / real gap", figures_dir / "fig_opening_ratio_over_steps", hline=1.0)
        _plot_series(summary_rows, "mean_cos_gap", "mean cos_bad - cos_good", figures_dir / "fig_cos_gap_over_steps", hline=0.0)
        _plot_series(summary_rows, "mean_noise_ratio", "mean noise ratio", figures_dir / "fig_noise_ratio_over_steps", hline=1.0)
        _plot_series(summary_rows, "mean_c_gap_norm", "mean predicted cost gap c_bad-c_good", figures_dir / "fig_cost_gap_over_steps", hline=0.0)
        _plot_topk(topk_rows, figures_dir / "fig_topK_collapse")
        _plot_case(pair_rows, 9, 0, figures_dir / "case_study_episode9_step0")
        _plot_case(pair_rows, 13, 0, figures_dir / "case_study_episode13_step0")
    print(f"[early_mode] processed_calls={processed_calls}, wrote outputs under {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Early mode-gap / noise-ball diagnostic for PushT CEM traces.")
    parser.add_argument("--trace_dir", default="results/cem_trace_state8_failure_full")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint_object.ckpt")
    parser.add_argument("--output_dir", default="results/early_mode_gap_diagnostic")
    parser.add_argument("--env_name", default="swm/PushT-v1")
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--planning_horizon_raw", type=int, default=25)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max_calls", type=int, default=-1)
    parser.add_argument("--step0_only", type=lambda value: str(value).lower() not in {"0", "false", "no"}, default=True)
    parser.add_argument("--candidate_actions_are_normalized", type=lambda value: str(value).lower() not in {"0", "false", "no"}, default=True)
    parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
