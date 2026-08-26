from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import hdf5plugin  # noqa: F401

    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_QUANTILES = (0.95, 0.99, 0.999)
DEFAULT_EPSILONS = (1e-6, 1e-4, 1e-2)
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


def _load_repo_helpers():
    from planner_success_reference_diagnostic import _encode_pixels, _find_key, _load_model, _read_h5_rows

    return _encode_pixels, _find_key, _load_model, _read_h5_rows


@dataclass(frozen=True)
class TransitionBatch:
    rows: np.ndarray
    next_rows: np.ndarray
    context_rows: np.ndarray
    episode_idx: np.ndarray
    step_idx: np.ndarray
    action_blocks: np.ndarray
    action_block_ids: np.ndarray
    context_action_blocks: np.ndarray
    action_blocks_norm: np.ndarray


def _parse_name_paths(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            path = Path(item)
            out[_safe_name(path.stem)] = path
            continue
        name, path = item.split("=", 1)
        out[_safe_name(name)] = Path(path)
    return out


def _parse_float_list(value: str, default: Sequence[float]) -> List[float]:
    if value is None or str(value).strip() == "":
        return list(default)
    return [float(item) for item in str(value).replace(",", " ").split()]


def _safe_name(value: str) -> str:
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    name = "".join(keep).strip("_")
    return name or "checkpoint"


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


def _write_parquet_or_csv(path: Path, rows: List[Dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _write_csv(path.with_suffix(".csv"), rows)
        return str(path.with_suffix(".csv"))
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        csv_path = path.with_suffix(".csv")
        note_path = path.with_suffix(".parquet_unavailable.txt")
        _write_csv(csv_path, rows)
        note_path.write_text(
            "Could not write parquet because pandas/pyarrow/fastparquet is unavailable or failed.\n"
            f"Fallback CSV: {csv_path.name}\n"
            f"Error: {exc}\n"
        )
        return str(csv_path)


def _action_block_ids(blocks: np.ndarray) -> np.ndarray:
    ids: Dict[bytes, int] = {}
    out = []
    for block in np.asarray(blocks, dtype=np.float32):
        key = np.ascontiguousarray(block).tobytes()
        if key not in ids:
            ids[key] = len(ids)
        out.append(ids[key])
    return np.asarray(out, dtype=np.int64)


def _has_raw_action_block(
    row_by_key: Dict[Tuple[int, int], int],
    episode: int,
    start_step: int,
    frameskip: int,
) -> bool:
    return all((episode, start_step + offset) in row_by_key for offset in range(frameskip))


def _sample_transition_rows(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    total_rows: int,
    frameskip: int,
    history_size: int,
    num_transitions: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_idx = np.asarray(episode_idx).reshape(-1)
    step_idx = np.asarray(step_idx).reshape(-1)
    row_by_key = {(int(ep), int(step)): row for row, (ep, step) in enumerate(zip(episode_idx, step_idx))}
    candidates = []
    next_rows = []
    context_rows = []
    for row in range(total_rows):
        episode = int(episode_idx[row])
        step = int(step_idx[row])
        next_row = row_by_key.get((episode, step + int(frameskip)))
        if next_row is None:
            continue
        context_steps = [step - (history_size - 1 - offset) * frameskip for offset in range(history_size)]
        context = [row_by_key.get((episode, context_step)) for context_step in context_steps]
        if any(context_row is None for context_row in context):
            continue
        if not all(_has_raw_action_block(row_by_key, episode, context_step, frameskip) for context_step in context_steps):
            continue
        candidates.append(row)
        next_rows.append(next_row)
        context_rows.append(context)
    candidates = np.asarray(candidates, dtype=np.int64)
    next_rows = np.asarray(next_rows, dtype=np.int64)
    context_rows = np.asarray(context_rows, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("No same-episode transitions with enough predictor history found. Check episode_idx/step_idx/frameskip/history_size.")
    rng = np.random.default_rng(seed)
    count = min(int(num_transitions), int(candidates.size))
    selected = np.sort(rng.choice(candidates.size, size=count, replace=False))
    return candidates[selected], next_rows[selected], context_rows[selected]


def _build_action_blocks(actions: np.ndarray, rows: np.ndarray, frameskip: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    blocks = []
    for row in np.asarray(rows, dtype=np.int64):
        block = actions[row : row + frameskip]
        if block.shape[0] != frameskip:
            raise ValueError(f"Row {row} does not have a full action block of length {frameskip}.")
        blocks.append(block.reshape(-1))
    return np.asarray(blocks, dtype=np.float32)


def _build_context_action_blocks(actions: np.ndarray, context_rows: np.ndarray, frameskip: int) -> np.ndarray:
    flat_rows = np.asarray(context_rows, dtype=np.int64).reshape(-1)
    flat_blocks = _build_action_blocks(actions, flat_rows, frameskip)
    return flat_blocks.reshape(context_rows.shape[0], context_rows.shape[1], -1)


def _normalize_blocks(blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(blocks, axis=0, keepdims=True)
    std = np.std(blocks, axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((blocks - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


def _normalize_actions(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    flat = actions.reshape(-1, actions.shape[-1])
    valid = ~np.isnan(flat).any(axis=1)
    mean = flat[valid].mean(axis=0, keepdims=True)
    std = flat[valid].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((actions - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


def _load_transition_batch(
    h5: h5py.File,
    num_transitions: int,
    frameskip: int,
    history_size: int,
    seed: int,
    action_key: str,
    episode_key: str,
    step_key: str,
) -> TransitionBatch:
    total_rows = int(h5[action_key].shape[0])
    episode_idx = np.asarray(h5[episode_key]).reshape(-1)
    step_idx = np.asarray(h5[step_key]).reshape(-1)
    rows, next_rows, context_rows = _sample_transition_rows(
        episode_idx,
        step_idx,
        total_rows,
        frameskip,
        history_size,
        num_transitions,
        seed,
    )
    actions, _action_mean, _action_std = _normalize_actions(np.asarray(h5[action_key]))
    action_blocks = _build_action_blocks(actions, rows, frameskip)
    context_action_blocks = _build_context_action_blocks(actions, context_rows, frameskip)
    action_blocks_norm, _mean, _std = _normalize_blocks(action_blocks)
    return TransitionBatch(
        rows=rows,
        next_rows=next_rows,
        context_rows=context_rows,
        episode_idx=episode_idx[rows].astype(np.int64),
        step_idx=step_idx[rows].astype(np.int64),
        action_blocks=action_blocks,
        action_block_ids=_action_block_ids(action_blocks),
        context_action_blocks=context_action_blocks,
        action_blocks_norm=action_blocks_norm,
    )


def _sample_action_matched_pairs(
    action_blocks_norm: np.ndarray,
    num_pairs: int,
    action_matching_threshold: float,
    seed: int,
    max_attempt_multiplier: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(action_blocks_norm.shape[0])
    if n < 2:
        raise ValueError("Need at least two transitions to sample pairs.")
    rng = np.random.default_rng(seed)
    pairs_i: List[int] = []
    pairs_j: List[int] = []
    action_dist: List[float] = []
    target = int(num_pairs)
    attempts = 0
    max_attempts = max(target * int(max_attempt_multiplier), 10_000)
    while len(pairs_i) < target and attempts < max_attempts:
        batch = min(max(4096, target - len(pairs_i)), 65536)
        left = rng.integers(0, n, size=batch)
        right = rng.integers(0, n, size=batch)
        mask = left != right
        left = left[mask]
        right = right[mask]
        dist = np.linalg.norm(action_blocks_norm[left] - action_blocks_norm[right], axis=1)
        keep = np.where(dist <= action_matching_threshold)[0]
        for idx in keep:
            pairs_i.append(int(left[idx]))
            pairs_j.append(int(right[idx]))
            action_dist.append(float(dist[idx]))
            if len(pairs_i) >= target:
                break
        attempts += batch
    if not pairs_i:
        raise ValueError(
            "No action-matched pairs sampled. Increase --action_matching_threshold or --num_transitions."
        )
    return np.asarray(pairs_i, dtype=np.int64), np.asarray(pairs_j, dtype=np.int64), np.asarray(action_dist, dtype=np.float32)


def _sample_exact_action_pairs(
    action_block_ids: np.ndarray,
    num_pairs: int,
    seed: int,
    max_attempt_multiplier: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: Dict[int, np.ndarray] = {}
    for action_block_id in np.unique(action_block_ids):
        idx = np.where(action_block_ids == action_block_id)[0]
        if idx.size >= 2:
            groups[int(action_block_id)] = idx
    if not groups:
        raise ValueError(
            "No exactly identical action-block pairs found. "
            "Use --action_matching_mode approximate only if you explicitly want thresholded continuous-action matching."
        )
    rng = np.random.default_rng(seed)
    group_ids = np.asarray(list(groups), dtype=np.int64)
    pairs_i: List[int] = []
    pairs_j: List[int] = []
    target = int(num_pairs)
    attempts = 0
    max_attempts = max(target * int(max_attempt_multiplier), 10_000)
    while len(pairs_i) < target and attempts < max_attempts:
        group = groups[int(rng.choice(group_ids))]
        left, right = rng.choice(group, size=2, replace=False)
        pairs_i.append(int(left))
        pairs_j.append(int(right))
        attempts += 1
    return (
        np.asarray(pairs_i, dtype=np.int64),
        np.asarray(pairs_j, dtype=np.int64),
        np.zeros(len(pairs_i), dtype=np.float32),
    )


def _l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _compute_ratios(d_now: np.ndarray, d_next: np.ndarray, epsilon: float) -> np.ndarray:
    return np.asarray(d_next, dtype=np.float64) / np.maximum(np.asarray(d_now, dtype=np.float64), float(epsilon))


def _quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def _quantile_label(q: float) -> str:
    if math.isclose(q, 0.95, rel_tol=0.0, abs_tol=1e-12):
        return "q95"
    if math.isclose(q, 0.99, rel_tol=0.0, abs_tol=1e-12):
        return "q99"
    if math.isclose(q, 0.999, rel_tol=0.0, abs_tol=1e-12):
        return "q999"
    return f"q{int(round(q * 1000))}"


def _episode_pair_groups(episode_i: np.ndarray, episode_j: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, (left, right) in enumerate(zip(episode_i, episode_j)):
        key = tuple(sorted((int(left), int(right))))
        groups.setdefault(key, []).append(idx)
    return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}


def _bootstrap_quantile_ci(
    ratios: np.ndarray,
    groups: Dict[Tuple[int, int], np.ndarray],
    quantile: float,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    if samples <= 0 or not groups:
        return float("nan"), float("nan")
    keys = list(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        indices = np.concatenate([groups[keys[int(idx)]] for idx in sampled])
        estimates.append(_quantile(ratios[indices], quantile))
    finite = np.asarray(estimates, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _latent_dim_from_model(name: str, z: np.ndarray) -> int:
    return int(MODEL_DIMS.get(name, z.shape[-1]))


def _apply_transition_bottleneck(model, pred_raw, anchor):
    tb = getattr(model, "transition_bottleneck", None)
    if tb is None:
        return pred_raw
    delta_raw = pred_raw - anchor
    if tb.__class__.__name__ == "StateConditionedTangentBottleneck":
        delta_rec, _, _ = tb(delta_raw, anchor)
    else:
        delta_rec, _ = tb(delta_raw)
    return anchor + delta_rec


def _predict_next_latents(model, context_emb: np.ndarray, context_actions: np.ndarray, batch_size: int, device) -> np.ndarray:
    import torch

    history_size = int(getattr(getattr(model, "predictor", None), "pos_embedding").shape[1])
    chunks = []
    with torch.no_grad():
        for start in range(0, context_emb.shape[0], batch_size):
            emb = torch.from_numpy(context_emb[start:start + batch_size]).float().to(device)
            action = torch.from_numpy(context_actions[start:start + batch_size]).float().to(device)
            act_emb = model.action_encoder(action)
            ctx_emb = emb[:, -history_size:]
            ctx_act = act_emb[:, -history_size:]
            pred_raw = model.predict(ctx_emb, ctx_act)[:, -1:]
            pred = _apply_transition_bottleneck(model, pred_raw, ctx_emb[:, -1:])
            chunks.append(pred[:, 0].detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _pair_arrays(
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    z_now: np.ndarray,
    z_next_true: np.ndarray,
    z_next_pred: np.ndarray,
    epsilon: float,
) -> Dict[str, np.ndarray]:
    d_now = _l2_distance(z_now[pairs_i], z_now[pairs_j])
    d_next_true = _l2_distance(z_next_true[pairs_i], z_next_true[pairs_j])
    d_next_pred = _l2_distance(z_next_pred[pairs_i], z_next_pred[pairs_j])
    r_true = _compute_ratios(d_now, d_next_true, epsilon)
    r_model = _compute_ratios(d_now, d_next_pred, epsilon)
    deficit = np.maximum(d_next_true - d_next_pred, 0.0)
    e_i = _l2_distance(z_next_pred[pairs_i], z_next_true[pairs_i])
    e_j = _l2_distance(z_next_pred[pairs_j], z_next_true[pairs_j])
    pair_error = np.sqrt((e_i * e_i + e_j * e_j) / 2.0)
    return {
        "d_now": d_now,
        "d_next_true": d_next_true,
        "d_next_pred": d_next_pred,
        "r_true": r_true,
        "r_model": r_model,
        "deficit": deficit,
        "lower_bound": deficit / 2.0,
        "e_i": e_i,
        "e_j": e_j,
        "pair_error": pair_error,
    }


def _pair_metric_rows(
    name: str,
    checkpoint: Path,
    latent_dim: int,
    transition_batch: TransitionBatch,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    action_dist: np.ndarray,
    arrays: Dict[str, np.ndarray],
    epsilons: Sequence[float],
    seed: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    primary_epsilon = float(epsilons[0]) if epsilons else 1e-8
    for pair_idx, (left, right) in enumerate(zip(pairs_i, pairs_j)):
        row = {
            "pair_idx": int(pair_idx),
            "transition_i": int(transition_batch.rows[left]),
            "transition_j": int(transition_batch.rows[right]),
            "next_transition_i": int(transition_batch.next_rows[left]),
            "next_transition_j": int(transition_batch.next_rows[right]),
            "episode_i": int(transition_batch.episode_idx[left]),
            "episode_j": int(transition_batch.episode_idx[right]),
            "step_i": int(transition_batch.step_idx[left]),
            "step_j": int(transition_batch.step_idx[right]),
            "timestep_i": int(transition_batch.step_idx[left]),
            "timestep_j": int(transition_batch.step_idx[right]),
            "action_block_id": int(transition_batch.action_block_ids[left])
            if transition_batch.action_block_ids[left] == transition_batch.action_block_ids[right]
            else -1,
            "action_block_i_id": int(transition_batch.action_block_ids[left]),
            "action_block_j_id": int(transition_batch.action_block_ids[right]),
            "action_distance": float(action_dist[pair_idx]),
            "action_norm": float(np.linalg.norm(transition_batch.action_blocks[left])),
            "d_now": float(arrays["d_now"][pair_idx]),
            "d_next_true": float(arrays["d_next_true"][pair_idx]),
            "d_next_pred": float(arrays["d_next_pred"][pair_idx]),
            "epsilon": primary_epsilon,
            "r_true": float(arrays["r_true"][pair_idx]),
            "r_model": float(arrays["r_model"][pair_idx]),
            "ratio": float(arrays["r_true"][pair_idx]),
            "deficit": float(arrays["deficit"][pair_idx]),
            "pair_error": float(arrays["pair_error"][pair_idx]),
            "lower_bound": float(arrays["lower_bound"][pair_idx]),
            "e_i": float(arrays["e_i"][pair_idx]),
            "e_j": float(arrays["e_j"][pair_idx]),
            "latent_dimension": int(latent_dim),
            "checkpoint": str(checkpoint),
            "model": name,
            "seed": int(seed),
        }
        for epsilon in epsilons:
            row[f"r_true_eps_{epsilon:g}"] = float(
                arrays["d_next_true"][pair_idx] / max(arrays["d_now"][pair_idx], float(epsilon))
            )
            row[f"r_model_eps_{epsilon:g}"] = float(
                arrays["d_next_pred"][pair_idx] / max(arrays["d_now"][pair_idx], float(epsilon))
            )
        rows.append(row)
    return rows


def _summary_rows_for_epsilons(
    name: str,
    checkpoint: Path,
    latent_dim: int,
    num_transitions: int,
    action_matching_mode: str,
    action_matching_threshold: float,
    arrays: Dict[str, np.ndarray],
    epsilons: Sequence[float],
    quantiles: Sequence[float],
    episode_i: np.ndarray,
    episode_j: np.ndarray,
    bootstrap_samples: int,
    seed: int,
    triangle_tol: float,
) -> List[Dict[str, object]]:
    groups = _episode_pair_groups(episode_i, episode_j)
    rows: List[Dict[str, object]] = []
    d_now = arrays["d_now"]
    for epsilon in epsilons:
        r_true = _compute_ratios(d_now, arrays["d_next_true"], epsilon)
        r_model = _compute_ratios(d_now, arrays["d_next_pred"], epsilon)
        triangle_violations = arrays["pair_error"] + triangle_tol < arrays["lower_bound"]
        row: Dict[str, object] = {
            "model": name,
            "checkpoint": str(checkpoint),
            "latent_dimension": int(latent_dim),
            "num_transitions": int(num_transitions),
            "num_pairs": int(r_true.size),
            "epsilon": float(epsilon),
            "action_matching_mode": action_matching_mode,
            "action_matching_threshold": float(action_matching_threshold),
            "median_r_true": _quantile(r_true, 0.5),
            "median_r_model": _quantile(r_model, 0.5),
            "median_ratio": _quantile(r_true, 0.5),
            "mean_ratio": float(np.nanmean(r_true)) if r_true.size else float("nan"),
            "fraction_small_d_now": float(np.mean(d_now < epsilon)) if d_now.size else float("nan"),
            "median_deficit": _quantile(arrays["deficit"], 0.5),
            "q95_deficit": _quantile(arrays["deficit"], 0.95),
            "q99_deficit": _quantile(arrays["deficit"], 0.99),
            "mean_pair_error": float(np.nanmean(arrays["pair_error"])) if arrays["pair_error"].size else float("nan"),
            "median_pair_error": _quantile(arrays["pair_error"], 0.5),
            "violation_rate": float(np.mean(arrays["deficit"] > 1e-9)) if arrays["deficit"].size else float("nan"),
            "triangle_bound_violations": float(np.mean(triangle_violations)) if triangle_violations.size else float("nan"),
            "bootstrap_group": "unordered_episode_pair",
            "num_bootstrap_groups": int(len(groups)),
        }
        for quantile in quantiles:
            label = _quantile_label(quantile)
            true_value = _quantile(r_true, quantile)
            model_value = _quantile(r_model, quantile)
            row[f"{label}_r_true"] = true_value
            row[f"{label}_r_model"] = model_value
            row[f"ratio_{label}"] = true_value
            row[f"ratio_{label}_true"] = true_value
            row[f"ratio_{label}_model"] = model_value
            if label in {"q95", "q99", "q999"}:
                row[f"{label}_deficit"] = _quantile(arrays["deficit"], quantile)
            low, high = _bootstrap_quantile_ci(r_true, groups, quantile, bootstrap_samples, seed + int(round(quantile * 10000)))
            row[f"ratio_{label}_ci95_low"] = low
            row[f"ratio_{label}_ci95_high"] = high
        rows.append(row)
    return rows


def _write_model_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    payload = {
        "summary_rows": rows,
        "primary_epsilon": rows[0]["epsilon"] if rows else None,
    }
    if rows:
        payload.update(rows[0])
    _write_json(path, payload)


def _plot_model_scatter(path: Path, d_now: np.ndarray, d_next: np.ndarray, name: str, latent_dim: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        path.with_suffix(".plot_unavailable.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    count = min(5000, d_now.size)
    idx = rng.choice(d_now.size, size=count, replace=False) if d_now.size > count else np.arange(d_now.size)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.scatter(d_now[idx], d_next[idx], s=7, alpha=0.35, linewidths=0)
    limit = float(np.nanmax([np.nanmax(d_now[idx]), np.nanmax(d_next[idx])])) if idx.size else 1.0
    ax.plot([0, limit], [0, limit], color="black", lw=1, alpha=0.5)
    ax.set_xlabel(r"$d_{\mathrm{now}}$")
    ax.set_ylabel(r"$d_{\mathrm{next}}$")
    ax.set_title(f"{name} ({latent_dim}D)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_gain_by_dim(path: Path, summary_rows: List[Dict[str, object]], primary_epsilon: float, quantiles: Sequence[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        path.with_suffix(".plot_unavailable.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return
    rows = [row for row in summary_rows if math.isclose(float(row["epsilon"]), float(primary_epsilon), rel_tol=0.0, abs_tol=1e-15)]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: int(row["latent_dimension"]))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for quantile in quantiles:
        key = f"ratio_{_quantile_label(quantile)}"
        low_key = f"{key}_ci95_low"
        high_key = f"{key}_ci95_high"
        x = np.asarray([int(row["latent_dimension"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        low = np.asarray([float(row.get(low_key, "nan")) for row in rows], dtype=np.float64)
        high = np.asarray([float(row.get(high_key, "nan")) for row in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", label=f"q={quantile:g}")
        if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
            ax.fill_between(x, low, high, alpha=0.15)
    max_quantile_key = f"ratio_{_quantile_label(max(quantiles))}"
    if np.nanmax([float(row[max_quantile_key]) for row in rows]) > 10:
        ax.set_yscale("log")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel(r"$\widehat L_q$")
    ax.set_title(f"Empirical transition gain, epsilon={primary_epsilon:g}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_deficit_error(path: Path, pair_rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        path.with_suffix(".plot_unavailable.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return
    if not pair_rows:
        return
    rng = np.random.default_rng(0)
    rows = pair_rows
    if len(rows) > 30000:
        keep = rng.choice(len(rows), size=30000, replace=False)
        rows = [rows[int(idx)] for idx in keep]
    dims = sorted({int(row["latent_dimension"]) for row in rows})
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for dim in dims:
        subset = [row for row in rows if int(row["latent_dimension"]) == dim]
        x = np.asarray([float(row["lower_bound"]) for row in subset], dtype=np.float64)
        y = np.asarray([float(row["pair_error"]) for row in subset], dtype=np.float64)
        ax.scatter(x, y, s=8, alpha=0.25, linewidths=0, label=f"{dim}D")
    limit = float(np.nanmax([float(row["pair_error"]) for row in rows] + [float(row["lower_bound"]) for row in rows]))
    ax.plot([0, limit], [0, limit], color="black", lw=1, alpha=0.75)
    ax.set_xlabel("deficit / 2")
    ax.set_ylabel("pair_error")
    ax.set_title("Expansion deficit lower-bounds prediction error")
    ax.grid(alpha=0.25)
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_deficit_summary(path: Path, summary_rows: List[Dict[str, object]], primary_epsilon: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        path.with_suffix(".plot_unavailable.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return
    rows = [row for row in summary_rows if math.isclose(float(row["epsilon"]), float(primary_epsilon), rel_tol=0.0, abs_tol=1e-15)]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: int(row["latent_dimension"]))
    x = np.asarray([int(row["latent_dimension"]) for row in rows], dtype=np.float64)
    q99_deficit = np.asarray([float(row["q99_deficit"]) for row in rows], dtype=np.float64)
    mean_error = np.asarray([float(row["mean_pair_error"]) for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    axes[0].plot(x, q99_deficit, marker="o")
    axes[0].set_xlabel("latent dimension")
    axes[0].set_ylabel("q99 deficit")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, mean_error, marker="o", color="tab:orange")
    axes[1].set_xlabel("latent dimension")
    axes[1].set_ylabel("mean pair_error")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Deficit and prediction error by latent dimension", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _discover_checkpoints(search_roots: Sequence[Path]) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    patterns = ["*state8*object.ckpt", "*state16*object.ckpt", "*state32*object.ckpt", "*state64*object.ckpt", "*baseline*object.ckpt"]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                name = path.stem.replace("_baseline_object", "").replace("_object", "")
                if "baseline" in path.name and "baseline192" not in discovered:
                    discovered["baseline192"] = path
                elif "state8" in path.name and "state8" not in discovered:
                    discovered["state8"] = path
                elif "state16" in path.name and "state16" not in discovered:
                    discovered["state16"] = path
                elif "state32" in path.name and "state32" not in discovered:
                    discovered["state32"] = path
                elif "state64" in path.name and "state64" not in discovered:
                    discovered["state64"] = path
                else:
                    discovered.setdefault(_safe_name(name), path)
    return discovered


def main() -> None:
    try:
        import h5py
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "eval_transition_gain.py requires the LeWorldModel runtime dependencies "
            "(at least h5py and torch). Run it inside the project Apptainer/venv."
        ) from exc

    _encode_pixels, _find_key, _load_model, _read_h5_rows = _load_repo_helpers()

    parser = argparse.ArgumentParser(description="Evaluate empirical same-action transition gain for LeWorldModel latents.")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="NAME=checkpoint_object.ckpt entries. Missing paths are skipped.")
    parser.add_argument("--discover_checkpoints", action="store_true", help="Search common local checkpoint roots if --checkpoints is omitted.")
    parser.add_argument("--checkpoint_search_roots", nargs="*", default=[".", "/home/jw3425/.stable_worldmodel/pusht"])
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--output_dir", default="outputs/lewm_gain")
    parser.add_argument("--num_transitions", type=int, default=2000)
    parser.add_argument("--num_pairs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", "--epsilons", dest="epsilons", default="1e-6,1e-4,1e-2")
    parser.add_argument("--action_matching_mode", choices=["exact", "approximate"], default="exact")
    parser.add_argument("--action_matching_threshold", type=float, default=0.5)
    parser.add_argument("--quantiles", default="0.95,0.99,0.999")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--history_size", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument("--triangle_tol", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--step_key", default="step_idx")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if HDF5PLUGIN_AVAILABLE:
        print("[transition_gain] hdf5plugin available; compressed HDF5 filters enabled.", flush=True)

    if args.smoke_test:
        args.num_transitions = min(args.num_transitions, 128)
        args.num_pairs = min(args.num_pairs, 512)
        args.bootstrap_samples = min(args.bootstrap_samples, 20)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epsilons = _parse_float_list(args.epsilons, DEFAULT_EPSILONS)
    quantiles = _parse_float_list(args.quantiles, DEFAULT_QUANTILES)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoints = _parse_name_paths(args.checkpoints)
    if args.discover_checkpoints and not checkpoints:
        checkpoints = _discover_checkpoints([Path(item) for item in args.checkpoint_search_roots])
    checkpoints = {name: path for name, path in checkpoints.items() if path.exists()}
    if not checkpoints:
        raise FileNotFoundError("No existing checkpoints found. Pass --checkpoints NAME=PATH or use --discover_checkpoints.")

    history_size = args.history_size
    if history_size is None:
        first_name, first_checkpoint = next(iter(checkpoints.items()))
        print(f"[transition_gain] inferring history_size from {first_name}: {first_checkpoint}", flush=True)
        first_model = _load_model(first_checkpoint, device)
        history_size = int(getattr(getattr(first_model, "predictor", None), "pos_embedding").shape[1])
        del first_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[transition_gain] using predictor history_size={history_size}", flush=True)

    with h5py.File(args.dataset, "r") as h5:
        print("[transition_gain] dataset keys and shapes:", flush=True)
        for key in h5.keys():
            print(f"  - {key}: {h5[key].shape}", flush=True)
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        action_key = _find_key(h5, [args.action_key, "action"])
        episode_key = _find_key(h5, [args.episode_key, "episode_idx", "ep_idx"])
        step_key = _find_key(h5, [args.step_key, "step_idx"])
        transitions = _load_transition_batch(
            h5,
            args.num_transitions,
            args.frameskip,
            int(history_size),
            args.seed,
            action_key,
            episode_key,
            step_key,
        )
        if args.action_matching_mode == "exact":
            pairs_i, pairs_j, action_dist = _sample_exact_action_pairs(
                transitions.action_block_ids,
                args.num_pairs,
                args.seed,
            )
        else:
            pairs_i, pairs_j, action_dist = _sample_action_matched_pairs(
                transitions.action_blocks_norm,
                args.num_pairs,
                args.action_matching_threshold,
                args.seed,
            )
        print(
            f"[transition_gain] transitions={len(transitions.rows)}, pairs={len(pairs_i)}, "
            f"frameskip={args.frameskip}, action_matching_mode={args.action_matching_mode}, "
            f"action_threshold={args.action_matching_threshold}",
            flush=True,
        )

        flat_context_pixels = _read_h5_rows(h5[pixels_key], transitions.context_rows)
        context_pixels = flat_context_pixels.reshape(*transitions.context_rows.shape, *flat_context_pixels.shape[1:])
        next_pixels = _read_h5_rows(h5[pixels_key], transitions.next_rows)

        transition_id_rows = [
            {
                "transition_index": int(idx),
                "row": int(row),
                "next_row": int(next_row),
                "episode_idx": int(ep),
                "step_idx": int(step),
            }
            for idx, (row, next_row, ep, step) in enumerate(
                zip(transitions.rows, transitions.next_rows, transitions.episode_idx, transitions.step_idx)
            )
        ]
        _write_csv(output_dir / "sampled_transition_ids.csv", transition_id_rows)
        _write_csv(
            output_dir / "sampled_pair_ids.csv",
            [
                {
                    "pair_idx": int(idx),
                    "transition_i_index": int(left),
                    "transition_j_index": int(right),
                    "transition_i": int(transitions.rows[left]),
                    "transition_j": int(transitions.rows[right]),
                    "action_distance": float(dist),
                    "action_matching_mode": args.action_matching_mode,
                    "action_block_i_id": int(transitions.action_block_ids[left]),
                    "action_block_j_id": int(transitions.action_block_ids[right]),
                }
                for idx, (left, right, dist) in enumerate(zip(pairs_i, pairs_j, action_dist))
            ],
        )

        all_summary_rows: List[Dict[str, object]] = []
        all_pair_rows: List[Dict[str, object]] = []
        for name, checkpoint in checkpoints.items():
            print(f"[transition_gain] loading {name}: {checkpoint}", flush=True)
            model = _load_model(checkpoint, device)
            if model.training:
                raise RuntimeError(f"Model {name} is not in eval mode.")
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError(f"Model {name} has trainable parameters enabled during evaluation.")
            with torch.no_grad():
                context_emb = _encode_pixels(model, context_pixels, args.img_size, args.batch_size, device).astype(np.float32)
                z_next = _encode_pixels(model, next_pixels, args.img_size, args.batch_size, device)[:, 0].astype(np.float32)
                z_pred = _predict_next_latents(
                    model,
                    context_emb,
                    transitions.context_action_blocks,
                    args.batch_size,
                    device,
                )
            z_now = context_emb[:, -1]
            latent_dim = _latent_dim_from_model(name, z_now)
            arrays = _pair_arrays(pairs_i, pairs_j, z_now, z_next, z_pred, epsilons[0])
            triangle_violations = arrays["pair_error"] + float(args.triangle_tol) < arrays["lower_bound"]
            if np.any(triangle_violations):
                raise RuntimeError(
                    f"Triangle inequality lower-bound violated for {name}: "
                    f"{int(np.sum(triangle_violations))}/{triangle_violations.size} pairs beyond tol={args.triangle_tol}."
                )
            ckpt_dir = output_dir / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            pair_rows = _pair_metric_rows(
                name,
                checkpoint,
                latent_dim,
                transitions,
                pairs_i,
                pairs_j,
                action_dist,
                arrays,
                epsilons,
                args.seed,
            )
            pair_path = _write_parquet_or_csv(ckpt_dir / "pair_metrics.parquet", pair_rows)
            summary_rows = _summary_rows_for_epsilons(
                name,
                checkpoint,
                latent_dim,
                len(transitions.rows),
                args.action_matching_mode,
                args.action_matching_threshold,
                arrays,
                epsilons,
                quantiles,
                transitions.episode_idx[pairs_i],
                transitions.episode_idx[pairs_j],
                args.bootstrap_samples,
                args.seed,
                args.triangle_tol,
            )
            for row in summary_rows:
                row["pair_metrics_path"] = pair_path
            _write_model_summary(ckpt_dir / "summary.json", summary_rows)
            _write_csv(ckpt_dir / "summary.csv", summary_rows)
            _plot_model_scatter(ckpt_dir / "d_now_vs_d_next_true", arrays["d_now"], arrays["d_next_true"], name, latent_dim)
            all_summary_rows.extend(summary_rows)
            all_pair_rows.extend(pair_rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    global_pair_path = _write_parquet_or_csv(output_dir / "pair_metrics.parquet", all_pair_rows)
    _write_csv(output_dir / "summary.csv", all_summary_rows)
    _write_csv(output_dir / "transition_gain_summary.csv", all_summary_rows)
    _write_json(
        output_dir / "run_metadata.json",
        {
            "dataset": str(args.dataset),
            "checkpoints": {name: str(path) for name, path in checkpoints.items()},
            "num_transitions_requested": int(args.num_transitions),
            "num_pairs_requested": int(args.num_pairs),
            "frameskip": int(args.frameskip),
            "history_size": int(history_size),
            "pair_metrics_path": global_pair_path,
            "action_matching_mode": args.action_matching_mode,
            "action_matching_rule": "exact_action_block" if args.action_matching_mode == "exact" else "normalized_action_block_l2_threshold",
            "action_matching_threshold": float(args.action_matching_threshold),
            "epsilons": epsilons,
            "quantiles": quantiles,
            "seed": int(args.seed),
        },
    )
    _plot_gain_by_dim(output_dir / "lewm_required_gain", all_summary_rows, epsilons[0], quantiles)
    _plot_gain_by_dim(output_dir / "transition_gain_vs_latent_dim", all_summary_rows, epsilons[0], quantiles)
    _plot_deficit_error(output_dir / "lewm_deficit_error", all_pair_rows)
    _plot_deficit_summary(output_dir / "lewm_deficit_summary", all_summary_rows, epsilons[0])
    print(f"[transition_gain] wrote outputs under {output_dir}", flush=True)
    print(f"[transition_gain] summary: {output_dir / 'summary.csv'}", flush=True)
    print(f"[transition_gain] pair metrics: {global_pair_path}", flush=True)
    print(f"[transition_gain] figures: {output_dir / 'lewm_required_gain.pdf'}, {output_dir / 'lewm_deficit_error.pdf'}, {output_dir / 'lewm_deficit_summary.pdf'}", flush=True)


if __name__ == "__main__":
    main()
