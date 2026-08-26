from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import hdf5plugin  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EPS = 1e-8


def _parse_name_paths(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty model name in {item!r}")
        out[name] = Path(path)
    return out


def _parse_name_floats(items: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        out[name.strip()] = float(value)
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _find_key(h5: h5py.File, candidates: Iterable[str]) -> str:
    for key in candidates:
        if key and key in h5:
            return key
    raise KeyError(f"None of these keys were found in dataset: {list(candidates)}")


def _read_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    values = np.asarray(dataset[unique_rows])
    return values[inverse]


def _to_chw_float(images: torch.Tensor) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError(f"Expected image batch (N,H,W,C) or (N,C,H,W), got {tuple(images.shape)}")
    if images.shape[1] not in (1, 3, 4) and images.shape[-1] in (1, 3, 4):
        images = images.permute(0, 3, 1, 2)
    if images.shape[1] == 4:
        images = images[:, :3]
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)
    images = images.float()
    if images.max() > 2.0:
        images = images / 255.0
    return images


def _preprocess_pixels(pixels: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(pixels).to(device)
    batch, steps = tensor.shape[:2]
    flat = tensor.reshape(batch * steps, *tensor.shape[2:])
    flat = _to_chw_float(flat)
    if flat.shape[-2:] != (img_size, img_size):
        flat = F.interpolate(flat, size=(img_size, img_size), mode="bilinear", align_corners=False)
    flat = (flat - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return flat.reshape(batch, steps, *flat.shape[1:])


def _load_model(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    model = obj.model if hasattr(obj, "model") else obj
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def _sample_reference_windows(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    num_trajectories: int,
    trajectory_len: int,
    trajectory_stride: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw_len = (trajectory_len - 1) * trajectory_stride + 1
    candidates: List[np.ndarray] = []
    for episode in np.unique(episode_idx):
        rows = np.where(episode_idx == episode)[0]
        rows = rows[np.argsort(step_idx[rows])]
        if rows.size < raw_len:
            continue
        for start in range(0, rows.size - raw_len + 1):
            raw_window = rows[start:start + raw_len]
            if np.all(np.diff(step_idx[raw_window]) == 1):
                candidates.append(raw_window[::trajectory_stride])
    if not candidates:
        raise ValueError(
            f"No reference windows found for trajectory_len={trajectory_len}, stride={trajectory_stride}. "
            "Try smaller --trajectory_len or --trajectory_stride."
        )
    if len(candidates) > num_trajectories:
        keep = rng.choice(len(candidates), size=num_trajectories, replace=False)
        candidates = [candidates[int(idx)] for idx in keep]
    return np.stack(candidates, axis=0).astype(np.int64)


def _parse_int_list(text: str) -> List[int]:
    return [int(item) for item in str(text).replace(",", " ").split() if item.strip()]


def _sample_reference_segments(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    num_trajectories: int,
    action_block: int,
    reference_raw_span: int,
    min_reference_raw_span: int,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    reservoir: List[np.ndarray] = []
    seen = 0
    for episode in np.unique(episode_idx):
        episode_rows = np.where(episode_idx == episode)[0]
        episode_rows = episode_rows[np.argsort(step_idx[episode_rows])]
        if episode_rows.size < 2:
            continue

        breaks = np.where(np.diff(step_idx[episode_rows]) != 1)[0] + 1
        runs = np.split(episode_rows, breaks)
        for run in runs:
            if run.size < min_reference_raw_span + 1:
                continue
            max_start = run.size - min_reference_raw_span - 1
            for start in range(max_start + 1):
                available = run.size - start
                segment_raw_len = min(reference_raw_span + 1, available)
                if segment_raw_len < min_reference_raw_span + 1:
                    continue
                raw_segment = run[start:start + segment_raw_len]
                sampled_segment = raw_segment[::action_block].astype(np.int64)
                seen += 1
                if len(reservoir) < num_trajectories:
                    reservoir.append(sampled_segment)
                else:
                    replace = int(rng.integers(seen))
                    if replace < num_trajectories:
                        reservoir[replace] = sampled_segment
    if not reservoir:
        raise ValueError(
            "No horizon-aware reference segments found. Try smaller --min_reference_raw_span "
            "or check episode continuity."
        )
    return reservoir


def _self_approaching_stats(z: np.ndarray) -> Dict[str, float]:
    if z.ndim != 3:
        raise ValueError(f"Expected latent trajectory shape (B,T,D), got {z.shape}")
    batch, steps, _dim = z.shape
    violation_rates = []
    terminal_violation_rates = []
    margin_values = []
    terminal_margin_values = []
    for item in range(batch):
        traj = z[item].astype(np.float64)
        violations = []
        terminal_violations = []
        for t in range(steps - 1):
            future = traj[t + 1:]
            before = np.sum(np.square(traj[t][None, :] - future), axis=-1)
            after = np.sum(np.square(traj[t + 1][None, :] - future), axis=-1)
            margin = after - before
            violations.append(margin > 0.0)
            margin_values.append(float(np.mean(margin)))

            before_terminal = float(np.sum(np.square(traj[t] - traj[-1])))
            after_terminal = float(np.sum(np.square(traj[t + 1] - traj[-1])))
            terminal_margin = after_terminal - before_terminal
            terminal_violations.append(terminal_margin > 0.0)
            terminal_margin_values.append(terminal_margin)
        violation_rates.append(float(np.mean(np.concatenate(violations))) if violations else float("nan"))
        terminal_violation_rates.append(float(np.mean(terminal_violations)) if terminal_violations else float("nan"))
    return {
        "rho_sa_mean": float(np.nanmean(violation_rates)),
        "rho_sa_std": float(np.nanstd(violation_rates)),
        "rho_sa_terminal_mean": float(np.nanmean(terminal_violation_rates)),
        "rho_sa_terminal_std": float(np.nanstd(terminal_violation_rates)),
        "sa_margin_mean": float(np.nanmean(margin_values)),
        "sa_terminal_margin_mean": float(np.nanmean(terminal_margin_values)),
        "num_trajectories": int(batch),
        "trajectory_len": int(steps),
    }


def _distance_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.square(a - b), axis=-1)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + EPS
    return numerator / denominator


def _horizon_self_approaching_rows(
    segments: List[np.ndarray],
    model_name: str,
    latent_dim: int,
    horizons_raw: List[int],
    action_block: int,
    tau_values: List[float],
    reference_mode: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if reference_mode not in {"terminal", "all_future"}:
        raise ValueError(f"Unknown sa_reference_mode={reference_mode!r}. Use 'terminal' or 'all_future'.")
    for horizon_raw in horizons_raw:
        if horizon_raw % action_block != 0:
            raise ValueError(f"h_raw={horizon_raw} must be divisible by action_block={action_block}.")
        h_blocks = horizon_raw // action_block
        deltas_norm = []
        positives_norm = []
        positive_only_norm = []
        tau_counts = {tau: 0 for tau in tau_values}
        total = 0
        for z in segments:
            traj = z.astype(np.float64)
            steps = traj.shape[0]
            if steps <= h_blocks + 1:
                continue
            for t_idx in range(0, steps - h_blocks - 1):
                if reference_mode == "terminal":
                    target = traj[-1]
                    before = np.asarray([float(_distance_sq(traj[t_idx], target))])
                    after = np.asarray([float(_distance_sq(traj[t_idx + h_blocks], target))])
                else:
                    r_start = t_idx + h_blocks + 1
                    future = traj[r_start:]
                    before = _distance_sq(traj[t_idx][None, :], future)
                    after = _distance_sq(traj[t_idx + h_blocks][None, :], future)
                delta = after - before
                denom = before + EPS
                normalized = delta / denom
                deltas_norm.extend(normalized.tolist())
                positives_norm.extend(np.maximum(normalized, 0.0).tolist())
                positive_only_norm.extend(normalized[delta > 0].tolist())
                for tau in tau_values:
                    tau_counts[tau] += int(np.sum(delta > tau * denom))
                total += int(delta.size)
        rho_sa = float(tau_counts.get(0.0, 0) / total) if total else float("nan")
        row = {
            "model_name": model_name,
            "dim": int(latent_dim),
            "h_raw": int(horizon_raw),
            "h_blocks": float(horizon_raw / action_block),
            "reference_mode": reference_mode,
            "num_valid_pairs": int(total),
            "rho_sa": rho_sa,
            "p_sa": float(1.0 - rho_sa) if total else float("nan"),
            "vplus_sa_norm": float(np.mean(positives_norm)) if positives_norm else float("nan"),
            "vplus_sa_norm_cond": float(np.mean(positive_only_norm)) if positive_only_norm else 0.0,
            "signed_progress_norm": float(np.mean(deltas_norm)) if deltas_norm else float("nan"),
        }
        for tau in tau_values:
            rho_tau = float(tau_counts[tau] / total) if total else float("nan")
            row[_tau_column(tau)] = rho_tau
            row[_p_tau_column(tau)] = float(1.0 - rho_tau) if total else float("nan")
        rows.append(row)
    return rows


def _progress_angle_rows(
    segments: List[np.ndarray],
    model_name: str,
    latent_dim: int,
    horizons_raw: List[int],
    action_block: int,
) -> Tuple[List[Dict[str, object]], Dict[int, np.ndarray]]:
    rows: List[Dict[str, object]] = []
    cos_by_horizon: Dict[int, np.ndarray] = {}
    for horizon_raw in horizons_raw:
        if horizon_raw % action_block != 0:
            raise ValueError(f"h_raw={horizon_raw} must be divisible by action_block={action_block}.")
        h_blocks = horizon_raw // action_block
        cos_values = []
        for z in segments:
            traj = z.astype(np.float64)
            steps = traj.shape[0]
            if steps <= h_blocks + 1:
                continue
            target = traj[-1]
            for t_idx in range(0, steps - h_blocks - 1):
                d = traj[t_idx + h_blocks] - traj[t_idx]
                q = target - traj[t_idx]
                cos_values.append(float(_cosine(d, q)))
        cos_array = np.asarray(cos_values, dtype=np.float64)
        cos_by_horizon[horizon_raw] = cos_array
        rows.append(
            {
                "model_name": model_name,
                "dim": int(latent_dim),
                "h_raw": int(horizon_raw),
                "h_blocks": float(horizon_raw / action_block),
                "num_valid_pairs": int(cos_array.size),
                "mean_cos_progress": float(np.mean(cos_array)) if cos_array.size else float("nan"),
                "progress_obtuse_rate": float(np.mean(cos_array < 0.0)) if cos_array.size else float("nan"),
                "mean_neg_cos_progress": float(np.mean(np.maximum(-cos_array, 0.0))) if cos_array.size else float("nan"),
                "cos_progress_q10": float(np.quantile(cos_array, 0.10)) if cos_array.size else float("nan"),
                "cos_progress_q25": float(np.quantile(cos_array, 0.25)) if cos_array.size else float("nan"),
                "cos_progress_q50": float(np.quantile(cos_array, 0.50)) if cos_array.size else float("nan"),
            }
        )
    return rows, cos_by_horizon


def _corner_angle_rows(
    segments: List[np.ndarray],
    model_name: str,
    latent_dim: int,
    horizons_raw: List[int],
    action_block: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for horizon_raw in horizons_raw:
        if horizon_raw % action_block != 0:
            raise ValueError(f"h_raw={horizon_raw} must be divisible by action_block={action_block}.")
        h_blocks = horizon_raw // action_block
        cos_values = []
        for z in segments:
            traj = z.astype(np.float64)
            steps = traj.shape[0]
            if steps <= 2 * h_blocks:
                continue
            for t_idx in range(h_blocks, steps - h_blocks):
                a = traj[t_idx - h_blocks] - traj[t_idx]
                b = traj[t_idx + h_blocks] - traj[t_idx]
                cos_values.append(float(_cosine(a, b)))
        cos_array = np.asarray(cos_values, dtype=np.float64)
        rows.append(
            {
                "model_name": model_name,
                "dim": int(latent_dim),
                "h_raw": int(horizon_raw),
                "h_blocks": float(horizon_raw / action_block),
                "num_valid_corners": int(cos_array.size),
                "mean_corner_cos": float(np.mean(cos_array)) if cos_array.size else float("nan"),
                "acute_corner_rate": float(np.mean(cos_array > 0.0)) if cos_array.size else float("nan"),
                "mean_acute_corner_severity": float(np.mean(np.maximum(cos_array, 0.0))) if cos_array.size else float("nan"),
            }
        )
    return rows


def _fixed_target_self_approaching_rows(
    segments: List[np.ndarray],
    model_name: str,
    latent_dim: int,
    horizons_raw: List[int],
    target_raws: List[int],
    action_block: int,
    tau_values: List[float],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for target_raw in target_raws:
        if target_raw % action_block != 0:
            raise ValueError(f"target_raw={target_raw} must be divisible by action_block={action_block}.")
        target_blocks = target_raw // action_block
        for horizon_raw in horizons_raw:
            if horizon_raw >= target_raw:
                continue
            if horizon_raw % action_block != 0:
                raise ValueError(f"h_raw={horizon_raw} must be divisible by action_block={action_block}.")
            h_blocks = horizon_raw // action_block
            deltas_norm = []
            positives_norm = []
            tau_counts = {tau: 0 for tau in tau_values}
            total = 0
            for z in segments:
                traj = z.astype(np.float64)
                steps = traj.shape[0]
                if steps <= target_blocks:
                    continue
                for t_idx in range(0, steps - target_blocks):
                    target = traj[t_idx + target_blocks]
                    before = float(_distance_sq(traj[t_idx], target))
                    after = float(_distance_sq(traj[t_idx + h_blocks], target))
                    delta = after - before
                    denom = before + EPS
                    normalized = delta / denom
                    deltas_norm.append(normalized)
                    positives_norm.append(max(normalized, 0.0))
                    for tau in tau_values:
                        tau_counts[tau] += int(delta > tau * denom)
                    total += 1
            row = {
                "model_name": model_name,
                "dim": int(latent_dim),
                "target_raw": int(target_raw),
                "target_blocks": float(target_raw / action_block),
                "h_raw": int(horizon_raw),
                "h_blocks": float(horizon_raw / action_block),
                "num_valid_pairs": int(total),
                "rho_sa": float(tau_counts.get(0.0, 0) / total) if total else float("nan"),
                "vplus_sa_norm": float(np.mean(positives_norm)) if positives_norm else float("nan"),
                "signed_progress_norm": float(np.mean(deltas_norm)) if deltas_norm else float("nan"),
            }
            for tau in tau_values:
                row[_tau_column(tau)] = float(tau_counts[tau] / total) if total else float("nan")
            rows.append(row)
    return rows


def _tau_column(tau: float) -> str:
    if tau == 0:
        return "rho_sa_tau0"
    if np.isclose(tau, 1e-4):
        return "rho_sa_tau1e4"
    if np.isclose(tau, 1e-3):
        return "rho_sa_tau1e3"
    if np.isclose(tau, 1e-2):
        return "rho_sa_tau1e2"
    return f"rho_sa_tau{tau:g}".replace("-", "m").replace(".", "p")


def _p_tau_column(tau: float) -> str:
    return _tau_column(tau).replace("rho_sa", "p_sa", 1)


def _encode_reference_trajectories(
    model,
    pixels_ds,
    windows: np.ndarray,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            end = min(start + batch_size, windows.shape[0])
            pixels = _read_h5_rows(pixels_ds, windows[start:end].reshape(-1))
            pixels = pixels.reshape(end - start, windows.shape[1], *pixels.shape[1:])
            pixels_tensor = _preprocess_pixels(pixels, img_size, device)
            emb = model.encode({"pixels": pixels_tensor})["emb"]
            chunks.append(emb.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _encode_reference_segments(
    model,
    pixels_ds,
    segments: List[np.ndarray],
    img_size: int,
    device: torch.device,
) -> List[np.ndarray]:
    encoded = []
    with torch.no_grad():
        for rows in segments:
            pixels = _read_h5_rows(pixels_ds, rows)
            pixels = pixels.reshape(1, rows.shape[0], *pixels.shape[1:])
            pixels_tensor = _preprocess_pixels(pixels, img_size, device)
            emb = model.encode({"pixels": pixels_tensor})["emb"]
            encoded.append(emb[0].detach().float().cpu().numpy())
    return encoded


def _encode_single_rows(
    model,
    pixels_ds,
    rows: np.ndarray,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    chunks = []
    with torch.no_grad():
        for start in range(0, rows.shape[0], batch_size):
            batch_rows = rows[start:start + batch_size]
            pixels = _read_h5_rows(pixels_ds, batch_rows)
            pixels = pixels.reshape(batch_rows.shape[0], 1, *pixels.shape[1:])
            pixels_tensor = _preprocess_pixels(pixels, img_size, device)
            emb = model.encode({"pixels": pixels_tensor})["emb"][:, 0]
            chunks.append(emb.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan")
    value = spearmanr(x, y).correlation
    return float(value) if value is not None else float("nan")


def _rank_1_based(values: np.ndarray, index: int, lower_is_better: bool = True) -> int:
    order = np.argsort(values if lower_is_better else -values)
    return int(np.where(order == index)[0][0] + 1)


def _reshape_candidate_array(array: np.ndarray, key: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        raise ValueError(f"Cannot infer candidate windows from 1D array {key} with shape {array.shape}")
    return array


def _load_true_scores(raw: np.lib.npyio.NpzFile, true_metric: str) -> Tuple[np.ndarray, bool]:
    if true_metric == "auto":
        if "progress" in raw:
            return _reshape_candidate_array(raw["progress"], "progress"), False
        if "terminal_cost" in raw:
            return _reshape_candidate_array(raw["terminal_cost"], "terminal_cost"), True
        raise KeyError("Raw pool must contain progress or terminal_cost.")
    if true_metric == "progress":
        return _reshape_candidate_array(raw["progress"], "progress"), False
    if true_metric == "terminal_cost":
        return _reshape_candidate_array(raw["terminal_cost"], "terminal_cost"), True
    raise ValueError(f"Unknown true_metric: {true_metric}")


def _candidate_rank_metrics(raw_pool: Path, model_name: str, true_metric: str, topk: int) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    raw = np.load(raw_pool)
    if "latent_scores" not in raw:
        raise KeyError(f"{raw_pool} does not contain latent_scores.")
    latent_scores = _reshape_candidate_array(raw["latent_scores"], "latent_scores").astype(np.float64)
    true_scores, true_lower_is_better = _load_true_scores(raw, true_metric)
    true_scores = true_scores.astype(np.float64)
    if latent_scores.shape != true_scores.shape:
        raise ValueError(f"latent_scores shape {latent_scores.shape} != true score shape {true_scores.shape}")

    per_window = []
    rank_errors = []
    oracle_ranks = []
    regrets = []
    topk_hits = []
    spearmans = []
    for window_idx in range(latent_scores.shape[0]):
        latent = latent_scores[window_idx]
        true = true_scores[window_idx]
        latent_best = int(np.argmin(latent))
        task_best = int(np.argmin(true) if true_lower_is_better else np.argmax(true))
        rank_error = int(latent_best != task_best)
        oracle_rank = _rank_1_based(latent, task_best, lower_is_better=True)
        if true_lower_is_better:
            regret = float(true[latent_best] - true[task_best])
            spearman_true = true
        else:
            regret = float(true[task_best] - true[latent_best])
            spearman_true = -true
        topk_hit = int(oracle_rank <= min(topk, latent.shape[0]))
        spearman = _spearman(latent, spearman_true)
        per_window.append(
            {
                "model": model_name,
                "window_idx": window_idx,
                "rank_error": rank_error,
                "latent_best_idx": latent_best,
                "task_best_idx": task_best,
                "latent_rank_of_task_best": oracle_rank,
                "topk_hit": topk_hit,
                "candidate_regret": regret,
                "spearman_latent_vs_task_score": spearman,
                "num_candidates": int(latent.shape[0]),
                "raw_pool": str(raw_pool),
            }
        )
        rank_errors.append(rank_error)
        oracle_ranks.append(oracle_rank)
        regrets.append(regret)
        topk_hits.append(topk_hit)
        spearmans.append(spearman)

    summary = {
        "model": model_name,
        "raw_pool": str(raw_pool),
        "num_windows": int(latent_scores.shape[0]),
        "num_candidates": int(latent_scores.shape[1]),
        "rank_error_rate": float(np.mean(rank_errors)),
        "latent_rank_of_task_best_mean": float(np.mean(oracle_ranks)),
        "latent_rank_of_task_best_median": float(np.median(oracle_ranks)),
        "topk_hit_rate": float(np.mean(topk_hits)),
        "candidate_regret_mean": float(np.mean(regrets)),
        "candidate_regret_median": float(np.median(regrets)),
        "spearman_mean": float(np.nanmean(spearmans)),
        "true_metric": true_metric,
    }
    oracle_payload = _oracle_candidate_sa_from_raw(raw, true_scores, true_lower_is_better, model_name)
    return summary, per_window, oracle_payload


def _candidate_angle_ranking_metrics(
    raw_pool: Path,
    model_name: str,
    true_metric: str,
    model_checkpoint: Path | None,
    dataset_path: Path,
    pixels_key_arg: str,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    raw = np.load(raw_pool)
    required = ["terminal_latents", "goal_latents"]
    missing = [key for key in required if key not in raw]
    if missing:
        return [], {
            "model": model_name,
            "raw_pool": str(raw_pool),
            "candidate_angle_available": False,
            "skip_reason": f"raw pool missing keys: {missing}",
        }
    terminal_latents = np.asarray(raw["terminal_latents"], dtype=np.float64)
    goal_latents = np.asarray(raw["goal_latents"], dtype=np.float64)
    if terminal_latents.ndim != 3:
        return [], {
            "model": model_name,
            "raw_pool": str(raw_pool),
            "candidate_angle_available": False,
            "skip_reason": f"terminal_latents has shape {terminal_latents.shape}; expected (windows,candidates,D)",
        }
    if goal_latents.ndim != 2 or goal_latents.shape[0] != terminal_latents.shape[0]:
        return [], {
            "model": model_name,
            "raw_pool": str(raw_pool),
            "candidate_angle_available": False,
            "skip_reason": f"goal_latents has shape {goal_latents.shape}; expected (windows,D)",
        }

    if "start_latents" in raw:
        start_latents = np.asarray(raw["start_latents"], dtype=np.float64)
    elif "context_latents" in raw:
        context_latents = np.asarray(raw["context_latents"], dtype=np.float64)
        start_latents = context_latents[:, -1] if context_latents.ndim == 3 else context_latents
    elif "start_rows" in raw and model_checkpoint is not None and model_checkpoint.exists():
        start_rows = np.asarray(raw["start_rows"], dtype=np.int64).reshape(-1)
        model = _load_model(model_checkpoint, device)
        with h5py.File(dataset_path, "r") as h5:
            pixels_key = _find_key(h5, [pixels_key_arg, "pixels", "observation/pixels"])
            start_latents = _encode_single_rows(model, h5[pixels_key], start_rows, img_size, batch_size, device).astype(np.float64)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        return [], {
            "model": model_name,
            "raw_pool": str(raw_pool),
            "candidate_angle_available": False,
            "skip_reason": "need start_latents/context_latents in raw pool or matching --models checkpoint plus start_rows",
        }

    if start_latents.ndim != 2 or start_latents.shape[0] != terminal_latents.shape[0]:
        return [], {
            "model": model_name,
            "raw_pool": str(raw_pool),
            "candidate_angle_available": False,
            "skip_reason": f"start_latents has shape {start_latents.shape}; expected (windows,D)",
        }

    true_scores, true_lower_is_better = _load_true_scores(raw, true_metric)
    true_scores = true_scores.astype(np.float64)
    if true_scores.shape != terminal_latents.shape[:2]:
        return [], {
            "model": model_name,
            "raw_pool": str(raw_pool),
            "candidate_angle_available": False,
            "skip_reason": f"true score shape {true_scores.shape} != terminal candidate shape {terminal_latents.shape[:2]}",
        }

    latent_costs = np.sum((terminal_latents - goal_latents[:, None, :]) ** 2, axis=-1)
    if "latent_scores" in raw:
        saved_scores = np.asarray(raw["latent_scores"], dtype=np.float64)
        max_score_diff = float(np.max(np.abs(saved_scores - latent_costs))) if saved_scores.shape == latent_costs.shape else float("nan")
    else:
        max_score_diff = float("nan")

    per_window: List[Dict[str, object]] = []
    for window_idx in range(terminal_latents.shape[0]):
        z0 = start_latents[window_idx]
        zg = goal_latents[window_idx]
        candidates = terminal_latents[window_idx]
        task_scores = true_scores[window_idx]
        costs = latent_costs[window_idx]

        latent_best = int(np.argmin(costs))
        task_best = int(np.argmin(task_scores) if true_lower_is_better else np.argmax(task_scores))
        rank_error = int(latent_best != task_best)
        if true_lower_is_better:
            regret = float(task_scores[latent_best] - task_scores[task_best])
            spearman_true = task_scores
        else:
            regret = float(task_scores[task_best] - task_scores[latent_best])
            spearman_true = -task_scores

        q = zg - z0
        d_latent = candidates[latent_best] - z0
        d_task = candidates[task_best] - z0
        cos_latent = float(_cosine(d_latent, q))
        cos_task = float(_cosine(d_task, q))
        cos_all = _cosine(candidates - z0[None, :], q[None, :])

        norm_diff = float(np.sum(d_latent ** 2) - np.sum(d_task ** 2))
        align_diff = float(-2.0 * np.dot(q, d_latent - d_task))
        score_gap = float(costs[latent_best] - costs[task_best])
        decomposition_residual = float(score_gap - (norm_diff + align_diff))
        angle_advantage = float(cos_latent - cos_task)
        angle_explained = int(rank_error and cos_latent > cos_task)
        alignment_dominated = int(rank_error and align_diff < 0.0 and abs(align_diff) > abs(min(norm_diff, 0.0)))

        per_window.append(
            {
                "model": model_name,
                "window_idx": int(window_idx),
                "raw_pool": str(raw_pool),
                "candidate_angle_available": True,
                "num_candidates": int(candidates.shape[0]),
                "rank_error": rank_error,
                "rank_regret": regret,
                "latent_best_idx": latent_best,
                "task_best_idx": task_best,
                "latent_cost_latent_best": float(costs[latent_best]),
                "latent_cost_task_best": float(costs[task_best]),
                "score_gap": score_gap,
                "norm_diff": norm_diff,
                "align_diff": align_diff,
                "decomposition_residual": decomposition_residual,
                "cos_latent": cos_latent,
                "cos_task": cos_task,
                "angle_latent_rad": float(np.arccos(np.clip(cos_latent, -1.0, 1.0))),
                "angle_task_rad": float(np.arccos(np.clip(cos_task, -1.0, 1.0))),
                "angle_advantage": angle_advantage,
                "angle_explained_error": angle_explained,
                "alignment_dominated_error": alignment_dominated,
                "spearman_latent_task": _spearman(costs, spearman_true),
                "mean_candidate_cos": float(np.mean(cos_all)),
                "max_saved_score_diff": max_score_diff,
            }
        )

    error_rows = [row for row in per_window if row["rank_error"]]
    summary = {
        "model": model_name,
        "raw_pool": str(raw_pool),
        "candidate_angle_available": True,
        "num_windows": int(len(per_window)),
        "num_candidates": int(terminal_latents.shape[1]),
        "latent_dim": int(terminal_latents.shape[-1]),
        "rank_error_rate": float(np.mean([row["rank_error"] for row in per_window])) if per_window else float("nan"),
        "mean_rank_regret": float(np.mean([row["rank_regret"] for row in per_window])) if per_window else float("nan"),
        "mean_spearman_latent_task": float(np.nanmean([row["spearman_latent_task"] for row in per_window])) if per_window else float("nan"),
        "angle_explained_error_rate": float(np.mean([row["angle_explained_error"] for row in error_rows])) if error_rows else 0.0,
        "alignment_dominated_error_rate": float(np.mean([row["alignment_dominated_error"] for row in error_rows])) if error_rows else 0.0,
        "mean_angle_advantage_on_errors": float(np.mean([row["angle_advantage"] for row in error_rows])) if error_rows else 0.0,
        "mean_score_gap_on_errors": float(np.mean([row["score_gap"] for row in error_rows])) if error_rows else 0.0,
        "mean_norm_diff_on_errors": float(np.mean([row["norm_diff"] for row in error_rows])) if error_rows else 0.0,
        "mean_align_diff_on_errors": float(np.mean([row["align_diff"] for row in error_rows])) if error_rows else 0.0,
        "max_saved_score_diff": max_score_diff,
        "true_metric": true_metric,
    }
    return per_window, summary


def _oracle_candidate_sa_from_raw(
    raw: np.lib.npyio.NpzFile,
    true_scores: np.ndarray,
    true_lower_is_better: bool,
    model_name: str,
) -> Dict[str, object]:
    path_keys = [
        "latent_paths",
        "candidate_latent_paths",
        "predicted_latent_paths",
        "rollout_latents",
        "predicted_emb",
    ]
    path_key = next((key for key in path_keys if key in raw), None)
    if path_key is None:
        return {
            "model": model_name,
            "oracle_candidate_sa_available": False,
            "skip_reason": "raw pool has terminal latents only; no full candidate latent paths",
        }
    paths = np.asarray(raw[path_key])
    if paths.ndim != 4:
        return {
            "model": model_name,
            "oracle_candidate_sa_available": False,
            "skip_reason": f"{path_key} has unsupported shape {paths.shape}; expected (windows,candidates,T,D)",
        }
    selected = []
    for window_idx in range(paths.shape[0]):
        task_best = int(np.argmin(true_scores[window_idx]) if true_lower_is_better else np.argmax(true_scores[window_idx]))
        selected.append(paths[window_idx, task_best])
    stats = _self_approaching_stats(np.stack(selected, axis=0))
    return {
        "model": model_name,
        "oracle_candidate_sa_available": True,
        "latent_path_key": path_key,
        **stats,
    }


def _write_markdown(
    path: Path,
    expert_rows: List[Dict[str, object]],
    horizon_rows: List[Dict[str, object]],
    fixed_target_rows: List[Dict[str, object]],
    rank_rows: List[Dict[str, object]],
    oracle_rows: List[Dict[str, object]],
) -> None:
    lines = [
        "# Self-Approaching Plannability Diagnostic",
        "",
        "This diagnostic separates task-correct trajectory geometry from candidate ranking.",
        "",
        "## What is measured",
        "",
        "- `rho_sa_mean`: legacy one-step diagnostic kept for comparison; it includes the immediate next point in the future set.",
        "- `rho_sa_terminal_mean`: legacy one-step diagnostic using only the final trajectory point as the reference.",
        "- `rho_sa` in `horizon_sa_diagnostics.csv`: by default, the h-step terminal-target violation rate `||z_{t+h}-z_R||^2 > ||z_t-z_R||^2`, where `z_R` is the final point of the expert window.",
        "- `p_sa`: the non-violation/self-approaching probability, equal to `1 - rho_sa`.",
        "- `vplus_sa_norm`: positive violation magnitude normalized by the starting latent distance; this equals violation probability times conditional violation severity.",
        "- `vplus_sa_norm_cond`: conditional normalized violation severity, averaged only over pairs with positive violation.",
        "- `rank_error_rate`: fraction of candidate sets where the latent terminal-distance best candidate differs from the task-metric best candidate.",
        "- `latent_rank_of_task_best_mean`: rank assigned by latent score to the task-oracle candidate.",
        "",
        "In continuous PushT we do not claim exact shortest paths. Expert demonstrations and task-oracle candidate rollouts are treated as task-correct reference rollouts.",
        "",
    ]
    if expert_rows:
        lines.extend([
            "## Legacy expert/reference self-approaching",
            "",
            "This table is kept for backward compatibility. The paper-facing horizon-aware metric is below.",
            "",
            "| model | dim | rho_SA | rho_SA_terminal | trajectories | T |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in expert_rows:
            lines.append(
                f"| {row['model']} | {row['latent_dim']} | {float(row['rho_sa_mean']):.4f} | "
                f"{float(row['rho_sa_terminal_mean']):.4f} | {row['num_trajectories']} | {row['trajectory_len']} |"
            )
        lines.append("")
    if horizon_rows:
        horizons = sorted({int(row["h_raw"]) for row in horizon_rows})
        models = sorted({str(row["model_name"]) for row in horizon_rows}, key=lambda name: min(int(row["dim"]) for row in horizon_rows if row["model_name"] == name))
        reference_modes = sorted({str(row.get("reference_mode", "terminal")) for row in horizon_rows})
        lines.extend([
            "## Horizon-aware self-approaching",
            "",
            f"Reference mode: `{','.join(reference_modes)}`. In terminal mode, the reference point is the final latent in the expert window.",
            "",
            "| model | dim | " + " | ".join(f"p h={h}" for h in horizons) + " |",
            "|---|---:" + "|---:" * len(horizons) + "|",
        ])
        for model in models:
            dim = min(int(row["dim"]) for row in horizon_rows if row["model_name"] == model)
            values = []
            for horizon in horizons:
                match = [row for row in horizon_rows if row["model_name"] == model and int(row["h_raw"]) == horizon]
                values.append(f"{float(match[0]['p_sa']):.4f}" if match else "nan")
            lines.append(f"| {model} | {dim} | " + " | ".join(values) + " |")
        lines.append("")
    if fixed_target_rows:
        lines.extend(["## Fixed future targets", "", "For this table, `h_raw=25` matches the PushT CEM planning horizon.", "", "| model | target=50 | target=100 | target=125 |", "|---|---:|---:|---:|"])
        models = sorted({str(row["model_name"]) for row in fixed_target_rows}, key=lambda name: min(int(row["dim"]) for row in fixed_target_rows if row["model_name"] == name))
        for model in models:
            values = []
            for target in [50, 100, 125]:
                match = [
                    row for row in fixed_target_rows
                    if row["model_name"] == model and int(row["h_raw"]) == 25 and int(row["target_raw"]) == target
                ]
                values.append(f"{float(match[0]['rho_sa']):.4f}" if match else "nan")
            lines.append(f"| {model} | " + " | ".join(values) + " |")
        lines.append("")
    if rank_rows:
        lines.extend(["## Candidate ranking", "", "| model | rank error | oracle latent rank | top-k hit | regret | spearman |", "|---|---:|---:|---:|---:|---:|"])
        for row in rank_rows:
            lines.append(
                f"| {row['model']} | {float(row['rank_error_rate']):.4f} | "
                f"{float(row['latent_rank_of_task_best_mean']):.2f} | {float(row['topk_hit_rate']):.4f} | "
                f"{float(row['candidate_regret_mean']):.4f} | {float(row['spearman_mean']):.4f} |"
            )
        lines.append("")
    if oracle_rows:
        lines.extend(["## Oracle-candidate self-approaching", "", "| model | available | rho_SA | reason |", "|---|---|---:|---|"])
        for row in oracle_rows:
            value = float(row["rho_sa_mean"]) if row.get("oracle_candidate_sa_available") else float("nan")
            lines.append(f"| {row['model']} | {row.get('oracle_candidate_sa_available')} | {value:.4f} | {row.get('skip_reason', '')} |")
        lines.append("")
    lines.extend(
        [
            "## Intended evidence chain",
            "",
            "`latent dimension ↑ -> self-approaching violations on task-correct rollouts ↓ -> candidate ranking error ↓ -> closed-loop SR ↑`.",
            "",
            "This should be interpreted as a planner-facing diagnostic, not as a claim that prediction loss is irrelevant.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _plot_horizon_diagnostics(
    rows: List[Dict[str, object]],
    fixed_rows: List[Dict[str, object]],
    output_dir: Path,
    model_order: List[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sa_diag] matplotlib unavailable; skipping figures.", flush=True)
        return
    if not rows:
        return
    figure_dir = output_dir / "paper_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted({int(row["h_raw"]) for row in rows})
    models = [model for model in model_order if any(row["model_name"] == model for row in rows)]
    if not models:
        models = sorted({str(row["model_name"]) for row in rows})
    dims = {str(row["model_name"]): int(row["dim"]) for row in rows}
    labels = [f"{model}\nD={dims.get(model, '')}" for model in models]
    x = np.arange(len(models), dtype=np.float64)
    width = min(0.14, 0.75 / max(len(horizons), 1))

    def value(model: str, horizon: int, key: str) -> float:
        for row in rows:
            if row["model_name"] == model and int(row["h_raw"]) == horizon:
                return float(row[key])
        return float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor="white")
    offsets = (np.arange(len(horizons)) - (len(horizons) - 1) / 2.0) * width
    for offset, horizon in zip(offsets, horizons):
        axes[0].bar(x + offset, [value(model, horizon, "p_sa") for model in models], width=width, label=f"h={horizon}")
        axes[1].bar(x + offset, [value(model, horizon, "vplus_sa_norm") for model in models], width=width, label=f"h={horizon}")
    axes[0].set_title("A. H-step self-approaching probability")
    axes[0].set_ylabel("p_sa = 1 - rho_sa")
    axes[1].set_title("B. Positive violation magnitude")
    axes[1].set_ylabel("vplus_sa_norm")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_horizon_sa_vs_dim.png", dpi=260)
    fig.savefig(figure_dir / "fig_horizon_sa_vs_dim.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), facecolor="white")
    for model in models:
        ax.plot(horizons, [value(model, horizon, "p_sa") for horizon in horizons], marker="o", label=model)
    ax.set_xlabel("h_raw")
    ax.set_ylabel("p_sa")
    ax.set_title("Horizon-aware self-approaching probability")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_horizon_sa_vs_horizon.png", dpi=260)
    fig.savefig(figure_dir / "fig_horizon_sa_vs_horizon.pdf")
    plt.close(fig)

    if fixed_rows:
        fixed_targets = [target for target in sorted({int(row["target_raw"]) for row in fixed_rows}) if target in {50, 100, 125}]
        fixed_h = 25
        fig, ax = plt.subplots(figsize=(8.0, 4.2), facecolor="white")
        width = min(0.22, 0.72 / max(len(fixed_targets), 1))
        offsets = (np.arange(len(fixed_targets)) - (len(fixed_targets) - 1) / 2.0) * width

        def fixed_value(model: str, target: int) -> float:
            for row in fixed_rows:
                if row["model_name"] == model and int(row["target_raw"]) == target and int(row["h_raw"]) == fixed_h:
                    return float(row["rho_sa"])
            return float("nan")

        for offset, target in zip(offsets, fixed_targets):
            ax.bar(x + offset, [fixed_value(model, target) for model in models], width=width, label=f"target={target}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("rho_sa after h_raw=25")
        ax.set_title("Fixed future target self-approaching")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "fig_fixed_target_sa.png", dpi=260)
        fig.savefig(figure_dir / "fig_fixed_target_sa.pdf")
        plt.close(fig)


def _plot_angle_diagnostics(
    angle_rows: List[Dict[str, object]],
    cos_distributions: Dict[str, Dict[int, np.ndarray]],
    candidate_summary_rows: List[Dict[str, object]],
    success_rates: Dict[str, float],
    output_dir: Path,
    model_order: List[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sa_diag] matplotlib unavailable; skipping angle figures.", flush=True)
        return
    figure_dir = output_dir / "paper_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    if angle_rows:
        horizons = sorted({int(row["h_raw"]) for row in angle_rows})
        models = [model for model in model_order if any(row["model_name"] == model for row in angle_rows)]
        if not models:
            models = sorted({str(row["model_name"]) for row in angle_rows})
        dims = {str(row["model_name"]): int(row["dim"]) for row in angle_rows}
        labels = [f"{model}\nD={dims.get(model, '')}" for model in models]
        x = np.arange(len(models), dtype=np.float64)
        width = min(0.14, 0.75 / max(len(horizons), 1))

        def value(model: str, horizon: int, key: str) -> float:
            for row in angle_rows:
                if row["model_name"] == model and int(row["h_raw"]) == horizon:
                    return float(row[key])
            return float("nan")

        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), facecolor="white")
        offsets = (np.arange(len(horizons)) - (len(horizons) - 1) / 2.0) * width
        for offset, horizon in zip(offsets, horizons):
            axes[0].bar(x + offset, [value(model, horizon, "mean_neg_cos_progress") for model in models], width=width, label=f"h={horizon}")
            axes[1].bar(x + offset, [value(model, horizon, "progress_obtuse_rate") for model in models], width=width, label=f"h={horizon}")
        h_box = 25 if 25 in horizons else horizons[0]
        box_values = [cos_distributions.get(model, {}).get(h_box, np.asarray([])) for model in models]
        axes[2].boxplot(box_values, labels=labels, showfliers=False)
        axes[0].set_title("A. Negative progress cosine")
        axes[0].set_ylabel("mean max(-cos, 0)")
        axes[1].set_title("B. Obtuse progress rate")
        axes[1].set_ylabel("P(cos < 0)")
        axes[2].set_title(f"C. cos_progress distribution, h={h_box}")
        axes[2].set_ylabel("cos_progress")
        for ax in axes[:2]:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.22)
            ax.legend(fontsize=8)
        axes[2].tick_params(axis="x", rotation=25)
        axes[2].grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(figure_dir / "fig_progress_angle_vs_dim.png", dpi=260)
        fig.savefig(figure_dir / "fig_progress_angle_vs_dim.pdf")
        plt.close(fig)

    available_candidate_rows = [row for row in candidate_summary_rows if row.get("candidate_angle_available")]
    if available_candidate_rows:
        models = [model for model in model_order if any(row["model"] == model for row in available_candidate_rows)]
        if not models:
            models = sorted({str(row["model"]) for row in available_candidate_rows})
        by_model = {str(row["model"]): row for row in available_candidate_rows}
        dims = {model: int(by_model[model].get("latent_dim", 0)) for model in models}
        labels = [f"{model}\nD={dims.get(model, '')}" for model in models]
        x = np.arange(len(models), dtype=np.float64)

        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), facecolor="white")
        axes[0].bar(x, [float(by_model[m].get("mean_neg_cos_progress_h25", np.nan)) for m in models])
        axes[0].set_title("A. Reference angle severity")
        axes[0].set_ylabel("mean_neg_cos_progress h=25")
        axes[1].bar(x, [float(by_model[m].get("angle_explained_error_rate", np.nan)) for m in models])
        axes[1].set_title("B. Angle-explained errors")
        axes[1].set_ylabel("rate among ranking errors")
        sr_values = [success_rates.get(m, float("nan")) for m in models]
        axes[2].bar(x, sr_values)
        axes[2].set_title("C. Success rate")
        axes[2].set_ylabel("SR")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(figure_dir / "fig_angle_ranking_sr.png", dpi=260)
        fig.savefig(figure_dir / "fig_angle_ranking_sr.pdf")
        plt.close(fig)

        if any(np.isfinite(float(by_model[m].get("mean_angle_advantage_on_errors", np.nan))) for m in models):
            fig, ax = plt.subplots(figsize=(5.0, 4.0), facecolor="white")
            for model in models:
                x_value = float(by_model[model].get("mean_angle_advantage_on_errors", np.nan))
                y_value = float(by_model[model].get("mean_rank_regret", np.nan))
                ax.scatter(x_value, y_value, s=45)
                ax.annotate(model, (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=8)
            ax.set_xlabel("mean_angle_advantage_on_errors")
            ax.set_ylabel("mean_rank_regret")
            ax.set_title("Angle advantage vs regret")
            ax.grid(alpha=0.22)
            fig.tight_layout()
            fig.savefig(figure_dir / "fig_angle_advantage_regret_scatter.png", dpi=260)
            fig.savefig(figure_dir / "fig_angle_advantage_regret_scatter.pdf")
            plt.close(fig)


def _print_horizon_summary(rows: List[Dict[str, object]], fixed_rows: List[Dict[str, object]], model_order: List[str]) -> None:
    if not rows:
        return
    horizons = sorted({int(row["h_raw"]) for row in rows})
    models = [model for model in model_order if any(row["model_name"] == model for row in rows)]
    if not models:
        models = sorted({str(row["model_name"]) for row in rows})
    print("\n[sa_diag] Horizon-aware self-approaching summary", flush=True)
    for model in models:
        model_rows = {int(row["h_raw"]): row for row in rows if row["model_name"] == model}
        rho = " ".join(f"h{h}={float(model_rows[h]['rho_sa']):.4f}" if h in model_rows else f"h{h}=N/A" for h in horizons)
        p_sa = " ".join(f"h{h}={float(model_rows[h]['p_sa']):.4f}" if h in model_rows else f"h{h}=N/A" for h in horizons)
        vplus = " ".join(f"h{h}={float(model_rows[h]['vplus_sa_norm']):.4f}" if h in model_rows else f"h{h}=N/A" for h in horizons)
        vplus_cond = " ".join(f"h{h}={float(model_rows[h]['vplus_sa_norm_cond']):.4f}" if h in model_rows else f"h{h}=N/A" for h in horizons)
        print(f"  {model} rho_sa: {rho}", flush=True)
        print(f"  {model} p_sa:   {p_sa}", flush=True)
        print(f"  {model} vplus:  {vplus}", flush=True)
        print(f"  {model} v+|err: {vplus_cond}", flush=True)
        if fixed_rows:
            fixed_bits = []
            for target in [50, 100, 125]:
                match = [
                    row for row in fixed_rows
                    if row["model_name"] == model and int(row["h_raw"]) == 25 and int(row["target_raw"]) == target
                ]
                if match:
                    fixed_bits.append(f"target{target}={float(match[0]['rho_sa']):.4f}")
                else:
                    fixed_bits.append(f"target{target}=N/A")
            print(f"  {model} fixed h25: {' '.join(fixed_bits)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-approaching and candidate-ranking diagnostics for LeWM PushT models.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--models", nargs="*", default=[], help="Model checkpoints as NAME=checkpoint_object.ckpt.")
    parser.add_argument("--raw_pools", nargs="*", default=[], help="Candidate raw pools as NAME=varyK_or_aliasing_raw.npz.")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--num_trajectories", type=int, default=100)
    parser.add_argument("--trajectory_len", type=int, default=16)
    parser.add_argument("--trajectory_stride", type=int, default=5)
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--step_key", default="step_idx")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--true_metric", choices=["auto", "progress", "terminal_cost"], default="auto")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--action_block", type=int, default=5)
    parser.add_argument("--horizon_raws", default="25,50,100")
    parser.add_argument("--angle_horizon_raws", default="25,50,100")
    parser.add_argument("--corner_horizon_raws", default="5,25")
    parser.add_argument("--target_raws", default="", help="Optional fixed-target offsets. Empty by default; terminal-target SA is the main metric.")
    parser.add_argument("--sa_reference_mode", choices=["terminal", "all_future"], default="terminal")
    parser.add_argument("--reference_raw_span", type=int, default=175)
    parser.add_argument("--min_reference_raw_span", type=int, default=-1, help="Minimum raw-step span for horizon windows. Default -1 requires full --reference_raw_span.")
    parser.add_argument("--tau_values", default="0,1e-4,1e-3,1e-2")
    parser.add_argument("--success_rates", nargs="*", default=[], help="Optional model=SR values for plotting, e.g. state8=36 baseline192=96.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[sa_diag] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[sa_diag] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models = _parse_name_paths(args.models)
    raw_pools = _parse_name_paths(args.raw_pools)
    success_rates = _parse_name_floats(args.success_rates)
    horizons_raw = _parse_int_list(args.horizon_raws)
    angle_horizons_raw = _parse_int_list(args.angle_horizon_raws)
    corner_horizons_raw = _parse_int_list(args.corner_horizon_raws)
    target_raws = _parse_int_list(args.target_raws)
    tau_values = [float(item) for item in str(args.tau_values).replace(",", " ").split() if item.strip()]
    all_reference_horizons = horizons_raw + angle_horizons_raw + corner_horizons_raw
    max_strict_horizon = max(all_reference_horizons) + args.action_block if all_reference_horizons else 0
    max_fixed_target = max(target_raws) if target_raws else 0
    min_reference_raw_span = args.reference_raw_span if args.min_reference_raw_span < 0 else args.min_reference_raw_span
    required_reference_raw_span = max(min_reference_raw_span, max_strict_horizon, max_fixed_target)
    effective_reference_raw_span = max(args.reference_raw_span, required_reference_raw_span)

    expert_rows: List[Dict[str, object]] = []
    horizon_sa_rows: List[Dict[str, object]] = []
    fixed_target_rows: List[Dict[str, object]] = []
    angle_reference_rows: List[Dict[str, object]] = []
    corner_angle_rows: List[Dict[str, object]] = []
    angle_cos_distributions: Dict[str, Dict[int, np.ndarray]] = {}
    per_window_rank_rows: List[Dict[str, object]] = []
    rank_summary_rows: List[Dict[str, object]] = []
    oracle_sa_rows: List[Dict[str, object]] = []
    candidate_angle_rows: List[Dict[str, object]] = []
    candidate_angle_summary_rows: List[Dict[str, object]] = []

    if models:
        with h5py.File(args.dataset, "r") as h5:
            pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
            episode_key = _find_key(h5, [args.episode_key, "episode_idx", "ep_idx"])
            step_key = _find_key(h5, [args.step_key, "step_idx"])
            episode_idx = np.asarray(h5[episode_key]).reshape(-1)
            step_idx = np.asarray(h5[step_key]).reshape(-1)
            windows = _sample_reference_windows(
                episode_idx,
                step_idx,
                args.num_trajectories,
                args.trajectory_len,
                args.trajectory_stride,
                args.seed,
            )
            print(f"[sa_diag] sampled reference windows: {windows.shape}", flush=True)
            horizon_segments = _sample_reference_segments(
                episode_idx,
                step_idx,
                args.num_trajectories,
                args.action_block,
                effective_reference_raw_span,
                required_reference_raw_span,
                args.seed,
            )
            print(
                f"[sa_diag] sampled horizon-aware reference segments: {len(horizon_segments)} "
                f"(lengths={[int(seg.shape[0]) for seg in horizon_segments[:5]]}..., "
                f"required_raw_span={required_reference_raw_span}, "
                f"reference_raw_span={effective_reference_raw_span})",
                flush=True,
            )
            pixels_ds = h5[pixels_key]
            for model_name, checkpoint in models.items():
                print(f"[sa_diag] encoding expert/reference trajectories for {model_name}", flush=True)
                model = _load_model(checkpoint, device)
                z = _encode_reference_trajectories(model, pixels_ds, windows, args.img_size, args.batch_size, device)
                stats = _self_approaching_stats(z)
                expert_rows.append(
                    {
                        "model": model_name,
                        "checkpoint_object": str(checkpoint),
                        "latent_dim": int(z.shape[-1]),
                        "trajectory_stride": int(args.trajectory_stride),
                        **stats,
                    }
                )
                z_segments = _encode_reference_segments(model, pixels_ds, horizon_segments, args.img_size, device)
                horizon_sa_rows.extend(
                    _horizon_self_approaching_rows(
                        z_segments,
                        model_name,
                        int(z.shape[-1]),
                        horizons_raw,
                        args.action_block,
                        tau_values,
                        args.sa_reference_mode,
                    )
                )
                fixed_target_rows.extend(
                    _fixed_target_self_approaching_rows(
                        z_segments,
                        model_name,
                        int(z.shape[-1]),
                        horizons_raw,
                        target_raws,
                        args.action_block,
                        tau_values,
                    )
                )
                model_angle_rows, model_cos_distribution = _progress_angle_rows(
                    z_segments,
                    model_name,
                    int(z.shape[-1]),
                    angle_horizons_raw,
                    args.action_block,
                )
                angle_reference_rows.extend(model_angle_rows)
                angle_cos_distributions[model_name] = model_cos_distribution
                corner_angle_rows.extend(
                    _corner_angle_rows(
                        z_segments,
                        model_name,
                        int(z.shape[-1]),
                        corner_horizons_raw,
                        args.action_block,
                    )
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    for model_name, raw_pool in raw_pools.items():
        print(f"[sa_diag] candidate ranking diagnostics for {model_name}", flush=True)
        summary, per_window, oracle_payload = _candidate_rank_metrics(raw_pool, model_name, args.true_metric, args.topk)
        rank_summary_rows.append(summary)
        per_window_rank_rows.extend(per_window)
        oracle_sa_rows.append(oracle_payload)
        angle_per_window, angle_summary = _candidate_angle_ranking_metrics(
            raw_pool,
            model_name,
            args.true_metric,
            models.get(model_name),
            Path(args.dataset),
            args.pixels_key,
            args.img_size,
            args.batch_size,
            device,
        )
        for angle_row in angle_reference_rows:
            if angle_row["model_name"] == model_name and int(angle_row["h_raw"]) == 25:
                angle_summary["mean_neg_cos_progress_h25"] = angle_row["mean_neg_cos_progress"]
                angle_summary["progress_obtuse_rate_h25"] = angle_row["progress_obtuse_rate"]
                break
        candidate_angle_rows.extend(angle_per_window)
        candidate_angle_summary_rows.append(angle_summary)

    _write_csv(output_dir / "expert_self_approaching_summary.csv", expert_rows)
    _write_csv(output_dir / "horizon_sa_diagnostics.csv", horizon_sa_rows)
    _write_csv(output_dir / "horizon_sa_fixed_targets.csv", fixed_target_rows)
    _write_csv(output_dir / "angle_reference_diagnostics.csv", angle_reference_rows)
    _write_csv(output_dir / "corner_angle_reference_diagnostics.csv", corner_angle_rows)
    _write_csv(output_dir / "candidate_ranking_summary.csv", rank_summary_rows)
    _write_csv(output_dir / "candidate_ranking_per_window.csv", per_window_rank_rows)
    _write_csv(output_dir / "candidate_angle_ranking_diagnostics.csv", candidate_angle_rows)
    _write_csv(output_dir / "candidate_angle_ranking_summary.csv", candidate_angle_summary_rows)
    _write_csv(output_dir / "oracle_candidate_self_approaching_summary.csv", oracle_sa_rows)
    with (output_dir / "self_approaching_plannability_summary.json").open("w") as file:
        json.dump(
            {
                "expert_self_approaching": expert_rows,
                "horizon_self_approaching": horizon_sa_rows,
                "horizon_fixed_targets": fixed_target_rows,
                "angle_reference": angle_reference_rows,
                "corner_angle_reference": corner_angle_rows,
                "candidate_ranking": rank_summary_rows,
                "candidate_angle_ranking": candidate_angle_summary_rows,
                "oracle_candidate_self_approaching": oracle_sa_rows,
                "args": {
                    **vars(args),
                    "min_reference_raw_span_effective": min_reference_raw_span,
                    "required_reference_raw_span": required_reference_raw_span,
                    "effective_reference_raw_span": effective_reference_raw_span,
                    "sa_reference_mode": args.sa_reference_mode,
                },
            },
            file,
            indent=2,
        )
    _write_markdown(
        output_dir / "self_approaching_plannability_summary.md",
        expert_rows,
        horizon_sa_rows,
        fixed_target_rows,
        rank_summary_rows,
        oracle_sa_rows,
    )
    _plot_horizon_diagnostics(horizon_sa_rows, fixed_target_rows, output_dir, list(models.keys()))
    _plot_angle_diagnostics(
        angle_reference_rows,
        angle_cos_distributions,
        candidate_angle_summary_rows,
        success_rates,
        output_dir,
        list(models.keys()) or list(raw_pools.keys()),
    )
    _print_horizon_summary(horizon_sa_rows, fixed_target_rows, list(models.keys()))
    print(f"[sa_diag] wrote outputs under {output_dir}")


if __name__ == "__main__":
    main()
