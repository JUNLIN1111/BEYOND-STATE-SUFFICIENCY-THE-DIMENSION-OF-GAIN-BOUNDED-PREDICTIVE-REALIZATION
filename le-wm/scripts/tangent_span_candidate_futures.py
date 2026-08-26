from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


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
            arr = np.asarray(values, dtype=np.float64)
            out[f"{metric_key}_mean"] = float(np.nanmean(arr))
            out[f"{metric_key}_std"] = float(np.nanstd(arr))
            out[f"{metric_key}_stderr"] = float(np.nanstd(arr) / math.sqrt(max(np.sum(np.isfinite(arr)), 1)))
        out_rows.append(out)
    return out_rows


def _rank_at_energy(singular_values: np.ndarray, threshold: float) -> int:
    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(np.sum(energy))
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(energy) / total, threshold) + 1)


def _effective_rank(singular_values: np.ndarray) -> float:
    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(np.sum(energy))
    if total <= 0:
        return float("nan")
    p = energy / total
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def _span_metrics_from_matrix(matrix: np.ndarray, prefix: str = "") -> Dict[str, object]:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return {
            f"{prefix}effective_rank": float("nan"),
            f"{prefix}rank50": float("nan"),
            f"{prefix}rank90": float("nan"),
            f"{prefix}rank95": float("nan"),
            f"{prefix}rank99": float("nan"),
            f"{prefix}singular_values": [],
            f"{prefix}squared_energy_cumsum": [],
        }
    singular_values = np.linalg.svd(matrix.astype(np.float32), compute_uv=False)
    singular_values = np.sort(singular_values)[::-1]
    energy = np.square(singular_values)
    energy_cumsum = np.cumsum(energy) / max(float(np.sum(energy)), 1e-12)
    return {
        f"{prefix}effective_rank": _effective_rank(singular_values),
        f"{prefix}rank50": _rank_at_energy(singular_values, 0.50),
        f"{prefix}rank90": _rank_at_energy(singular_values, 0.90),
        f"{prefix}rank95": _rank_at_energy(singular_values, 0.95),
        f"{prefix}rank99": _rank_at_energy(singular_values, 0.99),
        f"{prefix}singular_values": [float(value) for value in singular_values.tolist()],
        f"{prefix}squared_energy_cumsum": [float(value) for value in energy_cumsum.tolist()],
    }


def _global_pca_metrics(Z: np.ndarray) -> Dict[str, object]:
    centered = Z - np.mean(Z, axis=0, keepdims=True)
    metrics = _span_metrics_from_matrix(centered, prefix="global_")
    singular_values = np.asarray(metrics["global_singular_values"], dtype=np.float64)
    pr = float(np.square(np.sum(np.square(singular_values))) / max(np.sum(np.square(singular_values) ** 2), 1e-12))
    return {
        "global_participation_ratio": pr,
        "global_rank90": metrics["global_rank90"],
        "global_rank95": metrics["global_rank95"],
        "global_rank99": metrics["global_rank99"],
    }


def _local_tangent_metrics(Z: np.ndarray, num_neighbors: int, local_dim: int, device: torch.device) -> Dict[str, object]:
    if Z.ndim != 2:
        raise ValueError(f"Expected Z shape [N,D], got {Z.shape}")
    n, d = Z.shape
    k = min(int(num_neighbors), n - 1)
    if k <= 0:
        raise ValueError("Need at least two points for tangent span.")
    if local_dim <= 0 or local_dim > min(k, d):
        raise ValueError(f"local_dim={local_dim} incompatible with neighbors={k}, D={d}")
    Z_t = torch.from_numpy(Z.astype(np.float32)).to(device)
    distances = torch.cdist(Z_t, Z_t)
    neighbor_idx = torch.topk(distances, k=k + 1, largest=False).indices[:, 1:]
    bases = []
    for idx in range(n):
        neighbors = Z_t[neighbor_idx[idx]]
        centered = neighbors - neighbors.mean(dim=0, keepdim=True)
        _u, _s, vh = torch.linalg.svd(centered.float(), full_matrices=False)
        bases.append(vh[:local_dim].detach().cpu().numpy())
    V_all = np.concatenate(bases, axis=0)
    return _span_metrics_from_matrix(V_all, prefix="tangent_span_")


def _subsample(Z: np.ndarray, max_points: int | None, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = Z.shape[0]
    if max_points is None or max_points <= 0 or n <= max_points:
        idx = np.arange(n)
        return Z, idx
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return Z[idx], idx


def _kmeans_numpy(Z: np.ndarray, k: int, seed: int, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    init_idx = rng.choice(n, size=min(k, n), replace=False)
    centers = Z[init_idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dist2 = np.sum((Z[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_idx in range(centers.shape[0]):
            mask = labels == cluster_idx
            if np.any(mask):
                centers[cluster_idx] = Z[mask].mean(axis=0)
    return labels, centers


def _cluster_offset_metrics(Z: np.ndarray, k_values: Sequence[int], seed: int, min_cluster_size: int = 10) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    overall_mean = Z.mean(axis=0, keepdims=True)
    total_var = float(np.mean(np.sum((Z - overall_mean) ** 2, axis=1)))
    for k in k_values:
        if k <= 1 or k > Z.shape[0]:
            continue
        labels, centers = _kmeans_numpy(Z, k, seed + k)
        kept_centers = []
        weights = []
        within_weighted = 0.0
        between_weighted = 0.0
        for cluster_idx in range(centers.shape[0]):
            mask = labels == cluster_idx
            count = int(np.sum(mask))
            if count < min_cluster_size:
                continue
            cluster = Z[mask]
            center = cluster.mean(axis=0)
            kept_centers.append(center)
            weights.append(count)
            within_weighted += count * float(np.mean(np.sum((cluster - center[None]) ** 2, axis=1)))
            between_weighted += count * float(np.sum((center - overall_mean[0]) ** 2))
        if not kept_centers:
            continue
        mean_matrix = np.stack(kept_centers, axis=0) - np.mean(np.stack(kept_centers, axis=0), axis=0, keepdims=True)
        span = _span_metrics_from_matrix(mean_matrix, prefix="mean_span_")
        rows.append(
            {
                "kmeans_k": int(k),
                "num_clusters_kept": int(len(kept_centers)),
                "total_var": total_var,
                "within_var": float(within_weighted / max(np.sum(weights), 1)),
                "between_var": float(between_weighted / max(np.sum(weights), 1)),
                "between_frac": float((between_weighted / max(np.sum(weights), 1)) / max(total_var, 1e-12)),
                "mean_span_effective_rank": span["mean_span_effective_rank"],
                "mean_span_rank90": span["mean_span_rank90"],
                "mean_span_rank99": span["mean_span_rank99"],
            }
        )
    return rows


def _load_terminal_latents(path: Path) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=False)
    keys = sorted(data.files)
    print(f"[candidate_tangent] available keys in {path}: {keys}", flush=True)
    if "terminal_latents" not in data:
        raise KeyError(f"{path} does not contain terminal_latents")
    Z = np.asarray(data["terminal_latents"], dtype=np.float32)
    if Z.ndim == 2:
        Z = Z[None]
    if Z.ndim != 3:
        raise ValueError(f"Expected terminal_latents [W,N,D] or [N,D], got {Z.shape}")
    return Z, keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Tangent span analysis on candidate terminal future latents.")
    parser.add_argument("--raw_npz", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--num_neighbors", type=int, default=50)
    parser.add_argument("--local_dim", type=int, default=2)
    parser.add_argument("--max_points_per_window", type=int, default=0)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--chart_offset_output_csv", default="")
    parser.add_argument("--kmeans_values", default="4,8,16,32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    Z_all, keys = _load_terminal_latents(Path(args.raw_npz))
    num_windows = Z_all.shape[0] if args.max_windows <= 0 else min(args.max_windows, Z_all.shape[0])
    max_points = None if args.max_points_per_window <= 0 else args.max_points_per_window
    kmeans_values = [int(item) for item in str(args.kmeans_values).split(",") if item]
    per_window: List[Dict[str, object]] = []
    offset_rows: List[Dict[str, object]] = []
    for window_idx in range(num_windows):
        Z, selected_idx = _subsample(Z_all[window_idx], max_points, args.seed + window_idx)
        print(f"[candidate_tangent] {args.model_name} window {window_idx + 1}/{num_windows}: Z={Z.shape}", flush=True)
        tangent = _local_tangent_metrics(Z, args.num_neighbors, args.local_dim, device)
        global_metrics = _global_pca_metrics(Z)
        row = {
            "model": args.model_name,
            "window_idx": int(window_idx),
            "latent_dim": int(Z.shape[-1]),
            "num_points": int(Z.shape[0]),
            "num_neighbors": int(min(args.num_neighbors, Z.shape[0] - 1)),
            "local_dim": int(args.local_dim),
        }
        row.update({key: value for key, value in tangent.items() if not isinstance(value, list)})
        row.update(global_metrics)
        per_window.append(row)
        for offset_row in _cluster_offset_metrics(Z, kmeans_values, args.seed + 1000 * window_idx):
            offset_row.update({"model": args.model_name, "window_idx": int(window_idx), "latent_dim": int(Z.shape[-1])})
            offset_rows.append(offset_row)

    summary_rows = _aggregate(per_window, ["model", "latent_dim", "num_neighbors", "local_dim"])
    summary = {
        "model": args.model_name,
        "raw_npz": str(args.raw_npz),
        "available_keys": keys,
        "latent_dim": int(Z_all.shape[-1]),
        "num_windows": int(num_windows),
        "num_points_per_window_mean": float(np.mean([row["num_points"] for row in per_window])),
        "num_neighbors": int(args.num_neighbors),
        "local_dim": int(args.local_dim),
        "energy_definition": "rank thresholds use cumulative sum(s_k^2) / sum(s_j^2).",
        "summary": summary_rows[0] if summary_rows else {},
        "per_window": per_window,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as file:
        json.dump(summary, file, indent=2)
    _write_csv(Path(args.output_csv), per_window)
    chart_path = Path(args.chart_offset_output_csv) if args.chart_offset_output_csv else Path(args.output_csv).with_name(f"{args.model_name}_chart_offset_span.csv")
    _write_csv(chart_path, offset_rows)
    _write_csv(chart_path.with_name(chart_path.stem + "_summary.csv"), _aggregate(offset_rows, ["model", "latent_dim", "kmeans_k"]))
    print(f"[candidate_tangent] wrote {output_json}")
    print(f"[candidate_tangent] wrote {args.output_csv}")
    print(f"[candidate_tangent] wrote {chart_path}")


if __name__ == "__main__":
    main()
