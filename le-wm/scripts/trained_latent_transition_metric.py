from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aliasing_experiment import _find_key, _load_model, _preprocess_pixels, _read_rows  # noqa: E402


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        corr = spearmanr(x, y).correlation
        return float(corr) if corr is not None else float("nan")
    except ImportError:
        xr = np.argsort(np.argsort(x)).astype(np.float64)
        yr = np.argsort(np.argsort(y)).astype(np.float64)
        return float(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) > 1e-12 and np.std(yr) > 1e-12 else float("nan")


def _encode_rows(model, dataset_path: Path, rows: np.ndarray, device: torch.device, batch_size: int, img_size: int) -> np.ndarray:
    latents = []
    with h5py.File(dataset_path, "r") as h5:
        pixels_key = _find_key(h5, ["pixels", "observation/pixels"])
        pixels_ds = h5[pixels_key]
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start:start + batch_size]
                pixels_np = _read_rows(pixels_ds, batch_rows)[:, None]
                pixels = _preprocess_pixels(pixels_np, img_size, device)
                emb = model.encode({"pixels": pixels})["emb"][:, 0]
                latents.append(emb.detach().float().cpu().numpy())
    return np.concatenate(latents, axis=0)


def _pairwise_dist(Z: np.ndarray) -> np.ndarray:
    sq = np.sum(Z * Z, axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (Z @ Z.T), 0.0)
    return np.sqrt(D2)


def _metric_rows(model_name: str, Z: np.ndarray, D_graph: np.ndarray, q_far: float, q_near: float, k_neighbors: int, num_goals: int, seed: int):
    rows: List[Dict[str, object]] = []
    D_lat = _pairwise_dist(Z)
    left, right = np.triu_indices(D_graph.shape[0], k=1)
    graph = D_graph[left, right]
    lat = D_lat[left, right]
    graph_far_thr = float(np.quantile(graph, q_far))
    lat_near_thr = float(np.quantile(lat, q_near))
    graph_far = graph >= graph_far_thr
    lat_near = lat <= lat_near_thr
    scale = float(np.sum(lat * graph) / max(np.sum(graph * graph), 1e-12))
    stress = float(np.sum((lat - scale * graph) ** 2) / max(np.sum(graph ** 2), 1e-12))
    nearest = np.argsort(D_lat + np.eye(D_lat.shape[0]) * 1e9, axis=1)[:, :k_neighbors]
    for k in [1, 5, 10]:
        k_eff = min(k, nearest.shape[1])
        rows.append(
            {
                "model": model_name,
                "metric": f"latent_nearest_{k}_graph_distance",
                "value": float(np.median(np.take_along_axis(D_graph, nearest[:, :k_eff], axis=1))),
            }
        )
    rows.extend(
        [
            {"model": model_name, "metric": "latent_graph_spearman", "value": _spearman(lat, graph)},
            {"model": model_name, "metric": "latent_graph_pearson", "value": float(np.corrcoef(lat, graph)[0, 1])},
            {"model": model_name, "metric": "false_shortcut_rate", "value": float(np.mean(lat_near[graph_far])) if np.any(graph_far) else float("nan")},
            {"model": model_name, "metric": "false_shortcut_joint_rate", "value": float(np.mean(lat_near & graph_far))},
            {"model": model_name, "metric": "transition_distance_scaled_stress", "value": stress},
        ]
    )
    rng = np.random.default_rng(seed)
    goals = rng.choice(Z.shape[0], size=min(num_goals, Z.shape[0]), replace=False)
    goal_spearman = []
    goal_false = []
    for goal_idx in goals:
        d_graph_goal = D_graph[:, goal_idx]
        d_lat_goal = D_lat[:, goal_idx]
        mask = np.arange(Z.shape[0]) != goal_idx
        graph_far_goal = d_graph_goal[mask] >= np.quantile(d_graph_goal[mask], q_far)
        lat_near_goal = d_lat_goal[mask] <= np.quantile(d_lat_goal[mask], q_near)
        goal_spearman.append(_spearman(d_lat_goal[mask], d_graph_goal[mask]))
        goal_false.append(float(np.mean(lat_near_goal[graph_far_goal])) if np.any(graph_far_goal) else float("nan"))
    rows.extend(
        [
            {"model": model_name, "metric": "goal_relative_spearman", "value": float(np.nanmean(goal_spearman))},
            {"model": model_name, "metric": "goal_relative_false_shortcut_rate", "value": float(np.nanmean(goal_false))},
        ]
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure learned latent metric fidelity to graph/reachability distances.")
    parser.add_argument("--checkpoint_object", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph_distance_cache", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--q_far", type=float, default=0.75)
    parser.add_argument("--q_near", type=float, default=0.05)
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--num_goals", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cache = np.load(args.graph_distance_cache)
    sampled_rows = np.asarray(cache["sampled_rows"], dtype=np.int64)
    landmarks = np.asarray(cache["landmarks"], dtype=np.int64)
    rows = sampled_rows[landmarks]
    D_graph = np.asarray(cache["D_landmark"], dtype=np.float64)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _load_model(Path(args.checkpoint_object), device)
    Z = _encode_rows(model, Path(args.dataset), rows, device, args.batch_size, args.img_size)
    metric_rows = _metric_rows(args.model_name, Z, D_graph, args.q_far, args.q_near, args.k_neighbors, args.num_goals, args.seed)
    _write_csv(Path(args.output_csv), metric_rows)
    if args.summary_csv:
        summary_row = {"model": args.model_name, "latent_dim": int(Z.shape[-1])}
        for row in metric_rows:
            summary_row[row["metric"]] = row["value"]
        existing = []
        path = Path(args.summary_csv)
        if path.exists():
            with path.open() as file:
                existing = list(csv.DictReader(file))
            existing = [row for row in existing if row.get("model") != args.model_name]
        existing.append(summary_row)
        _write_csv(path, existing)
    print(f"[trained_metric] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
