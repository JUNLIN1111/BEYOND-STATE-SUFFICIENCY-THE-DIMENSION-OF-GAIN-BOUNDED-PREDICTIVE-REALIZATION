from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from candidate_future_transition_metric import (  # noqa: E402
    _load_goal_states,
    _nearest_landmarks,
    _window_rows,
    _write_csv,
)


def _parse_csv_ints(text: str) -> List[int]:
    return [int(item) for item in text.replace(",", " ").split() if item]


def _parse_csv_text(text: str) -> List[str]:
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def _fit_projection(X: np.ndarray, dim: int, method: str, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    full_dim = X.shape[-1]
    dim = min(dim, full_dim)
    if method == "identity":
        return np.eye(full_dim, dim), {"projection_method": method}
    if method == "random_gaussian":
        return rng.normal(0.0, 1.0 / np.sqrt(dim), size=(full_dim, dim)), {"projection_method": method}
    if method == "random_orthogonal":
        mat = rng.normal(size=(full_dim, full_dim))
        q, _r = np.linalg.qr(mat)
        return q[:, :dim], {"projection_method": method}
    if method in ("pca", "pca_whitened"):
        Xc = X - X.mean(axis=0, keepdims=True)
        _u, s, vh = np.linalg.svd(Xc, full_matrices=False)
        basis = vh[:dim].T
        if method == "pca_whitened":
            scale = np.sqrt(max(X.shape[0] - 1, 1)) / np.maximum(s[:dim], 1e-8)
            basis = basis * scale[None, :]
        return basis, {"projection_method": method, "retained_variance": float(np.sum(s[:dim] ** 2) / max(np.sum(s ** 2), 1e-12))}
    raise ValueError(f"Unknown projection method: {method}")


def _project_latents(terminal_latents: np.ndarray, goal_latents: np.ndarray, projection: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = terminal_latents.reshape(-1, terminal_latents.shape[-1])
    center = np.concatenate([flat, goal_latents], axis=0).mean(axis=0, keepdims=True)
    terminal_proj = ((flat - center) @ projection).reshape(terminal_latents.shape[0], terminal_latents.shape[1], -1)
    goal_proj = (goal_latents - center) @ projection
    return terminal_proj, goal_proj


def _save_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, int, str], List[float]] = {}
    for row in rows:
        key = (str(row["projection_method"]), int(row["effective_dim"]), str(row["metric"]))
        grouped.setdefault(key, []).append(float(row["value"]))
    out = []
    for (method, dim, metric), values in sorted(grouped.items()):
        out.append({"projection_method": method, "effective_dim": dim, "metric": metric, "value": float(np.nanmean(values))})
    _write_csv(path, out)


def _plot_summary(summary_csv: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not summary_csv.exists():
        return
    rows: List[Dict[str, str]]
    with summary_csv.open() as file:
        rows = list(csv.DictReader(file))
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in [
        ("candidate_false_shortcut_rate", "P[latent-near | graph-far]", "compression_false_shortcut_vs_dim.png"),
        ("candidate_goal_metric_spearman", "Spearman(score, graph distance)", "compression_graph_spearman_vs_dim.png"),
        ("candidate_pairwise_rank_acc", "Pairwise rank accuracy", "compression_pairwise_rank_acc_vs_dim.png"),
    ]:
        plt.figure(figsize=(7, 4.8))
        methods = sorted({row["projection_method"] for row in rows if row["metric"] == metric})
        for method in methods:
            selected = [row for row in rows if row["metric"] == metric and row["projection_method"] == method]
            selected.sort(key=lambda row: int(row["effective_dim"]))
            plt.plot([int(row["effective_dim"]) for row in selected], [float(row["value"]) for row in selected], marker="o", label=method)
        plt.xlabel("effective dimension")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc compression metric-fidelity diagnostic for candidate futures.")
    parser.add_argument("--raw_npz", required=True)
    parser.add_argument("--graph_distance_cache", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--dims", default="8,16,32,64,96,128,163,192")
    parser.add_argument("--projection_methods", default="pca,random_gaussian,random_orthogonal,pca_whitened")
    parser.add_argument("--num_random_seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q_far", type=float, default=0.75)
    parser.add_argument("--q_near", type=float, default=0.05)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--plot_dir", default="")
    args = parser.parse_args()

    raw = np.load(args.raw_npz)
    cache = np.load(args.graph_distance_cache)
    terminal_states = np.asarray(raw["terminal_states"], dtype=np.float64)
    terminal_latents = np.asarray(raw["terminal_latents"], dtype=np.float64)
    goal_latents = np.asarray(raw["goal_latents"], dtype=np.float64)
    goal_states = _load_goal_states(raw, args.dataset, args.state_key)

    X_landmark = np.asarray(cache["X_landmark"], dtype=np.float64)
    D_graph = np.asarray(cache["D_landmark"], dtype=np.float64)
    mean = np.asarray(cache["feature_mean"], dtype=np.float64)
    std = np.asarray(cache["feature_std"], dtype=np.float64)
    terminal_lm = _nearest_landmarks(terminal_states, X_landmark, mean, std)
    goal_lm = _nearest_landmarks(goal_states, X_landmark, mean, std).reshape(-1)

    dims = _parse_csv_ints(args.dims)
    methods = _parse_csv_text(args.projection_methods)
    X_fit = np.concatenate([terminal_latents.reshape(-1, terminal_latents.shape[-1]), goal_latents], axis=0)
    rows: List[Dict[str, object]] = []
    for method in methods:
        seed_count = args.num_random_seeds if method.startswith("random") else 1
        for seed_idx in range(seed_count):
            rng = np.random.default_rng(args.seed + seed_idx)
            for dim in dims:
                projection, meta = _fit_projection(X_fit, dim, method, rng)
                terminal_proj, goal_proj = _project_latents(terminal_latents, goal_latents, projection)
                for window_idx in range(terminal_states.shape[0]):
                    score = np.sum((terminal_proj[window_idx] - goal_proj[window_idx][None, :]) ** 2, axis=-1)
                    graph_cost = D_graph[terminal_lm[window_idx], goal_lm[window_idx]]
                    for row in _window_rows("baseline192_compressed", window_idx, graph_cost, score, args.q_far, args.q_near):
                        row.update(meta)
                        row["effective_dim"] = int(projection.shape[-1])
                        row["projection_seed"] = seed_idx
                        rows.append(row)

    output_csv = Path(args.output_csv)
    _write_csv(output_csv, rows)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_csv.with_name(output_csv.stem + "_summary.csv")
    _save_summary(summary_csv, rows)
    if args.plot_dir:
        _plot_summary(summary_csv, Path(args.plot_dir))
    print(f"[compression_metric] wrote {output_csv}")
    print(f"[compression_metric] wrote {summary_csv}")


if __name__ == "__main__":
    main()
