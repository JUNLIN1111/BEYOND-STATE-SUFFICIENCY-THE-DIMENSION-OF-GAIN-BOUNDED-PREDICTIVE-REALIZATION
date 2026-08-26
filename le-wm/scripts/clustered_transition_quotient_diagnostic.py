from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import hdf5plugin  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False

import h5py
import numpy as np

EPS = 1e-12


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _parse_ints(text: str) -> List[int]:
    return [int(item) for item in str(text).replace(",", " ").split() if item.strip()]


def _find_key(h5: h5py.File, candidates: Iterable[str]) -> str:
    for key in candidates:
        if key and key in h5:
            return key
    raise KeyError(f"Could not find any key among {list(candidates)}")


def _standardize(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = states.mean(axis=0, keepdims=True)
    std = states.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (states - mean) / std, mean, std


def _load_dataset(dataset: Path, state_key: str, episode_key: str, step_key: str):
    print(
        "[quotient] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[quotient] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    with h5py.File(dataset, "r") as h5:
        print("[quotient] dataset keys:")
        for key in h5.keys():
            print(f"  - {key}: {getattr(h5[key], 'shape', None)}")
        state_key = _find_key(h5, [state_key, "state", "states", "proprio"])
        episode_key = _find_key(h5, [episode_key, "episode_idx", "ep_idx", "traj_idx"])
        step_key = _find_key(h5, [step_key, "step_idx", "timestep", "step", "time"])
        states = np.asarray(h5[state_key], dtype=np.float64)
        episode_idx = np.asarray(h5[episode_key]).reshape(-1)
        step_idx = np.asarray(h5[step_key]).reshape(-1)
    if states.shape[0] != episode_idx.shape[0] or states.shape[0] != step_idx.shape[0]:
        raise ValueError("state, episode, and step arrays must have matching first dimensions.")
    return states, episode_idx, step_idx, {"state_key": state_key, "episode_key": episode_key, "step_key": step_key}


def _temporal_transition_rows(episode_idx: np.ndarray, step_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((step_idx, episode_idx))
    src = order[:-1]
    dst = order[1:]
    valid = (episode_idx[src] == episode_idx[dst]) & (step_idx[dst] == step_idx[src] + 1)
    return src[valid].astype(np.int64), dst[valid].astype(np.int64)


def _fit_kmeans(X: np.ndarray, n_clusters: int, seed: int, max_fit_points: int, batch_size: int, max_iter: int):
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError(
            "clustering_method=kmeans requires scikit-learn in this script. "
            "Install sklearn or use --clustering_method radius for a scipy-only fallback."
        ) from exc
    rng = np.random.default_rng(seed)
    fit_n = min(int(max_fit_points), X.shape[0]) if max_fit_points > 0 else X.shape[0]
    fit_idx = rng.choice(X.shape[0], size=fit_n, replace=False) if fit_n < X.shape[0] else np.arange(X.shape[0])
    kwargs = {
        "n_clusters": int(n_clusters),
        "random_state": int(seed),
        "batch_size": int(batch_size),
        "max_iter": int(max_iter),
        "reassignment_ratio": 0.01,
    }
    try:
        model = MiniBatchKMeans(n_init="auto", **kwargs)
    except TypeError:
        model = MiniBatchKMeans(n_init=3, **kwargs)
    print(f"[quotient] fitting MiniBatchKMeans: M={n_clusters}, fit_points={fit_n}", flush=True)
    model.fit(X[fit_idx])
    return np.asarray(model.cluster_centers_, dtype=np.float64), model


def _fit_radius_centers(X: np.ndarray, target_clusters: int, seed: int, max_fit_points: int, radius_epsilon: float):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("clustering_method=radius requires scipy.") from exc
    rng = np.random.default_rng(seed)
    fit_n = min(int(max_fit_points), X.shape[0]) if max_fit_points > 0 else X.shape[0]
    fit_idx = rng.choice(X.shape[0], size=fit_n, replace=False) if fit_n < X.shape[0] else np.arange(X.shape[0])
    sample = X[fit_idx]
    tree = cKDTree(sample)
    uncovered = np.ones(sample.shape[0], dtype=bool)
    centers = []
    order = rng.permutation(sample.shape[0])
    for idx in order:
        if not uncovered[idx]:
            continue
        centers.append(sample[idx])
        if len(centers) >= target_clusters:
            break
        nearby = tree.query_ball_point(sample[idx], r=radius_epsilon)
        uncovered[np.asarray(nearby, dtype=np.int64)] = False
    if len(centers) < target_clusters:
        remaining = np.where(uncovered)[0]
        take = min(target_clusters - len(centers), remaining.size)
        if take > 0:
            centers.extend(sample[rng.choice(remaining, size=take, replace=False)])
    print(f"[quotient] radius centers: requested={target_clusters}, actual={len(centers)}, radius={radius_epsilon}", flush=True)
    return np.asarray(centers, dtype=np.float64)


def _fit_kcenter_centers(X: np.ndarray, target_clusters: int, seed: int, max_fit_points: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fit_n = min(int(max_fit_points), X.shape[0]) if max_fit_points > 0 else X.shape[0]
    fit_idx = rng.choice(X.shape[0], size=fit_n, replace=False) if fit_n < X.shape[0] else np.arange(X.shape[0])
    sample = X[fit_idx]
    if target_clusters <= 0:
        raise ValueError("--num_clusters must be positive for kcenter clustering.")
    n_centers = min(int(target_clusters), sample.shape[0])
    centers = np.empty((n_centers, X.shape[1]), dtype=np.float64)
    first = int(rng.integers(sample.shape[0]))
    centers[0] = sample[first]
    min_sq_dist = np.sum(np.square(sample - centers[0]), axis=1)
    for idx in range(1, n_centers):
        farthest = int(np.argmax(min_sq_dist))
        centers[idx] = sample[farthest]
        candidate_sq_dist = np.sum(np.square(sample - centers[idx]), axis=1)
        min_sq_dist = np.minimum(min_sq_dist, candidate_sq_dist)
        if idx % 256 == 0:
            print(f"[quotient] k-center selected {idx}/{n_centers} centers", flush=True)
    print(f"[quotient] k-center centers: requested={target_clusters}, actual={n_centers}, fit_points={fit_n}", flush=True)
    return centers


def _assign_to_centers(X: np.ndarray, centers: np.ndarray, batch_size: int) -> np.ndarray:
    assignments = np.empty(X.shape[0], dtype=np.int32)
    center_norm = np.sum(np.square(centers), axis=1)
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        batch = X[start:end]
        distances = np.sum(np.square(batch), axis=1, keepdims=True) + center_norm[None, :] - 2.0 * batch @ centers.T
        assignments[start:end] = np.argmin(distances, axis=1).astype(np.int32)
    return assignments


def _cluster_states(
    X: np.ndarray,
    n_clusters: int,
    method: str,
    seed: int,
    max_fit_points: int,
    batch_size: int,
    max_iter: int,
    radius_epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "kmeans":
        centers, model = _fit_kmeans(X, n_clusters, seed, max_fit_points, batch_size, max_iter)
        assignments = np.empty(X.shape[0], dtype=np.int32)
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            assignments[start:end] = model.predict(X[start:end]).astype(np.int32)
        centers = _recompute_centers(X, assignments, centers.shape[0], centers)
        return centers, assignments
    if method in ("radius", "mutual_knn"):
        if method == "mutual_knn":
            print("[quotient] warning: mutual_knn currently uses conservative radius-cover clustering; no graph edges are added from kNN.", flush=True)
        centers = _fit_radius_centers(X, n_clusters, seed, max_fit_points, radius_epsilon)
        assignments = _assign_to_centers(X, centers, batch_size)
        centers = _recompute_centers(X, assignments, centers.shape[0], centers)
        return centers, assignments
    if method == "kcenter":
        centers = _fit_kcenter_centers(X, n_clusters, seed, max_fit_points)
        assignments = _assign_to_centers(X, centers, batch_size)
        centers = _recompute_centers(X, assignments, centers.shape[0], centers)
        return centers, assignments
    raise ValueError(f"Unknown clustering method: {method}")


def _recompute_centers(X: np.ndarray, assignments: np.ndarray, n_clusters: int, fallback_centers: np.ndarray) -> np.ndarray:
    centers = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    counts = np.bincount(assignments, minlength=n_clusters).astype(np.float64)
    for dim in range(X.shape[1]):
        centers[:, dim] = np.bincount(assignments, weights=X[:, dim], minlength=n_clusters)
    nonempty = counts > 0
    centers[nonempty] /= counts[nonempty, None]
    centers[~nonempty] = fallback_centers[~nonempty]
    return centers


def _build_quotient_graph(assignments: np.ndarray, src_rows: np.ndarray, dst_rows: np.ndarray, n_clusters: int):
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError("scipy is required for quotient graph construction.") from exc
    src_clusters = assignments[src_rows]
    dst_clusters = assignments[dst_rows]
    valid = src_clusters != dst_clusters
    src_clusters = src_clusters[valid].astype(np.int64)
    dst_clusters = dst_clusters[valid].astype(np.int64)
    if src_clusters.size == 0:
        return coo_matrix((n_clusters, n_clusters)).tocsr(), 0
    rows = np.concatenate([src_clusters, dst_clusters])
    cols = np.concatenate([dst_clusters, src_clusters])
    data = np.ones(rows.shape[0], dtype=np.float64)
    graph = coo_matrix((data, (rows, cols)), shape=(n_clusters, n_clusters)).tocsr()
    graph.data[:] = 1.0
    graph.eliminate_zeros()
    return graph, int(src_clusters.size)


def _cluster_radius_stats(X: np.ndarray, assignments: np.ndarray, centers: np.ndarray) -> Dict[str, object]:
    counts = np.bincount(assignments, minlength=centers.shape[0]).astype(np.int64)
    point_radius = np.linalg.norm(X - centers[assignments], axis=1)
    max_radius = np.zeros(centers.shape[0], dtype=np.float64)
    sum_radius = np.zeros(centers.shape[0], dtype=np.float64)
    np.maximum.at(max_radius, assignments, point_radius)
    np.add.at(sum_radius, assignments, point_radius)
    nonempty = counts > 0
    cluster_max_radius = max_radius[nonempty]
    cluster_mean_radius = sum_radius[nonempty] / np.maximum(counts[nonempty], 1)
    cluster_sizes = counts[nonempty]
    if not np.any(nonempty):
        return {
            "cluster_size_min": 0,
            "cluster_size_median": 0.0,
            "cluster_size_mean": 0.0,
            "cluster_size_p90": 0.0,
            "cluster_size_max": 0,
            "cluster_radius_mean": float("nan"),
            "cluster_radius_median": float("nan"),
            "cluster_radius_p90": float("nan"),
            "cluster_radius_max": float("nan"),
            "cluster_mean_point_radius_median": float("nan"),
        }
    return {
        "cluster_size_min": int(np.min(cluster_sizes)),
        "cluster_size_median": float(np.median(cluster_sizes)),
        "cluster_size_mean": float(np.mean(cluster_sizes)),
        "cluster_size_p90": float(np.quantile(cluster_sizes, 0.90)),
        "cluster_size_max": int(np.max(cluster_sizes)),
        "cluster_radius_mean": float(np.mean(cluster_max_radius)),
        "cluster_radius_median": float(np.median(cluster_max_radius)),
        "cluster_radius_p90": float(np.quantile(cluster_max_radius, 0.90)),
        "cluster_radius_max": float(np.max(cluster_max_radius)),
        "cluster_mean_point_radius_median": float(np.median(cluster_mean_radius)),
    }


def _graph_degree_stats(graph) -> Dict[str, object]:
    degrees = np.diff(graph.indptr).astype(np.int64)
    if degrees.size == 0:
        return {
            "num_quotient_edges": 0,
            "average_degree": 0.0,
            "degree_median": 0.0,
            "degree_p90": 0.0,
            "degree_max": 0,
        }
    return {
        "num_quotient_edges": int(graph.nnz // 2),
        "average_degree": float(np.mean(degrees)),
        "degree_median": float(np.median(degrees)),
        "degree_p90": float(np.quantile(degrees, 0.90)),
        "degree_max": int(np.max(degrees)),
    }


def _histogram(values: np.ndarray, bins: int = 30) -> Dict[str, List[float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.astype(int).tolist(), "bin_edges": edges.astype(float).tolist()}


def _largest_component(graph):
    try:
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:
        raise ImportError("scipy is required for connected components.") from exc
    n_components, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=n_components)
    largest = int(np.argmax(counts)) if counts.size else 0
    keep = np.where(labels == largest)[0].astype(np.int64)
    return keep, {"num_connected_components": int(n_components), "largest_connected_component_size": int(keep.size)}


def _shortest_paths(graph):
    try:
        from scipy.sparse.csgraph import shortest_path
    except ImportError as exc:
        raise ImportError("scipy is required for shortest paths.") from exc
    return shortest_path(graph, directed=False, unweighted=False)


def _classical_mds_spectrum(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ np.square(D) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def _spectrum_summary(evals: np.ndarray) -> Dict[str, object]:
    evals = np.asarray(evals, dtype=np.float64)
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = evals[evals > tol]
    neg = evals[evals < -tol]
    pos_energy = float(np.sum(pos))
    neg_abs = float(np.sum(np.abs(neg)))
    abs_all = float(np.sum(np.abs(evals)))
    cum = np.cumsum(pos) / max(pos_energy, EPS) if pos.size else np.asarray([])
    out: Dict[str, object] = {
        "positive_eigen_count": int(pos.size),
        "negative_eigen_count": int(neg.size),
        "positive_energy": pos_energy,
        "negative_abs_energy": neg_abs,
        "negative_energy_ratio": float(neg_abs / max(abs_all, EPS)),
        "positive_eigenvalues": [float(v) for v in pos.tolist()],
    }
    for label, q in [("d50", 0.50), ("d80", 0.80), ("d90", 0.90), ("d95", 0.95), ("d99", 0.99)]:
        out[label] = int(np.searchsorted(cum, q) + 1) if pos.size else 0
    out["d100"] = int(pos.size)
    return out


def _shortcut_candidates(centers: np.ndarray, D_graph: np.ndarray, epsilon_near: float) -> Dict[str, object]:
    left, right = np.triu_indices(D_graph.shape[0], k=1)
    graph_dist = D_graph[left, right]
    center_dist = np.linalg.norm(centers[left] - centers[right], axis=1)
    physical_thr = float(epsilon_near) if math.isfinite(epsilon_near) and epsilon_near > 0 else float(np.quantile(center_dist, 0.05))
    graph_thr = float(np.quantile(graph_dist, 0.75))
    physical_near = center_dist <= physical_thr
    transition_far = graph_dist >= graph_thr
    near_far = physical_near & transition_far
    return {
        "physical_near_threshold": physical_thr,
        "transition_far_threshold": graph_thr,
        "p_physical_near_given_transition_far": float(np.mean(physical_near[transition_far])) if np.any(transition_far) else float("nan"),
        "p_transition_far_given_physical_near": float(np.mean(transition_far[physical_near])) if np.any(physical_near) else float("nan"),
        "near_far_pair_count": int(np.sum(near_far)),
        "physical_near_pair_count": int(np.sum(physical_near)),
        "transition_far_pair_count": int(np.sum(transition_far)),
        "pair_count": int(graph_dist.size),
    }


def _shortest_path_stats(D: np.ndarray) -> Dict[str, object]:
    values = D[np.isfinite(D) & (D > 0)]
    if values.size == 0:
        return {
            "shortest_path_mean": float("nan"),
            "shortest_path_median": float("nan"),
            "shortest_path_p90": float("nan"),
            "shortest_path_max": float("nan"),
        }
    return {
        "shortest_path_mean": float(np.mean(values)),
        "shortest_path_median": float(np.median(values)),
        "shortest_path_p90": float(np.quantile(values, 0.90)),
        "shortest_path_max": float(np.max(values)),
    }


def _plot_outputs(rows: List[Dict[str, object]], output_dir: Path, previous_rows: List[Dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[quotient] matplotlib unavailable; skipping plots.", flush=True)
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["num_clusters"]))
    ms = np.asarray([int(row["num_clusters"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="white")
    for key, label in [("d90", "d90"), ("d95", "d95"), ("d99", "d99")]:
        ax.plot(ms, [float(row[key]) for row in rows], marker="o", label=f"clustered quotient {label}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("D_plan(q)")
    ax.set_title("Clustered quotient transition graph")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "clustered_quotient_d90_d95_d99_vs_clusters.png", dpi=260)
    fig.savefig(plots_dir / "clustered_quotient_d90_d95_d99_vs_clusters.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="white")
    for row in rows:
        eig = np.asarray(row.get("positive_eigenvalues", []), dtype=np.float64)
        if eig.size:
            ax.semilogy(np.arange(1, eig.size + 1), eig, label=f"M={row['num_clusters']}, d90={row['d90']}")
    ax.set_xlabel("MDS component")
    ax.set_ylabel("positive eigenvalue")
    ax.set_title("Clustered quotient positive spectra")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "clustered_quotient_positive_spectrum_by_M.png", dpi=260)
    fig.savefig(plots_dir / "clustered_quotient_positive_spectrum_by_M.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="white")
    ax.plot(ms, [float(row["p_physical_near_given_transition_far"]) for row in rows], marker="o", label="P(physical-near | transition-far)")
    ax.plot(ms, [float(row["p_transition_far_given_physical_near"]) for row in rows], marker="s", label="P(transition-far | physical-near)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("empirical shortcut-candidate rate")
    ax.set_title("Physical-near / transition-far candidates")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "clustered_quotient_shortcut_candidates_vs_clusters.png", dpi=260)
    fig.savefig(plots_dir / "clustered_quotient_shortcut_candidates_vs_clusters.pdf")
    plt.close(fig)

    if previous_rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor="white")
        ax.plot(ms, [float(row["d90"]) for row in rows], marker="o", label="clustered quotient d90")
        ax.plot(ms, [float(row["d95"]) for row in rows], marker="o", label="clustered quotient d95")
        for prev in previous_rows:
            mode = prev.get("graph_mode", "")
            if mode not in {"temporal_plus_knn_k10", "knn_only_k10", "temporal_only"}:
                continue
            d90 = _safe_float(prev.get("d90"))
            if math.isfinite(d90):
                style = "--" if mode != "knn_only_k10" else ":"
                ax.axhline(d90, linestyle=style, linewidth=1.2, label=f"{mode} d90={d90:g}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of quotient clusters")
        ax.set_ylabel("D_plan")
        ax.set_title("Quotient graph vs previous graph diagnostics")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / "clustered_quotient_comparison_previous_graphs.png", dpi=260)
        fig.savefig(plots_dir / "clustered_quotient_comparison_previous_graphs.pdf")
        plt.close(fig)


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_summary(output_dir: Path, rows: List[Dict[str, object]], previous_rows: List[Dict[str, str]], meta: Dict[str, object]) -> None:
    lines = [
        "# Clustered Transition-Quotient Diagnostic",
        "",
        "This diagnostic separates state similarity from transition reachability.",
        "",
        "- Similarity is used only to aggregate normalized physical states into clusters.",
        "- Quotient graph edges are added only when an observed temporal transition crosses between two clusters.",
        "- No kNN edge is treated as a feasible transition edge.",
        "",
        "This makes the diagnostic conceptually cleaner than temporal+KNN: it avoids treating physical/state-space proximity as reachability, which is exactly the false-shortcut failure mode studied in the paper.",
        "",
        f"- Dataset: `{meta['dataset']}`",
        f"- State key: `{meta['state_key']}`",
        f"- Episode key: `{meta['episode_key']}`",
        f"- Step key: `{meta['step_key']}`",
        f"- Clustering method: `{meta['clustering_method']}`",
        "",
        "| clusters | LCC | components | d90 | d95 | d99 | shortcut P(near|far) | shortcut P(far|near) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: int(item["num_clusters"])):
        lines.append(
            f"| {row['num_clusters']} | {row['largest_connected_component_size']} | {row['num_connected_components']} | "
            f"{row['d90']} | {row['d95']} | {row['d99']} | "
            f"{float(row['p_physical_near_given_transition_far']):.4f} | {float(row['p_transition_far_given_physical_near']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Comparison notes",
            "",
            "`temporal_only` can be disconnected across offline trajectories. `temporal_plus_knn_k10` and `knn_only_k10` are useful ablations, but they allow state similarity to create graph edges. The clustered quotient graph uses state similarity only for conservative aggregation; observed temporal transitions define all graph edges.",
            "",
            "The physical-near / transition-far quantities are empirical shortcut candidates, not proofs of impossibility: absence of an observed transition can reflect offline coverage.",
        ]
    )
    if previous_rows:
        lines.extend(["", "## Previous graph-method reference", "", "| graph mode | d90 | d95 | d99 | LCC |", "|---|---:|---:|---:|---:|"])
        for prev in previous_rows:
            if prev.get("graph_mode") in {"temporal_only", "temporal_plus_knn_k10", "knn_only_k10"}:
                lines.append(
                    f"| {prev.get('graph_mode')} | {prev.get('d90')} | {prev.get('d95')} | {prev.get('d99')} | {prev.get('largest_connected_component_size', prev.get('largest_component_size', ''))} |"
                )
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "clustered_quotient_diagnostic.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustered transition-quotient plannable-dimension diagnostic.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--step_key", default="step_idx")
    parser.add_argument("--num_clusters", default="256,512,1024,2048")
    parser.add_argument("--num_landmarks", default="", help="Alias for --num_clusters.")
    parser.add_argument("--clustering_method", choices=["kmeans", "radius", "mutual_knn", "kcenter"], default="kmeans")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_fit_points", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--kmeans_max_iter", type=int, default=100)
    parser.add_argument("--radius_epsilon", type=float, default=0.25)
    parser.add_argument("--epsilon_near", type=float, default=float("nan"))
    parser.add_argument("--previous_graph_csv", default="rollout_results/plannable_dim_evidence/spectra/pusht_graph_sensitivity.csv")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    states, episode_idx, step_idx, meta = _load_dataset(Path(args.dataset), args.state_key, args.episode_key, args.step_key)
    X, mean, std = _standardize(states)
    src_rows, dst_rows = _temporal_transition_rows(episode_idx, step_idx)
    print(f"[quotient] states={X.shape}, temporal_transitions={src_rows.size}", flush=True)
    cluster_values = _parse_ints(args.num_landmarks or args.num_clusters)
    rows: List[Dict[str, object]] = []
    spectra_payload: Dict[str, object] = {}

    for n_clusters in cluster_values:
        print(f"[quotient] processing M={n_clusters}", flush=True)
        centers, assignments = _cluster_states(
            X,
            n_clusters,
            args.clustering_method,
            args.seed,
            args.max_fit_points,
            args.batch_size,
            args.kmeans_max_iter,
            args.radius_epsilon,
        )
        actual_clusters = centers.shape[0]
        graph, observed_cross_edges = _build_quotient_graph(assignments, src_rows, dst_rows, actual_clusters)
        cluster_stats = _cluster_radius_stats(X, assignments, centers)
        degree_stats = _graph_degree_stats(graph)
        keep, component = _largest_component(graph)
        if keep.size < 2:
            print(f"[quotient] warning: M={n_clusters} LCC too small; skipping.", flush=True)
            continue
        graph_lcc = graph[keep][:, keep]
        centers_lcc = centers[keep]
        D = _shortest_paths(graph_lcc)
        D_raw = D.copy()
        finite = np.isfinite(D)
        if not np.all(finite):
            max_finite = float(np.max(D[finite])) if np.any(finite) else 1.0
            D = D.copy()
            D[~finite] = 2.0 * max_finite
        finite_positive = D[np.isfinite(D) & (D > 0)]
        median_distance = float(np.median(finite_positive)) if finite_positive.size else 1.0
        D_norm = D / max(median_distance, EPS)
        evals, _evecs = _classical_mds_spectrum(D_norm)
        spectrum = _spectrum_summary(evals)
        shortcuts = _shortcut_candidates(centers_lcc, D_norm, args.epsilon_near)
        shortest_path_stats = _shortest_path_stats(D_raw)
        counts = np.bincount(assignments, minlength=actual_clusters)
        row = {
            "graph_mode": f"clustered_quotient_M{actual_clusters}",
            "num_clusters": int(actual_clusters),
            "requested_clusters": int(n_clusters),
            "clustering_method": args.clustering_method,
            "random_seed": int(args.seed),
            "num_states": int(X.shape[0]),
            "num_temporal_transitions": int(src_rows.size),
            "observed_cross_cluster_temporal_edges": int(observed_cross_edges),
            "num_connected_components": component["num_connected_components"],
            "largest_connected_component_size": component["largest_connected_component_size"],
            "largest_connected_component_fraction": float(component["largest_connected_component_size"] / max(actual_clusters, 1)),
            "nonempty_cluster_count": int(np.sum(counts > 0)),
            "median_shortest_path_before_normalization": median_distance,
            "radius_epsilon": float(args.radius_epsilon) if args.clustering_method in {"radius", "mutual_knn"} else float("nan"),
            **cluster_stats,
            **degree_stats,
            **shortest_path_stats,
            **{key: value for key, value in spectrum.items() if key != "positive_eigenvalues"},
            **shortcuts,
        }
        rows.append(row)
        spectra_payload[str(actual_clusters)] = {
            **row,
            "positive_eigenvalues": spectrum["positive_eigenvalues"],
            "degree_histogram": _histogram(np.diff(graph_lcc.indptr)),
            "shortest_path_histogram": _histogram(D_raw[np.isfinite(D_raw) & (D_raw > 0)]),
        }
        cluster_path = output_dir / "spectra" / f"clustered_quotient_M{actual_clusters}_clusters.npz"
        cluster_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cluster_path,
            centers=centers.astype(np.float32),
            assignments=assignments.astype(np.int32),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            cluster_counts=counts.astype(np.int64),
        )

    previous_rows = _read_csv(Path(args.previous_graph_csv))
    _write_csv(output_dir / "spectra" / "clustered_quotient_diagnostic.csv", rows)
    with (output_dir / "spectra" / "clustered_quotient_spectra.json").open("w") as file:
        json.dump(spectra_payload, file, indent=2)
    _plot_outputs(rows, output_dir, previous_rows)
    _write_summary(
        output_dir,
        rows,
        previous_rows,
        {
            "dataset": args.dataset,
            "clustering_method": args.clustering_method,
            **meta,
        },
    )
    print(f"[quotient] wrote {output_dir / 'spectra' / 'clustered_quotient_diagnostic.csv'}")
    print(f"[quotient] wrote {output_dir / 'summaries' / 'clustered_quotient_diagnostic.md'}")


if __name__ == "__main__":
    main()
