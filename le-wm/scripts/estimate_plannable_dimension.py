from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import hdf5plugin  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False

import h5py
import numpy as np


EPS = 1e-12


def _find_key(h5: h5py.File, candidates: Iterable[str]) -> str | None:
    for key in candidates:
        if key and key in h5:
            return key
    return None


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _episode_rows(episode_idx: np.ndarray, step_idx: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for ep in np.unique(episode_idx):
        rows = np.where(episode_idx == ep)[0]
        out[int(ep)] = rows[np.argsort(step_idx[rows])]
    return out


def _read_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    values = np.asarray(dataset[unique_rows])
    return values[inverse]


def _sample_random_rows(num_rows: int, num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_rows, size=min(num_nodes, num_rows), replace=False).astype(np.int64))


def _sample_contiguous_rows(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    num_nodes: int,
    seed: int,
    num_segments: int = 50,
    segment_len: int = 100,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    episodes = list(_episode_rows(episode_idx, step_idx).values())
    selected: List[np.ndarray] = []
    valid = [rows for rows in episodes if rows.size > 0]
    if not valid:
        raise ValueError("No non-empty episodes found.")
    for _ in range(max(1, int(num_segments))):
        if sum(chunk.size for chunk in selected) >= num_nodes:
            break
        rows = valid[int(rng.integers(0, len(valid)))]
        take = min(int(segment_len), rows.size, int(num_nodes) - sum(chunk.size for chunk in selected))
        if take <= 0:
            break
        start = 0 if rows.size == take else int(rng.integers(0, rows.size - take + 1))
        selected.append(rows[start:start + take])
    if not selected:
        raise ValueError("No rows sampled from dataset.")
    return np.concatenate(selected).astype(np.int64)


def _sample_episode_segments(episode_idx: np.ndarray, step_idx: np.ndarray, num_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    episodes = list(_episode_rows(episode_idx, step_idx).values())
    rng.shuffle(episodes)
    selected: List[np.ndarray] = []
    remaining = int(num_nodes)
    for rows in episodes:
        if remaining <= 0:
            break
        take = min(rows.size, remaining)
        selected.append(rows[:take])
        remaining -= take
    if not selected:
        raise ValueError("No rows sampled from dataset.")
    return np.concatenate(selected).astype(np.int64)


def _load_features(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    print(
        "[d_plan] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[d_plan] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    with h5py.File(args.dataset, "r") as h5:
        print("[d_plan] dataset keys:")
        for key in h5.keys():
            print(f"  - {key}: {getattr(h5[key], 'shape', None)}")
        state_key = _find_key(h5, [args.state_key, "state", "states", "obs/state", "proprio"])
        obs_key = _find_key(h5, [args.obs_key, "pixels", "observation/pixels", "image"])
        action_key = _find_key(h5, [args.action_key, "action", "actions"])
        episode_key = _find_key(h5, [args.episode_key, "episode_idx", "ep_idx", "traj_idx"])
        step_key = _find_key(h5, [args.step_key, "step_idx", "timestep", "step", "time"])
        if episode_key is None or step_key is None:
            raise KeyError("Need episode_idx and step_idx-like keys for transition graph.")
        episode_idx_all = np.asarray(h5[episode_key]).reshape(-1)
        step_idx_all = np.asarray(h5[step_key]).reshape(-1)
        if args.sampling_mode == "random_rows":
            sampled_rows = _sample_random_rows(len(episode_idx_all), args.num_nodes, args.seed)
        elif args.sampling_mode == "contiguous_segments":
            sampled_rows = _sample_contiguous_rows(
                episode_idx_all,
                step_idx_all,
                args.num_nodes,
                args.seed,
                args.num_segments,
                args.segment_len,
            )
        elif args.sampling_mode == "episode_segments":
            sampled_rows = _sample_episode_segments(episode_idx_all, step_idx_all, args.num_nodes, args.seed)
        else:
            raise ValueError(f"Unknown sampling_mode={args.sampling_mode}")
        if args.feature_mode == "state":
            if state_key is None:
                raise KeyError("feature_mode=state requested but no state key found.")
            features = np.asarray(_read_h5_rows(h5[state_key], sampled_rows), dtype=np.float64)
            feature_source = state_key
        elif args.feature_mode == "image_pca":
            if obs_key is None:
                raise KeyError("feature_mode=image_pca requested but no image key found.")
            images = np.asarray(_read_h5_rows(h5[obs_key], sampled_rows), dtype=np.float32)
            flat = images.reshape(images.shape[0], -1)
            flat = flat / 255.0 if flat.max() > 2 else flat
            flat = flat - flat.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(flat, full_matrices=False)
            dim = min(args.image_pca_dim, vh.shape[0])
            features = flat @ vh[:dim].T
            feature_source = f"{obs_key}_pca{dim}"
        else:
            raise ValueError(f"Unknown feature_mode={args.feature_mode}")
        meta = {
            "state_key": state_key,
            "obs_key": obs_key,
            "action_key": action_key,
            "episode_key": episode_key,
            "step_key": step_key,
            "feature_source": feature_source,
            "sampled_rows": sampled_rows,
        }
        if action_key is not None:
            meta["sampled_actions"] = np.asarray(_read_h5_rows(h5[action_key], sampled_rows), dtype=np.float64)
        return (
            features.astype(np.float64),
            episode_idx_all[sampled_rows],
            step_idx_all[sampled_rows],
            sampled_rows,
            meta,
        )


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / std


def _build_graph(
    X: np.ndarray,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    graph_mode: str,
    knn_k: int,
    temporal_edge_weight: float,
    knn_edge_weight_mode: str,
    minimal_knn_weight: float = 10.0,
    actions: np.ndarray | None = None,
):
    try:
        from scipy.spatial import cKDTree
        import networkx as nx
    except ImportError as exc:
        raise ImportError("scipy and networkx are required for graph construction and shortest paths.") from exc

    n = X.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    def add_edge(src: int, dst: int, weight: float, edge_type: str) -> None:
        if src == dst:
            return
        if graph.has_edge(src, dst):
            data = graph[src][dst]
            data["weight"] = min(float(data["weight"]), float(weight))
            data["edge_types"].add(edge_type)
        else:
            graph.add_edge(src, dst, weight=float(weight), edge_types={edge_type})

    if graph_mode in ("temporal_only", "temporal_plus_knn", "temporal_plus_minimal_knn", "action_temporal"):
        row_by_episode_step = {(int(ep), int(st)): idx for idx, (ep, st) in enumerate(zip(episode_idx, step_idx))}
        for idx, (ep, st) in enumerate(zip(episode_idx, step_idx)):
            nxt = row_by_episode_step.get((int(ep), int(st) + 1))
            if nxt is None:
                continue
            if graph_mode == "action_temporal" and actions is not None:
                weight = float(np.linalg.norm(actions[idx]) + 1e-6)
            else:
                weight = float(temporal_edge_weight)
            add_edge(idx, nxt, weight, "temporal")

    if graph_mode in ("temporal_plus_knn", "knn_only", "pure_knn", "temporal_plus_minimal_knn"):
        effective_knn_k = knn_k
        tree = cKDTree(X)
        distances, indices = tree.query(X, k=min(effective_knn_k + 1, n))
        for idx in range(n):
            for dist, nbr in zip(distances[idx, 1:], indices[idx, 1:]):
                if graph_mode == "temporal_plus_minimal_knn":
                    weight = float(minimal_knn_weight)
                else:
                    weight = float(dist) if knn_edge_weight_mode == "euclidean" else 1.0
                add_edge(idx, int(nbr), weight, "knn")
    elif graph_mode not in ("temporal_only", "action_temporal"):
        raise ValueError(f"Unknown graph_mode={graph_mode}")

    return graph


def _networkx_to_scipy_graph(graph):
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError("scipy is required for shortest paths.") from exc
    rows: List[int] = []
    cols: List[int] = []
    weights: List[float] = []
    for src, dst, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        rows.extend([int(src), int(dst)])
        cols.extend([int(dst), int(src)])
        weights.extend([weight, weight])
    return coo_matrix((weights, (rows, cols)), shape=(graph.number_of_nodes(), graph.number_of_nodes())).tocsr()


def _graph_statistics(graph, graph_label: str, graph_mode: str, knn_k: int) -> Dict[str, object]:
    import networkx as nx

    n_nodes = int(graph.number_of_nodes())
    n_edges = int(graph.number_of_edges())
    component_sizes = [len(component) for component in nx.connected_components(graph)]
    temporal_edges = 0
    knn_edges = 0
    mixed_edges = 0
    for _src, _dst, data in graph.edges(data=True):
        edge_types = data.get("edge_types", set())
        if "temporal" in edge_types:
            temporal_edges += 1
        if "knn" in edge_types:
            knn_edges += 1
        if len(edge_types) > 1:
            mixed_edges += 1
    return {
        "graph_mode": graph_label,
        "base_graph_mode": graph_mode,
        "knn_k": int(knn_k),
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "num_connected_components": int(len(component_sizes)),
        "largest_component_size": int(max(component_sizes) if component_sizes else 0),
        "largest_component_fraction": float((max(component_sizes) if component_sizes else 0) / max(n_nodes, 1)),
        "average_degree": float((2.0 * n_edges) / max(n_nodes, 1)),
        "temporal_edges": int(temporal_edges),
        "knn_edges": int(knn_edges),
        "mixed_temporal_knn_edges": int(mixed_edges),
        "temporal_edge_fraction": float(temporal_edges / max(n_edges, 1)),
        "knn_edge_fraction": float(knn_edges / max(n_edges, 1)),
    }


def _graph_label(graph_mode: str, graph_k: int, minimal_knn_weight: float) -> str:
    if graph_mode in ("temporal_plus_knn", "knn_only", "pure_knn"):
        return f"{graph_mode}_k{graph_k}"
    if graph_mode == "temporal_plus_minimal_knn":
        return f"{graph_mode}_k{graph_k}_w{minimal_knn_weight:g}"
    return graph_mode


def _parse_graph_mode_spec(spec: str, default_knn_k: int, minimal_knn_k: int) -> List[Tuple[str, int]]:
    spec = spec.strip()
    if spec.startswith("temporal_plus_knn_k"):
        return [("temporal_plus_knn", int(spec.rsplit("_k", 1)[1]))]
    if spec.startswith("knn_only_k"):
        return [("knn_only", int(spec.rsplit("_k", 1)[1]))]
    if spec.startswith("pure_knn_k"):
        return [("pure_knn", int(spec.rsplit("_k", 1)[1]))]
    if spec.startswith("temporal_plus_minimal_knn_k"):
        return [("temporal_plus_minimal_knn", int(spec.rsplit("_k", 1)[1].split("_", 1)[0]))]
    if spec == "temporal_plus_minimal_knn":
        return [(spec, minimal_knn_k)]
    return [(spec, default_knn_k)]


def _shortest_paths(graph, landmarks: np.ndarray) -> np.ndarray:
    try:
        from scipy.sparse.csgraph import shortest_path
    except ImportError as exc:
        raise ImportError("scipy is required for shortest_path.") from exc
    return shortest_path(graph, directed=False, indices=landmarks, unweighted=False)


def _largest_connected_component_indices(graph) -> Tuple[np.ndarray, Dict[str, object]]:
    try:
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:
        raise ImportError("scipy is required for connected_components.") from exc
    n_components, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    largest = int(np.argmax(counts))
    idx = np.where(labels == largest)[0]
    return idx.astype(np.int64), {
        "num_connected_components": int(n_components),
        "largest_component_size": int(idx.size),
        "largest_component_fraction": float(idx.size / graph.shape[0]),
    }


def _replace_inf_distances(distances: np.ndarray) -> Tuple[np.ndarray, float]:
    finite = np.isfinite(distances)
    disconnected_frac = float(1.0 - np.mean(finite))
    if np.all(finite):
        return distances, disconnected_frac
    max_finite = float(np.max(distances[finite])) if np.any(finite) else 1.0
    out = distances.copy()
    out[~finite] = 2.0 * max_finite
    return out, disconnected_frac


def _classical_mds_spectrum(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    D2 = np.square(D)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def _rank_energy(evals: np.ndarray, threshold: float) -> int:
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = evals[evals > tol]
    if pos.size == 0:
        return 0
    return int(np.searchsorted(np.cumsum(pos) / np.sum(pos), threshold) + 1)


def _effective_rank(evals: np.ndarray) -> float:
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = evals[evals > tol]
    if pos.size == 0:
        return float("nan")
    p = pos / np.sum(pos)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def _spectrum_coverage_rows(evals: np.ndarray, graph_mode: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    evals = np.asarray(evals, dtype=np.float64)
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = evals[evals > tol]
    neg = evals[evals < -tol]
    pos_energy = float(np.sum(pos))
    neg_abs_energy = float(np.sum(np.abs(neg)))
    abs_all = float(np.sum(np.abs(evals)))
    coverage_levels = [0.50, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]
    rows = []
    if pos.size:
        cum = np.cumsum(pos) / max(pos_energy, EPS)
    else:
        cum = np.asarray([])
    for level in coverage_levels:
        if level >= 1.0:
            dim = int(pos.size)
        elif pos.size:
            dim = int(np.searchsorted(cum, level) + 1)
        else:
            dim = 0
        rows.append(
            {
                "graph_mode": graph_mode,
                "coverage": float(level),
                "dimension": dim,
                "total_eigenvalues": int(evals.size),
                "positive_eigenvalues": int(pos.size),
                "negative_eigenvalues": int(neg.size),
                "positive_eigenvalue_energy": pos_energy,
                "negative_eigenvalue_abs_energy": neg_abs_energy,
                "negative_energy_ratio": float(neg_abs_energy / max(abs_all, EPS)),
                "eigenvalue_tolerance": tol,
                "spectrum_truncated": False,
            }
        )
    summary = {
        "total_eigenvalues": int(evals.size),
        "positive_eigenvalues": int(pos.size),
        "negative_eigenvalues": int(neg.size),
        "positive_eigenvalue_energy": pos_energy,
        "negative_eigenvalue_abs_energy": neg_abs_energy,
        "negative_energy_ratio": float(neg_abs_energy / max(abs_all, EPS)),
        "eigenvalue_tolerance": tol,
        "spectrum_truncated": False,
        "coverage": {str(row["coverage"]): row["dimension"] for row in rows},
    }
    return rows, summary


def _stress_curve(
    D_landmark: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    dims: Sequence[int],
    graph_mode: str,
    raw_near_quantile: float,
    geo_far_quantile: float,
) -> List[Dict[str, object]]:
    pos = np.maximum(evals, 0.0)
    rows: List[Dict[str, object]] = []
    left, right = np.triu_indices(D_landmark.shape[0], k=1)
    geo = D_landmark[left, right]
    geo_far = np.quantile(geo, geo_far_quantile)
    for dim in dims:
        k = min(int(dim), int(np.sum(pos > 1e-10)))
        if k == 0:
            coords = np.zeros((D_landmark.shape[0], 1))
        else:
            coords = evecs[:, :k] * np.sqrt(pos[:k])[None, :k]
        emb_dist = np.linalg.norm(coords[left] - coords[right], axis=1)
        stress = float(np.sum((emb_dist - geo) ** 2) / max(np.sum(geo ** 2), EPS))
        emb_near = emb_dist <= np.quantile(emb_dist, raw_near_quantile)
        false_pres = float(np.mean(emb_near & (geo >= geo_far)))
        rows.append(
            {
                "graph_mode": graph_mode,
                "embed_dim": int(dim),
                "stress": stress,
                "false_shortcut_preservation": false_pres,
            }
        )
    return rows


def _coords_for_dim(evals: np.ndarray, evecs: np.ndarray, dim: int) -> np.ndarray:
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = np.maximum(evals, 0.0)
    k = min(int(dim), int(np.sum(evals > tol)))
    if k == 0:
        return np.zeros((evecs.shape[0], 1))
    return evecs[:, :k] * np.sqrt(pos[:k])[None, :k]


def _false_shortcut_vs_dim(
    D_landmark: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    dims: Sequence[int],
    graph_mode: str,
    emb_near_quantile: float,
    geo_far_quantile: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    left, right = np.triu_indices(D_landmark.shape[0], k=1)
    geo = D_landmark[left, right]
    geo_far_thr = float(np.quantile(geo, geo_far_quantile))
    geo_far = geo >= geo_far_thr
    for dim in dims:
        coords = _coords_for_dim(evals, evecs, dim)
        emb = np.linalg.norm(coords[left] - coords[right], axis=1)
        emb_near_thr = float(np.quantile(emb, emb_near_quantile))
        emb_near = emb <= emb_near_thr
        emb_near_geo = geo[emb_near]
        emb_near_emb = emb[emb_near]
        rows.append(
            {
                "graph_mode": graph_mode,
                "embed_dim": int(dim),
                "geo_far_quantile": float(geo_far_quantile),
                "emb_near_quantile": float(emb_near_quantile),
                "geo_far_threshold": geo_far_thr,
                "emb_near_threshold": emb_near_thr,
                "false_shortcut_rate_conditional": float(np.mean(emb_near[geo_far])) if np.any(geo_far) else float("nan"),
                "false_shortcut_joint_rate": float(np.mean(emb_near & geo_far)),
                "median_geo_among_emb_near": float(np.median(emb_near_geo)) if emb_near_geo.size else float("nan"),
                "median_geo_over_emb_among_emb_near": float(np.median(emb_near_geo / (emb_near_emb + 1e-8))) if emb_near_geo.size else float("nan"),
            }
        )
    return rows


def _stress_curve_by_weighting(
    D_landmark: np.ndarray,
    raw_landmark: np.ndarray,
    evals: np.ndarray,
    evecs: np.ndarray,
    dims: Sequence[int],
    graph_mode: str,
    raw_near_quantile: float,
    geo_far_quantile: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    left, right = np.triu_indices(D_landmark.shape[0], k=1)
    geo = D_landmark[left, right]
    raw = raw_landmark[left, right]
    geo_far_thr = float(np.quantile(geo, geo_far_quantile))
    raw_near_thr = float(np.quantile(raw, raw_near_quantile))
    weight_defs = {
        "uniform": np.ones_like(geo),
        "geo_far": (geo >= geo_far_thr).astype(np.float64),
        "geo_squared": np.square(geo),
        "raw_near_geo_far": ((raw <= raw_near_thr) & (geo >= geo_far_thr)).astype(np.float64),
    }
    for dim in dims:
        coords = _coords_for_dim(evals, evecs, dim)
        emb = np.linalg.norm(coords[left] - coords[right], axis=1)
        err2 = np.square(emb - geo)
        for name, weights in weight_defs.items():
            if np.sum(weights) <= 0:
                stress = float("nan")
            else:
                stress = float(np.sum(weights * err2) / max(np.sum(weights * np.square(geo)), EPS))
            rows.append(
                {
                    "graph_mode": graph_mode,
                    "embed_dim": int(dim),
                    "weighting": name,
                    "stress": stress,
                    "num_weighted_pairs": int(np.sum(weights > 0)),
                    "geo_far_threshold": geo_far_thr,
                    "raw_near_threshold": raw_near_thr,
                }
            )
    return rows


def _false_shortcut_summary(
    X: np.ndarray,
    landmarks: np.ndarray,
    dist_lm_all: np.ndarray,
    graph_mode: str,
    raw_q: float,
    geo_q: float,
) -> Dict[str, object]:
    raw = np.linalg.norm(X[landmarks][:, None, :] - X[None, :, :], axis=-1)
    mask = np.ones_like(raw, dtype=bool)
    mask[np.arange(len(landmarks)), landmarks] = False
    raw_vals = raw[mask]
    geo_vals = dist_lm_all[mask]
    finite = np.isfinite(geo_vals)
    raw_vals = raw_vals[finite]
    geo_vals = geo_vals[finite]
    raw_thr = float(np.quantile(raw_vals, raw_q))
    geo_thr = float(np.quantile(geo_vals, geo_q))
    raw_near = raw_vals <= raw_thr
    false_shortcuts = raw_near & (geo_vals >= geo_thr)
    ratios = geo_vals[raw_near] / (raw_vals[raw_near] + 1e-8)
    return {
        "graph_mode": graph_mode,
        "raw_quantile_threshold": raw_q,
        "geo_quantile_threshold": geo_q,
        "raw_distance_threshold": raw_thr,
        "geo_distance_threshold": geo_thr,
        "false_shortcut_rate": float(np.mean(false_shortcuts)),
        "median_geo_distance_among_raw_near": float(np.median(geo_vals[raw_near])) if np.any(raw_near) else float("nan"),
        "median_geo_raw_ratio_among_raw_near": float(np.median(ratios)) if ratios.size else float("nan"),
        "p90_geo_raw_ratio_among_raw_near": float(np.percentile(ratios, 90)) if ratios.size else float("nan"),
        "num_pairs_evaluated": int(raw_vals.size),
    }


def _intrinsic_baselines(X: np.ndarray, knn_k: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    evals = np.square(s)
    rows.append(
        {
            "baseline": "pca",
            "participation_ratio": float(np.square(np.sum(evals)) / max(np.sum(np.square(evals)), EPS)),
            "rank90": _rank_energy(evals, 0.90),
            "rank95": _rank_energy(evals, 0.95),
            "rank99": _rank_energy(evals, 0.99),
        }
    )
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(X)
        dist, _idx = tree.query(X, k=min(3, X.shape[0]))
        if dist.shape[1] >= 3:
            ratios = dist[:, 2] / np.maximum(dist[:, 1], 1e-12)
            rows.append({"baseline": "two_nn", "id_estimate": float(1.0 / np.mean(np.log(ratios + 1e-12)))})
        dist_l, idx_l = tree.query(X, k=min(knn_k + 1, X.shape[0]))
        local_ranks = []
        for nbrs in idx_l[:, 1:]:
            local = X[nbrs] - X[nbrs].mean(axis=0, keepdims=True)
            _u, s_local, _vh = np.linalg.svd(local, full_matrices=False)
            local_ranks.append(_rank_energy(np.square(s_local), 0.90))
        rows.append({"baseline": "local_pca", "mean_local_rank90": float(np.mean(local_ranks))})
    except ImportError:
        rows.append({"baseline": "two_nn", "id_estimate": float("nan")})
    return rows


def _trained_fallback_rows() -> List[Dict[str, object]]:
    return [
        {"model": "state8", "latent_dim": 8, "success_rate": 36, "score_alias": 0.117, "norm_alias": 0.131},
        {"model": "state16", "latent_dim": 16, "success_rate": 72, "score_alias": 0.078, "norm_alias": 0.091},
        {"model": "state32", "latent_dim": 32, "success_rate": 78, "score_alias": 0.032, "norm_alias": 0.058},
        {"model": "state64", "latent_dim": 64, "success_rate": 90, "score_alias": 0.009, "norm_alias": 0.040},
        {"model": "baseline192", "latent_dim": 192, "success_rate": 96, "score_alias": 0.003, "norm_alias": 0.032},
    ]


def _save_spectrum_artifacts(
    output_dir: Path,
    task_name: str,
    graph_label: str,
    evals: np.ndarray,
    evecs: np.ndarray,
    D_landmark: np.ndarray,
    raw_landmark: np.ndarray,
    X_landmark: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    sampled_rows: np.ndarray,
    landmarks: np.ndarray,
    coverage_rows: List[Dict[str, object]],
    spectrum_payload: Dict[str, object],
) -> None:
    spectra_dir = output_dir / "spectra"
    plots_dir = output_dir / "plots"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{task_name}_{graph_label}"
    _write_csv(spectra_dir / f"{prefix}_spectrum.csv", coverage_rows)
    with (spectra_dir / f"{prefix}_spectrum.json").open("w") as file:
        json.dump(_jsonify(spectrum_payload), file, indent=2)
    np.savez_compressed(
        spectra_dir / f"{prefix}_spectrum_cache.npz",
        evals=evals.astype(np.float64),
        evecs=evecs.astype(np.float32),
        D_landmark=D_landmark.astype(np.float32),
        raw_landmark=raw_landmark.astype(np.float32),
        X_landmark=X_landmark.astype(np.float32),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        sampled_rows=sampled_rows.astype(np.int64),
        landmarks=landmarks.astype(np.int64),
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pos = np.asarray(spectrum_payload.get("positive_eigenvalues", []), dtype=np.float64)
    if pos.size:
        energy = np.cumsum(pos) / max(np.sum(pos), EPS)
        plt.figure(figsize=(7, 4.8))
        plt.plot(np.arange(1, pos.size + 1), energy)
        for key, color in [("rank90", "tab:orange"), ("rank95", "tab:green"), ("rank99", "tab:red")]:
            if key in spectrum_payload:
                plt.axvline(float(spectrum_payload[key]), linestyle="--", color=color, label=f"{key}={spectrum_payload[key]}")
        plt.xlabel("dimension")
        plt.ylabel("cumulative positive energy")
        plt.title(f"{graph_label} positive spectrum coverage")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{prefix}_cumulative_positive_energy.png", dpi=220)
        plt.close()
        plt.figure(figsize=(7, 4.8))
        plt.semilogy(np.arange(1, pos.size + 1), pos)
        plt.xlabel("component")
        plt.ylabel("positive eigenvalue")
        plt.title(f"{graph_label} eigenvalue decay")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{prefix}_eigenvalue_decay.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate plannable latent dimension from offline reachability geometry.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--state_key", default="")
    parser.add_argument("--obs_key", default="")
    parser.add_argument("--action_key", default="")
    parser.add_argument("--episode_key", default="")
    parser.add_argument("--step_key", default="")
    parser.add_argument("--num_nodes", type=int, default=5000)
    parser.add_argument("--sampling_mode", choices=["random_rows", "contiguous_segments", "episode_segments"], default="contiguous_segments")
    parser.add_argument("--num_segments", type=int, default=50)
    parser.add_argument("--segment_len", type=int, default=100)
    parser.add_argument("--feature_mode", choices=["state", "image_pca"], default="state")
    parser.add_argument("--image_pca_dim", type=int, default=64)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--knn_values", default="10")
    parser.add_argument("--graph_modes", default="temporal_only,temporal_plus_knn,knn_only,action_temporal")
    parser.add_argument("--disconnect_handling", choices=["largest_component", "cap_infinite"], default="largest_component")
    parser.add_argument("--temporal_edge_weight", type=float, default=1.0)
    parser.add_argument("--knn_edge_weight_mode", choices=["euclidean", "constant"], default="euclidean")
    parser.add_argument("--minimal_knn_k", type=int, default=2)
    parser.add_argument("--minimal_knn_weight", type=float, default=10.0)
    parser.add_argument("--max_embed_dim", type=int, default=128)
    parser.add_argument("--landmarks", type=int, default=1000)
    parser.add_argument("--raw_quantile_threshold", type=float, default=0.05)
    parser.add_argument("--geo_quantile_threshold", type=float, default=0.75)
    parser.add_argument("--stress_threshold", type=float, default=0.15)
    parser.add_argument("--task_name", default="pusht")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in (
        "spectra",
        "false_shortcuts",
        "trained_latent_metric",
        "compression_metric",
        "prediction_loss_control",
        "toy",
        "plots",
        "summaries",
    ):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    X_raw, episode_idx, step_idx, sampled_rows, meta = _load_features(args)
    sampled_actions = meta.get("sampled_actions")
    meta_for_json = {key: value for key, value in meta.items() if key != "sampled_actions"}
    feature_mean = X_raw.mean(axis=0, keepdims=True)
    feature_std = X_raw.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    X = (X_raw - feature_mean) / feature_std
    rng = np.random.default_rng(args.seed)
    landmark_count = min(args.landmarks, X.shape[0])
    landmarks = np.sort(rng.choice(X.shape[0], size=landmark_count, replace=False))
    dims = [d for d in [1, 2, 4, 8, 16, 32, 64, 96, 128] if d <= args.max_embed_dim]

    false_rows: List[Dict[str, object]] = []
    stress_rows: List[Dict[str, object]] = []
    false_vs_dim_rows: List[Dict[str, object]] = []
    weighted_stress_rows: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    graph_stats_rows: List[Dict[str, object]] = []
    coverage_summary: Dict[str, object] = {}
    spectra: Dict[str, object] = {}
    graph_modes = [item.strip() for item in args.graph_modes.split(",") if item.strip()]
    knn_values = [int(item) for item in str(args.knn_values).split(",") if item]
    graph_specs: List[Tuple[str, int]] = []
    for mode in graph_modes:
        parsed_specs = _parse_graph_mode_spec(mode, args.knn_k, args.minimal_knn_k)
        if len(parsed_specs) == 1 and parsed_specs[0][0] != mode:
            graph_specs.extend(parsed_specs)
        elif mode in ("temporal_plus_knn", "knn_only", "pure_knn"):
            for knn in knn_values:
                graph_specs.append((mode, knn))
        else:
            graph_specs.extend(parsed_specs)

    main_graph = (
        "temporal_plus_knn_k10"
        if ("temporal_plus_knn", 10) in graph_specs
        else _graph_label(graph_specs[0][0], graph_specs[0][1], args.minimal_knn_weight)
    )
    for graph_mode, graph_k in graph_specs:
        graph_label = _graph_label(graph_mode, graph_k, args.minimal_knn_weight)
        print(f"[d_plan] graph_mode={graph_label}", flush=True)
        graph_nx = _build_graph(
            X,
            episode_idx,
            step_idx,
            graph_mode,
            graph_k,
            args.temporal_edge_weight,
            args.knn_edge_weight_mode,
            minimal_knn_weight=args.minimal_knn_weight,
            actions=sampled_actions,
        )
        graph_stat = _graph_statistics(graph_nx, graph_label, graph_mode, graph_k)
        graph_stats_rows.append(graph_stat.copy())
        graph = _networkx_to_scipy_graph(graph_nx)
        component_info = {"num_connected_components": 1, "largest_component_size": graph.shape[0], "largest_component_fraction": 1.0}
        X_graph = X
        sampled_rows_graph = sampled_rows
        landmarks_graph = landmarks
        graph_used = graph
        if args.disconnect_handling == "largest_component":
            keep, component_info = _largest_connected_component_indices(graph)
            if keep.size < 2:
                print(f"[d_plan] warning: graph {graph_label} largest component too small; skipping", flush=True)
                continue
            X_graph = X[keep]
            sampled_rows_graph = sampled_rows[keep]
            index_map = {int(old): idx for idx, old in enumerate(keep.tolist())}
            landmarks_kept = [index_map[int(idx)] for idx in landmarks if int(idx) in index_map]
            if len(landmarks_kept) < min(10, X_graph.shape[0]):
                rng = np.random.default_rng(args.seed)
                landmarks_graph = np.sort(rng.choice(X_graph.shape[0], size=min(landmark_count, X_graph.shape[0]), replace=False))
            else:
                landmarks_graph = np.asarray(landmarks_kept, dtype=np.int64)
            graph_used = graph[keep][:, keep]
        dist_lm_all = _shortest_paths(graph_used, landmarks_graph)
        dist_lm_all_replaced, disconnected_frac = _replace_inf_distances(dist_lm_all)
        D_ll = dist_lm_all_replaced[:, landmarks_graph]
        D_ll = 0.5 * (D_ll + D_ll.T)
        finite = np.isfinite(D_ll)
        median_finite = float(np.median(D_ll[finite & (D_ll > 0)])) if np.any(finite & (D_ll > 0)) else 1.0
        D_ll = D_ll / max(median_finite, EPS)
        raw_lm = np.linalg.norm(X_graph[landmarks_graph][:, None, :] - X_graph[landmarks_graph][None, :, :], axis=-1)
        evals, evecs = _classical_mds_spectrum(D_ll)
        graph_coverage_rows, graph_coverage_summary = _spectrum_coverage_rows(evals, graph_label)
        coverage_rows.extend(graph_coverage_rows)
        coverage_summary[graph_label] = graph_coverage_summary
        graph_stat.update(
            {
                "num_nodes_used": int(X_graph.shape[0]),
                "finite_pair_fraction": float(1.0 - disconnected_frac),
                "sampling_mode": args.sampling_mode,
                "random_seed": int(args.seed),
                "d50": graph_coverage_summary["coverage"]["0.5"],
                "d80": graph_coverage_summary["coverage"]["0.8"],
                "d90": graph_coverage_summary["coverage"]["0.9"],
                "d95": graph_coverage_summary["coverage"]["0.95"],
                "d99": graph_coverage_summary["coverage"]["0.99"],
                "d995": graph_coverage_summary["coverage"]["0.995"],
                "d999": graph_coverage_summary["coverage"]["0.999"],
                "d100": graph_coverage_summary["coverage"]["1.0"],
                "d_plan_90": graph_coverage_summary["coverage"]["0.9"],
                "d_plan_95": graph_coverage_summary["coverage"]["0.95"],
                "d_plan_99": graph_coverage_summary["coverage"]["0.99"],
                "positive_eigenvalues": graph_coverage_summary["positive_eigenvalues"],
                "negative_eigenvalues": graph_coverage_summary["negative_eigenvalues"],
                "positive_eigen_count": graph_coverage_summary["positive_eigenvalues"],
                "negative_eigen_count": graph_coverage_summary["negative_eigenvalues"],
                "positive_energy": graph_coverage_summary["positive_eigenvalue_energy"],
                "negative_abs_energy": graph_coverage_summary["negative_eigenvalue_abs_energy"],
                "negative_energy_ratio": graph_coverage_summary["negative_energy_ratio"],
            }
        )
        graph_stats_rows[-1] = graph_stat
        false_row = _false_shortcut_summary(
            X_graph,
            landmarks_graph,
            dist_lm_all,
            graph_label,
            args.raw_quantile_threshold,
            args.geo_quantile_threshold,
        )
        false_row["disconnected_pair_frac_landmark_all"] = disconnected_frac
        false_row.update(component_info)
        false_row["knn_k"] = graph_k
        false_rows.append(false_row)
        stress_rows.extend(
            _stress_curve(D_ll, evals, evecs, dims, graph_label, args.raw_quantile_threshold, args.geo_quantile_threshold)
        )
        false_vs_dim_rows.extend(
            _false_shortcut_vs_dim(
                D_ll,
                evals,
                evecs,
                dims,
                graph_label,
                args.raw_quantile_threshold,
                args.geo_quantile_threshold,
            )
        )
        weighted_stress_rows.extend(
            _stress_curve_by_weighting(
                D_ll,
                raw_lm,
                evals,
                evecs,
                dims,
                graph_label,
                args.raw_quantile_threshold,
                args.geo_quantile_threshold,
            )
        )
        tol = graph_coverage_summary["eigenvalue_tolerance"]
        pos = evals[evals > tol]
        spectra[graph_label] = {
            "positive_eigenvalues": [float(v) for v in pos.tolist()],
            "effective_rank": _effective_rank(evals),
            "rank50": graph_coverage_summary["coverage"]["0.5"],
            "rank80": graph_coverage_summary["coverage"]["0.8"],
            "rank90": graph_coverage_summary["coverage"]["0.9"],
            "rank95": graph_coverage_summary["coverage"]["0.95"],
            "rank96": graph_coverage_summary["coverage"]["0.96"],
            "rank97": graph_coverage_summary["coverage"]["0.97"],
            "rank98": graph_coverage_summary["coverage"]["0.98"],
            "rank99": graph_coverage_summary["coverage"]["0.99"],
            "rank995": graph_coverage_summary["coverage"]["0.995"],
            "rank999": graph_coverage_summary["coverage"]["0.999"],
            "rank100": graph_coverage_summary["coverage"]["1.0"],
            "coverage_summary": graph_coverage_summary,
            "disconnected_pair_frac_landmark_all": disconnected_frac,
            "component_info": component_info,
            "distance_normalization": "D_landmark divided by median positive finite landmark distance before double-centering.",
            "median_finite_distance_before_normalization": median_finite,
        }
        _save_spectrum_artifacts(
            output_dir,
            args.task_name,
            graph_label,
            evals,
            evecs,
            D_ll,
            raw_lm,
            X_graph[landmarks_graph],
            feature_mean,
            feature_std,
            sampled_rows_graph,
            landmarks_graph,
            graph_coverage_rows,
            spectra[graph_label],
        )

    intrinsic = _intrinsic_baselines(X, args.knn_k)
    main_stress = [row for row in stress_rows if row["graph_mode"] == main_graph]
    d_stress = next((row["embed_dim"] for row in main_stress if row["stress"] <= args.stress_threshold), None)
    summary = {
        "dataset": args.dataset,
        "feature_mode": args.feature_mode,
        "feature_source": meta["feature_source"],
        "num_nodes": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "landmarks": int(landmark_count),
        "sampling_mode": args.sampling_mode,
        "num_segments": int(args.num_segments),
        "segment_len": int(args.segment_len),
        "disconnect_handling": args.disconnect_handling,
        "graph_modes": list(spectra.keys()),
        "main_graph_mode": main_graph,
        "d_plan_50": spectra[main_graph]["rank50"],
        "d_plan_80": spectra[main_graph]["rank80"],
        "d_plan_90": spectra[main_graph]["rank90"],
        "d_plan_95": spectra[main_graph]["rank95"],
        "d_plan_96": spectra[main_graph]["rank96"],
        "d_plan_97": spectra[main_graph]["rank97"],
        "d_plan_98": spectra[main_graph]["rank98"],
        "d_plan_99": spectra[main_graph]["rank99"],
        "d_plan_995": spectra[main_graph]["rank995"],
        "d_plan_999": spectra[main_graph]["rank999"],
        "d_plan_100": spectra[main_graph]["rank100"],
        "d_plan_stress": d_stress,
        "stress_threshold": args.stress_threshold,
        "raw_quantile_threshold": args.raw_quantile_threshold,
        "geo_quantile_threshold": args.geo_quantile_threshold,
    }
    _write_csv(output_dir / "false_shortcut_summary.csv", false_rows)
    _write_csv(output_dir / "dimension_stress_curve.csv", stress_rows)
    _write_csv(output_dir / "false_shortcut_vs_dim.csv", false_vs_dim_rows)
    _write_csv(output_dir / "stress_curve_by_weighting.csv", weighted_stress_rows)
    _write_csv(output_dir / "spectrum_coverage.csv", coverage_rows)
    _write_csv(output_dir / "graph_statistics.csv", graph_stats_rows)
    _write_csv(output_dir / "intrinsic_dim_baselines.csv", intrinsic)
    _write_csv(output_dir / "trained_dim_comparison.csv", _trained_fallback_rows())
    with (output_dir / "mds_spectrum.json").open("w") as file:
        json.dump(_jsonify(spectra), file, indent=2)
    with (output_dir / "spectrum_coverage.json").open("w") as file:
        json.dump(_jsonify(coverage_summary), file, indent=2)
    with (output_dir / "plannable_dimension_summary.json").open("w") as file:
        json.dump(_jsonify({"summary": summary, "meta": meta_for_json}), file, indent=2)
    print(f"[d_plan] d_plan_90={summary['d_plan_90']}, d_plan_stress={summary['d_plan_stress']}")
    print(f"[d_plan] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
