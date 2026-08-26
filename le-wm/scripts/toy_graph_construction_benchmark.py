from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

EPS = 1e-12
INF = 1e12


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) < EPS or np.std(ry) < EPS:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _build_road_world(length: int, gap: float, vertical_length: int):
    coords: List[Tuple[float, float]] = []
    node_by_coord: Dict[Tuple[float, float], int] = {}
    edges: List[Tuple[int, int, float]] = []

    def node(coord: Tuple[float, float]) -> int:
        if coord not in node_by_coord:
            node_by_coord[coord] = len(coords)
            coords.append(coord)
        return node_by_coord[coord]

    def add_path(points: Iterable[Tuple[float, float]], weight: float = 1.0) -> None:
        points = list(points)
        for src, dst in zip(points[:-1], points[1:]):
            edges.append((node(src), node(dst), weight))

    main = [(float(x), 0.0) for x in range(-length, length + 1)]
    parallel = [(float(x), gap) for x in range(-length, length + 1)]
    vertical = [(0.0, float(y)) for y in range(-vertical_length, vertical_length + 1)]
    add_path(main)
    add_path(parallel)
    add_path(vertical)
    edges.append((node((0.0, 0.0)), node((0.0, gap)), 1.0))
    s_node = node((float(length), 0.0))
    g_node = node((float(length), gap))
    return np.asarray(coords, dtype=np.float64), edges, s_node, g_node


def _adjacency_from_edges(n: int, edges: List[Tuple[int, int, float]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for src, dst, _weight in edges:
        adj[src].append(dst)
        adj[dst].append(src)
    return adj


def _shortest_path(adj: List[List[int]], src: int, dst: int) -> List[int]:
    parent = {src: -1}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        if node == dst:
            break
        for nbr in adj[node]:
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)
    if dst not in parent:
        return []
    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    return path[::-1]


def _floyd_warshall_from_edges(n: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    D = np.full((n, n), INF, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for src, dst, weight in edges:
        D[src, dst] = min(D[src, dst], weight)
        D[dst, src] = min(D[dst, src], weight)
    for mid in range(n):
        D = np.minimum(D, D[:, mid:mid + 1] + D[mid:mid + 1, :])
    D[D >= INF / 2] = np.inf
    return D


def _sample_trajectories(
    coords: np.ndarray,
    oracle_edges: List[Tuple[int, int, float]],
    s_node: int,
    g_node: int,
    num_trajectories: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    adj = _adjacency_from_edges(coords.shape[0], oracle_edges)
    visited = {s_node, g_node}
    observed_edges = set()
    for _ in range(num_trajectories):
        cur = int(rng.integers(0, coords.shape[0]))
        steps = int(rng.integers(min_len, max_len + 1))
        visited.add(cur)
        prev = -1
        for _step in range(steps):
            nbrs = adj[cur]
            if prev >= 0 and len(nbrs) > 1 and rng.random() < 0.82:
                candidates = [nbr for nbr in nbrs if nbr != prev]
            else:
                candidates = nbrs
            nxt = int(candidates[int(rng.integers(0, len(candidates)))])
            observed_edges.add(tuple(sorted((cur, nxt))))
            visited.add(nxt)
            prev, cur = cur, nxt
    oracle_path = _shortest_path(adj, s_node, g_node)
    for src, dst in zip(oracle_path[:-1], oracle_path[1:]):
        observed_edges.add(tuple(sorted((src, dst))))
        visited.add(src)
        visited.add(dst)
    sampled_nodes = np.asarray(sorted(visited), dtype=np.int64)
    return sampled_nodes, sorted(observed_edges)


def _components_from_adj(adj: np.ndarray) -> Tuple[int, int]:
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    sizes = []
    for start in range(n):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        size = 0
        while queue:
            node = queue.pop()
            size += 1
            for nbr in np.where(adj[node])[0]:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(int(nbr))
        sizes.append(size)
    return len(sizes), max(sizes) if sizes else 0


def _knn_edges(points: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    edges = {}
    for idx in range(points.shape[0]):
        order = np.argsort(D[idx])
        for nbr in order[1:min(k + 1, points.shape[0])]:
            src, dst = sorted((idx, int(nbr)))
            edges[(src, dst)] = min(edges.get((src, dst), float("inf")), float(D[idx, nbr]))
    return [(src, dst, weight) for (src, dst), weight in edges.items()]


def _make_distance_graph(n: int, edges: List[Tuple[int, int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    adj = np.zeros((n, n), dtype=bool)
    for src, dst, _weight in edges:
        if src == dst:
            continue
        adj[src, dst] = True
        adj[dst, src] = True
    return adj, _floyd_warshall_from_edges(n, edges)


def _simple_kmeans(points: np.ndarray, k: int, seed: int, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(points.shape[0], size=min(k, points.shape[0]), replace=False)].copy()
    assignments = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(iters):
        dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
        new_assignments = np.argmin(dist, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for cluster in range(centers.shape[0]):
            members = points[assignments == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
    return centers, assignments


def _fixed_radius_clusters(points: np.ndarray, radius: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(points.shape[0])
    centers = []
    assignments = np.full(points.shape[0], -1, dtype=np.int64)
    for idx in order:
        if assignments[idx] >= 0:
            continue
        center_id = len(centers)
        centers.append(points[idx])
        dist = np.linalg.norm(points - points[idx], axis=1)
        assignments[(assignments < 0) & (dist <= radius)] = center_id
    centers_arr = np.asarray(centers, dtype=np.float64)
    dist = np.linalg.norm(points[:, None, :] - centers_arr[None, :, :], axis=-1)
    assignments = np.argmin(dist, axis=1)
    return centers_arr, assignments


def _kcenter_clusters(points: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, points.shape[0]))
    centers = [points[first]]
    min_dist = np.linalg.norm(points - points[first], axis=1)
    for _ in range(1, min(k, points.shape[0])):
        idx = int(np.argmax(min_dist))
        centers.append(points[idx])
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[idx], axis=1))
    centers_arr = np.asarray(centers, dtype=np.float64)
    dist = np.linalg.norm(points[:, None, :] - centers_arr[None, :, :], axis=-1)
    assignments = np.argmin(dist, axis=1)
    return centers_arr, assignments


def _quotient_distance(
    points: np.ndarray,
    observed_sample_edges: List[Tuple[int, int]],
    centers: np.ndarray,
    assignments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_edges = {}
    for src, dst in observed_sample_edges:
        c_src = int(assignments[src])
        c_dst = int(assignments[dst])
        if c_src == c_dst:
            continue
        key = tuple(sorted((c_src, c_dst)))
        q_edges[key] = 1.0
    q_edge_list = [(src, dst, weight) for (src, dst), weight in q_edges.items()]
    q_adj, D_q = _make_distance_graph(centers.shape[0], q_edge_list)
    D_est = D_q[assignments[:, None], assignments[None, :]]
    sample_adj = q_adj[assignments[:, None], assignments[None, :]]
    np.fill_diagonal(sample_adj, False)
    return sample_adj, D_est, q_adj


def _replace_inf_for_spectrum(D: np.ndarray) -> np.ndarray:
    finite = np.isfinite(D)
    out = D.copy()
    if np.all(finite):
        return out
    max_finite = float(np.max(out[finite])) if np.any(finite) else 1.0
    out[~finite] = 2.0 * max_finite
    return out


def _spectrum(D: np.ndarray) -> Dict[str, object]:
    D = _replace_inf_for_spectrum(D)
    finite_pos = D[np.isfinite(D) & (D > 0)]
    scale = float(np.median(finite_pos)) if finite_pos.size else 1.0
    D = D / max(scale, EPS)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ np.square(D) @ J
    evals = np.linalg.eigvalsh(B)[::-1]
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = evals[evals > tol]
    pos_energy = float(np.sum(pos))
    cum = np.cumsum(pos) / max(pos_energy, EPS) if pos.size else np.asarray([])
    out: Dict[str, object] = {"positive_eigenvalues": [float(v) for v in pos.tolist()], "positive_rank": int(pos.size)}
    for label, q in [("d90", 0.90), ("d95", 0.95), ("d99", 0.99)]:
        out[label] = int(np.searchsorted(cum, q) + 1) if pos.size else 0
    return out


def _edge_rates(est_adj: np.ndarray, oracle_adj: np.ndarray) -> Tuple[float, float]:
    tri = np.triu_indices(est_adj.shape[0], k=1)
    est = est_adj[tri]
    oracle = oracle_adj[tri]
    spurious = est & ~oracle
    missing = oracle & ~est
    spurious_rate = float(np.sum(spurious) / max(np.sum(est), 1))
    missing_rate = float(np.sum(missing) / max(np.sum(oracle), 1))
    return spurious_rate, missing_rate


def _distance_metrics(D_est: np.ndarray, D_true: np.ndarray, s_idx: int, g_idx: int) -> Dict[str, object]:
    tri = np.triu_indices(D_true.shape[0], k=1)
    true = D_true[tri]
    est = D_est[tri]
    finite = np.isfinite(est) & np.isfinite(true)
    mae = float(np.mean(np.abs(est[finite] - true[finite]))) if np.any(finite) else float("nan")
    rel = float(np.mean(np.abs(est[finite] - true[finite]) / np.maximum(true[finite], EPS))) if np.any(finite) else float("nan")
    spearman = _spearman(est[finite], true[finite]) if np.any(finite) else float("nan")
    true_far = true >= np.quantile(true, 0.75)
    if np.any(finite):
        est_near_thr = float(np.quantile(est[finite], 0.05))
        est_near = finite & (est <= est_near_thr)
        false_shortcut = float(np.mean(est_near[true_far])) if np.any(true_far) else float("nan")
    else:
        est_near_thr = float("nan")
        false_shortcut = float("nan")
    return {
        "finite_pair_fraction": float(np.mean(finite)),
        "spearman_est_vs_true": spearman,
        "mae_distance": mae,
        "relative_error_distance": rel,
        "d_est_sg": float(D_est[s_idx, g_idx]) if np.isfinite(D_est[s_idx, g_idx]) else float("inf"),
        "d_true_sg": float(D_true[s_idx, g_idx]),
        "graph_false_shortcut_rate": false_shortcut,
        "est_near_threshold": est_near_thr,
    }


def _spectral_error(method_pos: List[float], oracle_pos: List[float]) -> float:
    a = np.asarray(method_pos, dtype=np.float64)
    b = np.asarray(oracle_pos, dtype=np.float64)
    n = max(a.size, b.size)
    if n == 0:
        return float("nan")
    aa = np.zeros(n)
    bb = np.zeros(n)
    aa[:a.size] = a / max(float(np.sum(a)), EPS)
    bb[:b.size] = b / max(float(np.sum(b)), EPS)
    return float(np.linalg.norm(aa - bb))


def _evaluate_method(
    name: str,
    est_adj: np.ndarray,
    D_est: np.ndarray,
    oracle_adj: np.ndarray,
    D_true: np.ndarray,
    s_idx: int,
    g_idx: int,
    oracle_spectrum: Dict[str, object],
) -> Dict[str, object]:
    components, lcc = _components_from_adj(est_adj)
    spurious, missing = _edge_rates(est_adj, oracle_adj)
    spec = _spectrum(D_est)
    metrics = _distance_metrics(D_est, D_true, s_idx, g_idx)
    return {
        "method": name,
        "connected_components": components,
        "largest_connected_component_size": lcc,
        "num_edges": int(np.sum(np.triu(est_adj, k=1))),
        "spurious_edge_rate": spurious,
        "missing_edge_rate": missing,
        **metrics,
        "positive_rank": spec["positive_rank"],
        "d90": spec["d90"],
        "d95": spec["d95"],
        "d99": spec["d99"],
        "spectral_error_vs_oracle": _spectral_error(spec["positive_eigenvalues"], oracle_spectrum["positive_eigenvalues"]),
        "positive_eigenvalues": spec["positive_eigenvalues"],
    }


def _plot_outputs(
    output_dir: Path,
    coords_all: np.ndarray,
    oracle_edges_all: List[Tuple[int, int, float]],
    sampled_nodes: np.ndarray,
    sample_coords: np.ndarray,
    s_idx: int,
    g_idx: int,
    path_nodes_all: List[int],
    method_graphs: Dict[str, np.ndarray],
    rows: List[Dict[str, object]],
    D_true: np.ndarray,
    method_distances: Dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[toy_benchmark] matplotlib unavailable; skipping plots.")
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(6.2, 3.2), facecolor="white")
    for src, dst, _weight in oracle_edges_all:
        ax.plot([coords_all[src, 0], coords_all[dst, 0]], [coords_all[src, 1], coords_all[dst, 1]], color="0.75", lw=1.4)
    if path_nodes_all:
        path = coords_all[path_nodes_all]
        ax.plot(path[:, 0], path[:, 1], color="#F28E2B", lw=2.6, label="oracle path")
    ax.scatter(sample_coords[:, 0], sample_coords[:, 1], s=12, color="0.25", alpha=0.6, label="sampled states")
    ax.scatter(sample_coords[[s_idx], 0], sample_coords[[s_idx], 1], s=80, color="#4E79A7", label="s")
    ax.scatter(sample_coords[[g_idx], 0], sample_coords[[g_idx], 1], s=80, color="#E15759", label="g")
    ax.plot([sample_coords[s_idx, 0], sample_coords[g_idx, 0]], [sample_coords[s_idx, 1], sample_coords[g_idx, 1]], "--", color="0.2", lw=1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Toy road layout with oracle transition path")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "toy_layout_with_oracle_path.png", dpi=260)
    fig.savefig(plots_dir / "toy_layout_with_oracle_path.pdf")
    plt.close(fig)

    selected = ["oracle_graph", "knn_only", "temporal_only", "temporal_plus_knn", "kmeans_quotient", "fixed_radius_quotient", "kcenter_quotient"]
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.2), facecolor="white")
    axes = axes.ravel()
    for ax, method in zip(axes, selected):
        adj = method_graphs.get(method)
        ax.scatter(sample_coords[:, 0], sample_coords[:, 1], s=8, color="0.35", zorder=3)
        if adj is not None:
            tri = np.transpose(np.triu_indices(adj.shape[0], k=1))
            for src, dst in tri[adj[np.triu_indices(adj.shape[0], k=1)]]:
                ax.plot([sample_coords[src, 0], sample_coords[dst, 0]], [sample_coords[src, 1], sample_coords[dst, 1]], color="0.55", lw=0.5, alpha=0.35)
        ax.scatter(sample_coords[[s_idx], 0], sample_coords[[s_idx], 1], s=42, color="#4E79A7")
        ax.scatter(sample_coords[[g_idx], 0], sample_coords[[g_idx], 1], s=42, color="#E15759")
        ax.set_title(method, fontsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "estimated_graph_comparison.png", dpi=260)
    fig.savefig(plots_dir / "estimated_graph_comparison.pdf")
    plt.close(fig)

    tri = np.triu_indices(D_true.shape[0], k=1)
    true = D_true[tri]
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor="white")
    for row in rows:
        method = str(row["method"])
        est = method_distances[method][tri]
        finite = np.isfinite(est)
        if np.any(finite):
            ax.scatter(true[finite], est[finite], s=8, alpha=0.22, label=method)
    lim = float(np.nanmax(true))
    ax.plot([0, lim], [0, lim], color="0.2", lw=1, linestyle="--")
    ax.set_xlabel("true oracle transition distance")
    ax.set_ylabel("estimated graph distance")
    ax.set_title("Estimated vs true transition distances")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(plots_dir / "d_est_vs_d_true_scatter.png", dpi=260)
    fig.savefig(plots_dir / "d_est_vs_d_true_scatter.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor="white")
    methods = [str(row["method"]) for row in rows]
    ax.bar(methods, [float(row["graph_false_shortcut_rate"]) for row in rows], color="#4E79A7")
    ax.set_ylabel("P(est-near | true-far)")
    ax.set_title("Graph false shortcut rate by method")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(plots_dir / "false_shortcut_rate_by_method.png", dpi=260)
    fig.savefig(plots_dir / "false_shortcut_rate_by_method.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor="white")
    x = np.arange(len(methods))
    width = 0.25
    for offset, key, label in [(-width, "d90", "d90"), (0.0, "d95", "d95"), (width, "d99", "d99")]:
        ax.bar(x + offset, [float(row[key]) for row in rows], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("D_plan(q)")
    ax.set_title("Positive-spectrum dimensions by method")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(plots_dir / "d90_d95_d99_by_method.png", dpi=260)
    fig.savefig(plots_dir / "d90_d95_d99_by_method.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor="white")
    for row in rows:
        eig = np.asarray(row["positive_eigenvalues"], dtype=np.float64)
        if eig.size:
            ax.semilogy(np.arange(1, eig.size + 1), eig / max(float(np.sum(eig)), EPS), label=str(row["method"]))
    ax.set_xlabel("MDS component")
    ax.set_ylabel("normalized positive eigenvalue")
    ax.set_title("Spectrum comparison by graph construction method")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(plots_dir / "spectrum_comparison_by_method.png", dpi=260)
    fig.savefig(plots_dir / "spectrum_comparison_by_method.pdf")
    plt.close(fig)


def _write_summary(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Toy Graph Construction Benchmark",
        "",
        "This benchmark compares graph-construction methods in a toy road environment where physical proximity can create false shortcuts.",
        "",
        "The oracle road graph defines true transition distance. Synthetic trajectories are generated only along feasible road edges, and graph methods are evaluated by how well their estimated shortest-path distances recover the oracle distances.",
        "",
        "| method | components | spurious edge rate | missing edge rate | Spearman | MAE | rel err | d_est(s,g) | d_true(s,g) | false shortcut | d90/d95/d99 | spectral err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['connected_components']} | {float(row['spurious_edge_rate']):.3f} | "
            f"{float(row['missing_edge_rate']):.3f} | {float(row['spearman_est_vs_true']):.3f} | "
            f"{float(row['mae_distance']):.3f} | {float(row['relative_error_distance']):.3f} | "
            f"{row['d_est_sg']} | {row['d_true_sg']} | {float(row['graph_false_shortcut_rate']):.3f} | "
            f"{row['d90']}/{row['d95']}/{row['d99']} | {float(row['spectral_error_vs_oracle']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `oracle_graph` is the gold-standard transition metric on sampled road states.",
            "- `knn_only` is expected to create false shortcut edges because physical closeness does not imply transition reachability.",
            "- `temporal_only` can miss feasible edges if the synthetic trajectory set does not observe them.",
            "- Quotient methods use physical similarity only to aggregate states, then use observed temporal transitions as graph edges.",
            "",
            "This benchmark is for method selection and debugging. Do not turn it directly into paper conclusions without checking the generated plots.",
        ]
    )
    (output_dir / "toy_graph_construction_summary.md").write_text("\n".join(lines) + "\n")


def _write_parallel_appendix_summary(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    by_method = {str(row["method"]): row for row in rows}
    lines = [
        "# Parallel-Road Graph-Construction Sanity Check",
        "",
        "This appendix toy isolates the graph-construction failure mode: physical proximity does not imply transition reachability.",
        "",
        "| method | d_est(s,g) | d_true(s,g) | false shortcut rate | spurious edge rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ["oracle_graph", "knn_only", "temporal_plus_knn", "fixed_radius_quotient"]:
        row = by_method.get(method)
        if not row:
            continue
        lines.append(
            f"| {method} | {row['d_est_sg']} | {row['d_true_sg']} | "
            f"{float(row['graph_false_shortcut_rate']):.3f} | {float(row['spurious_edge_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- kNN-only fails because physical proximity is incorrectly treated as transition reachability.",
            "- temporal+knn can also fail when the kNN shortcut enters the shortest path.",
            "- fixed-radius quotient succeeds here when the radius is smaller than the corridor gap, because similarity is used only for aggregation and edges come only from observed temporal transitions.",
        ]
    )
    (output_dir / "parallel_road_graph_construction_summary.md").write_text("\n".join(lines) + "\n")


def _copy_parallel_appendix_plot(output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    for suffix in ("png", "pdf"):
        src = plots_dir / f"estimated_graph_comparison.{suffix}"
        dst = plots_dir / f"parallel_road_graph_construction_comparison.{suffix}"
        if src.exists():
            shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy benchmark for transition graph construction under physical false shortcuts.")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence/toy_graph_benchmark")
    parser.add_argument("--length", type=int, default=18)
    parser.add_argument("--gap", type=float, default=0.16)
    parser.add_argument("--vertical_length", type=int, default=8)
    parser.add_argument("--num_trajectories", type=int, default=90)
    parser.add_argument("--min_traj_len", type=int, default=16)
    parser.add_argument("--max_traj_len", type=int, default=56)
    parser.add_argument("--knn_k", type=int, default=4)
    parser.add_argument("--quotient_clusters", type=int, default=64)
    parser.add_argument("--radius", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coords_all, oracle_edges_all, s_node_all, g_node_all = _build_road_world(args.length, args.gap, args.vertical_length)
    oracle_D_all = _floyd_warshall_from_edges(coords_all.shape[0], oracle_edges_all)
    sampled_nodes, observed_edges_all = _sample_trajectories(
        coords_all,
        oracle_edges_all,
        s_node_all,
        g_node_all,
        args.num_trajectories,
        args.min_traj_len,
        args.max_traj_len,
        args.seed,
    )
    node_to_sample = {int(node): idx for idx, node in enumerate(sampled_nodes)}
    sample_coords = coords_all[sampled_nodes]
    s_idx = node_to_sample[int(s_node_all)]
    g_idx = node_to_sample[int(g_node_all)]
    D_true = oracle_D_all[sampled_nodes[:, None], sampled_nodes[None, :]]

    oracle_sample_edges = []
    for src, dst, _weight in oracle_edges_all:
        if src in node_to_sample and dst in node_to_sample:
            oracle_sample_edges.append((node_to_sample[src], node_to_sample[dst], 1.0))
    observed_sample_pairs = []
    temporal_edges = []
    for src, dst in observed_edges_all:
        if src in node_to_sample and dst in node_to_sample:
            a, b = node_to_sample[src], node_to_sample[dst]
            observed_sample_pairs.append((a, b))
            temporal_edges.append((a, b, 1.0))

    oracle_adj, _D_oracle_induced = _make_distance_graph(sample_coords.shape[0], oracle_sample_edges)
    D_oracle = D_true
    oracle_spectrum = _spectrum(D_oracle)

    method_graphs: Dict[str, np.ndarray] = {}
    method_distances: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []

    def add_method(name: str, adj: np.ndarray, D_est: np.ndarray) -> None:
        method_graphs[name] = adj
        method_distances[name] = D_est
        rows.append(_evaluate_method(name, adj, D_est, oracle_adj, D_true, s_idx, g_idx, oracle_spectrum))

    add_method("oracle_graph", oracle_adj, D_oracle)
    knn_edges = _knn_edges(sample_coords, args.knn_k)
    add_method("knn_only", *_make_distance_graph(sample_coords.shape[0], knn_edges))
    add_method("temporal_only", *_make_distance_graph(sample_coords.shape[0], temporal_edges))
    temporal_plus_knn = temporal_edges + knn_edges
    add_method("temporal_plus_knn", *_make_distance_graph(sample_coords.shape[0], temporal_plus_knn))

    centers, assignments = _simple_kmeans(sample_coords, args.quotient_clusters, args.seed)
    add_method("kmeans_quotient", *_quotient_distance(sample_coords, observed_sample_pairs, centers, assignments)[:2])
    centers, assignments = _fixed_radius_clusters(sample_coords, args.radius, args.seed)
    add_method("fixed_radius_quotient", *_quotient_distance(sample_coords, observed_sample_pairs, centers, assignments)[:2])
    centers, assignments = _kcenter_clusters(sample_coords, args.quotient_clusters, args.seed)
    add_method("kcenter_quotient", *_quotient_distance(sample_coords, observed_sample_pairs, centers, assignments)[:2])

    csv_rows = [{key: value for key, value in row.items() if key != "positive_eigenvalues"} for row in rows]
    _write_csv(output_dir / "toy_graph_construction_benchmark.csv", csv_rows)
    _write_summary(output_dir, csv_rows)
    path_nodes_all = _shortest_path(_adjacency_from_edges(coords_all.shape[0], oracle_edges_all), s_node_all, g_node_all)
    _plot_outputs(
        output_dir,
        coords_all,
        oracle_edges_all,
        sampled_nodes,
        sample_coords,
        s_idx,
        g_idx,
        path_nodes_all,
        method_graphs,
        rows,
        D_true,
        method_distances,
    )
    _write_parallel_appendix_summary(output_dir, csv_rows)
    _copy_parallel_appendix_plot(output_dir)
    print(f"[toy_benchmark] sampled states: {sample_coords.shape[0]}")
    print(f"[toy_benchmark] d_true(s,g): {D_true[s_idx, g_idx]}")
    print(f"[toy_benchmark] wrote {output_dir / 'toy_graph_construction_benchmark.csv'}")
    print(f"[toy_benchmark] wrote {output_dir / 'toy_graph_construction_summary.md'}")


if __name__ == "__main__":
    main()
