from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

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


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) < EPS or np.std(ry) < EPS:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _build_crossroad_parallel_graph(
    road_length: int,
    vertical_length: int,
    gap: float,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]], Tuple[int, int]]:
    coords: List[Tuple[float, float]] = []
    node_by_coord: Dict[Tuple[float, float], int] = {}
    edges: List[Tuple[int, int, float]] = []

    def node(coord: Tuple[float, float]) -> int:
        if coord not in node_by_coord:
            node_by_coord[coord] = len(coords)
            coords.append(coord)
        return node_by_coord[coord]

    def add_path(points: List[Tuple[float, float]], weight: float = 1.0) -> None:
        for src, dst in zip(points[:-1], points[1:]):
            edges.append((node(src), node(dst), weight))

    main_road = [(float(x), 0.0) for x in range(-road_length, road_length + 1)]
    vertical_road = [(0.0, float(y)) for y in range(-vertical_length, vertical_length + 1)]
    parallel_road = [(float(x), gap) for x in range(-road_length, road_length + 1)]
    add_path(main_road)
    add_path(vertical_road)
    add_path(parallel_road)
    edges.append((node((0.0, 0.0)), node((0.0, gap)), 1.0))

    shortcut_pair = (node((float(road_length), 0.0)), node((float(road_length), gap)))
    return np.asarray(coords, dtype=np.float64), edges, shortcut_pair


def _shortest_path_matrix(num_nodes: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    D = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for src, dst, weight in edges:
        D[src, dst] = min(D[src, dst], weight)
        D[dst, src] = min(D[dst, src], weight)
    for k in range(num_nodes):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def _classical_mds(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ np.square(D) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def _coverage(evals: np.ndarray) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = evals[evals > tol]
    cum = np.cumsum(pos) / max(float(np.sum(pos)), EPS) if pos.size else np.asarray([])
    levels = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    rows = []
    ranks: Dict[str, int] = {}
    for level in levels:
        dim = int(pos.size) if level >= 1.0 else int(np.searchsorted(cum, level) + 1 if pos.size else 0)
        key = f"d{int(level * 100)}" if level < 1.0 else "d100"
        ranks[key] = dim
        rows.append({"coverage": level, "dimension": dim, "positive_eigen_count": int(pos.size)})
    return rows, ranks


def _coords(evals: np.ndarray, evecs: np.ndarray, dim: int) -> np.ndarray:
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    k = min(int(dim), int(np.sum(evals > tol)))
    if k <= 0:
        return np.zeros((evecs.shape[0], 1), dtype=np.float64)
    return evecs[:, :k] * np.sqrt(np.maximum(evals[:k], 0.0))[None, :]


def _false_shortcut_rows(evals: np.ndarray, evecs: np.ndarray, D_graph: np.ndarray, dims: List[int]) -> List[Dict[str, object]]:
    left, right = np.triu_indices(D_graph.shape[0], k=1)
    graph_dist = D_graph[left, right]
    graph_far_thr = float(np.quantile(graph_dist, 0.75))
    graph_far = graph_dist >= graph_far_thr
    rows = []
    for dim in dims:
        emb = _coords(evals, evecs, dim)
        emb_dist = np.linalg.norm(emb[left] - emb[right], axis=1)
        emb_near_thr = float(np.quantile(emb_dist, 0.05))
        emb_near = emb_dist <= emb_near_thr
        stress = float(np.sum(np.square(emb_dist - graph_dist)) / max(np.sum(np.square(graph_dist)), EPS))
        rows.append(
            {
                "embed_dim": int(dim),
                "false_shortcut_rate": float(np.mean(emb_near[graph_far])) if np.any(graph_far) else float("nan"),
                "false_shortcut_joint_rate": float(np.mean(emb_near & graph_far)),
                "spearman_embedding_vs_graph": _spearman(emb_dist, graph_dist),
                "stress": stress,
                "graph_far_threshold": graph_far_thr,
                "embedding_near_threshold": emb_near_thr,
            }
        )
    return rows


def _save_plots(
    coords: np.ndarray,
    edges: List[Tuple[int, int, float]],
    shortcut_pair: Tuple[int, int],
    evals: np.ndarray,
    false_rows: List[Dict[str, object]],
    plots_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(5.8, 2.2), facecolor="white")
    for src, dst, _weight in edges:
        ax.plot([coords[src, 0], coords[dst, 0]], [coords[src, 1], coords[dst, 1]], color="0.65", lw=1.5)
    s, g = shortcut_pair
    ax.scatter(coords[:, 0], coords[:, 1], s=12, color="0.25", zorder=3)
    ax.scatter([coords[s, 0]], [coords[s, 1]], s=70, color="#4E79A7", label="s", zorder=4)
    ax.scatter([coords[g, 0]], [coords[g, 1]], s=70, color="#E15759", label="g", zorder=4)
    ax.plot([coords[s, 0], coords[g, 0]], [coords[s, 1], coords[g, 1]], "--", color="0.2", lw=1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Crossroad + parallel road: close points, long transition path")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(plots_dir / "controlled_toy_physical_layout.png", dpi=260)
    fig.savefig(plots_dir / "controlled_toy_physical_layout.pdf")
    plt.close(fig)

    pos = evals[evals > 1e-10 * max(1.0, float(np.max(np.abs(evals))))]
    fig, ax = plt.subplots(figsize=(4.8, 3.2), facecolor="white")
    ax.semilogy(np.arange(1, pos.size + 1), pos, marker="o", ms=3)
    ax.set_xlabel("MDS component")
    ax.set_ylabel("positive eigenvalue")
    ax.set_title("Controlled toy transition spectrum")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "controlled_toy_spectrum.png", dpi=260)
    fig.savefig(plots_dir / "controlled_toy_spectrum.pdf")
    plt.close(fig)

    dims = [row["embed_dim"] for row in false_rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.2), facecolor="white")
    ax.plot(dims, [row["false_shortcut_rate"] for row in false_rows], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("P(embedding-near | graph-far)")
    ax.set_title("False shortcuts decrease with dimension")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "controlled_toy_false_shortcut_vs_dim.png", dpi=260)
    fig.savefig(plots_dir / "controlled_toy_false_shortcut_vs_dim.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.2), facecolor="white")
    ax.plot(dims, [row["spearman_embedding_vs_graph"] for row in false_rows], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("Spearman(distance, d_T)")
    ax.set_title("Transition-distance rank fidelity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "controlled_toy_spearman_vs_dim.png", dpi=260)
    fig.savefig(plots_dir / "controlled_toy_spearman_vs_dim.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled transition-geometry toy for plannable-dimension diagnostics.")
    parser.add_argument("--road_length", type=int, default=16)
    parser.add_argument("--vertical_length", type=int, default=8)
    parser.add_argument("--gap", type=float, default=0.16)
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    toy_dir = output_dir / "toy"
    plots_dir = output_dir / "plots"
    coords, edges, shortcut_pair = _build_crossroad_parallel_graph(args.road_length, args.vertical_length, args.gap)
    D_phys = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    D_graph = _shortest_path_matrix(coords.shape[0], edges)
    evals, evecs = _classical_mds(D_graph)
    spectrum_rows, ranks = _coverage(evals)
    false_rows = _false_shortcut_rows(evals, evecs, D_graph, [1, 2, 3, 4, 8, 16])

    _write_csv(toy_dir / "controlled_toy_spectrum.csv", spectrum_rows)
    _write_csv(toy_dir / "controlled_toy_false_shortcuts.csv", false_rows)
    _save_plots(coords, edges, shortcut_pair, evals, false_rows, plots_dir)

    s, g = shortcut_pair
    physical_close = float(D_phys[s, g])
    transition_far = float(D_graph[s, g])
    summary = [
        "# Controlled Transition Geometry Toy",
        "",
        "This toy is a numerical counterpart to the false-shortcut concept figure.",
        "",
        f"- Physical coordinate dimension: 2",
        f"- Number of graph states: {coords.shape[0]}",
        f"- Geometry: crossroad plus one nearby parallel road",
        f"- Main/parallel road half-length: {args.road_length}",
        f"- Vertical road half-length: {args.vertical_length}",
        f"- Parallel road gap: {args.gap}",
        f"- Physical distance between selected nearby states: {physical_close:.4f}",
        f"- Transition shortest-path distance between the same states: {transition_far:.4f}",
        f"- Positive MDS rank: {ranks['d100']}",
        f"- d50/d80/d90/d95/d99/d100: {ranks['d50']}/{ranks['d80']}/{ranks['d90']}/{ranks['d95']}/{ranks['d99']}/{ranks['d100']}",
        "",
        "Interpretation: the graph is drawn in 2D, but the transition distance is not the same object as 2D physical distance. Low-dimensional Euclidean embeddings can contract graph-far states into embedding-near false shortcuts.",
    ]
    (toy_dir / "controlled_toy_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (toy_dir / "controlled_toy_summary.md").write_text("\n".join(summary) + "\n")
    print(f"[controlled_toy] wrote outputs under {toy_dir} and {plots_dir}")


if __name__ == "__main__":
    main()
