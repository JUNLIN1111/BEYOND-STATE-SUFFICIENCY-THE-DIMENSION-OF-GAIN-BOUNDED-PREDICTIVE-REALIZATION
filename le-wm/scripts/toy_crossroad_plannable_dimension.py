from __future__ import annotations

import argparse
import csv
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


def _build_crossroad(radius: int) -> Tuple[np.ndarray, List[Tuple[int, int, float]], Dict[str, int]]:
    coords: List[Tuple[float, float]] = [(0.0, 0.0)]
    node_by_coord = {(0.0, 0.0): 0}
    edges: List[Tuple[int, int, float]] = []

    def node(coord: Tuple[float, float]) -> int:
        if coord not in node_by_coord:
            node_by_coord[coord] = len(coords)
            coords.append(coord)
        return node_by_coord[coord]

    def add_arm(name: str, direction: Tuple[int, int]) -> int:
        prev = 0
        endpoint = 0
        for step in range(1, radius + 1):
            cur = node((float(direction[0] * step), float(direction[1] * step)))
            edges.append((prev, cur, 1.0))
            prev = cur
            endpoint = cur
        return endpoint

    endpoints = {
        "L": add_arm("L", (-1, 0)),
        "R": add_arm("R", (1, 0)),
        "U": add_arm("U", (0, 1)),
        "D": add_arm("D", (0, -1)),
    }
    return np.asarray(coords, dtype=np.float64), edges, endpoints


def _floyd(n: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for src, dst, weight in edges:
        D[src, dst] = min(D[src, dst], weight)
        D[dst, src] = min(D[dst, src], weight)
    for mid in range(n):
        D = np.minimum(D, D[:, mid:mid + 1] + D[mid:mid + 1, :])
    return D


def _spectrum(D: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ np.square(D) @ J
    evals = np.linalg.eigvalsh(B)[::-1]
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = evals[evals > tol]
    neg = evals[evals < -tol]
    pos_energy = float(np.sum(pos))
    neg_abs_energy = float(np.sum(np.abs(neg)))
    abs_energy = float(np.sum(np.abs(evals)))
    cum = np.cumsum(pos) / max(pos_energy, EPS) if pos.size else np.asarray([])
    summary: Dict[str, object] = {
        "positive_rank": int(pos.size),
        "positive_energy": pos_energy,
        "negative_eigen_count": int(neg.size),
        "negative_abs_energy": neg_abs_energy,
        "negative_energy_ratio": float(neg_abs_energy / max(abs_energy, EPS)),
        "positive_eigenvalues": [float(v) for v in pos.tolist()],
        "negative_eigenvalues": [float(v) for v in neg.tolist()],
    }
    for label, q in [("d90", 0.90), ("d95", 0.95), ("d99", 0.99)]:
        summary[label] = int(np.searchsorted(cum, q) + 1) if pos.size else 0
    summary["d100"] = int(pos.size)
    return evals, summary


def _coords_from_spectrum(D: np.ndarray, dim: int) -> np.ndarray:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ np.square(D) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    k = min(dim, int(np.sum(evals > tol)))
    if k <= 0:
        return np.zeros((n, 1), dtype=np.float64)
    return evecs[:, :k] * np.sqrt(np.maximum(evals[:k], 0.0))[None, :]


def _embedding_metrics(D_true: np.ndarray, dims: List[int], endpoint_pairs: List[Tuple[int, int]]) -> List[Dict[str, object]]:
    left, right = np.triu_indices(D_true.shape[0], k=1)
    true = D_true[left, right]
    true_far = true >= np.quantile(true, 0.75)
    rows = []
    for dim in dims:
        emb = _coords_from_spectrum(D_true, dim)
        dist = np.linalg.norm(emb[left] - emb[right], axis=1)
        emb_near_thr = float(np.quantile(dist, 0.05))
        emb_near = dist <= emb_near_thr
        endpoint_errors = []
        for src, dst in endpoint_pairs:
            endpoint_errors.append(abs(float(np.linalg.norm(emb[src] - emb[dst])) - float(D_true[src, dst])) / max(float(D_true[src, dst]), EPS))
        rows.append(
            {
                "embed_dim": dim,
                "false_shortcut_rate": float(np.mean(emb_near[true_far])) if np.any(true_far) else float("nan"),
                "emb_near_threshold": emb_near_thr,
                "mean_endpoint_relative_distortion": float(np.mean(endpoint_errors)),
                "max_endpoint_relative_distortion": float(np.max(endpoint_errors)),
            }
        )
    return rows


def _plot_endpoint_concept(coords: np.ndarray, edges: List[Tuple[int, int, float]], endpoints: Dict[str, int], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    endpoint_order = ["L", "R", "U", "D"]
    tetra = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    fig = plt.figure(figsize=(8.6, 3.8), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1)
    for src, dst, _weight in edges:
        ax1.plot([coords[src, 0], coords[dst, 0]], [coords[src, 1], coords[dst, 1]], color="0.65", lw=1.5)
    for name in endpoint_order:
        idx = endpoints[name]
        color = "#4E79A7" if name == "R" else "#E15759" if name == "U" else "0.25"
        ax1.scatter(coords[idx, 0], coords[idx, 1], s=70, color=color)
        ax1.text(coords[idx, 0], coords[idx, 1], f" {name}", va="center")
    ax1.plot([coords[endpoints["R"], 0], coords[endpoints["U"], 0]], [coords[endpoints["R"], 1], coords[endpoints["U"], 1]], "--", color="0.2", lw=1)
    ax1.set_aspect("equal")
    ax1.set_axis_off()
    ax1.set_title("2D physical crossroad")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for idx, name in enumerate(endpoint_order):
        color = "#4E79A7" if name == "R" else "#E15759" if name == "U" else "0.35"
        ax2.scatter(*tetra[idx], s=70, color=color)
        ax2.text(*(tetra[idx] * 1.08), name)
    for i in range(4):
        for j in range(i + 1, 4):
            ax2.plot([tetra[i, 0], tetra[j, 0]], [tetra[i, 1], tetra[j, 1]], [tetra[i, 2], tetra[j, 2]], color="0.75", lw=1)
    ax2.set_title("3D branch-mode tetrahedron")
    ax2.set_axis_off()
    fig.tight_layout()
    fig.savefig(plots_dir / "crossroad_endpoint_tetrahedron_concept.png", dpi=260)
    fig.savefig(plots_dir / "crossroad_endpoint_tetrahedron_concept.pdf")
    plt.close(fig)


def _plot_full_outputs(D_true: np.ndarray, evals: np.ndarray, summary: Dict[str, object], metrics: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    eig = np.asarray(summary["positive_eigenvalues"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.8, 3.6), facecolor="white")
    bars = ax.bar(np.arange(1, eig.size + 1), eig, color="#4E79A7", width=0.65)
    ax.set_xticks(np.arange(1, eig.size + 1))
    ax.set_xlabel("positive MDS component")
    ax.set_ylabel("positive eigenvalue")
    ax.set_title("Full crossroad transition spectrum")
    if eig.size == 3 and np.allclose(eig, eig[0]):
        ax.text(
            0.5,
            0.93,
            "three equal branch-contrast modes",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="0.25",
        )
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.0f}", ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "crossroad_full_spectrum.png", dpi=260)
    fig.savefig(plots_dir / "crossroad_full_spectrum.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.6), facecolor="white")
    signed = np.asarray(evals, dtype=np.float64)
    nonzero = signed[np.abs(signed) > 1e-8]
    colors = np.where(nonzero >= 0, "#4E79A7", "#E15759")
    ax.bar(np.arange(1, nonzero.size + 1), nonzero, color=colors, width=0.8)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlabel("MDS component")
    ax.set_ylabel("signed eigenvalue")
    ax.set_title("Crossroad signed MDS spectrum")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(plots_dir / "crossroad_signed_spectrum.png", dpi=260)
    fig.savefig(plots_dir / "crossroad_signed_spectrum.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4), facecolor="white")
    ax.plot([row["embed_dim"] for row in metrics], [row["false_shortcut_rate"] for row in metrics], marker="o")
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("P(emb-near | true-far)")
    ax.set_title("Crossroad false shortcuts vs dimension")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "crossroad_false_shortcut_vs_dim.png", dpi=260)
    fig.savefig(plots_dir / "crossroad_false_shortcut_vs_dim.pdf")
    plt.close(fig)

    emb2 = _coords_from_spectrum(D_true, 2)
    emb3 = _coords_from_spectrum(D_true, 3)
    fig = plt.figure(figsize=(8.4, 3.8), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(emb2[:, 0], emb2[:, 1], s=14, color="#4E79A7")
    ax1.set_title("Top-2 positive embedding")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(alpha=0.2)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(emb3[:, 0], emb3[:, 1], emb3[:, 2], s=14, color="#4E79A7")
    ax2.set_title("Top-3 positive embedding")
    ax2.set_axis_off()
    fig.tight_layout()
    fig.savefig(plots_dir / "crossroad_2d_vs_3d_embedding.png", dpi=260)
    fig.savefig(plots_dir / "crossroad_2d_vs_3d_embedding.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Main-paper crossroad plannable-dimension toy.")
    parser.add_argument("--radius", type=int, default=18)
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    toy_dir = output_dir / "toy"
    coords, edges, endpoints = _build_crossroad(args.radius)
    D_full = _floyd(coords.shape[0], edges)

    endpoint_names = ["L", "R", "U", "D"]
    endpoint_idx = np.asarray([endpoints[name] for name in endpoint_names], dtype=np.int64)
    D_endpoint = D_full[endpoint_idx[:, None], endpoint_idx[None, :]]
    endpoint_coords = coords[endpoint_idx]
    _endpoint_evals, endpoint_summary = _spectrum(D_endpoint)
    endpoint_rows = []
    for idx, value in enumerate(endpoint_summary["positive_eigenvalues"], start=1):
        endpoint_rows.append({"component": idx, "positive_eigenvalue": value})
    _write_csv(toy_dir / "crossroad_endpoint_spectrum.csv", endpoint_rows)

    adjacent_physical = float(np.linalg.norm(coords[endpoints["R"]] - coords[endpoints["U"]]))
    adjacent_transition = float(D_full[endpoints["R"], endpoints["U"]])
    opposite_physical = float(np.linalg.norm(coords[endpoints["L"]] - coords[endpoints["R"]]))
    endpoint_md = [
        "# Crossroad Endpoint Branch-Mode Analysis",
        "",
        f"- Radius R: {args.radius}",
        f"- Endpoint names: {endpoint_names}",
        f"- Positive rank: {endpoint_summary['positive_rank']}",
        f"- d90/d95/d99/d100: {endpoint_summary['d90']}/{endpoint_summary['d95']}/{endpoint_summary['d99']}/{endpoint_summary['d100']}",
        f"- Negative energy ratio: {endpoint_summary['negative_energy_ratio']:.3e}",
        f"- Adjacent endpoint physical distance: {adjacent_physical:.6f}",
        f"- Adjacent endpoint transition distance: {adjacent_transition:.6f}",
        f"- Opposite endpoint physical distance: {opposite_physical:.6f}",
        "",
        "The four endpoint transition distances are all equal to 2R, so the endpoint metric forms a regular tetrahedron and requires a 3D Euclidean embedding.",
    ]
    (toy_dir / "crossroad_endpoint_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (toy_dir / "crossroad_endpoint_summary.md").write_text("\n".join(endpoint_md) + "\n")
    _plot_endpoint_concept(coords, edges, endpoints, output_dir)

    _full_evals, full_summary = _spectrum(D_full)
    full_rows = [{"component": idx, "positive_eigenvalue": value} for idx, value in enumerate(full_summary["positive_eigenvalues"], start=1)]
    _write_csv(toy_dir / "crossroad_full_spectrum.csv", full_rows)
    endpoint_pairs = [(endpoints["R"], endpoints["U"]), (endpoints["R"], endpoints["D"]), (endpoints["L"], endpoints["U"]), (endpoints["L"], endpoints["D"])]
    metrics = _embedding_metrics(D_full, [2, 3], endpoint_pairs)
    _write_csv(toy_dir / "crossroad_full_embedding_metrics.csv", metrics)
    signed_rows = [{"component": idx, "eigenvalue": value} for idx, value in enumerate(_full_evals.tolist(), start=1)]
    _write_csv(toy_dir / "crossroad_full_signed_spectrum.csv", signed_rows)
    _plot_full_outputs(D_full, _full_evals, full_summary, metrics, output_dir)
    full_md = [
        "# Full Crossroad Graph Analysis",
        "",
        f"- Number of road nodes: {coords.shape[0]}",
        f"- Positive rank: {full_summary['positive_rank']}",
        f"- d90/d95/d99/d100: {full_summary['d90']}/{full_summary['d95']}/{full_summary['d99']}/{full_summary['d100']}",
        f"- Positive energy: {full_summary['positive_energy']:.6f}",
        f"- Negative abs energy: {full_summary['negative_abs_energy']:.6f}",
        f"- Negative energy ratio: {full_summary['negative_energy_ratio']:.6f}",
        "",
        "Main message: a planar crossroad already has a 3D positive transition spectrum because the four branch modes form a tetrahedral structure. A 2D latent collapses one branch-contrast direction, while 3D preserves the full positive branch-mode geometry. Any remaining negative spectrum is irreducible non-Euclidean mismatch.",
    ]
    (toy_dir / "crossroad_full_summary.md").write_text("\n".join(full_md) + "\n")
    print(f"[crossroad_toy] wrote outputs under {toy_dir} and {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
