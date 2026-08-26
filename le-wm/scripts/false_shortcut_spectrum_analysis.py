from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def _parse_list(value: str, caster=float):
    return [caster(item) for item in str(value).split(",") if item.strip()]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _coords(evals: np.ndarray, evecs: np.ndarray, dim: int) -> np.ndarray:
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = np.maximum(evals, 0.0)
    k = min(int(dim), int(np.sum(evals > tol)))
    return evecs[:, :k] * np.sqrt(pos[:k])[None, :] if k > 0 else np.zeros((evecs.shape[0], 1))


def _rank_energy(evals: np.ndarray, q: float) -> int:
    tol = 1e-10 * max(1.0, float(np.max(np.abs(evals))) if evals.size else 1.0)
    pos = evals[evals > tol]
    if pos.size == 0:
        return 0
    return int(np.searchsorted(np.cumsum(pos) / np.sum(pos), q) + 1)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        corr = spearmanr(x, y).correlation
        return float(corr) if corr is not None else float("nan")
    except ImportError:
        xr = np.argsort(np.argsort(x)).astype(np.float64)
        yr = np.argsort(np.argsort(y)).astype(np.float64)
        if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
            return float("nan")
        return float(np.corrcoef(xr, yr)[0, 1])


def _stress(emb: np.ndarray, geo: np.ndarray, weights: np.ndarray) -> float:
    if np.sum(weights) <= 0:
        return float("nan")
    return float(np.sum(weights * np.square(emb - geo)) / max(np.sum(weights * np.square(geo)), 1e-12))


def _plot(rows: List[Dict[str, object]], output_dir: Path, graph_label: str, evals: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for q_far, q_near in sorted({(row["q_far"], row["q_near"]) for row in rows}):
        group = sorted([row for row in rows if row["q_far"] == q_far and row["q_near"] == q_near], key=lambda r: r["embed_dim"])
        plt.plot([row["embed_dim"] for row in group], [row["false_shortcut_rate_conditional"] for row in group], marker="o", label=f"far={q_far}, near={q_near}")
    for q, color in [(0.90, "tab:orange"), (0.95, "tab:green"), (0.99, "tab:red")]:
        plt.axvline(_rank_energy(evals, q), linestyle="--", color=color, linewidth=1, label=f"d{int(q * 100)}")
    plt.xscale("log", base=2)
    plt.xlabel("embedding dimension")
    plt.ylabel("P[emb-near | graph-far]")
    plt.title(f"False shortcut rate vs dim: {graph_label}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "false_shortcut_rate_vs_dim.png", dpi=220)
    plt.close()

    default = sorted([row for row in rows if row["q_far"] == 0.75 and row["q_near"] == 0.05], key=lambda r: r["embed_dim"])
    if default:
        for key, ylabel, filename in [
            ("spearman_d_emb_d_graph", "Spearman(d_emb,d_graph)", "spearman_vs_dim.png"),
            ("stress_uniform", "uniform stress", "stress_vs_dim.png"),
            ("stress_graph_far", "graph-far weighted stress", "graph_far_weighted_stress_vs_dim.png"),
        ]:
            plt.figure(figsize=(7, 4.8))
            plt.plot([row["embed_dim"] for row in default], [row[key] for row in default], marker="o")
            plt.xscale("log", base=2)
            plt.xlabel("embedding dimension")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel}: {graph_label}")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=220)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="False-shortcut spectrum sweep from cached MDS graph distances.")
    parser.add_argument("--spectrum_cache", required=True)
    parser.add_argument("--task", default="pusht")
    parser.add_argument("--graph_mode", default="")
    parser.add_argument("--dims", default="1,2,4,8,16,32,64,96,128,163,192,302,512")
    parser.add_argument("--q_far_values", default="0.70,0.75,0.80,0.90")
    parser.add_argument("--q_near_values", default="0.01,0.05,0.10")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--plot_dir", default="")
    args = parser.parse_args()

    data = np.load(args.spectrum_cache)
    evals = np.asarray(data["evals"], dtype=np.float64)
    evecs = np.asarray(data["evecs"], dtype=np.float64)
    geo_matrix = np.asarray(data["D_landmark"], dtype=np.float64)
    left, right = np.triu_indices(geo_matrix.shape[0], k=1)
    geo = geo_matrix[left, right]
    dims = [int(d) for d in _parse_list(args.dims, int)]
    q_far_values = _parse_list(args.q_far_values, float)
    q_near_values = _parse_list(args.q_near_values, float)
    graph_label = args.graph_mode or Path(args.spectrum_cache).stem.replace("_spectrum_cache", "")
    rows: List[Dict[str, object]] = []
    max_pos = int(np.sum(evals > (1e-10 * max(1.0, float(np.max(np.abs(evals)))))))
    for dim in dims:
        if dim > max_pos:
            continue
        Y = _coords(evals, evecs, dim)
        emb = np.linalg.norm(Y[left] - Y[right], axis=1)
        for q_far in q_far_values:
            geo_far_thr = float(np.quantile(geo, q_far))
            geo_far = geo >= geo_far_thr
            for q_near in q_near_values:
                emb_near_thr = float(np.quantile(emb, q_near))
                emb_near = emb <= emb_near_thr
                row = {
                    "task": args.task,
                    "graph_mode": graph_label,
                    "embed_dim": int(dim),
                    "q_far": float(q_far),
                    "q_near": float(q_near),
                    "geo_far_threshold": geo_far_thr,
                    "emb_near_threshold": emb_near_thr,
                    "false_shortcut_rate_conditional": float(np.mean(emb_near[geo_far])) if np.any(geo_far) else float("nan"),
                    "false_shortcut_joint_rate": float(np.mean(emb_near & geo_far)),
                    "median_graph_distance_among_emb_near": float(np.median(geo[emb_near])) if np.any(emb_near) else float("nan"),
                    "mean_graph_distance_among_emb_near": float(np.mean(geo[emb_near])) if np.any(emb_near) else float("nan"),
                    "median_graph_over_embedding_ratio_among_emb_near": float(np.median(geo[emb_near] / (emb[emb_near] + 1e-8))) if np.any(emb_near) else float("nan"),
                    "spearman_d_emb_d_graph": _spearman(emb, geo),
                    "stress_uniform": _stress(emb, geo, np.ones_like(geo)),
                    "stress_graph_far": _stress(emb, geo, geo_far.astype(np.float64)),
                    "stress_false_shortcut": _stress(emb, geo, (emb_near & geo_far).astype(np.float64)),
                }
                rows.append(row)
    _write_csv(Path(args.output_csv), rows)
    if args.plot_dir:
        _plot(rows, Path(args.plot_dir), graph_label, evals)
    print(f"[false_shortcut_spectrum] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
