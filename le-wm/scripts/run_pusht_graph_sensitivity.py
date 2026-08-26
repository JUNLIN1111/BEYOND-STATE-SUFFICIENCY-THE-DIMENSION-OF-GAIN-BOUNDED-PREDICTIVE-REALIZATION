from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


GRAPH_MODES = [
    "temporal_only",
    "temporal_plus_knn_k5",
    "temporal_plus_knn_k10",
    "temporal_plus_knn_k20",
    "knn_only_k10",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _run_estimator(args, work_dir: Path) -> None:
    estimator = Path(__file__).with_name("estimate_plannable_dimension.py")
    command = [
        sys.executable,
        str(estimator),
        "--dataset",
        args.dataset,
        "--feature_mode",
        "state",
        "--state_key",
        args.state_key,
        "--episode_key",
        args.episode_key,
        "--num_nodes",
        str(args.num_nodes),
        "--sampling_mode",
        args.sampling_mode,
        "--num_segments",
        str(args.num_segments),
        "--segment_len",
        str(args.segment_len),
        "--landmarks",
        str(args.landmarks),
        "--max_embed_dim",
        str(args.max_embed_dim),
        "--graph_modes",
        ",".join(GRAPH_MODES),
        "--disconnect_handling",
        "largest_component",
        "--output_dir",
        str(work_dir),
        "--seed",
        str(args.seed),
    ]
    print("[graph_sensitivity] running estimator:")
    print("  " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _plot_dplan(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["graph_mode"]) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.0), facecolor="white")
    width = 0.24
    for offset, key, name in [(-width, "d90", "d90"), (0.0, "d95", "d95"), (width, "d99", "d99")]:
        ax.bar(x + offset, [float(row[key]) for row in rows], width=width, label=name)
    ax.axhline(7, color="0.25", linestyle="--", linewidth=1, label="physical state dim = 7")
    ax.axhline(2, color="0.45", linestyle=":", linewidth=1, label="local ID ≈ 2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("D_plan(q)")
    ax.set_title("PushT graph-construction sensitivity")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_graph_sensitivity_d90_d95_d99.png", dpi=260)
    fig.savefig(plots_dir / "pusht_graph_sensitivity_d90_d95_d99.pdf")
    plt.close(fig)


def _plot_spectra(work_dir: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    with (work_dir / "mds_spectrum.json").open() as file:
        spectra = json.load(file)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), facecolor="white")
    for graph_mode, payload in spectra.items():
        eig = np.asarray(payload.get("positive_eigenvalues", []), dtype=np.float64)
        if eig.size:
            label = f"{graph_mode} d90={payload.get('rank90')}"
            if graph_mode.startswith("knn_only"):
                label += " (ablation)"
            ax.semilogy(np.arange(1, eig.size + 1), eig, label=label)
    ax.set_xlabel("MDS component")
    ax.set_ylabel("positive eigenvalue")
    ax.set_title("PushT transition-distance spectra")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_graph_sensitivity_spectra.png", dpi=260)
    fig.savefig(plots_dir / "pusht_graph_sensitivity_spectra.pdf")
    plt.close(fig)


def _write_summary(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# PushT Graph-Construction Sensitivity",
        "",
        "This table checks whether the plannable-dimension diagnostic is qualitatively stable under reasonable transition-graph construction choices.",
        "",
        "`knn_only_k10` is an Isomap-style ablation, not the main transition graph.",
        "",
        "| graph mode | LCC size | components | d90 | d95 | d99 | negative energy ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['graph_mode']} | {row['largest_connected_component_size']} | {row['num_connected_components']} | "
            f"{row['d90']} | {row['d95']} | {row['d99']} | {float(row['negative_energy_ratio']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: exact D_plan values can move with graph construction, but the diagnostic is useful when reasonable transition graphs remain far above physical state dimension (7) and local intrinsic dimension (~2). If a mode has a tiny largest connected component, treat its spectrum as local to that component rather than global.",
        ]
    )
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "pusht_graph_sensitivity.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PushT graph-construction sensitivity for plannable dimension.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--work_dir", default="")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--num_nodes", type=int, default=5000)
    parser.add_argument("--sampling_mode", default="contiguous_segments")
    parser.add_argument("--num_segments", type=int, default=50)
    parser.add_argument("--segment_len", type=int, default=100)
    parser.add_argument("--landmarks", type=int, default=1000)
    parser.add_argument("--max_embed_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_estimate", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / "graph_sensitivity_work"
    if not args.skip_estimate:
        _run_estimator(args, work_dir)

    stats_rows = _read_csv(work_dir / "graph_statistics.csv")
    rows: List[Dict[str, object]] = []
    for row in stats_rows:
        if row.get("graph_mode") not in GRAPH_MODES:
            continue
        rows.append(
            {
                "graph_mode": row.get("graph_mode"),
                "k": row.get("knn_k"),
                "sampling_mode": row.get("sampling_mode"),
                "random_seed": row.get("random_seed"),
                "num_nodes_used": row.get("num_nodes_used"),
                "largest_connected_component_size": row.get("largest_component_size"),
                "finite_pair_fraction": row.get("finite_pair_fraction"),
                "num_connected_components": row.get("num_connected_components"),
                "positive_eigen_count": row.get("positive_eigen_count"),
                "negative_eigen_count": row.get("negative_eigen_count"),
                "positive_energy": row.get("positive_energy"),
                "negative_abs_energy": row.get("negative_abs_energy"),
                "negative_energy_ratio": row.get("negative_energy_ratio"),
                "d50": row.get("d50"),
                "d80": row.get("d80"),
                "d90": row.get("d90"),
                "d95": row.get("d95"),
                "d99": row.get("d99"),
                "d995": row.get("d995"),
                "d999": row.get("d999"),
                "d100": row.get("d100"),
            }
        )
    _write_csv(output_dir / "spectra" / "pusht_graph_sensitivity.csv", rows)
    _plot_dplan(rows, output_dir)
    _plot_spectra(work_dir, output_dir)
    _write_summary(output_dir, rows)
    print(f"[graph_sensitivity] wrote {output_dir / 'spectra' / 'pusht_graph_sensitivity.csv'}")


if __name__ == "__main__":
    main()
