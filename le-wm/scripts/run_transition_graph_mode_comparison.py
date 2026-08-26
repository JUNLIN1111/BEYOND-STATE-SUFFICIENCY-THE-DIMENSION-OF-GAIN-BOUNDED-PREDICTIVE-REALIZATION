from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open() as file:
        return json.load(file)


def _plot_spectra(spectra: Dict[str, object], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[graph_compare] matplotlib unavailable; skipping combined spectrum plot.", flush=True)
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))
    for graph_mode, payload in spectra.items():
        eig = np.asarray(payload.get("positive_eigenvalues", []), dtype=np.float64)
        if eig.size == 0:
            continue
        plt.semilogy(np.arange(1, eig.size + 1), eig, label=f"{graph_mode} d90={payload.get('rank90')}")
    plt.xlabel("MDS component")
    plt.ylabel("positive eigenvalue")
    plt.title("Transition graph spectrum comparison")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "transition_graph_mode_spectrum_comparison.png", dpi=240)
    plt.savefig(plots_dir / "transition_graph_mode_spectrum_comparison.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare transition graph construction modes for PushT d_plan.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--output_dir", default="rollout_results/transition_graph_mode_comparison")
    parser.add_argument("--feature_mode", choices=["state", "image_pca"], default="state")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--num_nodes", type=int, default=5000)
    parser.add_argument("--sampling_mode", choices=["random_rows", "contiguous_segments", "episode_segments"], default="contiguous_segments")
    parser.add_argument("--num_segments", type=int, default=50)
    parser.add_argument("--segment_len", type=int, default=100)
    parser.add_argument("--landmarks", type=int, default=1000)
    parser.add_argument("--max_embed_dim", type=int, default=512)
    parser.add_argument("--pure_knn_k", type=int, default=10)
    parser.add_argument("--minimal_knn_k", type=int, default=2)
    parser.add_argument("--minimal_knn_weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_estimate", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = Path(__file__).with_name("estimate_plannable_dimension.py")

    if not args.skip_estimate:
        command = [
            sys.executable,
            str(estimator),
            "--dataset",
            args.dataset,
            "--feature_mode",
            args.feature_mode,
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
            "temporal_only,temporal_plus_minimal_knn,pure_knn",
            "--knn_values",
            str(args.pure_knn_k),
            "--minimal_knn_k",
            str(args.minimal_knn_k),
            "--minimal_knn_weight",
            str(args.minimal_knn_weight),
            "--disconnect_handling",
            "largest_component",
            "--output_dir",
            str(output_dir),
            "--seed",
            str(args.seed),
        ]
        print("[graph_compare] running estimator:")
        print("  " + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    spectra = _load_json(output_dir / "mds_spectrum.json")
    rows = []
    for graph_mode, payload in spectra.items():
        rows.append(
            {
                "graph_mode": graph_mode,
                "d_plan_q90": payload.get("rank90"),
                "d_plan_q95": payload.get("rank95"),
                "d_plan_q99": payload.get("rank99"),
                "effective_rank": payload.get("effective_rank"),
                "positive_eigenvalues": payload.get("coverage_summary", {}).get("positive_eigenvalues"),
                "largest_component_size": payload.get("component_info", {}).get("largest_component_size"),
                "num_connected_components": payload.get("component_info", {}).get("num_connected_components"),
            }
        )
    _write_csv(output_dir / "d_plan_graph_mode_comparison.csv", rows)
    _plot_spectra(spectra, output_dir)

    print("\n[graph_compare] D_plan(q=0.9)")
    print("graph_mode | d90 | d95 | d99 | largest_component")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['graph_mode']} | {row['d_plan_q90']} | {row['d_plan_q95']} | "
            f"{row['d_plan_q99']} | {row['largest_component_size']}"
        )
    print(f"\n[graph_compare] wrote table: {output_dir / 'd_plan_graph_mode_comparison.csv'}")
    print(f"[graph_compare] graph stats: {output_dir / 'graph_statistics.csv'}")
    print(f"[graph_compare] spectra plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
