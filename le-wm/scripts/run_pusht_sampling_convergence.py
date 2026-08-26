from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


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


def _parse_ints(text: str) -> List[int]:
    return [int(item) for item in str(text).replace(",", " ").split() if item.strip()]


def _run_one(args, output_dir: Path, num_nodes: int, seed: int) -> Path:
    run_dir = output_dir / "sampling_convergence_work" / f"n{num_nodes}_seed{seed}"
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
        str(num_nodes),
        "--sampling_mode",
        args.sampling_mode,
        "--num_segments",
        str(args.num_segments),
        "--segment_len",
        str(args.segment_len),
        "--landmarks",
        str(min(args.landmarks, num_nodes)),
        "--max_embed_dim",
        str(args.max_embed_dim),
        "--graph_modes",
        args.graph_mode,
        "--disconnect_handling",
        "largest_component",
        "--output_dir",
        str(run_dir),
        "--seed",
        str(seed),
    ]
    print(f"[sampling_convergence] running n={num_nodes}, seed={seed}", flush=True)
    subprocess.run(command, check=True)
    return run_dir


def _plot(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.0), facecolor="white")
    for key, label in [("d80", "d80"), ("d90", "d90"), ("d95", "d95"), ("d99", "d99")]:
        grouped = {}
        for row in rows:
            grouped.setdefault(int(row["num_nodes"]), []).append(float(row[key]))
        xs = sorted(grouped)
        means = [float(np.mean(grouped[x])) for x in xs]
        lows = [float(np.min(grouped[x])) for x in xs]
        highs = [float(np.max(grouped[x])) for x in xs]
        ax.plot(xs, means, marker="o", label=label)
        ax.fill_between(xs, lows, highs, alpha=0.12)
    ax.axhline(7, color="0.25", linestyle="--", linewidth=1, label="physical state dim = 7")
    ax.axhline(2, color="0.45", linestyle=":", linewidth=1, label="local ID ≈ 2")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sampled nodes")
    ax.set_ylabel("D_plan(q)")
    ax.set_title("PushT sampling convergence")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_sampling_convergence_dplan.png", dpi=260)
    fig.savefig(plots_dir / "pusht_sampling_convergence_dplan.pdf")
    plt.close(fig)


def _write_summary(rows: List[Dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# PushT Sampling Convergence",
        "",
        "This diagnostic checks whether D_plan is a stable sampled-geometry estimate rather than an artifact of one sample size.",
        "",
        "| num nodes | seeds | mean d90 | mean d95 | mean d99 | mean LCC |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for num_nodes in sorted({int(row["num_nodes"]) for row in rows}):
        group = [row for row in rows if int(row["num_nodes"]) == num_nodes]
        lines.append(
            f"| {num_nodes} | {len(group)} | {np.mean([float(r['d90']) for r in group]):.1f} | "
            f"{np.mean([float(r['d95']) for r in group]):.1f} | {np.mean([float(r['d99']) for r in group]):.1f} | "
            f"{np.mean([float(r['largest_connected_component_size']) for r in group]):.1f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: if D_plan grows with sample size, report it honestly as a sampled empirical estimate. The key robustness check is whether the estimates remain far above physical state dimension and local intrinsic dimension over the sampled range.",
        ]
    )
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "pusht_sampling_convergence.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PushT D_plan sampling-convergence diagnostic.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--num_nodes_values", default="256,512,1024,2048")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--graph_mode", default="temporal_plus_knn_k10")
    parser.add_argument("--sampling_mode", default="contiguous_segments")
    parser.add_argument("--num_segments", type=int, default=50)
    parser.add_argument("--segment_len", type=int, default=100)
    parser.add_argument("--landmarks", type=int, default=1000)
    parser.add_argument("--max_embed_dim", type=int, default=512)
    parser.add_argument("--skip_estimate", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows: List[Dict[str, object]] = []
    for num_nodes in _parse_ints(args.num_nodes_values):
        for seed in _parse_ints(args.seeds):
            run_dir = output_dir / "sampling_convergence_work" / f"n{num_nodes}_seed{seed}"
            if not args.skip_estimate:
                run_dir = _run_one(args, output_dir, num_nodes, seed)
            stats = _read_csv(run_dir / "graph_statistics.csv")
            if not stats:
                print(f"[sampling_convergence] warning: missing graph_statistics.csv in {run_dir}")
                continue
            row = stats[0]
            rows.append(
                {
                    "num_nodes": num_nodes,
                    "seed": seed,
                    "graph_mode": row.get("graph_mode"),
                    "d80": row.get("d80"),
                    "d90": row.get("d90"),
                    "d95": row.get("d95"),
                    "d99": row.get("d99"),
                    "d100": row.get("d100"),
                    "positive_eigen_count": row.get("positive_eigen_count"),
                    "largest_connected_component_size": row.get("largest_component_size"),
                    "negative_energy_ratio": row.get("negative_energy_ratio"),
                }
            )
    _write_csv(output_dir / "spectra" / "pusht_sampling_convergence.csv", rows)
    _plot(rows, output_dir)
    _write_summary(rows, output_dir)
    print(f"[sampling_convergence] wrote {output_dir / 'spectra' / 'pusht_sampling_convergence.csv'}")


if __name__ == "__main__":
    main()
