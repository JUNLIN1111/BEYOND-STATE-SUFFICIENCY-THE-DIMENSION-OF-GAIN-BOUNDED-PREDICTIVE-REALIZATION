from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _parse_csv(path: Path) -> List[Dict[str, str]]:
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


def _parse_list(text: str, cast):
    return [cast(item) for item in str(text).replace(",", " ").split() if item.strip()]


def _run_command(cmd: List[str]) -> None:
    print("[quotient_runner] running:")
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _collect_run(work_dir: Path, method_label: str, resolution_label: str) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    csv_path = work_dir / "spectra" / "clustered_quotient_diagnostic.csv"
    json_path = work_dir / "spectra" / "clustered_quotient_spectra.json"
    rows: List[Dict[str, object]] = []
    for row in _parse_csv(csv_path):
        enriched: Dict[str, object] = dict(row)
        enriched["method_label"] = method_label
        enriched["resolution_label"] = resolution_label
        rows.append(enriched)
    payload = {}
    if json_path.exists():
        with json_path.open() as file:
            payload = json.load(file)
    return rows, {f"{method_label}:{resolution_label}": payload}


def _float(row: Dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _plot_outputs(rows: List[Dict[str, object]], hist_payload: Dict[str, object], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[quotient_runner] matplotlib unavailable; skipping plots.", flush=True)
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    method_order = ["fixed_radius", "kcenter"]
    colors = {"fixed_radius": "#4C78A8", "kcenter": "#F58518"}
    markers = {"fixed_radius": "o", "kcenter": "s"}

    fig, ax = plt.subplots(figsize=(6.7, 4.0), facecolor="white")
    for method in method_order:
        subset = [row for row in rows if row.get("method_label") == method]
        subset = sorted(subset, key=lambda row: _float(row, "num_clusters"))
        if not subset:
            continue
        x = [_float(row, "num_clusters") for row in subset]
        for key, alpha in [("d90", 1.0), ("d95", 0.75), ("d99", 0.55)]:
            ax.plot(
                x,
                [_float(row, key) for row in subset],
                marker=markers[method],
                color=colors[method],
                alpha=alpha,
                label=f"{method} {key}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of quotient clusters")
    ax.set_ylabel("D_plan(q)")
    ax.set_title("PushT quotient graph spectra")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_quotient_d90_d95_d99_vs_resolution.png", dpi=260)
    fig.savefig(plots_dir / "pusht_quotient_d90_d95_d99_vs_resolution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.7, 4.0), facecolor="white")
    for method in method_order:
        subset = [row for row in rows if row.get("method_label") == method and np.isfinite(_float(row, "cluster_radius_median"))]
        subset = sorted(subset, key=lambda row: _float(row, "cluster_radius_median"))
        if not subset:
            continue
        x = [_float(row, "cluster_radius_median") for row in subset]
        ax.plot(x, [_float(row, "d90") for row in subset], marker=markers[method], label=f"{method} d90")
        ax.plot(x, [_float(row, "d95") for row in subset], marker=markers[method], alpha=0.7, label=f"{method} d95")
        ax.plot(x, [_float(row, "d99") for row in subset], marker=markers[method], alpha=0.5, label=f"{method} d99")
    ax.set_xlabel("median cluster radius in standardized state space")
    ax.set_ylabel("D_plan(q)")
    ax.set_title("Resolution-normalized quotient spectra")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_quotient_d90_d95_d99_vs_median_radius.png", dpi=260)
    fig.savefig(plots_dir / "pusht_quotient_d90_d95_d99_vs_median_radius.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="white")
    for method in method_order:
        subset = [row for row in rows if row.get("method_label") == method]
        subset = sorted(subset, key=lambda row: _float(row, "num_clusters"))
        if subset:
            ax.plot(
                [_float(row, "num_clusters") for row in subset],
                [_float(row, "negative_energy_ratio") for row in subset],
                marker=markers[method],
                color=colors[method],
                label=method,
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of quotient clusters")
    ax.set_ylabel("negative energy ratio")
    ax.set_title("Non-Euclidean distance energy")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_quotient_negative_energy_vs_resolution.png", dpi=260)
    fig.savefig(plots_dir / "pusht_quotient_negative_energy_vs_resolution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="white")
    for method in method_order:
        subset = [row for row in rows if row.get("method_label") == method]
        subset = sorted(subset, key=lambda row: _float(row, "num_clusters"))
        if subset:
            ax.plot(
                [_float(row, "num_clusters") for row in subset],
                [_float(row, "average_degree") for row in subset],
                marker=markers[method],
                color=colors[method],
                label=method,
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of quotient clusters")
    ax.set_ylabel("average quotient degree")
    ax.set_title("Quotient graph degree")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_quotient_degree_distribution_by_method.png", dpi=260)
    fig.savefig(plots_dir / "pusht_quotient_degree_distribution_by_method.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="white")
    for method in method_order:
        subset = [row for row in rows if row.get("method_label") == method]
        subset = sorted(subset, key=lambda row: _float(row, "num_clusters"))
        if subset:
            ax.plot(
                [_float(row, "num_clusters") for row in subset],
                [_float(row, "shortest_path_median") for row in subset],
                marker=markers[method],
                color=colors[method],
                label=f"{method} median",
            )
            ax.plot(
                [_float(row, "num_clusters") for row in subset],
                [_float(row, "shortest_path_p90") for row in subset],
                marker=markers[method],
                linestyle="--",
                color=colors[method],
                alpha=0.75,
                label=f"{method} p90",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of quotient clusters")
    ax.set_ylabel("shortest-path length")
    ax.set_title("Quotient shortest-path distribution")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_quotient_shortest_path_histogram_by_method.png", dpi=260)
    fig.savefig(plots_dir / "pusht_quotient_shortest_path_histogram_by_method.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="white")
    for method in method_order:
        subset = [row for row in rows if row.get("method_label") == method]
        subset = sorted(subset, key=lambda row: _float(row, "num_clusters"))
        if subset:
            ax.plot(
                [_float(row, "num_clusters") for row in subset],
                [_float(row, "cluster_radius_median") for row in subset],
                marker=markers[method],
                color=colors[method],
                label=f"{method} median",
            )
            ax.plot(
                [_float(row, "num_clusters") for row in subset],
                [_float(row, "cluster_radius_p90") for row in subset],
                marker=markers[method],
                linestyle="--",
                color=colors[method],
                alpha=0.75,
                label=f"{method} p90",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of quotient clusters")
    ax.set_ylabel("cluster radius")
    ax.set_title("Quotient cluster radius distribution")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "pusht_quotient_cluster_radius_distribution.png", dpi=260)
    fig.savefig(plots_dir / "pusht_quotient_cluster_radius_distribution.pdf")
    plt.close(fig)

    with (output_dir / "spectra" / "pusht_quotient_fixed_radius_kcenter_histograms.json").open("w") as file:
        json.dump(hist_payload, file, indent=2)


def _write_summary(rows: List[Dict[str, object]], output_dir: Path) -> None:
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PushT Fixed-Radius and K-Center Transition Quotient",
        "",
        "This diagnostic uses physical/state similarity only for aggregation. Quotient graph edges are induced only by observed temporal transitions crossing cluster assignments.",
        "",
        "| method | clusters | LCC | components | median radius | avg degree | d90 | d95 | d99 | negative energy | P(near|far) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item.get("method_label", "")), _float(item, "num_clusters"))):
        lines.append(
            f"| {row.get('method_label')} | {int(_float(row, 'num_clusters'))} | "
            f"{int(_float(row, 'largest_connected_component_size'))} | {int(_float(row, 'num_connected_components'))} | "
            f"{_float(row, 'cluster_radius_median'):.3f} | {_float(row, 'average_degree'):.2f} | "
            f"{int(_float(row, 'd90'))} | {int(_float(row, 'd95'))} | {int(_float(row, 'd99'))} | "
            f"{_float(row, 'negative_energy_ratio'):.4f} | {_float(row, 'p_physical_near_given_transition_far'):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a quotient transition-distance estimate, not a new abstraction algorithm. Similarity chooses bins; observed feasible transitions define graph edges. Stable high D_plan under several conservative quotient resolutions would support the claim that PushT transition geometry requires more directions than the raw 7D physical state or local 2D intrinsic estimates suggest.",
            "",
            "Caveat: physical-near / transition-far pairs are empirical shortcut candidates because offline coverage can miss feasible transitions.",
        ]
    )
    (summary_dir / "pusht_quotient_fixed_radius_kcenter.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PushT fixed-radius and k-center quotient diagnostics.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--step_key", default="step_idx")
    parser.add_argument("--radius_epsilons", default="0.12,0.18,0.25,0.35")
    parser.add_argument("--radius_max_clusters", type=int, default=4096)
    parser.add_argument("--kcenter_clusters", default="256,512,1024,2048")
    parser.add_argument("--max_fit_points", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run quotient jobs even if their CSV exists.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    script = Path(__file__).resolve().parent / "clustered_transition_quotient_diagnostic.py"
    all_rows: List[Dict[str, object]] = []
    hist_payload: Dict[str, object] = {}

    for epsilon in _parse_list(args.radius_epsilons, float):
        label = f"eps{epsilon:g}".replace(".", "p")
        work_dir = output_dir / "quotient_work" / f"fixed_radius_{label}"
        csv_path = work_dir / "spectra" / "clustered_quotient_diagnostic.csv"
        if args.force or not csv_path.exists():
            _run_command(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    args.dataset,
                    "--state_key",
                    args.state_key,
                    "--episode_key",
                    args.episode_key,
                    "--step_key",
                    args.step_key,
                    "--clustering_method",
                    "radius",
                    "--num_clusters",
                    str(args.radius_max_clusters),
                    "--radius_epsilon",
                    str(epsilon),
                    "--max_fit_points",
                    str(args.max_fit_points),
                    "--batch_size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                    "--output_dir",
                    str(work_dir),
                ]
            )
        rows, payload = _collect_run(work_dir, "fixed_radius", label)
        all_rows.extend(rows)
        hist_payload.update(payload)

    for n_clusters in _parse_list(args.kcenter_clusters, int):
        label = f"M{n_clusters}"
        work_dir = output_dir / "quotient_work" / f"kcenter_{label}"
        csv_path = work_dir / "spectra" / "clustered_quotient_diagnostic.csv"
        if args.force or not csv_path.exists():
            _run_command(
                [
                    sys.executable,
                    str(script),
                    "--dataset",
                    args.dataset,
                    "--state_key",
                    args.state_key,
                    "--episode_key",
                    args.episode_key,
                    "--step_key",
                    args.step_key,
                    "--clustering_method",
                    "kcenter",
                    "--num_clusters",
                    str(n_clusters),
                    "--max_fit_points",
                    str(args.max_fit_points),
                    "--batch_size",
                    str(args.batch_size),
                    "--seed",
                    str(args.seed),
                    "--output_dir",
                    str(work_dir),
                ]
            )
        rows, payload = _collect_run(work_dir, "kcenter", label)
        all_rows.extend(rows)
        hist_payload.update(payload)

    combined_csv = output_dir / "spectra" / "pusht_quotient_fixed_radius_kcenter.csv"
    _write_csv(combined_csv, all_rows)
    _plot_outputs(all_rows, hist_payload, output_dir)
    _write_summary(all_rows, output_dir)
    print(f"[quotient_runner] wrote {combined_csv}")
    print(f"[quotient_runner] wrote {output_dir / 'summaries' / 'pusht_quotient_fixed_radius_kcenter.md'}")


if __name__ == "__main__":
    main()
