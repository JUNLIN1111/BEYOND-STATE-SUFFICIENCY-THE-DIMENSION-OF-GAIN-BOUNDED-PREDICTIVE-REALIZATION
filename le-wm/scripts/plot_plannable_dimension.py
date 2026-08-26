from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as file:
        return list(csv.DictReader(file))


def _float(row: Dict[str, str], key: str, default=float("nan")) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def _plot_stress(stress_rows: List[Dict[str, str]], output_dir: Path) -> None:
    plt.figure(figsize=(7.5, 5))
    for mode in sorted({row["graph_mode"] for row in stress_rows}):
        rows = sorted([row for row in stress_rows if row["graph_mode"] == mode], key=lambda r: _float(r, "embed_dim"))
        plt.plot([_float(r, "embed_dim") for r in rows], [_float(r, "stress") for r in rows], marker="o", label=mode)
    plt.xscale("log", base=2)
    plt.xlabel("embedding dimension")
    plt.ylabel("MDS stress")
    plt.title("Reachability-distance embedding stress")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_dir / "stress_curve_vs_dim")


def _plot_weighted_stress(rows: List[Dict[str, str]], output_dir: Path, graph_mode: str = "temporal_plus_knn") -> None:
    if not rows:
        return
    plt.figure(figsize=(8, 5))
    for weighting in sorted({row["weighting"] for row in rows if row.get("graph_mode") == graph_mode}):
        group = sorted(
            [row for row in rows if row.get("graph_mode") == graph_mode and row.get("weighting") == weighting],
            key=lambda row: _float(row, "embed_dim"),
        )
        plt.plot([_float(row, "embed_dim") for row in group], [_float(row, "stress") for row in group], marker="o", label=weighting)
    plt.xscale("log", base=2)
    plt.xlabel("embedding dimension")
    plt.ylabel("weighted MDS stress")
    plt.title(f"Planner-weighted stress curves ({graph_mode})")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_dir / "stress_curve_by_weighting")


def _plot_false_shortcut_vs_dim(rows: List[Dict[str, str]], output_dir: Path, graph_mode: str = "temporal_plus_knn") -> None:
    if not rows:
        return
    group = sorted([row for row in rows if row.get("graph_mode") == graph_mode], key=lambda row: _float(row, "embed_dim"))
    if not group:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(
        [_float(row, "embed_dim") for row in group],
        [_float(row, "false_shortcut_rate_conditional") for row in group],
        marker="o",
        label="P(emb near | geo far)",
    )
    plt.plot(
        [_float(row, "embed_dim") for row in group],
        [_float(row, "false_shortcut_joint_rate") for row in group],
        marker="s",
        label="P(emb near and geo far)",
    )
    plt.xscale("log", base=2)
    plt.xlabel("embedding dimension")
    plt.ylabel("false shortcut rate")
    plt.title(f"False shortcuts vs embedding dimension ({graph_mode})")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_dir / "false_shortcut_vs_dim")


def _plot_spectrum(spectrum: Dict[str, object], output_dir: Path) -> None:
    plt.figure(figsize=(7.5, 5))
    for mode, data in spectrum.items():
        eig = np.asarray(data.get("positive_eigenvalues", []), dtype=np.float64)
        if eig.size == 0:
            continue
        energy = np.cumsum(eig) / np.sum(eig)
        plt.plot(np.arange(1, eig.size + 1), energy, label=f"{mode} rank90={data.get('rank90')}")
    plt.xlabel("positive MDS component")
    plt.ylabel("cumulative positive-spectrum energy")
    plt.title("MDS positive spectrum")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    _savefig(output_dir / "mds_positive_spectrum")


def _plot_false_shortcuts(false_rows: List[Dict[str, str]], output_dir: Path) -> None:
    labels = [row["graph_mode"] for row in false_rows]
    values = [_float(row, "false_shortcut_rate") for row in false_rows]
    plt.figure(figsize=(6.5, 4.5))
    plt.bar(labels, values)
    plt.ylabel("false shortcut rate")
    plt.title("Raw-near but graph-far state pairs")
    plt.grid(axis="y", alpha=0.25)
    _savefig(output_dir / "false_shortcut_rate")


def _plot_trained(summary: Dict[str, object], trained_rows: List[Dict[str, str]], output_dir: Path) -> None:
    rows = sorted(trained_rows, key=lambda row: _float(row, "latent_dim"))
    dims = [_float(row, "latent_dim") for row in rows]
    sr = [_float(row, "success_rate") for row in rows]
    score_alias = [_float(row, "score_alias") for row in rows]
    d_plan = float(summary.get("d_plan_90", float("nan")))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(dims, sr, marker="o", color="tab:blue", label="SR")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("trained latent dimension")
    ax1.set_ylabel("closed-loop SR", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(dims, score_alias, marker="s", color="tab:red", label="ScoreAlias")
    ax2.set_ylabel("ScoreAlias", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    if np.isfinite(d_plan):
        ax1.axvline(d_plan, linestyle="--", color="black", alpha=0.7, label=f"d_plan90={d_plan:g}")
    fig.suptitle("Estimated plannable dimension vs trained model outcomes")
    fig.tight_layout()
    fig.savefig(output_dir / "estimated_d_plan_vs_trained_dims.png", dpi=220)
    fig.savefig(output_dir / "estimated_d_plan_vs_trained_dims.pdf")
    plt.close(fig)


def _make_toy(output_dir: Path) -> None:
    # Two close parallel chains in 2D; graph movement between chains only through one branch point.
    n = 80
    x = np.linspace(0, 1, n)
    top = np.stack([x, np.zeros_like(x)], axis=1)
    bottom = np.stack([x, 0.08 * np.ones_like(x)], axis=1)
    points = np.concatenate([top, bottom], axis=0)
    graph = np.full((2 * n, 2 * n), np.inf)
    np.fill_diagonal(graph, 0)
    for offset in [0, n]:
        for i in range(n - 1):
            graph[offset + i, offset + i + 1] = 1
            graph[offset + i + 1, offset + i] = 1
    graph[0, n] = graph[n, 0] = 1
    for k in range(2 * n):
        graph = np.minimum(graph, graph[:, [k]] + graph[[k], :])
    evals, evecs = _classical_mds(graph)
    coords3 = evecs[:, :3] * np.sqrt(np.maximum(evals[:3], 0))[None, :]
    stress_rows = []
    left, right = np.triu_indices(2 * n, k=1)
    geo = graph[left, right]
    for d in [1, 2, 3, 4, 8]:
        coords = evecs[:, :d] * np.sqrt(np.maximum(evals[:d], 0))[None, :]
        dist = np.linalg.norm(coords[left] - coords[right], axis=1)
        stress_rows.append((d, np.sum((dist - geo) ** 2) / np.sum(geo ** 2)))
    plt.figure(figsize=(5, 4))
    plt.scatter(points[:, 0], points[:, 1], s=12)
    plt.title("Toy physical projection: false shortcut")
    _savefig(output_dir / "toy_physical_projection")
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords3[:, 0], coords3[:, 1], coords3[:, 2], s=12)
    fig.tight_layout()
    fig.savefig(output_dir / "toy_unfolded_embedding_3d.png", dpi=220)
    fig.savefig(output_dir / "toy_unfolded_embedding_3d.pdf")
    plt.close(fig)
    plt.figure(figsize=(5, 4))
    plt.plot([x[0] for x in stress_rows], [x[1] for x in stress_rows], marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("embedding dim")
    plt.ylabel("stress")
    plt.title("Toy stress curve")
    _savefig(output_dir / "toy_stress_curve")


def _classical_mds(D: np.ndarray):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ np.square(D) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def _write_summary_md(
    input_dir: Path,
    output_dir: Path,
    summary: Dict[str, object],
    false_rows: List[Dict[str, str]],
    intrinsic_rows: List[Dict[str, str]],
    coverage_rows: List[Dict[str, str]],
) -> None:
    pca = next((row for row in intrinsic_rows if row.get("baseline") == "pca"), {})
    two_nn = next((row for row in intrinsic_rows if row.get("baseline") == "two_nn"), {})
    main_false = next((row for row in false_rows if row.get("graph_mode") == summary.get("main_graph_mode")), {})
    lines = [
        "# Plannable Dimension Summary",
        "",
        f"Feature mode: `{summary.get('feature_mode')}` from `{summary.get('feature_source')}`.",
        f"Nodes: {summary.get('num_nodes')}, feature dim: {summary.get('feature_dim')}, landmarks: {summary.get('landmarks')}.",
        "",
        "## Raw / Local Dimension Baselines",
        "",
        f"- PCA rank90: {pca.get('rank90', 'N/A')}",
        f"- PCA rank95: {pca.get('rank95', 'N/A')}",
        f"- PCA participation ratio: {pca.get('participation_ratio', 'N/A')}",
        f"- TwoNN ID estimate: {two_nn.get('id_estimate', 'N/A')}",
        "",
        "## False Shortcuts",
        "",
        f"- Main graph mode: `{summary.get('main_graph_mode')}`",
        f"- False shortcut rate: {main_false.get('false_shortcut_rate', 'N/A')}",
        f"- Median graph/raw ratio among raw-near pairs: {main_false.get('median_geo_raw_ratio_among_raw_near', 'N/A')}",
        "",
        "## Estimated Plannable Dimension",
        "",
        f"- d_plan_50: {summary.get('d_plan_50')}",
        f"- d_plan_80: {summary.get('d_plan_80')}",
        f"- d_plan_90: {summary.get('d_plan_90')}",
        f"- d_plan_95: {summary.get('d_plan_95')}",
        f"- d_plan_96: {summary.get('d_plan_96')}",
        f"- d_plan_97: {summary.get('d_plan_97')}",
        f"- d_plan_98: {summary.get('d_plan_98')}",
        f"- d_plan_99: {summary.get('d_plan_99')}",
        f"- d_plan_99.5: {summary.get('d_plan_995')}",
        f"- d_plan_99.9: {summary.get('d_plan_999')}",
        f"- d_plan_100: {summary.get('d_plan_100')}",
        f"- d_plan_stress: {summary.get('d_plan_stress')}",
        "",
        "## Why d_plan_stress can disagree with d_plan_90",
        "",
        "The average MDS stress threshold is an average-distance distortion criterion. It can be passed in low dimension when the coarse global trend of graph distances is easy to approximate.",
        "The spectral criterion asks how many positive MDS components are needed to explain reachability-distance energy. A broad positive spectrum means many weaker but non-negligible directions exist.",
        "Planner failure is more sensitive to false shortcuts: graph-far pairs that remain Euclidean-near after embedding.",
        "",
        "## Cautious Interpretation",
        "",
        "This diagnostic estimates how many Euclidean dimensions are needed to preserve a data-derived reachability geometry.",
        "It is not a theorem that exactly predicts optimal latent width.",
        "The intended comparison is whether physical/local dimension underestimates the width needed for planning, and whether trained model performance saturates near the estimated plannable dimension.",
        "",
        "Suggested language:",
        "",
        "> The average MDS stress threshold is passed at dimension 2, indicating that the coarse global trend of graph distances can be approximated in low dimension. However, the positive MDS spectrum is broad: 96 dimensions are required to explain 90% of the reachability-distance energy. This suggests that the planning geometry contains many weaker but non-negligible directions. Since latent planners are sensitive to false shortcuts rather than average distance error, we additionally report the rate at which graph-far pairs remain Euclidean-near as a function of embedding dimension.",
    ]
    (input_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot plannable dimension diagnostic outputs.")
    parser.add_argument("--input_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--make_toy", action="store_true")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (input_dir / "plannable_dimension_summary.json").open() as file:
        summary = json.load(file)["summary"]
    with (input_dir / "mds_spectrum.json").open() as file:
        spectrum = json.load(file)
    stress_rows = _read_csv(input_dir / "dimension_stress_curve.csv")
    weighted_stress_rows = _read_csv(input_dir / "stress_curve_by_weighting.csv")
    false_vs_dim_rows = _read_csv(input_dir / "false_shortcut_vs_dim.csv")
    coverage_rows = _read_csv(input_dir / "spectrum_coverage.csv")
    false_rows = _read_csv(input_dir / "false_shortcut_summary.csv")
    intrinsic_rows = _read_csv(input_dir / "intrinsic_dim_baselines.csv")
    trained_rows = _read_csv(input_dir / "trained_dim_comparison.csv")

    _plot_stress(stress_rows, output_dir)
    _plot_weighted_stress(weighted_stress_rows, output_dir, summary.get("main_graph_mode", "temporal_plus_knn"))
    _plot_false_shortcut_vs_dim(false_vs_dim_rows, output_dir, summary.get("main_graph_mode", "temporal_plus_knn"))
    _plot_spectrum(spectrum, output_dir)
    _plot_false_shortcuts(false_rows, output_dir)
    _plot_trained(summary, trained_rows, output_dir)
    if args.make_toy:
        _make_toy(output_dir)
    _write_summary_md(input_dir, output_dir, summary, false_rows, intrinsic_rows, coverage_rows)
    print(f"[plot_d_plan] wrote plots to {output_dir}")
    print(f"[plot_d_plan] wrote summary to {input_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
