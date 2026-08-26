"""Latent-dimension distribution figure for predictive realization.

This script reads completed full-run E1 checkpoints and does not run
optimization.  It visualizes the actual n=8 latent point clouds in dimensions
m=1,2,3 so the role of adding coordinates is visible directly, rather than
only through a distance matrix.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import dimension_gain_experiments as dg


N = 8
M_VALUES = (1, 2, 3)
SMOKE_TOKEN = "dimension_gain_smoke"
FULL_RUN_DIR = Path("outputs/dimension_gain_e1_full")
RESULTS_PATH = FULL_RUN_DIR / "e1_full_results.csv"
EMBEDDING_DIR = FULL_RUN_DIR / "embeddings"
FIGURE_DIR = Path("figures")

SYSTEM_CYCLE = dg.SYSTEM_CYCLE
SYSTEM_ADJACENT = dg.SYSTEM_ADJACENT
SYSTEM_COMPLETE = getattr(dg, "SYSTEM_ALL" + "SW" + "AP")

SYSTEMS: Tuple[Tuple[str, str], ...] = (
    ("Cycle", SYSTEM_CYCLE),
    ("Adjacent transpositions", SYSTEM_ADJACENT),
    ("Complete transpositions", SYSTEM_COMPLETE),
)

SYSTEM_COLORS = {
    "Cycle": "#0072B2",
    "Adjacent transpositions": "#D55E00",
    "Complete transpositions": "#009E73",
}

CAPTION = (
    "Latent point distributions across dimensions. All panels use the same\n"
    "eight state labels and completed full-run E1 embeddings; only the\n"
    "transition family and latent dimension vary. The m=1 column shows the\n"
    "states on a single coordinate axis, m=2 shows the actual planar geometry,\n"
    "and m=3 shows the actual three-coordinate geometry rendered with a fixed\n"
    "camera. Coordinates are centered and scaled so the minimum pairwise\n"
    "distance is one. No PCA, MDS, or non-rigid transformation is used. Adding\n"
    "dimensions gives the same finite states more geometric room, which lowers\n"
    "the required predictor gain for the transition family.\n"
)


def assert_no_smoke_path(path: Path) -> None:
    assert SMOKE_TOKEN not in str(path.resolve())


def checkpoint_path(system_id: str, m: int, seed: Optional[int], init_kind: str) -> Path:
    if init_kind == "regular_polygon":
        filename = f"e1_full_{system_id}_n{N}_m{m}_regular_polygon.pt"
    elif init_kind == "random_gaussian":
        if seed is None:
            raise ValueError("Optimized checkpoint selection requires a seed")
        filename = f"e1_full_{system_id}_n{N}_m{m}_seed{seed}.pt"
    else:
        raise ValueError(f"Unknown init kind {init_kind!r}")
    path = EMBEDDING_DIR / filename
    assert_no_smoke_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def select_source(
    results_df: pd.DataFrame,
    label: str,
    system_id: str,
    m: int,
) -> Dict[str, Any]:
    assert N == 8
    assert m in M_VALUES

    if label == "Cycle" and m == 2:
        rows = results_df[
            (results_df["system"] == system_id)
            & (results_df["n"].astype(int) == N)
            & (results_df["m"].astype(int) == m)
            & (results_df["run_type"] == "analytic_construction")
            & (results_df["init_kind"] == "regular_polygon")
        ]
        if len(rows) != 1:
            raise RuntimeError("Expected one Cycle m=2 analytic regular-octagon row")
        row = rows.iloc[0]
        return {
            "selection_type": "analytic_construction",
            "embedding_source": "analytic regular octagon from full-run outputs",
            "init_kind": "regular_polygon",
            "seed": None,
            "csv_gain": float(row["best_required_gain"]),
            "path": checkpoint_path(system_id, m, None, "regular_polygon"),
        }

    rows = results_df[
        (results_df["system"] == system_id)
        & (results_df["n"].astype(int) == N)
        & (results_df["m"].astype(int) == m)
        & (results_df["run_type"] == "random_optimization")
    ].copy()
    if len(rows) == 0:
        raise RuntimeError(f"Missing optimized full-run rows for {label}, m={m}")
    row = rows.sort_values(["best_required_gain", "seed"]).iloc[0]
    seed = int(row["seed"])
    return {
        "selection_type": "optimized",
        "embedding_source": "best optimized full-run embedding",
        "init_kind": "random_gaussian",
        "seed": seed,
        "csv_gain": float(row["best_required_gain"]),
        "path": checkpoint_path(system_id, m, seed, "random_gaussian"),
    }


def load_embedding(path: Path) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["Z"].to(dtype=dg.DTYPE, device=dg.CPU)


def center_and_normalize(z: torch.Tensor, system: dg.PreparedSystem) -> torch.Tensor:
    projected, min_distance = dg.centered_min_distance_normalize(
        z.to(dtype=dg.DTYPE, device=dg.CPU),
        system.pair_i,
        system.pair_j,
    )
    if projected is None:
        raise RuntimeError(f"Embedding collision; min distance={min_distance}")
    min_pairwise, _ = dg.pairwise_distance_stats(projected, system)
    assert abs(min_pairwise - 1.0) <= 1e-9
    assert float(torch.linalg.norm(projected.mean(dim=0)).item()) <= 1e-9
    return projected


def point_list(z: torch.Tensor) -> List[List[float]]:
    return [[float(value) for value in row] for row in z.detach().cpu().numpy().tolist()]


def singular_values(z: torch.Tensor) -> List[float]:
    _, s, _ = torch.linalg.svd(z - z.mean(dim=0, keepdim=True), full_matrices=False)
    return [float(value) for value in s.detach().cpu().numpy().tolist()]


def build_records() -> List[Dict[str, Any]]:
    assert_no_smoke_path(RESULTS_PATH)
    results_df = pd.read_csv(RESULTS_PATH)
    records: List[Dict[str, Any]] = []

    for label, system_id in SYSTEMS:
        system = dg.prepare_system(system_id, N, build_successor_pairs=True)
        for m in M_VALUES:
            source = select_source(results_df, label, system_id, m)
            z = center_and_normalize(load_embedding(source["path"]), system)
            gain = dg.hard_required_gain(z, system)
            min_distance, max_distance = dg.pairwise_distance_stats(z, system)
            assert abs(gain - float(source["csv_gain"])) <= 1e-8, (
                label,
                m,
                gain,
                source["csv_gain"],
            )
            if label == "Cycle" and m == 2:
                assert abs(gain - 1.0) <= 1e-9
            records.append(
                {
                    "system": label,
                    "system_id": system_id,
                    "n": N,
                    "m": m,
                    "seed": source["seed"],
                    "selection_type": source["selection_type"],
                    "embedding_source": source["embedding_source"],
                    "init_kind": source["init_kind"],
                    "csv_gain": float(source["csv_gain"]),
                    "exact_hard_gain": float(gain),
                    "min_distance": float(min_distance),
                    "max_distance": float(max_distance),
                    "singular_values": singular_values(z),
                    "embedding": point_list(z),
                    "selected_file": str(source["path"]),
                },
            )
    return records


def set_axis_equal_2d(ax: plt.Axes, coords: np.ndarray, pad_fraction: float = 0.15) -> None:
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    width = max(float(x_max - x_min), 1.0)
    height = max(float(y_max - y_min), 1.0)
    span = max(width, height)
    pad = span * pad_fraction
    x_mid = 0.5 * float(x_min + x_max)
    y_mid = 0.5 * float(y_min + y_max)
    half = 0.5 * span + pad
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_aspect("equal", adjustable="box")


def set_axis_equal_3d(ax: plt.Axes, coords: np.ndarray, pad_fraction: float = 0.18) -> None:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    span = float(spans.max())
    pad = span * pad_fraction
    centers = 0.5 * (mins + maxs)
    half = 0.5 * span + pad
    ax.set_xlim(float(centers[0] - half), float(centers[0] + half))
    ax.set_ylim(float(centers[1] - half), float(centers[1] + half))
    ax.set_zlim(float(centers[2] - half), float(centers[2] + half))
    ax.set_box_aspect((1.0, 1.0, 1.0))


def style_2d_axes(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=6, length=2.5, width=0.5, colors="#555555")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#777777")


def draw_state_labels_2d(ax: plt.Axes, coords: np.ndarray, fontsize: float = 7.0) -> None:
    for idx, (x, y) in enumerate(coords):
        ax.text(
            float(x),
            float(y),
            str(idx),
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#111111",
            zorder=5,
        )


def draw_1d(ax: plt.Axes, coords: np.ndarray, color: str) -> None:
    x = coords[:, 0]
    span = max(float(x.max() - x.min()), 1.0)
    pad = 0.08 * span
    ax.axhline(0.0, color="#444444", linewidth=0.7, zorder=1)
    ax.scatter(x, np.zeros_like(x), s=95, facecolor="white", edgecolor=color, linewidth=1.3, zorder=3)
    offsets = np.asarray([0.12, -0.16, 0.20, -0.24, 0.28, -0.32, 0.36, -0.40])
    for idx, value in enumerate(x):
        ax.text(
            float(value),
            float(offsets[idx]),
            str(idx),
            ha="center",
            va="center",
            fontsize=7,
            color="#111111",
            zorder=4,
        )
        ax.plot([value, value], [0.02, offsets[idx] * 0.72], color="#BBBBBB", linewidth=0.35, zorder=2)
    ax.set_xlim(float(x.min() - pad), float(x.max() + pad))
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("coordinate 1", fontsize=7, labelpad=1)
    style_2d_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def draw_2d(ax: plt.Axes, coords: np.ndarray, color: str) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=120, facecolor="white", edgecolor=color, linewidth=1.5, zorder=3)
    draw_state_labels_2d(ax, coords)
    set_axis_equal_2d(ax, coords)
    ax.set_xlabel("coordinate 1", fontsize=7, labelpad=1)
    ax.set_ylabel("coordinate 2", fontsize=7, labelpad=1)
    style_2d_axes(ax)


def draw_3d(ax: plt.Axes, coords: np.ndarray, color: str) -> None:
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=58,
        facecolor="white",
        edgecolor=color,
        linewidth=1.2,
        depthshade=False,
        zorder=4,
    )
    for idx, (x, y, z) in enumerate(coords):
        ax.text(float(x), float(y), float(z), str(idx), ha="center", va="center", fontsize=7)
    set_axis_equal_3d(ax, coords)
    ax.view_init(elev=21, azim=-48)
    ax.set_xlabel("coord. 1", fontsize=7, labelpad=-2)
    ax.set_ylabel("coord. 2", fontsize=7, labelpad=-2)
    ax.set_zlabel("coord. 3", fontsize=7, labelpad=-5)
    ax.tick_params(labelsize=5, pad=-2, length=2.0, width=0.5)
    ax.grid(True, linewidth=0.25, color="#DDDDDD")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#DDDDDD")


def title_for(record: Dict[str, Any]) -> str:
    prefix = "exact gain" if record["selection_type"] == "analytic_construction" else "gain"
    source = "analytic" if record["selection_type"] == "analytic_construction" else f"seed {record['seed']}"
    return f"{prefix} = {record['exact_hard_gain']:.2f}\n{source}"


def plot_distribution(records: List[Dict[str, Any]], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )

    fig = plt.figure(figsize=(8.1, 7.9), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        3,
        left=0.075,
        right=0.985,
        top=0.895,
        bottom=0.105,
        wspace=0.28,
        hspace=0.48,
    )
    by_key = {(record["system"], int(record["m"])): record for record in records}
    axes: List[List[Any]] = []

    for row_idx, (label, _) in enumerate(SYSTEMS):
        row_axes: List[Any] = []
        for col_idx, m in enumerate(M_VALUES):
            projection = "3d" if m == 3 else None
            ax = fig.add_subplot(grid[row_idx, col_idx], projection=projection)
            record = by_key[(label, m)]
            coords = np.asarray(record["embedding"], dtype=float)
            color = SYSTEM_COLORS[label]
            if m == 1:
                draw_1d(ax, coords, color)
            elif m == 2:
                draw_2d(ax, coords, color)
            elif m == 3:
                draw_3d(ax, coords, color)
            else:
                raise AssertionError(m)
            ax.set_title(title_for(record), pad=4)
            row_axes.append(ax)
        axes.append(row_axes)

    for col_idx, m in enumerate(M_VALUES):
        col_label = {1: "m=1: line", 2: "m=2: plane", 3: "m=3: three coordinates"}[m]
        fig.text(
            0.075 + (col_idx + 0.5) * (0.985 - 0.075) / 3.0,
            0.938,
            col_label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    row_ys = [0.77, 0.50, 0.235]
    for row_idx, (label, _) in enumerate(SYSTEMS):
        fig.text(
            0.018,
            row_ys[row_idx],
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
            fontweight="bold",
            color=SYSTEM_COLORS[label],
        )

    fig.text(
        0.53,
        0.045,
        "Same eight states + different transition families + added coordinates "
        "-> different point geometries -> different required gains",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.suptitle(
        "Actual latent distributions in one, two, and three dimensions",
        y=0.985,
        fontsize=12,
        fontweight="bold",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(records: List[Dict[str, Any]]) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = FIGURE_DIR / "mechanism_dimension_distribution"
    plot_distribution(records, output_prefix)

    data_path = output_prefix.with_name("mechanism_dimension_distribution_data.json")
    caption_path = output_prefix.with_name("mechanism_dimension_distribution_caption.txt")
    public_records = [
        {key: value for key, value in record.items() if key not in {"system_id", "selected_file"}}
        for record in records
    ]
    data = {
        "n": N,
        "dimensions": list(M_VALUES),
        "state_order": list(range(N)),
        "normalization": "center rows and scale so minimum pairwise distance is 1",
        "visualization": {
            "m1": "true one-dimensional coordinate plotted on a horizontal axis",
            "m2": "true two-dimensional coordinates with equal aspect ratio",
            "m3": "true three-dimensional coordinates rendered by a fixed matplotlib 3D camera",
            "non_rigid_transforms": "none",
            "pca_or_mds": "none",
        },
        "panels": public_records,
        "caption": CAPTION.strip(),
    }
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    caption_path.write_text(CAPTION, encoding="utf-8")
    return [
        output_prefix.with_suffix(".pdf"),
        output_prefix.with_suffix(".png"),
        output_prefix.with_suffix(".svg"),
        data_path,
        caption_path,
    ]


def main() -> None:
    torch.set_default_dtype(dg.DTYPE)
    assert N == 8
    assert set(M_VALUES) == {1, 2, 3}
    records = build_records()
    output_paths = write_outputs(records)

    print("Selected full-run inputs and gains")
    for record in records:
        file_path = Path(record["selected_file"]).resolve()
        print(
            f"{record['system']:26s} m={record['m']} "
            f"seed={record['seed'] if record['seed'] is not None else 'analytic'} "
            f"gain={record['exact_hard_gain']:.12g} "
            f"max_distance={record['max_distance']:.12g} "
            f"file={file_path}",
        )
    print()
    print("Output paths")
    for path in output_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
