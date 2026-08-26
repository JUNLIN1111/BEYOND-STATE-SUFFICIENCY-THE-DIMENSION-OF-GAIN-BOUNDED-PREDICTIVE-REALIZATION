"""Dimension-wise transition-arrow mechanism figure.

This version makes the nesting of dimensions explicit. For a target dimension
m, it considers completed full-run embeddings from every source dimension
k <= m, pads lower-dimensional embeddings with zero coordinates, and selects
the candidate with the smallest L=1 finite-state oracle prediction MSE.

No embedding optimization or predictor training is performed.
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
from matplotlib.patches import FancyArrowPatch

import dimension_gain_experiments as dg
import make_mechanism_prediction_arrows as pa


N = 8
M_VALUES = (1, 2, 3)
GAIN_BUDGET = 1.0
RESULTS_PATH = Path("outputs/dimension_gain_e1_full/e1_full_results.csv")
FIGURE_DIR = Path("figures")
OUTPUT_PREFIX = FIGURE_DIR / "mechanism_dimension_transition_arrows"
ERROR_TABLE_PATH = FIGURE_DIR / "mechanism_prediction_error_table_monotone.csv"
SMOKE_TOKEN = "dimension_gain_smoke"

DISPLAY_NAME = {
    "Cycle": "Cycle",
    "Adjacent transpositions": "Adjacent pair actions",
    "Complete transpositions": "All-pair actions",
}
SYSTEM_ORDER = ("Cycle", "Adjacent transpositions", "Complete transpositions")
SYSTEM_COLORS = {
    "Cycle": "#0072B2",
    "Adjacent transpositions": "#D55E00",
    "Complete transpositions": "#009E73",
}

CAPTION = (
    "Dimension-wise transition-arrow mechanism. Columns show target latent\n"
    "dimensions m=1,2,3 for the same eight states. For each target dimension,\n"
    "completed full-run embeddings from lower dimensions are also allowed by\n"
    "padding unused coordinates with zeros, so the plotted profile respects the\n"
    "nested feasible sets. Colored arrows show one true transition action in\n"
    "the selected latent geometry. The first column in each row summarizes the\n"
    "transition family: one cyclic action, one action for each neighboring pair,\n"
    "or one action for every unordered pair. Reported error is the finite-state\n"
    "numerical oracle MSE at gain budget L=1. When a higher-dimensional optimizer\n"
    "finds a worse predictor-error geometry, the figure uses the lower-dimensional\n"
    "geometry with the extra coordinate set to zero.\n"
)


def assert_no_smoke_path(path: Path) -> None:
    assert SMOKE_TOKEN not in str(path.resolve())


def circle_points(radius: float = 1.0) -> np.ndarray:
    angles = np.linspace(0.5 * math.pi, 0.5 * math.pi - 2.0 * math.pi, N, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def add_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    arrowstyle: str = "-|>",
    linewidth: float = 1.1,
    alpha: float = 1.0,
    shrink: float = 0.13,
    rad: float = 0.0,
    mutation_scale: float = 8.5,
    zorder: int = 2,
) -> None:
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length <= 1e-12:
        return
    unit = vec / length
    patch = FancyArrowPatch(
        start + shrink * unit,
        end - shrink * unit,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(patch)


def draw_nodes(ax: plt.Axes, coords: np.ndarray, color: str, size: float = 118.0) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=size, facecolor="white", edgecolor=color, linewidth=1.3, zorder=4)
    for idx, (x, y) in enumerate(coords):
        ax.text(float(x), float(y), str(idx), ha="center", va="center", fontsize=7.2, zorder=5)


def clean_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_family_schematic(ax: plt.Axes, system_label: str) -> None:
    display = DISPLAY_NAME[system_label]
    color = SYSTEM_COLORS[system_label]
    if system_label == "Cycle":
        coords = circle_points()
        for i in range(N):
            add_arrow(ax, coords[i], coords[(i + 1) % N], color=color, shrink=0.18, linewidth=1.15)
        draw_nodes(ax, coords, color, size=118.0)
        ax.text(0.5, -0.10, "1 cyclic action", transform=ax.transAxes, ha="center", fontsize=7.6)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.32, 1.42)
    elif system_label == "Adjacent transpositions":
        coords = np.stack([np.linspace(-2.15, 2.15, N), np.zeros(N)], axis=1)
        for i in range(N - 1):
            add_arrow(
                ax,
                coords[i],
                coords[i + 1],
                color=color,
                arrowstyle="<->",
                shrink=0.07,
                linewidth=1.12,
                mutation_scale=8.0,
            )
        draw_nodes(ax, coords, color, size=102.0)
        ax.text(0.5, -0.12, "7 neighboring-pair actions", transform=ax.transAxes, ha="center", fontsize=7.6)
        ax.set_xlim(-2.45, 2.45)
        ax.set_ylim(-0.76, 0.76)
    else:
        coords = circle_points()
        for i in range(N):
            for j in range(i + 1, N):
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color=color,
                    linewidth=0.62,
                    alpha=0.22,
                    zorder=1,
                )
        draw_nodes(ax, coords, color, size=118.0)
        ax.text(0.5, -0.10, "28 all-pair actions", transform=ax.transAxes, ha="center", fontsize=7.6)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.32, 1.42)
    ax.set_title(display, fontsize=9.2, fontweight="bold", pad=4)
    clean_axis(ax)


def source_candidate(
    results_df: pd.DataFrame,
    label: str,
    system_id: str,
    system: dg.PreparedSystem,
    target_m: int,
    source_m: int,
) -> Dict[str, Any]:
    source = pa.select_source(results_df, label, system_id, source_m)
    z_raw = pa.load_embedding(source["path"])
    z_padded = dg.pad_embedding(z_raw, target_m)
    z = pa.center_and_normalize(z_padded, system)
    gain = dg.hard_required_gain(z, system)
    witness = pa.find_hard_gain_witness(z, system, label)
    oracle = pa.oracle_mse_at_budget(z, system, GAIN_BUDGET)
    assert oracle["num_failed_actions"] == 0, (label, target_m, source_m)
    assert abs(float(witness["ratio"]) - gain) <= 1e-10
    if source_m == target_m:
        assert abs(gain - float(source["csv_gain"])) <= 1e-8, (
            label,
            target_m,
            gain,
            source["csv_gain"],
        )
    selected_action = int(witness["action_index"])
    min_distance, max_distance = dg.pairwise_distance_stats(z, system)
    return {
        "system": label,
        "display_system": DISPLAY_NAME[label],
        "n": N,
        "target_m": int(target_m),
        "source_m": int(source_m),
        "padded_from_lower_dimension": bool(source_m < target_m),
        "seed": source["seed"],
        "selection_type": source["selection_type"],
        "embedding_source": source["embedding_source"],
        "init_kind": source["init_kind"],
        "source_csv_gain": float(source["csv_gain"]),
        "exact_hard_gain": float(gain),
        "gain_budget": float(GAIN_BUDGET),
        "oracle_mse_at_budget": float(oracle["oracle_mse"]),
        "oracle_solver": oracle["solver"],
        "oracle_num_actions": oracle["num_actions"],
        "oracle_max_constraint_violation": oracle["max_constraint_violation"],
        "min_distance": float(min_distance),
        "max_distance": float(max_distance),
        "witness": witness,
        "drawn_action_index": selected_action,
        "drawn_action_label": pa.action_label(label, system, selected_action),
        "drawn_transition": system.transitions_np[selected_action].astype(int).tolist(),
        "embedding": pa.point_list(z),
        "selected_file": str(source["path"]),
    }


def choose_best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(
        candidates,
        key=lambda item: (
            round(float(item["oracle_mse_at_budget"]), 15),
            round(float(item["exact_hard_gain"]), 12),
            int(item["source_m"]),
            -1 if item["seed"] is None else int(item["seed"]),
        ),
    )[0]


def build_records() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    assert N == 8
    assert set(M_VALUES) == {1, 2, 3}
    assert_no_smoke_path(RESULTS_PATH)
    results_df = pd.read_csv(RESULTS_PATH)
    selected_records: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []
    system_ids = dict(pa.SYSTEMS)

    for label in SYSTEM_ORDER:
        system_id = system_ids[label]
        system = dg.prepare_system(system_id, N, build_successor_pairs=True)
        previous_mse = math.inf
        for target_m in M_VALUES:
            candidates = [
                source_candidate(results_df, label, system_id, system, target_m, source_m)
                for source_m in range(1, target_m + 1)
            ]
            all_candidates.extend(candidates)
            selected = choose_best_candidate(candidates)
            selected = dict(selected)
            selected["candidate_count"] = len(candidates)
            selected["selection_rule"] = (
                "minimum L=1 oracle MSE among completed source dimensions k<=target_m, "
                "after zero-padding lower-dimensional embeddings"
            )
            selected_records.append(selected)
            assert float(selected["oracle_mse_at_budget"]) <= previous_mse + 1e-8, (
                label,
                target_m,
                selected["oracle_mse_at_budget"],
                previous_mse,
            )
            previous_mse = min(previous_mse, float(selected["oracle_mse_at_budget"]))
    return selected_records, all_candidates


def action_edges(transition: List[int]) -> List[Tuple[int, int]]:
    return [(idx, int(dst)) for idx, dst in enumerate(transition) if int(dst) != idx]


def set_axis_equal_2d(ax: plt.Axes, coords: np.ndarray, pad_fraction: float = 0.17) -> None:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = max(float((maxs - mins).max()), 1.0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * span + pad_fraction * span
    ax.set_xlim(float(center[0] - half), float(center[0] + half))
    ax.set_ylim(float(center[1] - half), float(center[1] + half))
    ax.set_aspect("equal", adjustable="box")


def set_axis_equal_3d(ax: plt.Axes, coords: np.ndarray, pad_fraction: float = 0.18) -> None:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    span = float(spans.max())
    center = 0.5 * (mins + maxs)
    half = 0.5 * span + pad_fraction * span
    ax.set_xlim(float(center[0] - half), float(center[0] + half))
    ax.set_ylim(float(center[1] - half), float(center[1] + half))
    ax.set_zlim(float(center[2] - half), float(center[2] + half))
    ax.set_box_aspect((1.0, 1.0, 1.0))


def style_2d_axes(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=5.2, length=2.1, width=0.45, colors="#555555")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#8A8A8A")


def draw_1d_distribution(ax: plt.Axes, record: Dict[str, Any]) -> None:
    coords = np.asarray(record["embedding"], dtype=float)
    color = SYSTEM_COLORS[record["system"]]
    x = coords[:, 0]
    span = max(float(x.max() - x.min()), 1.0)
    pad = 0.08 * span
    ax.axhline(0.0, color="#555555", linewidth=0.65, zorder=1)
    for edge_idx, (src, dst) in enumerate(action_edges(record["drawn_transition"])):
        height = 0.20 + 0.045 * (edge_idx % 5)
        start = np.asarray([x[src], height])
        end = np.asarray([x[dst], height])
        add_arrow(
            ax,
            start,
            end,
            color=color,
            linewidth=0.78,
            alpha=0.62,
            shrink=0.06,
            rad=0.35 if edge_idx % 2 == 0 else -0.35,
            mutation_scale=7.0,
            zorder=2,
        )
    ax.scatter(x, np.zeros_like(x), s=82, facecolor="white", edgecolor=color, linewidth=1.25, zorder=4)
    offsets = np.asarray([0.10, -0.15, 0.18, -0.22, 0.26, -0.30, 0.34, -0.38])
    for idx, value in enumerate(x):
        ax.text(float(value), float(offsets[idx]), str(idx), ha="center", va="center", fontsize=6.2, zorder=5)
        ax.plot([value, value], [0.02, offsets[idx] * 0.70], color="#BFBFBF", linewidth=0.30, zorder=2)
    ax.set_xlim(float(x.min() - pad), float(x.max() + pad))
    ax.set_ylim(-0.55, 0.68)
    ax.set_yticks([])
    ax.set_xlabel("coord. 1", fontsize=6.2, labelpad=1)
    style_2d_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def draw_2d_distribution(ax: plt.Axes, record: Dict[str, Any]) -> None:
    coords = np.asarray(record["embedding"], dtype=float)
    color = SYSTEM_COLORS[record["system"]]
    for edge_idx, (src, dst) in enumerate(action_edges(record["drawn_transition"])):
        add_arrow(
            ax,
            coords[src],
            coords[dst],
            color=color,
            linewidth=1.02,
            alpha=0.72,
            shrink=0.13,
            rad=0.12 if record["system"] == "Cycle" else (0.24 if edge_idx % 2 == 0 else -0.24),
            mutation_scale=8.0,
            zorder=2,
        )
    ax.scatter(coords[:, 0], coords[:, 1], s=96, facecolor="white", edgecolor=color, linewidth=1.35, zorder=4)
    for idx, (x, y) in enumerate(coords):
        ax.text(float(x), float(y), str(idx), ha="center", va="center", fontsize=6.3, zorder=5)
    set_axis_equal_2d(ax, coords)
    ax.set_xlabel("coord. 1", fontsize=6.2, labelpad=1)
    ax.set_ylabel("coord. 2", fontsize=6.2, labelpad=1)
    style_2d_axes(ax)


def draw_3d_distribution(ax: plt.Axes, record: Dict[str, Any]) -> None:
    coords = np.asarray(record["embedding"], dtype=float)
    color = SYSTEM_COLORS[record["system"]]
    for src, dst in action_edges(record["drawn_transition"]):
        delta = coords[dst] - coords[src]
        if float(np.linalg.norm(delta)) <= 1e-12:
            continue
        ax.quiver(
            coords[src, 0],
            coords[src, 1],
            coords[src, 2],
            delta[0],
            delta[1],
            delta[2],
            arrow_length_ratio=0.13,
            color=color,
            linewidth=0.88,
            alpha=0.66,
            normalize=False,
        )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=47,
        facecolor="white",
        edgecolor=color,
        linewidth=1.12,
        depthshade=False,
        zorder=4,
    )
    for idx, (x, y, z) in enumerate(coords):
        ax.text(float(x), float(y), float(z), str(idx), ha="center", va="center", fontsize=6.1)
    set_axis_equal_3d(ax, coords)
    ax.view_init(elev=21, azim=-48)
    ax.set_xlabel("c1", fontsize=6.0, labelpad=-4)
    ax.set_ylabel("c2", fontsize=6.0, labelpad=-4)
    ax.set_zlabel("c3", fontsize=6.0, labelpad=-6)
    ax.tick_params(labelsize=4.8, pad=-3, length=1.8, width=0.45)
    ax.grid(True, linewidth=0.24, color="#DDDDDD")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#DDDDDD")


def mse_text(value: float) -> str:
    if abs(value) < 5e-11:
        return "0"
    if value < 0.01:
        return f"{value:.1e}"
    return f"{value:.3f}"


def panel_title(record: Dict[str, Any]) -> str:
    title = f"gain {record['exact_hard_gain']:.2f} | MSE {mse_text(float(record['oracle_mse_at_budget']))}"
    if record["padded_from_lower_dimension"]:
        title += f"\nfrom m={record['source_m']}, extra coord.=0"
    else:
        title += f"\nsource m={record['source_m']}"
    return title


def plot_figure(records: List[Dict[str, Any]], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.titlesize": 7.6,
            "axes.labelsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig = plt.figure(figsize=(10.35, 7.15), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        4,
        left=0.070,
        right=0.988,
        top=0.875,
        bottom=0.075,
        wspace=0.34,
        hspace=0.52,
        width_ratios=(0.90, 1.0, 1.0, 1.0),
    )

    by_key = {(record["system"], int(record["target_m"])): record for record in records}
    for row_idx, system_label in enumerate(SYSTEM_ORDER):
        schematic_ax = fig.add_subplot(grid[row_idx, 0])
        draw_family_schematic(schematic_ax, system_label)
        for col_idx, m in enumerate(M_VALUES, start=1):
            projection = "3d" if m == 3 else None
            ax = fig.add_subplot(grid[row_idx, col_idx], projection=projection)
            record = by_key[(system_label, m)]
            if m == 1:
                draw_1d_distribution(ax, record)
            elif m == 2:
                draw_2d_distribution(ax, record)
            else:
                draw_3d_distribution(ax, record)
            ax.set_title(panel_title(record), pad=3)

    column_labels = ["transition family", "m=1", "m=2", "m=3"]
    x_positions = [0.105, 0.380, 0.610, 0.840]
    for x_pos, label in zip(x_positions, column_labels):
        fig.text(x_pos, 0.915, label, ha="center", va="center", fontsize=10, fontweight="bold")
    fig.text(
        0.520,
        0.030,
        "L=1 MSE is selected from completed source dimensions k<=m; a higher dimension may leave extra coordinates unused.",
        ha="center",
        va="center",
        fontsize=8.2,
    )
    fig.suptitle(
        "Transition families across one, two, and three latent dimensions",
        y=0.975,
        fontsize=12,
        fontweight="bold",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_error_table(selected_records: List[Dict[str, Any]], all_candidates: List[Dict[str, Any]]) -> pd.DataFrame:
    raw_by_key = {
        (item["system"], int(item["target_m"])): item
        for item in all_candidates
        if int(item["source_m"]) == int(item["target_m"])
    }
    rows: List[Dict[str, Any]] = []
    for selected in selected_records:
        raw = raw_by_key[(selected["system"], int(selected["target_m"]))]
        rows.append(
            {
                "system": selected["display_system"],
                "n": N,
                "m": int(selected["target_m"]),
                "selected_source_m": int(selected["source_m"]),
                "padded_from_lower_dimension": bool(selected["padded_from_lower_dimension"]),
                "selected_seed": "" if selected["seed"] is None else int(selected["seed"]),
                "selected_exact_hard_required_gain": float(selected["exact_hard_gain"]),
                "gain_budget_L": float(GAIN_BUDGET),
                "selected_oracle_mse_at_L1": float(selected["oracle_mse_at_budget"]),
                "raw_m_specific_seed": "" if raw["seed"] is None else int(raw["seed"]),
                "raw_m_specific_exact_hard_required_gain": float(raw["exact_hard_gain"]),
                "raw_m_specific_oracle_mse_at_L1": float(raw["oracle_mse_at_budget"]),
                "oracle_solver": selected["oracle_solver"],
                "oracle_num_actions": int(selected["oracle_num_actions"]),
                "oracle_max_constraint_violation": float(selected["oracle_max_constraint_violation"]),
                "drawn_action": selected["drawn_action_label"],
            },
        )
    table = pd.DataFrame(rows)
    table.to_csv(ERROR_TABLE_PATH, index=False, float_format="%.15g")
    return table


def write_outputs(selected_records: List[Dict[str, Any]], all_candidates: List[Dict[str, Any]]) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_figure(selected_records, OUTPUT_PREFIX)
    table = write_error_table(selected_records, all_candidates)
    data_path = OUTPUT_PREFIX.with_name("mechanism_dimension_transition_arrows_data.json")
    caption_path = OUTPUT_PREFIX.with_name("mechanism_dimension_transition_arrows_caption.txt")
    public_records = [
        {key: value for key, value in record.items() if key not in {"system", "selected_file"}}
        for record in selected_records
    ]
    public_candidates = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "system",
                "selected_file",
                "embedding",
                "drawn_transition",
                "witness",
            }
        }
        for record in all_candidates
    ]
    data = {
        "n": N,
        "target_dimensions": list(M_VALUES),
        "gain_budget": GAIN_BUDGET,
        "state_order": list(range(N)),
        "used_smoke_outputs": False,
        "selection_rule": (
            "For each target m, evaluate completed source dimensions k<=m after zero-padding; "
            "select the smallest L=1 oracle MSE."
        ),
        "caption": CAPTION.strip(),
        "selected_panels": public_records,
        "all_candidates_without_embeddings": public_candidates,
        "monotone_error_table": table.to_dict(orient="records"),
        "outputs": {
            "pdf": str(OUTPUT_PREFIX.with_suffix(".pdf")),
            "png": str(OUTPUT_PREFIX.with_suffix(".png")),
            "svg": str(OUTPUT_PREFIX.with_suffix(".svg")),
            "data_json": str(data_path),
            "caption_txt": str(caption_path),
            "error_table_csv": str(ERROR_TABLE_PATH),
        },
    }
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    caption_path.write_text(CAPTION, encoding="utf-8")
    return [
        OUTPUT_PREFIX.with_suffix(".pdf"),
        OUTPUT_PREFIX.with_suffix(".png"),
        OUTPUT_PREFIX.with_suffix(".svg"),
        data_path,
        caption_path,
        ERROR_TABLE_PATH,
    ]


def main() -> None:
    torch.set_default_dtype(dg.DTYPE)
    selected_records, all_candidates = build_records()
    output_paths = write_outputs(selected_records, all_candidates)
    table = pd.read_csv(ERROR_TABLE_PATH)
    print("Monotone best-available L=1 oracle prediction error table")
    print(
        table[
            [
                "system",
                "m",
                "selected_source_m",
                "selected_exact_hard_required_gain",
                "selected_oracle_mse_at_L1",
                "raw_m_specific_exact_hard_required_gain",
                "raw_m_specific_oracle_mse_at_L1",
            ]
        ].to_string(index=False),
    )
    print()
    print("Output paths")
    for path in output_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
