"""Clean mechanism figure plus precise prediction-error table.

This script reuses completed E1 full-run embeddings and the small numerical
oracle from make_mechanism_prediction_arrows.py. It does not rerun embedding
optimization and does not train a predictor.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

import make_mechanism_prediction_arrows as pa


N = 8
FIGURE_DIR = Path("figures")
OUTPUT_PREFIX = FIGURE_DIR / "mechanism_clean_transition_error"
ERROR_TABLE_PATH = FIGURE_DIR / "mechanism_prediction_error_table.csv"

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
    "Mechanism view for the prediction-error result. The top row shows the\n"
    "transition family on the same eight states: a single cyclic action, one\n"
    "neighboring-pair action per adjacent pair, or one pair action for every\n"
    "unordered state pair. The bottom row shows the corresponding m=2 latent\n"
    "geometry from the completed full-run outputs. Colored arrows show one true\n"
    "action in latent space; fixed states are not annotated. The reported error\n"
    "is the finite-state numerical oracle MSE at gain budget L=1, while gain is\n"
    "the recomputed exact hard required gain. The systems differ not because the\n"
    "state set changes, but because their transition families impose different\n"
    "geometric constraints on the same eight labels.\n"
)


def circle_points(radius: float = 1.0) -> np.ndarray:
    angles = np.linspace(0.5 * math.pi, 0.5 * math.pi - 2.0 * math.pi, N, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def add_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    arrowstyle: str = "-|>",
    linewidth: float = 1.15,
    alpha: float = 1.0,
    shrink: float = 0.13,
    rad: float = 0.0,
    mutation_scale: float = 9.0,
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


def draw_nodes(ax: plt.Axes, coords: np.ndarray, color: str, size: float = 148.0) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=size, facecolor="white", edgecolor=color, linewidth=1.4, zorder=4)
    for idx, (x, y) in enumerate(coords):
        ax.text(float(x), float(y), str(idx), ha="center", va="center", fontsize=8, zorder=5)


def clean_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_top_cycle(ax: plt.Axes) -> None:
    color = SYSTEM_COLORS["Cycle"]
    coords = circle_points()
    for i in range(N):
        add_arrow(ax, coords[i], coords[(i + 1) % N], color=color, shrink=0.17, linewidth=1.25)
    draw_nodes(ax, coords, color)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.32, 1.45)
    ax.text(0.5, -0.09, "1 action: i -> i+1 mod 8", transform=ax.transAxes, ha="center", fontsize=8)
    clean_axis(ax)


def draw_top_adjacent(ax: plt.Axes) -> None:
    color = SYSTEM_COLORS["Adjacent transpositions"]
    coords = np.stack([np.linspace(-2.25, 2.25, N), np.zeros(N)], axis=1)
    for i in range(N - 1):
        add_arrow(
            ax,
            coords[i],
            coords[i + 1],
            color=color,
            arrowstyle="<->",
            shrink=0.075,
            linewidth=1.25,
            mutation_scale=8.5,
        )
    draw_nodes(ax, coords, color, size=130.0)
    ax.set_xlim(-2.55, 2.55)
    ax.set_ylim(-0.72, 0.72)
    ax.text(0.5, -0.10, "7 actions: one neighboring pair", transform=ax.transAxes, ha="center", fontsize=8)
    clean_axis(ax)


def draw_top_complete(ax: plt.Axes) -> None:
    color = SYSTEM_COLORS["Complete transpositions"]
    coords = circle_points()
    for i in range(N):
        for j in range(i + 1, N):
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=color,
                linewidth=0.72,
                alpha=0.22,
                zorder=1,
            )
    draw_nodes(ax, coords, color)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.32, 1.45)
    ax.text(0.5, -0.09, "28 actions: any unordered pair", transform=ax.transAxes, ha="center", fontsize=8)
    clean_axis(ax)


def set_axis_equal(ax: plt.Axes, coords: np.ndarray, pad_fraction: float = 0.18) -> None:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = max(float((maxs - mins).max()), 1.0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * span + pad_fraction * span
    ax.set_xlim(float(center[0] - half), float(center[0] + half))
    ax.set_ylim(float(center[1] - half), float(center[1] + half))
    ax.set_aspect("equal", adjustable="box")


def action_edges(transition: List[int]) -> List[Tuple[int, int]]:
    return [(idx, int(dst)) for idx, dst in enumerate(transition) if int(dst) != idx]


def draw_bottom_panel(ax: plt.Axes, record: Dict[str, Any]) -> None:
    label = record["system"]
    color = SYSTEM_COLORS[label]
    coords = np.asarray(record["embedding"], dtype=float)
    transition = record["drawn_transition"]

    for edge_idx, (src, dst) in enumerate(action_edges(transition)):
        rad = 0.10
        if label != "Cycle":
            rad = 0.26 if edge_idx % 2 == 0 else -0.26
        add_arrow(
            ax,
            coords[src],
            coords[dst],
            color=color,
            linewidth=1.25,
            alpha=0.78,
            shrink=0.13,
            rad=rad,
            mutation_scale=9.0,
            zorder=2,
        )

    draw_nodes(ax, coords, color, size=152.0)
    set_axis_equal(ax, coords)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#888888")
    mse = float(record["oracle_mse_at_budget"])
    mse_text = "0" if abs(mse) < 5e-11 else f"{mse:.3f}"
    ax.set_title(f"gain {float(record['exact_hard_gain']):.2f} | L=1 MSE {mse_text}", fontsize=9, pad=4)


def public_error_rows(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        witness = record["witness"]
        rows.append(
            {
                "system": DISPLAY_NAME[record["system"]],
                "n": int(record["n"]),
                "m": int(record["m"]),
                "embedding_source": record["embedding_source"],
                "seed": "" if record["seed"] is None else int(record["seed"]),
                "exact_hard_required_gain": float(record["exact_hard_gain"]),
                "gain_budget_L": float(record["gain_budget"]),
                "oracle_mse_at_L1": float(record["oracle_mse_at_budget"]),
                "oracle_solver": record["oracle_solver"],
                "oracle_num_actions": int(record["oracle_num_actions"]),
                "oracle_max_constraint_violation": float(record["oracle_max_constraint_violation"]),
                "drawn_action": record["drawn_action_label"],
                "witness_current_pair": "{" + ",".join(str(x) for x in witness["current_pair"]) + "}",
                "witness_successor_pair": "{" + ",".join(str(x) for x in witness["successor_pair_unordered"]) + "}",
                "witness_ratio": float(witness["ratio"]),
            },
        )
    return pd.DataFrame(rows)


def make_figure(records: List[Dict[str, Any]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig = plt.figure(figsize=(9.4, 5.55), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        top=0.835,
        bottom=0.105,
        wspace=0.34,
        hspace=0.38,
        height_ratios=(0.72, 1.0),
    )
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)])
    top_drawers = [draw_top_cycle, draw_top_adjacent, draw_top_complete]
    for col, system_name in enumerate(SYSTEM_ORDER):
        top_drawers[col](axes[0, col])

    by_key = {(record["system"], int(record["m"])): record for record in records}
    for col, system_name in enumerate(SYSTEM_ORDER):
        draw_bottom_panel(axes[1, col], by_key[(system_name, 2)])

    x_positions = [0.075 + (col + 0.5) * (0.985 - 0.075) / 3.0 for col in range(3)]
    for x_pos, system_name in zip(x_positions, SYSTEM_ORDER):
        fig.text(
            x_pos,
            0.872,
            DISPLAY_NAME[system_name],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    fig.text(0.018, 0.645, "transition family", rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
    fig.text(0.018, 0.290, "m=2 latent action", rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
    fig.suptitle(
        "Same states, different transition families, different prediction error",
        y=0.970,
        fontsize=12,
        fontweight="bold",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(records: List[Dict[str, Any]]) -> List[Path]:
    make_figure(records)
    error_df = public_error_rows(records)
    error_df.to_csv(ERROR_TABLE_PATH, index=False, float_format="%.15g")

    data_path = OUTPUT_PREFIX.with_name("mechanism_clean_transition_error_data.json")
    caption_path = OUTPUT_PREFIX.with_name("mechanism_clean_transition_error_caption.txt")
    public_records = []
    for record in records:
        item = {key: value for key, value in record.items() if key not in {"system_id", "selected_file"}}
        item["display_system"] = DISPLAY_NAME[record["system"]]
        del item["system"]
        public_records.append(item)
    data = {
        "n": N,
        "figure_dimension": 2,
        "error_table_dimensions": list(pa.M_VALUES),
        "gain_budget": pa.GAIN_BUDGET,
        "state_order": list(range(N)),
        "used_smoke_outputs": False,
        "caption": CAPTION.strip(),
        "panels": public_records,
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
    records = pa.build_records()
    output_paths = write_outputs(records)
    table = public_error_rows(records)
    print("Precise L=1 oracle prediction error table")
    print(
        table[
            [
                "system",
                "m",
                "seed",
                "exact_hard_required_gain",
                "oracle_mse_at_L1",
                "oracle_max_constraint_violation",
            ]
        ].to_string(index=False),
    )
    print()
    print("Output paths")
    for path in output_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
