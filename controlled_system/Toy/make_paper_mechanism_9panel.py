"""Paper-style 3x3 mechanism schematic.

The figure is intentionally schematic rather than a raw checkpoint scatter:
the goal is to make the transition families and the role of latent dimension
legible in a paper figure. Error/gain values are read from the monotone
full-run oracle table generated from completed E1 outputs; no optimization is
run here.
"""

from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


N = 8
M_VALUES = (1, 2, 3)
FIGURE_DIR = Path("figures")
ERROR_TABLE_PATH = FIGURE_DIR / "mechanism_prediction_error_table_monotone.csv"
OUTPUT_PREFIX = FIGURE_DIR / "mechanism_paper_9panel"
SYSTEMS = ("Cycle", "Adjacent pair actions", "All-pair actions")
COLORS = {
    "Cycle": "#0072B2",
    "Adjacent pair actions": "#D55E00",
    "All-pair actions": "#009E73",
}

CAPTION = (
    "Schematic mechanism figure for the finite-state prediction experiment.\n"
    "Every panel contains the same eight state labels. Rows differ only in the\n"
    "transition family: one cyclic action, seven neighboring-pair actions, or\n"
    "twenty-eight all-pair actions. Columns show schematic latent geometries in\n"
    "one, two, and three dimensions. All transition arrows/edges for the family\n"
    "are drawn in each panel. Numbers report the best-available full-run\n"
    "required gain and finite-state numerical oracle MSE at gain budget L=1;\n"
    "the coordinates are schematic for readability rather than raw checkpoint\n"
    "coordinates.\n"
)


def load_error_table() -> pd.DataFrame:
    if not ERROR_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ERROR_TABLE_PATH}. Run make_mechanism_dimension_transition_arrows.py first.",
        )
    table = pd.read_csv(ERROR_TABLE_PATH)
    required = {
        "system",
        "m",
        "selected_source_m",
        "selected_exact_hard_required_gain",
        "selected_oracle_mse_at_L1",
    }
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"Error table missing columns: {sorted(missing)}")
    return table


def circle_points(radius: float = 1.0) -> np.ndarray:
    angles = np.linspace(0.5 * math.pi, 0.5 * math.pi - 2.0 * math.pi, N, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def line_points(width: float = 3.2) -> np.ndarray:
    return np.stack([np.linspace(-width, width, N), np.zeros(N)], axis=1)


def adjacent_plane_points() -> np.ndarray:
    x = np.linspace(-2.65, 2.65, N)
    y = np.asarray([-0.90, 0.22, -0.42, 0.70, -0.08, 0.88, 0.10, 0.58])
    return np.stack([x, y], axis=1)


def complete_plane_points() -> np.ndarray:
    return circle_points(radius=1.45)


def project_3d(points: np.ndarray) -> np.ndarray:
    """Simple isometric-style projection for schematic 3D point clouds."""

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack([x + 0.58 * z, y + 0.36 * z], axis=1)


def cube_points(scale: float = 1.0) -> np.ndarray:
    coords = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    return scale * coords


def schematic_points(system: str, m: int) -> Tuple[np.ndarray, str]:
    if m == 1:
        return line_points(), "1d"
    if m == 2:
        if system == "Cycle":
            return circle_points(radius=1.45), "2d"
        if system == "Adjacent pair actions":
            return adjacent_plane_points(), "2d"
        return complete_plane_points(), "2d"

    if system == "Cycle":
        base = np.pad(circle_points(radius=1.35), ((0, 0), (0, 1)))
        return project_3d(base), "3d_flat"
    if system == "Adjacent pair actions":
        base2 = adjacent_plane_points()
        base = np.column_stack([base2[:, 0], base2[:, 1], np.zeros(N)])
        return project_3d(base), "3d_flat"
    return project_3d(cube_points(scale=1.18)), "3d_volume"


def add_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    arrowstyle: str = "-|>",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    shrink: float = 0.10,
    rad: float = 0.0,
    mutation_scale: float = 7.5,
    zorder: int = 2,
) -> None:
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        return
    unit = delta / length
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


def draw_dimension_guide(ax: plt.Axes, coords: np.ndarray, kind: str) -> None:
    gray = "#B8B8B8"
    if kind == "1d":
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        ax.plot([x_min - 0.25, x_max + 0.25], [0, 0], color="#666666", linewidth=0.8, zorder=0)
    elif kind == "2d":
        x0, y0 = -0.88, -0.88
        ax.arrow(x0, y0, 0.45, 0.0, head_width=0.035, head_length=0.045, color=gray, linewidth=0.7)
        ax.arrow(x0, y0, 0.0, 0.45, head_width=0.035, head_length=0.045, color=gray, linewidth=0.7)
    else:
        x0, y0 = -0.96, -0.86
        ax.arrow(x0, y0, 0.40, 0.0, head_width=0.035, head_length=0.045, color=gray, linewidth=0.7)
        ax.arrow(x0, y0, 0.0, 0.40, head_width=0.035, head_length=0.045, color=gray, linewidth=0.7)
        ax.arrow(x0, y0, 0.27, 0.18, head_width=0.035, head_length=0.045, color=gray, linewidth=0.7)
        if kind == "3d_flat":
            ax.text(0.04, 0.06, "extra coordinate unused", transform=ax.transAxes, fontsize=6.1, color="#777777")


def transition_edges(system: str) -> Iterable[Tuple[int, int]]:
    if system == "Cycle":
        return [(i, (i + 1) % N) for i in range(N)]
    if system == "Adjacent pair actions":
        return [(i, i + 1) for i in range(N - 1)]
    return combinations(range(N), 2)


def draw_transitions(ax: plt.Axes, coords: np.ndarray, system: str, m: int) -> None:
    color = COLORS[system]
    if system == "Cycle":
        for i, j in transition_edges(system):
            rad = 0.12 if m != 1 else 0.54
            if m == 1 and i == N - 1:
                rad = 0.34
            add_arrow(
                ax,
                coords[i],
                coords[j],
                color=color,
                linewidth=1.28,
                alpha=0.82,
                shrink=0.16 if m != 1 else 0.17,
                rad=rad,
                mutation_scale=8.4,
                zorder=2,
            )
        return

    if system == "Adjacent pair actions":
        for i, j in transition_edges(system):
            rad = 0.52 if m == 1 else 0.08
            add_arrow(
                ax,
                coords[i],
                coords[j],
                color=color,
                arrowstyle="<->",
                linewidth=1.25,
                alpha=0.84,
                shrink=0.13,
                rad=rad,
                mutation_scale=7.5,
                zorder=2,
            )
        return

    for i, j in transition_edges(system):
        distance = abs(i - j)
        if m == 1:
            rad = 0.20 + 0.060 * distance
            alpha = 0.13
            linewidth = 0.58
        else:
            rad = 0.0
            alpha = 0.17
            linewidth = 0.58
        add_arrow(
            ax,
            coords[i],
            coords[j],
            color=color,
            arrowstyle="<->",
            linewidth=linewidth,
            alpha=alpha,
            shrink=0.12,
            rad=rad,
            mutation_scale=5.2,
            zorder=1,
        )


def draw_nodes(ax: plt.Axes, coords: np.ndarray, system: str) -> None:
    color = COLORS[system]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=92,
        facecolor="white",
        edgecolor=color,
        linewidth=1.45,
        zorder=4,
    )
    for idx, (x, y) in enumerate(coords):
        ax.text(float(x), float(y), str(idx), ha="center", va="center", fontsize=6.5, zorder=5)


def format_mse(value: float) -> str:
    if abs(value) < 5e-11:
        return "0"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def panel_note(table: pd.DataFrame, system: str, m: int) -> str:
    row = table[(table["system"] == system) & (table["m"].astype(int) == m)]
    if len(row) != 1:
        raise RuntimeError(f"Missing row for {system}, m={m}")
    item = row.iloc[0]
    gain = float(item["selected_exact_hard_required_gain"])
    mse = float(item["selected_oracle_mse_at_L1"])
    return f"gain {gain:.2f}   L=1 MSE {format_mse(mse)}"


def set_limits(ax: plt.Axes, coords: np.ndarray, system: str, kind: str) -> None:
    if kind == "1d":
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        ax.set_xlim(x_min - 0.55, x_max + 0.55)
        ax.set_ylim(-0.62, 2.10 if system != "All-pair actions" else 2.28)
        return
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = max(float((maxs - mins).max()), 1.0)
    pad = 0.20 * span
    if system == "All-pair actions":
        pad = 0.28 * span
    center = 0.5 * (mins + maxs)
    ax.set_xlim(float(center[0] - 0.5 * span - pad), float(center[0] + 0.5 * span + pad))
    ax.set_ylim(float(center[1] - 0.5 * span - pad), float(center[1] + 0.5 * span + pad))
    ax.set_aspect("equal", adjustable="box")


def draw_panel(ax: plt.Axes, table: pd.DataFrame, system: str, m: int) -> None:
    coords, kind = schematic_points(system, m)
    draw_dimension_guide(ax, coords, kind)
    draw_transitions(ax, coords, system, m)
    draw_nodes(ax, coords, system)
    ax.text(
        0.5,
        -0.095,
        panel_note(table, system, m),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
    )
    set_limits(ax, coords, system, kind)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(table: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.2), layout="constrained")
    for row_idx, system in enumerate(SYSTEMS):
        for col_idx, m in enumerate(M_VALUES):
            draw_panel(axes[row_idx, col_idx], table, system, m)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"m={m}", fontsize=10, fontweight="bold", pad=6)
            if col_idx == 0:
                axes[row_idx, col_idx].text(
                    -0.20,
                    0.5,
                    system,
                    transform=axes[row_idx, col_idx].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9.4,
                    fontweight="bold",
                    color=COLORS[system],
                )
    fig.suptitle(
        "Same states, different transition families, different useful dimensions",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.012,
        "Schematic coordinates for readability; numbers come from completed full-run embeddings and the L=1 oracle error table.",
        ha="center",
        va="top",
        fontsize=7.2,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(table: pd.DataFrame) -> List[Path]:
    make_figure(table)
    data_path = OUTPUT_PREFIX.with_name("mechanism_paper_9panel_data.json")
    caption_path = OUTPUT_PREFIX.with_name("mechanism_paper_9panel_caption.txt")
    data = {
        "n": N,
        "dimensions": list(M_VALUES),
        "systems": list(SYSTEMS),
        "coordinates": "schematic, not raw checkpoint coordinates",
        "transition_drawing": {
            "Cycle": "all 8 directed arrows i -> i+1 mod 8",
            "Adjacent pair actions": "all 7 neighboring-pair bidirectional edges",
            "All-pair actions": "all 28 unordered-pair bidirectional edges",
        },
        "numbers_source": str(ERROR_TABLE_PATH),
        "panels": table.to_dict(orient="records"),
        "caption": CAPTION.strip(),
        "outputs": {
            "pdf": str(OUTPUT_PREFIX.with_suffix(".pdf")),
            "png": str(OUTPUT_PREFIX.with_suffix(".png")),
            "svg": str(OUTPUT_PREFIX.with_suffix(".svg")),
            "data_json": str(data_path),
            "caption_txt": str(caption_path),
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
    ]


def main() -> None:
    table = load_error_table()
    outputs = write_outputs(table)
    print("Generated paper-style 3x3 mechanism figure")
    for system in SYSTEMS:
        for m in M_VALUES:
            row = table[(table["system"] == system) & (table["m"].astype(int) == m)].iloc[0]
            print(
                f"{system:22s} m={m} "
                f"gain={float(row['selected_exact_hard_required_gain']):.12g} "
                f"L1_MSE={float(row['selected_oracle_mse_at_L1']):.12g} "
                f"source_m={int(row['selected_source_m'])}",
            )
    print()
    print("Output paths")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
