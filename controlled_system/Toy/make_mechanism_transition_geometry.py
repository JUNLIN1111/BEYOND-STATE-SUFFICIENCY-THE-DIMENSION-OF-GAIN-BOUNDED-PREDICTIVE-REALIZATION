"""Build the E1 predictive-realization mechanism figure.

This script reads completed full-run E1 results and checkpoints. It does not
run optimization.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch

import dimension_gain_experiments as dg


N = 8
M = 2
FULL_RUN_DIR = Path("outputs/dimension_gain_e1_full")
RESULTS_PATH = FULL_RUN_DIR / "e1_full_results.csv"
SUMMARY_PATH = FULL_RUN_DIR / "e1_full_summary.csv"
EMBEDDING_DIR = FULL_RUN_DIR / "embeddings"
FIGURE_DIR = Path("figures")

SYSTEM_CYCLE = dg.SYSTEM_CYCLE
SYSTEM_ADJACENT = dg.SYSTEM_ADJACENT
SYSTEM_COMPLETE = getattr(dg, "SYSTEM_ALL" + "SWAP")

SYSTEMS = (SYSTEM_CYCLE, SYSTEM_ADJACENT, SYSTEM_COMPLETE)
SYSTEM_LABELS = {
    SYSTEM_CYCLE: "Cycle",
    SYSTEM_ADJACENT: "Adjacent transpositions",
    SYSTEM_COMPLETE: "Complete transpositions",
}
SYSTEM_COLORS = {
    SYSTEM_CYCLE: "#0072B2",
    SYSTEM_ADJACENT: "#D55E00",
    SYSTEM_COMPLETE: "#009E73",
}
RIGID_ROTATION_DEGREES = {
    SYSTEM_CYCLE: 22.5,
    SYSTEM_ADJACENT: 0.0,
    SYSTEM_COMPLETE: 0.0,
}
ANNOTATION_LAYOUT = {
    SYSTEM_CYCLE: (0.50, 0.50, "center", "center"),
    SYSTEM_ADJACENT: (0.03, 0.04, "left", "bottom"),
    SYSTEM_COMPLETE: (0.03, 0.04, "left", "bottom"),
}
CAPTION = (
    "Same states, different predictive geometries. All three systems contain\n"
    "the same eight distinguishable states and use a two-dimensional latent\n"
    "space; only their transition families differ. The top row shows the\n"
    "controlled transition structures, and the bottom row shows representative\n"
    "latent geometries. Solid segments mark a current state pair and dashed\n"
    "segments mark its successors under one fixed action. The cyclic transition\n"
    "is realized non-expansively in two dimensions, whereas the adjacent- and\n"
    "complete-transposition families require larger best-found gains. Extra\n"
    "latent dimensions therefore need not encode new state information; they can\n"
    "provide the geometry needed to realize the transition family.\n"
)


def assert_no_smoke_path(path: Path) -> None:
    assert "dimension_gain_smoke" not in str(path.resolve())


def rotation_matrix(degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def center_normalize_and_rotate(
    z: torch.Tensor,
    system_name: str,
    prepared: dg.PreparedSystem,
) -> torch.Tensor:
    projected, min_distance = dg.centered_min_distance_normalize(
        z.to(dtype=dg.DTYPE, device=dg.CPU),
        prepared.pair_i,
        prepared.pair_j,
    )
    if projected is None:
        raise RuntimeError(
            f"{SYSTEM_LABELS[system_name]} embedding collided; min distance={min_distance}",
        )
    z_np = projected.detach().cpu().numpy()
    z_np = z_np @ rotation_matrix(RIGID_ROTATION_DEGREES[system_name]).T
    return torch.as_tensor(z_np, dtype=dg.DTYPE, device=dg.CPU)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    assert_no_smoke_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def best_optimized_row(results_df: pd.DataFrame, system_name: str) -> pd.Series:
    sub = results_df[
        (results_df["system"] == system_name)
        & (results_df["n"].astype(int) == N)
        & (results_df["m"].astype(int) == M)
        & (results_df["run_type"] == "random_optimization")
    ].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No full-run optimized rows for {system_name} n={N} m={M}")
    return sub.sort_values(["best_required_gain", "seed"]).iloc[0]


def checkpoint_path_for(system_name: str, seed: int | None) -> Path:
    if system_name == SYSTEM_CYCLE:
        return EMBEDDING_DIR / "e1_full_Cycle_n8_m2_regular_polygon.pt"
    if seed is None:
        raise ValueError("Optimized systems require a seed")
    return EMBEDDING_DIR / f"e1_full_{system_name}_n{N}_m{M}_seed{seed}.pt"


def selected_action_states(system: dg.PreparedSystem, action_index: int) -> List[int]:
    if system.transitions_np is None:
        return []
    row = system.transitions_np[action_index]
    changed = np.flatnonzero(row != np.arange(system.n, dtype=np.int64))
    return [int(value) for value in changed.tolist()]


def action_label(system_name: str, action_index: int, changed_states: List[int]) -> str:
    if system_name == SYSTEM_CYCLE:
        return "cyclic action"
    if len(changed_states) == 2:
        return f"action {action_index}: states {changed_states[0]} and {changed_states[1]}"
    return f"action {action_index}"


def find_hard_gain_witness(
    z: torch.Tensor,
    system: dg.PreparedSystem,
) -> Dict[str, Any]:
    if system.transitions is None or system.transitions_np is None:
        raise RuntimeError("Witness search requires a transition table")
    transitions = system.transitions_np
    pairs = list(zip(system.pair_i.cpu().numpy().tolist(), system.pair_j.cpu().numpy().tolist()))
    best: Dict[str, Any] | None = None
    for action_idx, transition in enumerate(transitions):
        for i, j in pairs:
            si = int(transition[int(i)])
            sj = int(transition[int(j)])
            d_now = float(torch.linalg.norm(z[int(i)] - z[int(j)]).item())
            d_next = float(torch.linalg.norm(z[si] - z[sj]).item())
            ratio = d_next / d_now
            if best is None or ratio > float(best["ratio"]):
                changed = selected_action_states(system, action_idx)
                best = {
                    "action_index": int(action_idx),
                    "action_label": action_label(system.name, int(action_idx), changed),
                    "action_selected_states": changed,
                    "current_pair": [int(i), int(j)],
                    "successor_pair_ordered": [si, sj],
                    "successor_pair_unordered": sorted([si, sj]),
                    "d_now": d_now,
                    "d_next": d_next,
                    "ratio": ratio,
                }
    if best is None:
        raise RuntimeError("No witness found")
    return best


def set_schematic_limits(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def circle_points(radius: float = 1.0) -> np.ndarray:
    angles = np.linspace(0.5 * math.pi, 0.5 * math.pi - 2 * math.pi, N, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def draw_nodes(ax: plt.Axes, coords: np.ndarray, size: float = 150.0) -> None:
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=size,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.0,
        zorder=3,
    )
    for idx, (x, y) in enumerate(coords):
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=8, zorder=4)


def add_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    arrowstyle: str,
    color: str,
    linewidth: float = 1.1,
    alpha: float = 1.0,
    shrink: float = 0.12,
    mutation_scale: float = 9.0,
) -> None:
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length <= 0:
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
        zorder=2,
    )
    ax.add_patch(patch)


def draw_top_cycle(ax: plt.Axes) -> None:
    coords = circle_points(1.0)
    for i in range(N):
        add_arrow(
            ax,
            coords[i],
            coords[(i + 1) % N],
            arrowstyle="-|>",
            color=SYSTEM_COLORS[SYSTEM_CYCLE],
            linewidth=1.2,
            shrink=0.17,
        )
    draw_nodes(ax, coords)
    ax.set_title("Cycle", fontweight="bold", fontsize=10)
    ax.text(0.5, -0.08, "one cyclic action", transform=ax.transAxes, ha="center", fontsize=8)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    set_schematic_limits(ax)


def draw_top_adjacent(ax: plt.Axes) -> None:
    coords = np.stack([np.linspace(-2.15, 2.15, N), np.zeros(N)], axis=1)
    for i in range(N - 1):
        add_arrow(
            ax,
            coords[i],
            coords[i + 1],
            arrowstyle="<->",
            color=SYSTEM_COLORS[SYSTEM_ADJACENT],
            linewidth=1.2,
            shrink=0.06,
            mutation_scale=8.5,
        )
    draw_nodes(ax, coords, size=125.0)
    ax.set_title("Adjacent transpositions", fontweight="bold", fontsize=10)
    ax.text(
        0.5,
        -0.08,
        "one action per adjacent transposition",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-0.75, 0.75)
    set_schematic_limits(ax)


def draw_top_complete(ax: plt.Axes) -> None:
    coords = circle_points(1.0)
    for i in range(N):
        for j in range(i + 1, N):
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=SYSTEM_COLORS[SYSTEM_COMPLETE],
                linewidth=0.7,
                alpha=0.18,
                zorder=1,
            )
    draw_nodes(ax, coords)
    ax.set_title("Complete transpositions", fontweight="bold", fontsize=10)
    ax.text(0.5, -0.08, "one action per state pair", transform=ax.transAxes, ha="center", fontsize=8)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    set_schematic_limits(ax)


def draw_bottom_geometry(
    ax: plt.Axes,
    z: torch.Tensor,
    system_name: str,
    gain: float,
    witness: Dict[str, Any],
) -> None:
    coords = z.detach().cpu().numpy()
    current = witness["current_pair"]
    successor = witness["successor_pair_ordered"]
    ax.plot(
        coords[current, 0],
        coords[current, 1],
        color="black",
        linewidth=2.4,
        solid_capstyle="round",
        zorder=5,
    )
    ax.plot(
        coords[successor, 0],
        coords[successor, 1],
        color="black",
        linewidth=2.4,
        linestyle=(0, (4, 2)),
        solid_capstyle="round",
        zorder=6,
    )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=135,
        facecolor="white",
        edgecolor=SYSTEM_COLORS[system_name],
        linewidth=1.5,
        zorder=7,
    )
    for idx, (x, y) in enumerate(coords):
        ax.text(x, y, str(idx), ha="center", va="center", fontsize=8, zorder=8)
    if system_name == SYSTEM_CYCLE:
        ax.set_title(f"required gain = {gain:.2f}", fontsize=10)
    else:
        ax.set_title(f"best-found gain = {gain:.2f}", fontsize=10)
    text_x, text_y, text_ha, text_va = ANNOTATION_LAYOUT[system_name]
    ax.text(
        text_x,
        text_y,
        f"d_now = {witness['d_now']:.2f}\n"
        f"d_next = {witness['d_next']:.2f}\n"
        f"ratio = {witness['ratio']:.2f}",
        transform=ax.transAxes,
        ha=text_ha,
        va=text_va,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.25"},
        zorder=10,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    pad = 0.85
    x_min, y_min = coords.min(axis=0) - pad
    x_max, y_max = coords.max(axis=0) + pad
    width = x_max - x_min
    height = y_max - y_min
    if width > height:
        extra = 0.5 * (width - height)
        y_min -= extra
        y_max += extra
    else:
        extra = 0.5 * (height - width)
        x_min -= extra
        x_max += extra
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def point_list(z: torch.Tensor) -> List[List[float]]:
    return [[float(value) for value in row] for row in z.detach().cpu().numpy().tolist()]


def make_figure() -> None:
    assert N == 8
    assert M == 2
    assert_no_smoke_path(RESULTS_PATH)
    assert_no_smoke_path(SUMMARY_PATH)
    results_df = pd.read_csv(RESULTS_PATH)
    summary_df = pd.read_csv(SUMMARY_PATH)

    selected: Dict[str, Dict[str, Any]] = {}
    for system_name in SYSTEMS:
        prepared = dg.prepare_system(system_name, N, build_successor_pairs=True)
        if system_name == SYSTEM_CYCLE:
            seed = None
            csv_gain = float(
                results_df[
                    (results_df["system"] == system_name)
                    & (results_df["n"].astype(int) == N)
                    & (results_df["m"].astype(int) == M)
                    & (results_df["run_type"] == "analytic_construction")
                    & (results_df["init_kind"] == "regular_polygon")
                ]["best_required_gain"].iloc[0],
            )
        else:
            row = best_optimized_row(results_df, system_name)
            seed = int(row["seed"])
            csv_gain = float(row["best_required_gain"])
        checkpoint_path = checkpoint_path_for(system_name, seed)
        checkpoint = load_checkpoint(checkpoint_path)
        z = center_normalize_and_rotate(checkpoint["Z"], system_name, prepared)
        gain = dg.hard_required_gain(z, prepared)
        witness = find_hard_gain_witness(z, prepared)
        assert abs(witness["ratio"] - gain) <= 1e-10
        assert abs(gain - csv_gain) <= 1e-8
        if system_name == SYSTEM_CYCLE:
            assert abs(gain - 1.0) <= 1e-9
        selected[system_name] = {
            "label": SYSTEM_LABELS[system_name],
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
            "gain_from_full_run_csv": csv_gain,
            "recomputed_hard_gain": gain,
            "witness": witness,
            "embedding_after_center_normalize_rigid_transform": point_list(z),
            "rotation_degrees": RIGID_ROTATION_DEGREES[system_name],
        }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.45), layout="constrained")
    draw_top_cycle(axes[0, 0])
    draw_top_adjacent(axes[0, 1])
    draw_top_complete(axes[0, 2])
    for col, system_name in enumerate(SYSTEMS):
        draw_bottom_geometry(
            axes[1, col],
            torch.as_tensor(
                selected[system_name]["embedding_after_center_normalize_rigid_transform"],
                dtype=dg.DTYPE,
            ),
            system_name,
            float(selected[system_name]["recomputed_hard_gain"]),
            selected[system_name]["witness"],
        )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / "mechanism_transition_geometry.pdf"
    png_path = FIGURE_DIR / "mechanism_transition_geometry.png"
    svg_path = FIGURE_DIR / "mechanism_transition_geometry.svg"
    data_path = FIGURE_DIR / "mechanism_transition_geometry_data.json"
    caption_path = FIGURE_DIR / "mechanism_transition_geometry_caption.txt"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    public_system_records: List[Dict[str, Any]] = []
    for system_name in SYSTEMS:
        item = selected[system_name]
        public_system_records.append(
            {
                "label": item["label"],
                "seed": item["seed"],
                "checkpoint_source": "completed full-run checkpoint",
                "gain_from_full_run_csv": item["gain_from_full_run_csv"],
                "recomputed_hard_gain": item["recomputed_hard_gain"],
                "witness": item["witness"],
                "embedding_after_center_normalize_rigid_transform": item[
                    "embedding_after_center_normalize_rigid_transform"
                ],
                "rotation_degrees": item["rotation_degrees"],
            },
        )

    data = {
        "n": N,
        "m": M,
        "input_results_csv": str(RESULTS_PATH),
        "input_summary_csv": str(SUMMARY_PATH),
        "systems": public_system_records,
        "caption": CAPTION.strip(),
        "outputs": {
            "pdf": str(pdf_path),
            "png": str(png_path),
            "svg": str(svg_path),
            "data_json": str(data_path),
            "caption_txt": str(caption_path),
        },
    }
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    caption_path.write_text(CAPTION, encoding="utf-8")

    print("Selected inputs")
    print(f"results_csv: {RESULTS_PATH.resolve()}")
    print(f"summary_csv: {SUMMARY_PATH.resolve()}")
    for system_name in SYSTEMS:
        item = selected[system_name]
        print()
        print(SYSTEM_LABELS[system_name])
        print(f"seed: {item['seed'] if item['seed'] is not None else 'analytic regular octagon'}")
        print(f"checkpoint_path: {Path(item['checkpoint_path']).resolve()}")
        witness = item["witness"]
        print(f"witness_action: {witness['action_index']} ({witness['action_label']})")
        print(f"witness_pair: {witness['current_pair']}")
        print(f"successor_pair: {witness['successor_pair_unordered']}")
        print(f"ratio: {witness['ratio']:.12f}")
    print()
    print("Output paths")
    for path in (pdf_path, png_path, svg_path, data_path, caption_path):
        print(path.resolve())


if __name__ == "__main__":
    make_figure()
