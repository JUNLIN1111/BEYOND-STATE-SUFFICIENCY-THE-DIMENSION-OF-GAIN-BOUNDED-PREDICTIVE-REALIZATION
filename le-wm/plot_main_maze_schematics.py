from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from learned_branching_maze_gain import RoadMap, build_maps, graph_degrees, validate_map_family


MAP_ORDER = [
    "path_0",
    "one_t_junction",
    "comb_4",
    "comb_8",
    "comb_12",
    "double_comb_12",
    "hierarchical_tree_8",
]


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )
    return plt


def draw_map(ax, road_map: RoadMap, complexity: Dict[str, object], show_stats: bool) -> None:
    coords = road_map.coords.astype(float)
    degrees = graph_degrees(road_map)
    for src, dst in road_map.edges:
        xy = coords[[src, dst]]
        ax.plot(xy[:, 0], xy[:, 1], color="#b8bec7", lw=0.65, solid_capstyle="round", zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=8, marker="s", color="#375f7f", edgecolors="none", zorder=2)
    junctions = np.where(degrees >= 3)[0]
    if junctions.size:
        ax.scatter(coords[junctions, 0], coords[junctions, 1], s=14, marker="s", color="#b33f3a", edgecolors="none", zorder=3)
    ax.set_aspect("equal", adjustable="box")
    pad = 2.0
    ax.set_xlim(float(coords[:, 0].min() - pad), float(coords[:, 0].max() + pad))
    ax.set_ylim(float(coords[:, 1].min() - pad), float(coords[:, 1].max() + pad))
    ax.set_xticks([])
    ax.set_yticks([])
    title = road_map.name
    if show_stats:
        title += (
            f"\nstates={int(complexity['num_states'])}, "
            f"branch excess={int(complexity['branch_excess'])}, "
            f"junctions={int(complexity['junction_count'])}"
        )
    ax.set_title(title, fontsize=8, pad=4)


def draw_legend_panel(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.92, "Primitive actions", fontsize=8, weight="bold", va="top")
    arrows = [
        ((0.23, 0.50), (0.23, 0.72), "up"),
        ((0.23, 0.50), (0.23, 0.28), "down"),
        ((0.23, 0.50), (0.05, 0.50), "left"),
        ((0.23, 0.50), (0.41, 0.50), "right"),
    ]
    for start, end, label in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#555"})
        ax.text(end[0] + 0.02, end[1], label, fontsize=7, va="center")
    ax.scatter([0.64], [0.62], s=24, marker="s", color="#375f7f", edgecolors="none")
    ax.text(0.70, 0.62, "valid state", fontsize=7, va="center")
    ax.scatter([0.64], [0.45], s=30, marker="s", color="#b33f3a", edgecolors="none")
    ax.text(0.70, 0.45, "junction", fontsize=7, va="center")
    ax.plot([0.56, 0.68], [0.28, 0.28], color="#b8bec7", lw=0.9)
    ax.text(0.70, 0.28, "local adjacency", fontsize=7, va="center")
    ax.text(
        0.02,
        0.08,
        "Macro actions are composed from primitive moves;\nthis schematic shows the physical state graph.",
        fontsize=6.7,
        va="bottom",
    )


def save_main_layout(maps: Sequence[RoadMap], complexity: Dict[str, Dict[str, object]], out_dir: Path) -> List[Path]:
    plt = _matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.8), facecolor="white")
    axes_flat = axes.flat
    for ax, road_map in zip(axes_flat, maps):
        draw_map(ax, road_map, complexity[road_map.name], show_stats=False)
    draw_legend_panel(axes_flat[-1])
    fig.tight_layout(pad=0.8)
    paths = []
    for suffix in ["pdf", "png"]:
        path = out_dir / f"maze_schematics_main.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def save_appendix_layout(maps: Sequence[RoadMap], complexity: Dict[str, Dict[str, object]], out_dir: Path) -> List[Path]:
    plt = _matplotlib()
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 7.2), facecolor="white")
    axes_flat = list(axes.flat)
    for ax, road_map in zip(axes_flat, maps):
        draw_map(ax, road_map, complexity[road_map.name], show_stats=True)
    for ax in axes_flat[len(maps) :]:
        ax.set_axis_off()
    fig.tight_layout(pad=1.0)
    paths = []
    for suffix in ["pdf", "png"]:
        path = out_dir / f"maze_schematics_appendix.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def save_individual_maps(maps: Sequence[RoadMap], complexity: Dict[str, Dict[str, object]], out_dir: Path) -> List[Path]:
    plt = _matplotlib()
    maze_dir = out_dir / "mazes"
    maze_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for road_map in maps:
        fig, ax = plt.subplots(figsize=(4.0, 2.6), facecolor="white")
        draw_map(ax, road_map, complexity[road_map.name], show_stats=True)
        fig.tight_layout()
        path = maze_dir / f"{road_map.name}.pdf"
        fig.savefig(path)
        paths.append(path)
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw publication schematics for the seven representative maze maps.")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--num-states", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = build_maps(args.num_states, MAP_ORDER)
    map_by_name = {road_map.name: road_map for road_map in maps}
    missing = [name for name in MAP_ORDER if name not in map_by_name]
    if missing:
        raise ValueError(f"Unknown or missing map names: {missing}")
    maps = [map_by_name[name] for name in MAP_ORDER]
    complexity = validate_map_family(maps, args.num_states)
    paths: List[Path] = []
    paths.extend(save_main_layout(maps, complexity, out_dir))
    paths.extend(save_appendix_layout(maps, complexity, out_dir))
    paths.extend(save_individual_maps(maps, complexity, out_dir))
    print("wrote:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
