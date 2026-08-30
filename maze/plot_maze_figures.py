#!/usr/bin/env python3
"""Create publication-quality figures from direct maze realization outputs.

This script is intentionally read-only with respect to scientific data: it
loads existing CSV outputs, validates them, and writes figure/caption files.
It never reruns optimization and never changes reported values.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from statistics import median

_mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "maze_figure_mplconfig"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from direct_maze_realization import (  # noqa: E402
    ACTION_ORDER,
    CERTIFIED_GAIN_ONE_LOWER_BOUNDS,
    CHECKPOINT_VERSION,
    MAZE_NAMES,
    N,
    WALLS_BY_MAZE,
)


COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
MARKERS = {"A": "o", "B": "s", "C": "^"}
LINESTYLES = {"A": "-", "B": "-", "C": "-"}
GRID_COLOR = "#b8b8b8"
WALL_COLOR = "#111111"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing required CSV: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def q(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = quantile * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def grouped(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    out: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault((row["maze"], int(row["dimension"])), []).append(row)
    return out


def close(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(a - b) <= tol


def validate_data(summary_rows: list[dict[str, str]], raw_rows: list[dict[str, str]]) -> None:
    raw_by_key = grouped(raw_rows)
    summary_by_key = grouped(summary_rows)
    required_inits = {"random", "continuation_baseline", "continuation_padded_baseline", "continuation_jittered", "simplex_analytic"}

    for key, rows in raw_by_key.items():
        for row in rows:
            if row.get("checkpoint_version") != CHECKPOINT_VERSION:
                raise AssertionError(f"{key}: raw row has incompatible checkpoint_version={row.get('checkpoint_version')}")
            if row["initialization_type"] not in required_inits:
                raise AssertionError(f"{key}: unknown initialization_type={row['initialization_type']}")
            hard_gain = f(row, "hard_gain") if row.get("hard_gain") else f(row, "best_hard_gain")
            if not close(hard_gain, f(row, "best_hard_gain")):
                raise AssertionError(f"{key}: hard_gain and best_hard_gain disagree")

    for key, rows in summary_by_key.items():
        if len(rows) != 1:
            raise AssertionError(f"{key}: expected one summary row, got {len(rows)}")
        summary = rows[0]
        candidates = raw_by_key.get(key)
        if not candidates:
            raise AssertionError(f"{key}: no raw candidate rows")
        ok_candidates = [r for r in candidates if r["status"] == "ok" and math.isfinite(f(r, "best_hard_gain"))]
        if not ok_candidates:
            raise AssertionError(f"{key}: no valid candidates")
        raw_best = min(f(r, "best_hard_gain") for r in ok_candidates)
        if not close(raw_best, f(summary, "best_hard_gain")):
            raise AssertionError(f"{key}: summary best {summary['best_hard_gain']} != raw best {raw_best}")

        random_values = sorted(
            f(r, "best_hard_gain")
            for r in candidates
            if r["initialization_type"] == "random" and r["status"] == "ok" and math.isfinite(f(r, "best_hard_gain"))
        )
        if random_values:
            checks = {
                "median_random_restart_hard_gain": median(random_values),
                "q25_random_restart_hard_gain": q(random_values, 0.25),
                "q75_random_restart_hard_gain": q(random_values, 0.75),
            }
            for field, expected in checks.items():
                if not close(expected, f(summary, field)):
                    raise AssertionError(f"{key}: {field} uses non-random or stale rows")

    for maze in MAZE_NAMES:
        maze_rows = sorted([r[0] for key, r in summary_by_key.items() if key[0] == maze], key=lambda r: int(r["dimension"]))
        previous = None
        for row in maze_rows:
            gain = f(row, "best_hard_gain")
            if not math.isfinite(gain):
                raise AssertionError(f"Maze {maze} m={row['dimension']}: non-finite best gain")
            if gain < 1.0 - 1e-6:
                raise AssertionError(f"Maze {maze} m={row['dimension']}: materially sub-one gain")
            if previous is not None and gain > previous + 1e-10:
                raise AssertionError(f"Maze {maze}: best frontier is not non-increasing")
            previous = gain

        simplex_rows = [
            r
            for r in raw_rows
            if r["maze"] == maze and int(r["dimension"]) == 63 and r["initialization_type"] == "simplex_analytic"
        ]
        if len(simplex_rows) != 1:
            raise AssertionError(f"Maze {maze}: expected one m=63 simplex_analytic row")
        if abs(f(simplex_rows[0], "best_hard_gain") - 1.0) > 1e-10:
            raise AssertionError(f"Maze {maze}: simplex analytic gain is not 1")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def draw_maze(ax: plt.Axes, maze: str, panel_label: bool = False) -> None:
    ax.set_xlim(0, N)
    ax.set_ylim(N, -0.55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for x in range(N + 1):
        ax.plot([x, x], [0, N], color=GRID_COLOR, lw=0.45, solid_capstyle="butt", zorder=1)
    for y in range(N + 1):
        ax.plot([0, N], [y, y], color=GRID_COLOR, lw=0.45, solid_capstyle="butt", zorder=1)

    for kind, r, c in sorted(WALLS_BY_MAZE[maze]):
        if kind == "V":
            ax.plot([c + 1, c + 1], [r, r + 1], color=WALL_COLOR, lw=2.1, solid_capstyle="butt", zorder=3)
        elif kind == "H":
            ax.plot([c, c + 1], [r + 1, r + 1], color=WALL_COLOR, lw=2.1, solid_capstyle="butt", zorder=3)
        else:
            raise ValueError(f"unknown wall kind: {kind}")

    ax.set_title(f"Maze {maze}", color=COLORS[maze], pad=3)
    bound = CERTIFIED_GAIN_ONE_LOWER_BOUNDS[maze]
    ax.text(
        4,
        8.42,
        rf"$\mathrm{{d}}_{{\mathrm{{PR}}}}(T;1)\geq {bound}$",
        color=COLORS[maze],
        ha="center",
        va="top",
        fontsize=7.2,
    )
    if panel_label:
        ax.text(-0.11, 1.08, "A", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def plot_structures(fig: plt.Figure, spec: gridspec.SubplotSpec, panel_label: bool = False) -> list[plt.Axes]:
    sub = spec.subgridspec(1, 3, wspace=0.18)
    axes = []
    for i, maze in enumerate(MAZE_NAMES):
        ax = fig.add_subplot(sub[0, i])
        draw_maze(ax, maze, panel_label=panel_label and i == 0)
        axes.append(ax)
    return axes


def rows_for_maze(summary_rows: list[dict[str, str]], maze: str) -> list[dict[str, str]]:
    return sorted([row for row in summary_rows if row["maze"] == maze], key=lambda r: int(r["dimension"]))


def plot_frontier(
    ax: plt.Axes,
    summary_rows: list[dict[str, str]],
    panel_label: bool = False,
    yscale: str = "log",
) -> None:
    for maze in MAZE_NAMES:
        rows = rows_for_maze(summary_rows, maze)
        xs = [int(r["dimension"]) for r in rows]
        best = [f(r, "best_hard_gain") for r in rows]
        med = [f(r, "median_random_restart_hard_gain") for r in rows]
        q25 = [f(r, "q25_random_restart_hard_gain") for r in rows]
        q75 = [f(r, "q75_random_restart_hard_gain") for r in rows]
        ax.fill_between(xs, q25, q75, color=COLORS[maze], alpha=0.11, lw=0)
        ax.plot(
            xs,
            med,
            color=COLORS[maze],
            lw=1.0,
            ls="--",
            marker=MARKERS[maze],
            markersize=2.4,
            markerfacecolor="white",
            markeredgewidth=0.8,
            alpha=0.8,
        )
        ax.plot(
            xs,
            best,
            color=COLORS[maze],
            lw=1.75,
            ls=LINESTYLES[maze],
            marker=MARKERS[maze],
            markersize=3.4,
            label=f"Maze {maze}",
        )

    ax.axhline(1.0, color="#222222", lw=0.9, ls=(0, (2, 2)), zorder=0)
    ax.annotate(
        "analytic simplex",
        xy=(63, 1.0),
        xytext=(-27, 11),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.6,
        arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "#333333"},
    )

    xmin, xmax = 2, 63
    ax.set_xlim(xmin - 1.0, xmax + 1.0)
    ymax = max(f(row, "q75_random_restart_hard_gain") for row in summary_rows)
    ax.set_ylim(0.985, ymax * 1.06)
    if yscale == "log":
        ax.set_yscale("log")
        ticks = [1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0]
        ticks = [tick for tick in ticks if 0.985 <= tick <= ymax * 1.06]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in ticks]))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Ambient dimension $m$", labelpad=2)
    ax.set_ylabel("Exact hard gain", labelpad=2)
    ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_xticks([2, 4, 8, 16, 24, 29, 40, 47, 52, 63])

    bar_transform = blended_transform_factory(ax.transData, ax.transAxes)
    y0, dy = -0.16, -0.075
    for offset, maze in enumerate(MAZE_NAMES):
        bound = CERTIFIED_GAIN_ONE_LOWER_BOUNDS[maze]
        y = y0 + offset * dy
        ax.plot(
            [xmin, bound - 0.4],
            [y, y],
            color=COLORS[maze],
            lw=2.0,
            alpha=0.85,
            solid_capstyle="butt",
            transform=bar_transform,
            clip_on=False,
        )
        ax.text(
            bound + 0.45,
            y,
            rf"{maze}: gain 1 infeasible for $m<{bound}$",
            color=COLORS[maze],
            va="center",
            ha="left",
            fontsize=6.2,
            transform=bar_transform,
            clip_on=False,
        )

    handles = [
        Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], lw=1.75, markersize=3.4, label=f"Maze {m}")
        for m in MAZE_NAMES
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#333333", lw=1.75, label="best constructive"),
            Line2D([0], [0], color="#333333", lw=1.0, ls="--", label="random median"),
        ]
    )
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=1, handlelength=1.9)
    ax.text(
        0.0,
        -0.40,
        "Certified bounds mark infeasible regions for gain-one realization; they are not exact thresholds.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#333333",
    )
    if panel_label:
        ax.text(-0.1, 1.03, "B", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save_fig(fig: plt.Figure, path_no_suffix: Path, dpi: int) -> None:
    fig.savefig(path_no_suffix.with_suffix(".pdf"))
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def write_caption(path: Path, yscale: str) -> None:
    scale_sentence = " The gain axis uses a logarithmic scale." if yscale == "log" else ""
    caption = (
        "Transition structure induces different geometric realization requirements. "
        "Left: three $8\\times 8$ deterministic systems with identical state spaces "
        "and primitive action vocabularies but different wall-induced transition "
        "tables; labels report certified lower bounds on the dimension required "
        "for gain-one realization. Right: exact hard gain obtained by direct "
        "optimization of the latent state geometry as the ambient dimension varies. "
        "Solid curves show the best constructive gain over all valid candidates; "
        "dashed curves and bands summarize median and interquartile range over "
        "random restarts only."
        f"{scale_sentence} The horizontal reference marks gain one, and the "
        "$m=63$ regular simplex gives an analytic gain-one construction. The "
        "certified bounds identify dimensions where gain-one realization is "
        "impossible; they are lower bounds, not exact realization thresholds."
    )
    path.write_text(caption + "\n")


def make_figures(summary_rows: list[dict[str, str]], figure_dir: Path, dpi: int, yscale: str) -> None:
    setup_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir / "mplconfig"))

    fig = plt.figure(figsize=(7.35, 3.08))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.08, 1.35],
        left=0.035,
        right=0.995,
        bottom=0.20,
        top=0.93,
        wspace=0.22,
    )
    plot_structures(fig, outer[0], panel_label=True)
    ax = fig.add_subplot(outer[1])
    plot_frontier(ax, summary_rows, panel_label=True, yscale=yscale)
    save_fig(fig, figure_dir / "maze_main_figure", dpi)

    fig = plt.figure(figsize=(3.55, 5.15))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.35],
        left=0.10,
        right=0.99,
        bottom=0.14,
        top=0.95,
        hspace=0.28,
    )
    plot_structures(fig, outer[0], panel_label=True)
    ax = fig.add_subplot(outer[1])
    plot_frontier(ax, summary_rows, panel_label=True, yscale=yscale)
    save_fig(fig, figure_dir / "maze_main_figure_compact", dpi)

    fig = plt.figure(figsize=(4.9, 1.95))
    plot_structures(fig, fig.add_gridspec(1, 1, left=0.03, right=0.99, bottom=0.20, top=0.90)[0], panel_label=False)
    save_fig(fig, figure_dir / "maze_structures", dpi)

    fig, ax = plt.subplots(figsize=(4.85, 3.05))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.24, top=0.94)
    plot_frontier(ax, summary_rows, panel_label=False, yscale=yscale)
    save_fig(fig, figure_dir / "maze_dimension_gain", dpi)
    write_caption(figure_dir / "maze_main_figure_caption.tex", yscale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/direct_maze_realization_v2")
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--yscale", choices=["log", "linear"], default="log")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir) if args.figure_dir else output_dir / "paper_figures"
    summary_rows = read_csv(output_dir / "summary.csv")
    raw_rows = read_csv(output_dir / "raw_restart_results.csv")
    validate_data(summary_rows, raw_rows)
    make_figures(summary_rows, figure_dir, args.dpi, args.yscale)
    print(f"validated {len(summary_rows)} summary rows and {len(raw_rows)} raw candidate rows")
    print(f"wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
