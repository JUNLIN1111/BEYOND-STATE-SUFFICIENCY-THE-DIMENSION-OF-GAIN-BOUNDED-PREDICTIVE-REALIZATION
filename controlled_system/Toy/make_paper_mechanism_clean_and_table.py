"""Clean ICLR-style mechanism schematic and separate numeric table.

This script produces:
1. A 3x3 mechanism schematic with no numeric annotations inside panels.
2. A standalone table figure and LaTeX table for gain/error values.

It reads the monotone prediction-error table derived from completed full-run
outputs. It does not run optimization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import make_paper_mechanism_9panel as base


FIGURE_DIR = Path("figures")
ERROR_TABLE_PATH = FIGURE_DIR / "mechanism_prediction_error_table_monotone.csv"
MECHANISM_PREFIX = FIGURE_DIR / "mechanism_paper_9panel_clean"
TABLE_PREFIX = FIGURE_DIR / "mechanism_prediction_error_table"
SYSTEMS = base.SYSTEMS
M_VALUES = base.M_VALUES

MECHANISM_CAPTION = (
    "Clean mechanism schematic. Each panel contains the same eight state labels;\n"
    "rows vary the transition family and columns vary the latent dimension.\n"
    "All transition arrows or pair-action edges for the corresponding family are\n"
    "shown. The coordinates are schematic for readability: one-dimensional line,\n"
    "two-dimensional plane, and a three-dimensional schematic projection. Numeric\n"
    "gain and prediction-error values are reported separately in the table.\n"
)

TABLE_CAPTION = (
    "Prediction-error summary for the mechanism figure. Values are computed from\n"
    "completed full-run E1 embeddings and the finite-state numerical oracle at\n"
    "gain budget L=1. For each target dimension m, lower-dimensional embeddings\n"
    "are allowed by zero-padding, so source dim records which completed embedding\n"
    "was selected for the monotone best-available error profile.\n"
)


def load_table() -> pd.DataFrame:
    table = base.load_error_table()
    table = table.copy()
    table["m"] = table["m"].astype(int)
    table["selected_source_m"] = table["selected_source_m"].astype(int)
    return table


def draw_clean_panel(ax: plt.Axes, system: str, m: int) -> None:
    coords, kind = base.schematic_points(system, m)
    base.draw_transitions(ax, coords, system, m)
    base.draw_nodes(ax, coords, system)
    base.set_limits(ax, coords, system, kind)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_clean_mechanism() -> List[Path]:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.15), layout="constrained")
    for row_idx, system in enumerate(SYSTEMS):
        for col_idx, m in enumerate(M_VALUES):
            ax = axes[row_idx, col_idx]
            draw_clean_panel(ax, system, m)
            if row_idx == 0:
                ax.set_title(f"m={m}", fontsize=10, fontweight="bold", pad=5)
            if col_idx == 0:
                ax.text(
                    -0.18,
                    0.5,
                    system,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    fontweight="bold",
                    color=base.COLORS[system],
                )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MECHANISM_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(MECHANISM_PREFIX.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(MECHANISM_PREFIX.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    caption_path = MECHANISM_PREFIX.with_name("mechanism_paper_9panel_clean_caption.txt")
    data_path = MECHANISM_PREFIX.with_name("mechanism_paper_9panel_clean_data.json")
    caption_path.write_text(MECHANISM_CAPTION, encoding="utf-8")
    data_path.write_text(
        json.dumps(
            {
                "n": base.N,
                "dimensions": list(M_VALUES),
                "systems": list(SYSTEMS),
                "coordinates": "schematic, not raw checkpoint coordinates",
                "transition_drawing": {
                    "Cycle": "all 8 directed arrows",
                    "Adjacent pair actions": "all 7 neighboring-pair bidirectional edges",
                    "All-pair actions": "all 28 unordered-pair bidirectional edges",
                },
                "numeric_values": str(ERROR_TABLE_PATH),
                "caption": MECHANISM_CAPTION.strip(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [
        MECHANISM_PREFIX.with_suffix(".pdf"),
        MECHANISM_PREFIX.with_suffix(".png"),
        MECHANISM_PREFIX.with_suffix(".svg"),
        data_path,
        caption_path,
    ]


def format_gain(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_mse(value: float) -> str:
    if abs(value) < 5e-13:
        return "0"
    if value >= 100:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def table_rows(table: pd.DataFrame) -> List[List[str]]:
    rows: List[List[str]] = []
    for system in SYSTEMS:
        for m in M_VALUES:
            row = table[(table["system"] == system) & (table["m"] == m)].iloc[0]
            rows.append(
                [
                    system,
                    str(m),
                    str(int(row["selected_source_m"])),
                    format_gain(float(row["selected_exact_hard_required_gain"])),
                    format_mse(float(row["selected_oracle_mse_at_L1"])),
                ],
            )
    return rows


def make_table_figure(table: pd.DataFrame) -> List[Path]:
    columns = ["System", "m", "source dim", "required gain", "L=1 oracle MSE"]
    rows = table_rows(table)
    fig, ax = plt.subplots(figsize=(6.9, 2.55), layout="constrained")
    ax.axis("off")
    tab = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.08, 0.15, 0.19, 0.24],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.0)
    tab.scale(1.0, 1.22)
    for (row_idx, col_idx), cell in tab.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#C7C7C7")
        if row_idx == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold")
        elif col_idx == 0:
            system = rows[row_idx - 1][0]
            cell.set_text_props(color=base.COLORS[system], weight="bold")
            cell.set_facecolor("#FFFFFF")
        else:
            cell.set_facecolor("#FFFFFF")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TABLE_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(TABLE_PREFIX.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(TABLE_PREFIX.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    caption_path = TABLE_PREFIX.with_name("mechanism_prediction_error_table_caption.txt")
    tex_path = TABLE_PREFIX.with_suffix(".tex")
    caption_path.write_text(TABLE_CAPTION, encoding="utf-8")
    tex_path.write_text(make_latex_table(table), encoding="utf-8")
    return [
        TABLE_PREFIX.with_suffix(".pdf"),
        TABLE_PREFIX.with_suffix(".png"),
        TABLE_PREFIX.with_suffix(".svg"),
        tex_path,
        caption_path,
    ]


def make_latex_table(table: pd.DataFrame) -> str:
    rows = table_rows(table)
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "System & $m$ & source dim & required gain & $L=1$ oracle MSE \\\\",
        "\\midrule",
    ]
    for system, m, source_m, gain, mse in rows:
        lines.append(f"{system} & {m} & {source_m} & {gain} & {mse} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    table = load_table()
    outputs = make_clean_mechanism()
    outputs.extend(make_table_figure(table))
    print("Generated clean mechanism figure and separate table")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
