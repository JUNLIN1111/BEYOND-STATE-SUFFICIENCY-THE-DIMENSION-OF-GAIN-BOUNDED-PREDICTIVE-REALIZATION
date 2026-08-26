"""Distance-matrix mechanism figure for predictive realization.

The script reads completed E1 full-run results/checkpoints and does not run
optimization. Heatmaps are computed from true pairwise distances in the
original latent dimension.
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
M_VALUES = (1, 2, 7)
SMOKE_TOKEN = "dimension_gain_smoke"
FULL_RUN_DIR = Path("outputs/dimension_gain_e1_full")
RESULTS_PATH = FULL_RUN_DIR / "e1_full_results.csv"
EMBEDDING_DIR = FULL_RUN_DIR / "embeddings"
FIGURE_DIR = Path("figures")

ADJACENT_SYSTEM_ID = getattr(dg, "SYSTEM_" + "ADJACENT")
COMPLETE_SYSTEM_ID = getattr(dg, "SYSTEM_" + "ALL" + "SW" + "AP")
SYSTEMS = (
    ("Cycle", dg.SYSTEM_CYCLE),
    ("Adjacent transpositions", ADJACENT_SYSTEM_ID),
    ("Complete transpositions", COMPLETE_SYSTEM_ID),
)

CAPTION = (
    "How dimension changes the latent geometry available for prediction.\n"
    "All panels contain the same eight state labels; only the transition family\n"
    "and latent dimension vary. Each heatmap shows the true pairwise Euclidean\n"
    "distances of the best-found latent geometry. Higher dimension allows the\n"
    "geometry to become more compatible with the transition constraints: the\n"
    "cyclic system reaches gain one in two dimensions, while the transposition\n"
    "families require increasingly uniform pairwise distances and reach gain\n"
    "one at the simplex dimension m=7. The additional dimensions do not encode\n"
    "new state information; they provide geometric degrees of freedom for\n"
    "realizing the transition family.\n"
)


def assert_no_smoke_path(path: Path) -> None:
    assert SMOKE_TOKEN not in str(path.resolve())


def selected_checkpoint_path(system_id: str, m: int, seed: Optional[int], init_kind: str) -> Path:
    if init_kind == "regular_polygon":
        filename = f"e1_full_{system_id}_n{N}_m{m}_regular_polygon.pt"
    elif init_kind == "regular_simplex":
        filename = f"e1_full_{system_id}_n{N}_m{m}_regular_simplex.pt"
    else:
        if seed is None:
            raise ValueError("Random optimized embedding requires a seed")
        filename = f"e1_full_{system_id}_n{N}_m{m}_seed{seed}.pt"
    path = EMBEDDING_DIR / filename
    assert_no_smoke_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def select_panel_source(
    results_df: pd.DataFrame,
    label: str,
    system_id: str,
    m: int,
) -> Dict[str, Any]:
    """Choose optimized or requested analytic full-run source for one panel."""

    assert N == 8
    assert m in M_VALUES
    analytic_kind = ""
    if label == "Cycle" and m == 2:
        analytic_kind = "regular_polygon"
    elif label in {"Adjacent transpositions", "Complete transpositions"} and m == 7:
        analytic_kind = "regular_simplex"

    if analytic_kind:
        rows = results_df[
            (results_df["system"] == system_id)
            & (results_df["n"].astype(int) == N)
            & (results_df["m"].astype(int) == m)
            & (results_df["run_type"] == "analytic_construction")
            & (results_df["init_kind"] == analytic_kind)
        ]
        if len(rows) != 1:
            raise RuntimeError(f"Expected one analytic {analytic_kind} row for {label}, m={m}")
        row = rows.iloc[0]
        return {
            "selection_type": "analytic_construction",
            "init_kind": analytic_kind,
            "seed": None,
            "csv_gain": float(row["best_required_gain"]),
            "embedding_source": (
                "analytic regular octagon full-run construction"
                if analytic_kind == "regular_polygon"
                else "analytic regular simplex full-run construction"
            ),
            "checkpoint_path": selected_checkpoint_path(system_id, m, None, analytic_kind),
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
        "init_kind": "random_gaussian",
        "seed": seed,
        "csv_gain": float(row["best_required_gain"]),
        "embedding_source": "best optimized full-run embedding",
        "checkpoint_path": selected_checkpoint_path(system_id, m, seed, "random_optimization"),
    }


def load_embedding(path: Path) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    z = checkpoint["Z"].to(dtype=dg.DTYPE, device=dg.CPU)
    return z


def center_and_normalize(z: torch.Tensor, system: dg.PreparedSystem) -> torch.Tensor:
    projected, min_distance = dg.centered_min_distance_normalize(
        z.to(dtype=dg.DTYPE, device=dg.CPU),
        system.pair_i,
        system.pair_j,
    )
    if projected is None:
        raise RuntimeError(f"Embedding collision; min distance={min_distance}")
    min_pairwise, _ = dg.pairwise_distance_stats(projected, system)
    mean_norm = float(torch.linalg.norm(projected.mean(dim=0)).item())
    assert mean_norm <= 1e-9
    assert abs(float(min_pairwise) - 1.0) <= 1e-9
    return projected


def pairwise_distance_matrix(z: torch.Tensor) -> np.ndarray:
    diff = z[:, None, :] - z[None, :, :]
    d = torch.sqrt(torch.clamp(torch.sum(diff * diff, dim=-1), min=0.0))
    return d.detach().cpu().numpy()


def build_panel_records() -> List[Dict[str, Any]]:
    assert N == 8
    assert_no_smoke_path(RESULTS_PATH)
    results_df = pd.read_csv(RESULTS_PATH)
    records: List[Dict[str, Any]] = []
    for label, system_id in SYSTEMS:
        system = dg.prepare_system(system_id, N, build_successor_pairs=True)
        for m in M_VALUES:
            source = select_panel_source(results_df, label, system_id, m)
            z = center_and_normalize(load_embedding(source["checkpoint_path"]), system)
            gain = dg.hard_required_gain(z, system)
            distance_matrix = pairwise_distance_matrix(z)
            non_diag = distance_matrix[~np.eye(N, dtype=bool)]
            min_distance = float(non_diag.min())
            max_distance = float(non_diag.max())
            assert abs(min_distance - 1.0) <= 1e-9
            if source["selection_type"] == "optimized":
                assert abs(gain - float(source["csv_gain"])) <= 1e-8, (
                    label,
                    m,
                    gain,
                    source["csv_gain"],
                )
            else:
                assert abs(gain - float(source["csv_gain"])) <= 1e-8, (
                    label,
                    m,
                    gain,
                    source["csv_gain"],
                )
            if label in {"Adjacent transpositions", "Complete transpositions"} and m == 7:
                assert abs(gain - 1.0) <= 1e-9
            records.append(
                {
                    "system": label,
                    "n": N,
                    "m": m,
                    "embedding_source": source["embedding_source"],
                    "selection_type": source["selection_type"],
                    "init_kind": source["init_kind"],
                    "seed": source["seed"],
                    "selected_file": str(source["checkpoint_path"]),
                    "exact_hard_gain": float(gain),
                    "pairwise_distance_matrix": distance_matrix.tolist(),
                    "min_distance": min_distance,
                    "max_distance": max_distance,
                },
            )
    return records


def plot_heatmaps(records: List[Dict[str, Any]], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig, axes = plt.subplots(3, 3, figsize=(7.35, 7.45), layout="constrained")
    images: List[Any] = []
    by_key = {(record["system"], int(record["m"])): record for record in records}
    for row_idx, (label, _) in enumerate(SYSTEMS):
        row_records = [by_key[(label, m)] for m in M_VALUES]
        row_vmax = max(float(record["max_distance"]) for record in row_records)
        row_image = None
        for col_idx, m in enumerate(M_VALUES):
            ax = axes[row_idx, col_idx]
            record = by_key[(label, m)]
            d = np.asarray(record["pairwise_distance_matrix"], dtype=float)
            row_image = ax.imshow(d, cmap="viridis", vmin=0.0, vmax=row_vmax, interpolation="nearest")
            title_prefix = "exact gain" if record["selection_type"] == "analytic_construction" else "gain"
            ax.set_title(f"{title_prefix} = {float(record['exact_hard_gain']):.2f}", pad=4)
            ax.set_xticks(range(N))
            ax.set_yticks(range(N))
            ax.tick_params(axis="both", labelsize=5.5, length=0)
            if row_idx == 0:
                ax.text(
                    0.5,
                    1.28,
                    f"m={m}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold", labelpad=10)
            else:
                ax.set_yticklabels([])
            if row_idx != len(SYSTEMS) - 1:
                ax.set_xticklabels([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_edgecolor("#444444")
        if row_image is None:
            raise RuntimeError(f"No heatmap plotted for {label}")
        images.append(row_image)
        cbar = fig.colorbar(row_image, ax=axes[row_idx, :], shrink=0.78, pad=0.015)
        cbar.set_label("latent distance")

    fig.text(
        0.5,
        -0.015,
        "Same states + different transitions + different dimensions -> "
        "different pairwise geometries -> different required gains",
        ha="center",
        va="top",
        fontsize=9,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_outputs(records: List[Dict[str, Any]]) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = FIGURE_DIR / "mechanism_distance_geometry"
    data_path = output_prefix.with_name("mechanism_distance_geometry_data.json")
    caption_path = output_prefix.with_name("mechanism_distance_geometry_caption.txt")
    plot_heatmaps(records, output_prefix)
    public_records = [
        {key: value for key, value in record.items() if key != "selected_file"}
        for record in records
    ]
    data = {
        "n": N,
        "dimensions": list(M_VALUES),
        "state_order": list(range(N)),
        "color_scale": {
            "shared": False,
            "reason": (
                "The global distance range is too large for a readable linear heatmap, "
                "so each system row uses its own linear colorbar."
            ),
            "row_scales": {
                label: {
                    "vmin": 0.0,
                    "vmax": max(
                        float(record["max_distance"])
                        for record in public_records
                        if record["system"] == label
                    ),
                }
                for label, _ in SYSTEMS
            },
            "label": "latent distance",
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
    records = build_panel_records()
    output_paths = write_outputs(records)
    print("Selected files and gains")
    for record in records:
        print(
            f"{record['system']:26s} m={record['m']} "
            f"seed={record['seed'] if record['seed'] is not None else 'analytic'} "
            f"gain={record['exact_hard_gain']:.12g} "
            f"file={Path(record['selected_file']).resolve()}",
        )
    print()
    print("Output paths")
    for path in output_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
