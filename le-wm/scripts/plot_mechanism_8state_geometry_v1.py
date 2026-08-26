from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch


N = 8
M_VALUES = (1, 2, 3)
SYSTEM_LABELS = ("Cycle", "Adjacent transpositions", "Complete transpositions")
SYSTEM_COLORS = {
    "Cycle": "#0072B2",
    "Adjacent transpositions": "#D55E00",
    "Complete transpositions": "#009E73",
}
OUTPUT_STEM = "mechanism_8state_geometry_v1"
CAPTION = (
    "How dimension changes the geometry available to the same state set. All\n"
    "panels contain the same eight distinguishable states; rows differ only in\n"
    "the transition family and columns differ in latent dimension. In one\n"
    "dimension all states must lie on a line. As dimension increases, the\n"
    "encoder can realize richer pairwise geometries. The cyclic transition can\n"
    "be organized non-expansively in two dimensions, while the transposition\n"
    "families require more geometric freedom because their action constraints\n"
    "couple many state pairs. Edge overlays indicate the controlled transition\n"
    "structure; complete transpositions impose all-to-all pair constraints. For\n"
    "readability, one-dimensional panels use state-ordered equal display spacing;\n"
    "reported gains are recomputed from the normalized E1 embeddings."
)
BOTTOM_STATEMENT = "Same states, higher dimension -> more flexible geometry -> lower required gain"
DTYPE = torch.float64
TOL = 1e-8


def _assert_full_run_path(path: Path) -> None:
    resolved = path.resolve()
    text = str(resolved)
    assert "smoke" not in text.lower(), f"Refusing to use smoke outputs: {resolved}"
    assert "dryrun" not in text.lower(), f"Refusing to use dry-run outputs: {resolved}"
    assert resolved.name == "dimension_gain_e1_full", f"Expected full-run E1 directory, got {resolved}"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _label_from_row(row: Dict[str, object]) -> str:
    label = str(row.get("system_label", "")).strip()
    if label:
        return label
    raw = str(row.get("system", ""))
    if raw == "Cycle":
        return "Cycle"
    if raw.startswith("Adjacent"):
        return "Adjacent transpositions"
    return "Complete transpositions"


def _float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "":
        return float("nan")
    return float(text)


def _int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() == "nan":
        return None
    return int(float(text))


def _summary_index(summary_rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, int], Dict[str, str]]:
    out: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in summary_rows:
        label = _label_from_row(row)
        n = int(float(row["n"]))
        m = int(float(row["m"]))
        if n == N and m in M_VALUES and label in SYSTEM_LABELS:
            out[(label, m)] = row
    expected = {(label, m) for label in SYSTEM_LABELS for m in M_VALUES}
    missing = expected - set(out)
    if missing:
        raise RuntimeError(f"E1 summary is missing cells: {sorted(missing)}")
    return out


def _best_result_row(results_rows: Iterable[Dict[str, str]], label: str, m: int) -> Dict[str, str]:
    candidates = []
    for row in results_rows:
        if _label_from_row(row) != label:
            continue
        if int(float(row["n"])) != N or int(float(row["m"])) != m:
            continue
        if str(row["run_type"]) != "random_optimization":
            continue
        candidates.append(row)
    if not candidates:
        raise RuntimeError(f"No optimized E1 result row for {label}, m={m}.")
    return sorted(candidates, key=lambda row: (_float_or_nan(row["best_required_gain"]), _float_or_nan(row["seed"])))[0]


def _load_checkpoint(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    lower = str(path.resolve()).lower()
    assert "smoke" not in lower and "dryrun" not in lower, f"Refusing non-full checkpoint: {path}"
    return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint_label(payload: Dict[str, object]) -> str:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RuntimeError("Checkpoint metadata is missing or malformed.")
    return _label_from_row(metadata)


def _candidate_checkpoints(embedding_dir: Path, label: str, m: int) -> List[Tuple[Path, Dict[str, object]]]:
    candidates: List[Tuple[Path, Dict[str, object]]] = []
    for path in sorted(embedding_dir.glob(f"e1_full_*_n{N}_m{m}_*.pt")):
        payload = _load_checkpoint(path)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if _checkpoint_label(payload) != label:
            continue
        if int(metadata.get("n", -1)) != N or int(metadata.get("m", -1)) != m:
            continue
        candidates.append((path, payload))
    if not candidates:
        raise RuntimeError(f"No E1 full checkpoint found for {label}, m={m}.")
    return candidates


def _select_embedding(
    embedding_dir: Path,
    summary_row: Dict[str, str],
    best_result: Optional[Dict[str, str]],
) -> Tuple[torch.Tensor, Dict[str, object], str]:
    label = _label_from_row(summary_row)
    m = int(float(summary_row["m"]))
    solution_source = str(summary_row["best_solution_source"]).strip()
    target_gain = float(summary_row["best_available_upper_bound"])
    candidates = _candidate_checkpoints(embedding_dir, label, m)

    if solution_source == "analytic":
        analytic = []
        for path, payload in candidates:
            metadata = payload["metadata"]
            if isinstance(metadata, dict) and metadata.get("run_type") == "analytic_construction":
                analytic.append((path, payload))
        if not analytic:
            raise RuntimeError(f"No analytic checkpoint found for {label}, m={m}.")
        path, payload = sorted(
            analytic,
            key=lambda item: abs(float(item[1]["best_required_gain"]) - target_gain),
        )[0]
        source_text = "E1 full-run analytic regular-octagon construction"
    else:
        if best_result is None:
            raise RuntimeError(f"Missing optimized result row for {label}, m={m}.")
        target_seed = _int_or_none(best_result["seed"])
        optimized = []
        for path, payload in candidates:
            metadata = payload["metadata"]
            if not isinstance(metadata, dict):
                continue
            if metadata.get("run_type") != "random_optimization":
                continue
            if _int_or_none(metadata.get("seed")) == target_seed:
                optimized.append((path, payload))
        if not optimized:
            raise RuntimeError(f"No optimized checkpoint found for {label}, m={m}, seed={target_seed}.")
        path, payload = optimized[0]
        source_text = "E1 full-run optimized embedding"

    metadata = dict(payload["metadata"]) if isinstance(payload.get("metadata"), dict) else {}
    Z = payload["Z"].detach().cpu().to(dtype=DTYPE)
    assert tuple(Z.shape) == (N, m), f"Unexpected embedding shape for {label}, m={m}: {tuple(Z.shape)}"
    return Z, metadata, source_text


def _transitions(label: str) -> np.ndarray:
    identity = np.arange(N, dtype=np.int64)
    if label == "Cycle":
        return ((identity + 1) % N).reshape(1, N)
    if label == "Adjacent transpositions":
        rows = []
        for i in range(N - 1):
            row = identity.copy()
            row[i], row[i + 1] = row[i + 1], row[i]
            rows.append(row)
        return np.stack(rows, axis=0)
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            row = identity.copy()
            row[i], row[j] = row[j], row[i]
            rows.append(row)
    return np.stack(rows, axis=0)


def _transition_edges(label: str) -> List[Dict[str, object]]:
    if label == "Cycle":
        return [{"edge": [i, (i + 1) % N], "directed": True} for i in range(N)]
    if label == "Adjacent transpositions":
        return [{"edge": [i, i + 1], "directed": False} for i in range(N - 1)]
    return [{"edge": [i, j], "directed": False} for i in range(N) for j in range(i + 1, N)]


def _normalize_embedding(Z: torch.Tensor) -> torch.Tensor:
    Z = Z.detach().cpu().to(dtype=DTYPE).clone()
    Z -= Z.mean(dim=0, keepdim=True)
    distances = torch.cdist(Z, Z)
    eye = torch.eye(Z.shape[0], dtype=torch.bool)
    min_distance = distances.masked_fill(eye, float("inf")).min()
    if not torch.isfinite(min_distance) or float(min_distance) <= 0.0:
        raise RuntimeError("Embedding has non-finite or zero minimum pairwise distance.")
    Z /= min_distance
    Z -= Z.mean(dim=0, keepdim=True)
    normalized_distances = torch.cdist(Z, Z).masked_fill(eye, float("inf"))
    assert abs(float(normalized_distances.min()) - 1.0) <= 1e-8
    assert float(torch.max(torch.abs(Z.mean(dim=0)))) <= 1e-10
    return Z


def _hard_gain_and_witness(Z: torch.Tensor, transitions: np.ndarray) -> Tuple[float, Dict[str, object]]:
    coords = Z.detach().cpu().numpy().astype(np.float64)
    pair_i, pair_j = np.triu_indices(N, k=1)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    best_gain = -math.inf
    best: Dict[str, object] = {}
    for action_idx, transition in enumerate(transitions):
        for i, j in zip(pair_i, pair_j):
            d_now = float(d[i, j])
            si = int(transition[int(i)])
            sj = int(transition[int(j)])
            d_next = float(d[si, sj])
            ratio = d_next / d_now
            if ratio > best_gain:
                best_gain = ratio
                best = {
                    "action_index": int(action_idx),
                    "current_pair": [int(i), int(j)],
                    "successor_pair_ordered": [si, sj],
                    "d_now": d_now,
                    "d_next": d_next,
                    "ratio": ratio,
                }
    if not math.isfinite(best_gain):
        raise RuntimeError("Hard-gain recomputation failed.")
    return float(best_gain), best


def _display_coords_from_normalized(Z: torch.Tensor, m: int) -> np.ndarray:
    if m == 1:
        x = np.linspace(-3.5, 3.5, N, dtype=np.float64)
        return np.column_stack([x, np.zeros(N)])
    return Z.detach().cpu().numpy().astype(np.float64)


def _point_labels_1d(ax: plt.Axes, coords: np.ndarray) -> None:
    offsets = [0.20, -0.24, 0.34, -0.38, 0.20, -0.24, 0.34, -0.38]
    for idx, (x, _y) in enumerate(coords):
        ax.text(x, offsets[idx], str(idx), ha="center", va="center", fontsize=7.5, color="0.1")


def _point_labels_2d(ax: plt.Axes, coords: np.ndarray) -> None:
    for idx, (x, y) in enumerate(coords):
        ax.annotate(str(idx), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color="0.1")


def _point_labels_3d(ax: plt.Axes, coords: np.ndarray) -> None:
    for idx, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(idx), fontsize=7.0, color="0.1")


def _scatter_points(ax: plt.Axes, coords: np.ndarray, m: int) -> None:
    if m == 3:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=34,
            c="#222222",
            edgecolors="white",
            linewidths=0.7,
            depthshade=False,
            zorder=5,
        )
        _point_labels_3d(ax, coords)
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=38,
            facecolor="#222222",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        if m == 1:
            _point_labels_1d(ax, coords)
        else:
            _point_labels_2d(ax, coords)


def _draw_arc(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    directed: bool,
    alpha: float,
    lw: float,
    rad: float,
) -> None:
    arrowstyle = "-|>" if directed else "-"
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        mutation_scale=7,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)


def _draw_edges_1d(ax: plt.Axes, coords: np.ndarray, label: str, color: str) -> None:
    if label == "Cycle":
        for i in range(N):
            start = coords[i].copy()
            end = coords[(i + 1) % N].copy()
            rad = 0.28 if end[0] >= start[0] else -0.28
            _draw_arc(ax, start, end, color, True, 0.62, 0.9, rad)
        return
    if label == "Adjacent transpositions":
        for i in range(N - 1):
            _draw_arc(ax, coords[i], coords[i + 1], color, False, 0.58, 1.2, 0.18)
        return
    for i in range(N):
        for j in range(i + 1, N):
            span = abs(j - i)
            rad = 0.08 + 0.035 * span
            _draw_arc(ax, coords[i], coords[j], color, False, 0.14, 0.6, rad)


def _draw_edges_2d(ax: plt.Axes, coords: np.ndarray, label: str, color: str) -> None:
    if label == "Cycle":
        for i in range(N):
            start = coords[i]
            end = coords[(i + 1) % N]
            vec = end - start
            patch = FancyArrowPatch(
                start + 0.08 * vec,
                end - 0.12 * vec,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.9,
                color=color,
                alpha=0.56,
                zorder=1,
            )
            ax.add_patch(patch)
        return
    if label == "Adjacent transpositions":
        for i in range(N - 1):
            ax.plot(
                [coords[i, 0], coords[i + 1, 0]],
                [coords[i, 1], coords[i + 1, 1]],
                color=color,
                alpha=0.55,
                linewidth=1.2,
                zorder=1,
            )
        return
    for i in range(N):
        for j in range(i + 1, N):
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=color,
                alpha=0.13,
                linewidth=0.55,
                zorder=1,
            )


def _draw_edges_3d(ax: plt.Axes, coords: np.ndarray, label: str, color: str) -> None:
    if label == "Cycle":
        for i in range(N):
            start = coords[i]
            end = coords[(i + 1) % N]
            vec = end - start
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                alpha=0.35,
                linewidth=0.9,
                zorder=1,
            )
            ax.quiver(
                start[0] + 0.72 * vec[0],
                start[1] + 0.72 * vec[1],
                start[2] + 0.72 * vec[2],
                0.16 * vec[0],
                0.16 * vec[1],
                0.16 * vec[2],
                color=color,
                alpha=0.55,
                linewidth=0.8,
                arrow_length_ratio=0.45,
                normalize=False,
            )
        return
    if label == "Adjacent transpositions":
        for i in range(N - 1):
            ax.plot(
                [coords[i, 0], coords[i + 1, 0]],
                [coords[i, 1], coords[i + 1, 1]],
                [coords[i, 2], coords[i + 1, 2]],
                color=color,
                alpha=0.55,
                linewidth=1.15,
                zorder=1,
            )
        return
    for i in range(N):
        for j in range(i + 1, N):
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                [coords[i, 2], coords[j, 2]],
                color=color,
                alpha=0.12,
                linewidth=0.5,
                zorder=1,
            )


def _set_clean_2d_axes(ax: plt.Axes, coords: np.ndarray, m: int) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if m == 1:
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        pad = max(0.8, 0.08 * (x_max - x_min))
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(-0.62, 0.72)
        ax.axhline(0.0, color="0.72", lw=0.7, zorder=0)
    else:
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        center = np.asarray([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        span = max(float(x_max - x_min), float(y_max - y_min), 1.0)
        pad = 0.20 * span
        ax.set_xlim(center[0] - span / 2.0 - pad, center[0] + span / 2.0 + pad)
        ax.set_ylim(center[1] - span / 2.0 - pad, center[1] + span / 2.0 + pad)
        ax.set_aspect("equal", adjustable="box")


def _set_clean_3d_axes(ax: plt.Axes, coords: np.ndarray) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    span = max(float(np.max(maxs - mins)), 1.0)
    pad = 0.18 * span
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - span / 2.0 - pad, c + span / 2.0 + pad)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-55)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))


def _gain_title(cell: Dict[str, object]) -> str:
    gain = float(cell["exact_hard_gain"])
    if bool(cell["analytic_or_optimized"] == "analytic") and abs(gain - 1.0) <= 1e-8:
        return "exact gain = 1.00"
    return f"best-found gain = {gain:.2f}"


def _plot_cell(ax: plt.Axes, cell: Dict[str, object]) -> None:
    label = str(cell["system"])
    m = int(cell["m"])
    color = SYSTEM_COLORS[label]
    plot_coords = np.asarray(cell["display_coordinates_used_in_figure"], dtype=np.float64)
    if m == 1:
        _draw_edges_1d(ax, plot_coords, label, color)
        _scatter_points(ax, plot_coords, m)
        _set_clean_2d_axes(ax, plot_coords, m)
    elif m == 2:
        _draw_edges_2d(ax, plot_coords, label, color)
        _scatter_points(ax, plot_coords, m)
        _set_clean_2d_axes(ax, plot_coords, m)
    else:
        _draw_edges_3d(ax, plot_coords, label, color)
        _scatter_points(ax, plot_coords, m)
        _set_clean_3d_axes(ax, plot_coords)

    if label == "Complete transpositions" and m == 1:
        ax.text(
            0.02,
            0.04,
            "each edge denotes one\ntransposition action",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.25",
        )


def _json_coordinates(Z: torch.Tensor) -> List[List[float]]:
    return [[float(x) for x in row] for row in Z.detach().cpu().numpy().tolist()]


def _json_array(arr: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in np.asarray(arr, dtype=np.float64).tolist()]


def _build_cells(e1_dir: Path) -> List[Dict[str, object]]:
    _assert_full_run_path(e1_dir)
    summary_path = e1_dir / "e1_full_summary.csv"
    results_path = e1_dir / "e1_full_results.csv"
    embedding_dir = e1_dir / "embeddings"
    for path in (summary_path, results_path, embedding_dir):
        if not path.exists():
            raise FileNotFoundError(path)
        _assert_full_run_path(e1_dir)

    summary = _summary_index(_read_csv(summary_path))
    results_rows = _read_csv(results_path)
    cells: List[Dict[str, object]] = []
    for label in SYSTEM_LABELS:
        for m in M_VALUES:
            assert N == 8
            assert m in {1, 2, 3}
            summary_row = summary[(label, m)]
            solution_source = str(summary_row["best_solution_source"]).strip()
            best_result = None if solution_source == "analytic" else _best_result_row(results_rows, label, m)
            Z_raw, metadata, source_text = _select_embedding(embedding_dir, summary_row, best_result)
            Z = _normalize_embedding(Z_raw)
            gain, witness = _hard_gain_and_witness(Z, _transitions(label))
            summary_gain = float(summary_row["best_available_upper_bound"])
            assert abs(gain - summary_gain) <= TOL, (label, m, gain, summary_gain)
            metadata_gain = float(metadata.get("best_required_gain", gain))
            assert abs(gain - metadata_gain) <= TOL, (label, m, gain, metadata_gain)
            seed = _int_or_none(metadata.get("seed"))
            display_coords = _display_coords_from_normalized(Z, m)
            layout_note = (
                "state-ordered equal-spacing schematic; gain recomputed from normalized E1 embedding"
                if m == 1
                else "actual normalized E1 coordinates"
            )
            cell = {
                "system": label,
                "n": N,
                "m": m,
                "embedding_source": source_text,
                "embedding source": source_text,
                "seed": seed,
                "exact_hard_gain": gain,
                "exact hard gain": gain,
                "coordinates_after_normalization": _json_coordinates(Z),
                "coordinates after normalization": _json_coordinates(Z),
                "display_coordinates_used_in_figure": _json_array(display_coords),
                "visualization_layout": layout_note,
                "transition_edges_drawn": _transition_edges(label),
                "transition edges drawn": _transition_edges(label),
                "analytic_or_optimized": "analytic" if solution_source == "analytic" else "optimized",
                "whether analytic or optimized": "analytic" if solution_source == "analytic" else "optimized",
                "hard_gain_witness_not_plotted": witness,
            }
            cells.append(cell)
    return cells


def _make_figure(cells: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(9.4, 7.9), facecolor="white")
    grid = GridSpec(3, 3, figure=fig, left=0.075, right=0.985, bottom=0.085, top=0.925, wspace=0.06, hspace=0.20)

    axes: Dict[Tuple[int, int], plt.Axes] = {}
    for row_idx, label in enumerate(SYSTEM_LABELS):
        for col_idx, m in enumerate(M_VALUES):
            projection = "3d" if m == 3 else None
            ax = fig.add_subplot(grid[row_idx, col_idx], projection=projection)
            axes[(row_idx, col_idx)] = ax
            cell = next(item for item in cells if item["system"] == label and int(item["m"]) == m)
            _plot_cell(ax, cell)
            title = _gain_title(cell)
            if row_idx == 0:
                title = f"m = {m}\n{title}"
            ax.set_title(title, pad=2.5)

    row_y = [0.79, 0.50, 0.21]
    row_labels = ["Cycle", "Adjacent\ntranspositions", "Complete\ntranspositions"]
    for y, label in zip(row_y, row_labels):
        fig.text(0.022, y, label, rotation=90, ha="center", va="center", fontsize=9.5, fontweight="bold")
    fig.text(0.53, 0.030, BOTTOM_STATEMENT, ha="center", va="center", fontsize=8.3, color="0.25")

    paths = [
        output_dir / f"{OUTPUT_STEM}.pdf",
        output_dir / f"{OUTPUT_STEM}.png",
        output_dir / f"{OUTPUT_STEM}.svg",
    ]
    for path in paths:
        if path.suffix == ".png":
            fig.savefig(path, dpi=320)
        else:
            fig.savefig(path)
    plt.close(fig)
    return paths


def _write_outputs(cells: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    data_path = output_dir / f"{OUTPUT_STEM}_data.json"
    caption_path = output_dir / f"{OUTPUT_STEM}_caption.txt"
    payload = {
        "figure": OUTPUT_STEM,
        "n": N,
        "m_values": list(M_VALUES),
        "source": "completed E1 full-run outputs",
        "cells": cells,
    }
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    caption_path.write_text(CAPTION + "\n")
    return [data_path, caption_path]


def _print_validation(cells: List[Dict[str, object]]) -> None:
    print("Selected E1 full-run cells:")
    for label in SYSTEM_LABELS:
        for m in M_VALUES:
            cell = next(item for item in cells if item["system"] == label and int(item["m"]) == m)
            seed = cell["seed"]
            seed_text = "analytic" if seed is None else str(seed)
            print(
                f"  {label:26s} m={m} seed={seed_text:>8s} "
                f"source={cell['analytic_or_optimized']:9s} gain={float(cell['exact_hard_gain']):.12g}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the 8-state mechanism geometry figure from completed E1 full-run outputs.")
    parser.add_argument(
        "--e1-dir",
        default="/home/junlin/SemNev/Toy/outputs/dimension_gain_e1_full",
        help="Completed E1 full-run output directory.",
    )
    parser.add_argument("--output-dir", default="figures")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    e1_dir = Path(args.e1_dir)
    output_dir = Path(args.output_dir)
    cells = _build_cells(e1_dir)
    _print_validation(cells)
    figure_paths = _make_figure(cells, output_dir)
    extra_paths = _write_outputs(cells, output_dir)
    print("\nOutput paths:")
    for path in figure_paths + extra_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
