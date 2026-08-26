"""Prediction-error mechanism figure with transition arrows.

The script uses completed E1 full-run embeddings only. It does not optimize
latent embeddings and does not train a neural predictor. For each fixed
embedding it computes a small finite-state numerical oracle: the best predicted
successor points under gain budget L=1, averaged over the transition family.

The arrows drawn in each panel show one representative true transition action:
the single cyclic action for Cycle, and the hard-gain witness action for the
two transposition families.
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
from scipy.optimize import minimize

import dimension_gain_experiments as dg


N = 8
M_VALUES = (1, 2, 3)
GAIN_BUDGET = 1.0
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
    "Transition arrows and bounded-gain prediction error. All panels use the\n"
    "same eight state labels and completed full-run E1 embeddings. Arrows show\n"
    "one true transition action in the latent geometry: the cyclic action for\n"
    "Cycle and the hard-gain witness action for the two transposition families.\n"
    "Solid black segments mark the current witness pair, and dashed black\n"
    "segments mark its successor pair under that same action. The reported\n"
    "prediction error is the finite-state numerical oracle MSE at gain budget\n"
    "L=1, averaged over all actions in the transition family. As dimension gives\n"
    "the states more geometric room, the required gain and the constrained\n"
    "prediction error both decrease.\n"
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


def pairwise_arrays(system: dg.PreparedSystem) -> Tuple[np.ndarray, np.ndarray]:
    return (
        system.pair_i.detach().cpu().numpy().astype(np.int64),
        system.pair_j.detach().cpu().numpy().astype(np.int64),
    )


def selected_action_states(system: dg.PreparedSystem, action_index: int) -> List[int]:
    if system.transitions_np is None:
        return []
    row = system.transitions_np[action_index]
    changed = np.flatnonzero(row != np.arange(system.n, dtype=np.int64))
    return [int(value) for value in changed.tolist()]


def action_label(label: str, system: dg.PreparedSystem, action_index: int) -> str:
    if label == "Cycle":
        return "cyclic action"
    changed = selected_action_states(system, action_index)
    if len(changed) == 2:
        return f"action on states {changed[0]} and {changed[1]}"
    return f"action {action_index}"


def find_hard_gain_witness(z: torch.Tensor, system: dg.PreparedSystem, label: str) -> Dict[str, Any]:
    if system.transitions_np is None:
        raise RuntimeError("Transition table required for witness search")
    pair_i, pair_j = pairwise_arrays(system)
    best: Optional[Dict[str, Any]] = None
    for action_idx, transition in enumerate(system.transitions_np):
        for i, j in zip(pair_i, pair_j):
            si = int(transition[int(i)])
            sj = int(transition[int(j)])
            d_now = float(torch.linalg.norm(z[int(i)] - z[int(j)]).item())
            d_next = float(torch.linalg.norm(z[si] - z[sj]).item())
            ratio = d_next / d_now
            if best is None or ratio > float(best["ratio"]):
                best = {
                    "action_index": int(action_idx),
                    "action_label": action_label(label, system, int(action_idx)),
                    "action_changed_states": selected_action_states(system, int(action_idx)),
                    "current_pair": [int(i), int(j)],
                    "successor_pair_ordered": [si, sj],
                    "successor_pair_unordered": sorted([si, sj]),
                    "d_now": d_now,
                    "d_next": d_next,
                    "ratio": ratio,
                }
    if best is None:
        raise RuntimeError("No hard-gain witness found")
    return best


def action_gain(z_np: np.ndarray, transition: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> float:
    base = np.linalg.norm(z_np[pair_i] - z_np[pair_j], axis=1)
    target = z_np[transition.astype(np.int64)]
    succ = np.linalg.norm(target[pair_i] - target[pair_j], axis=1)
    return float(np.max(succ / base))


def solve_oracle_action_slsqp(
    z_np: np.ndarray,
    transition: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    budget: float,
) -> Dict[str, Any]:
    target = z_np[transition.astype(np.int64)]
    n, m = target.shape
    base = np.linalg.norm(z_np[pair_i] - z_np[pair_j], axis=1)
    target_pair_dist = np.linalg.norm(target[pair_i] - target[pair_j], axis=1)
    this_action_gain = float(np.max(target_pair_dist / base))

    if budget >= this_action_gain - 1e-10:
        return {
            "mse": 0.0,
            "action_gain": this_action_gain,
            "success": True,
            "status": "target_feasible",
            "iterations": 0,
            "max_ratio": this_action_gain,
            "max_constraint_violation": 0.0,
        }

    alpha = min(1.0, 0.98 * budget / max(this_action_gain, 1e-12))
    mean_target = target.mean(axis=0, keepdims=True)
    x0 = (mean_target + alpha * (target - mean_target)).reshape(-1)
    constraint_limits_squared = (budget * base) ** 2

    def objective(x: np.ndarray) -> float:
        y = x.reshape(n, m)
        residual = y - target
        return float(np.sum(residual * residual) / float(n))

    def objective_jacobian(x: np.ndarray) -> np.ndarray:
        y = x.reshape(n, m)
        return (2.0 * (y - target) / float(n)).reshape(-1)

    def constraints(x: np.ndarray) -> np.ndarray:
        y = x.reshape(n, m)
        diff = y[pair_i] - y[pair_j]
        return constraint_limits_squared - np.sum(diff * diff, axis=1)

    def constraints_jacobian(x: np.ndarray) -> np.ndarray:
        y = x.reshape(n, m)
        diff = y[pair_i] - y[pair_j]
        jac = np.zeros((len(pair_i), n * m), dtype=np.float64)
        for row_idx, (i, j) in enumerate(zip(pair_i, pair_j)):
            jac[row_idx, int(i) * m : (int(i) + 1) * m] = -2.0 * diff[row_idx]
            jac[row_idx, int(j) * m : (int(j) + 1) * m] = 2.0 * diff[row_idx]
        return jac

    result = minimize(
        objective,
        x0,
        jac=objective_jacobian,
        constraints={"type": "ineq", "fun": constraints, "jac": constraints_jacobian},
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )
    y = result.x.reshape(n, m)
    pred_pair_dist = np.linalg.norm(y[pair_i] - y[pair_j], axis=1)
    max_ratio = float(np.max(pred_pair_dist / base))
    max_violation = max(0.0, max_ratio - budget)
    return {
        "mse": float(objective(result.x)),
        "action_gain": this_action_gain,
        "success": bool(result.success) and max_violation <= 5e-6,
        "status": str(result.message),
        "iterations": int(result.nit),
        "max_ratio": max_ratio,
        "max_constraint_violation": max_violation,
    }


def oracle_mse_at_budget(z: torch.Tensor, system: dg.PreparedSystem, budget: float) -> Dict[str, Any]:
    if system.transitions_np is None:
        raise RuntimeError("Transition table required for oracle prediction error")
    z_np = z.detach().cpu().numpy().astype(np.float64)
    pair_i, pair_j = pairwise_arrays(system)
    action_results = [
        solve_oracle_action_slsqp(z_np, transition, pair_i, pair_j, budget)
        for transition in system.transitions_np
    ]
    failures = [item for item in action_results if not item["success"]]
    mse = float(np.mean([item["mse"] for item in action_results]))
    return {
        "oracle_mse": mse,
        "budget": float(budget),
        "solver": "scipy SLSQP",
        "num_actions": int(len(action_results)),
        "num_failed_actions": int(len(failures)),
        "max_constraint_violation": float(
            max(item["max_constraint_violation"] for item in action_results),
        ),
        "max_action_gain": float(max(item["action_gain"] for item in action_results)),
        "action_results": action_results,
    }


def point_list(z: torch.Tensor) -> List[List[float]]:
    return [[float(value) for value in row] for row in z.detach().cpu().numpy().tolist()]


def build_records() -> List[Dict[str, Any]]:
    assert N == 8
    assert set(M_VALUES) == {1, 2, 3}
    assert_no_smoke_path(RESULTS_PATH)
    results_df = pd.read_csv(RESULTS_PATH)
    records: List[Dict[str, Any]] = []

    for label, system_id in SYSTEMS:
        system = dg.prepare_system(system_id, N, build_successor_pairs=True)
        for m in M_VALUES:
            source = select_source(results_df, label, system_id, m)
            z = center_and_normalize(load_embedding(source["path"]), system)
            gain = dg.hard_required_gain(z, system)
            assert abs(gain - float(source["csv_gain"])) <= 1e-8, (
                label,
                m,
                gain,
                source["csv_gain"],
            )
            witness = find_hard_gain_witness(z, system, label)
            assert abs(float(witness["ratio"]) - gain) <= 1e-10
            oracle = oracle_mse_at_budget(z, system, GAIN_BUDGET)
            assert oracle["num_failed_actions"] == 0, (label, m, oracle["num_failed_actions"])
            assert abs(float(oracle["max_action_gain"]) - gain) <= 1e-8
            if gain <= GAIN_BUDGET + 1e-8:
                assert oracle["oracle_mse"] <= 1e-10, (label, m, gain, oracle["oracle_mse"])
            else:
                assert oracle["oracle_mse"] > 1e-8, (label, m, gain, oracle["oracle_mse"])
            min_distance, max_distance = dg.pairwise_distance_stats(z, system)
            selected_action = int(witness["action_index"])
            records.append(
                {
                    "system": label,
                    "system_id": system_id,
                    "n": N,
                    "m": int(m),
                    "seed": source["seed"],
                    "selection_type": source["selection_type"],
                    "embedding_source": source["embedding_source"],
                    "init_kind": source["init_kind"],
                    "csv_gain": float(source["csv_gain"]),
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
                    "drawn_action_label": action_label(label, system, selected_action),
                    "drawn_transition": system.transitions_np[selected_action].astype(int).tolist(),
                    "embedding": point_list(z),
                    "selected_file": str(source["path"]),
                },
            )
    return records


def format_mse(value: float) -> str:
    if abs(value) < 5e-11:
        return "0"
    if value < 0.01:
        return f"{value:.1e}"
    return f"{value:.3f}"


def style_2d_axes(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=5.5, length=2.5, width=0.5, colors="#555555")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#777777")


def set_axis_equal_2d(ax: plt.Axes, coords: np.ndarray, pad_fraction: float = 0.17) -> None:
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


def add_arrow_2d(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    rad: float,
    linewidth: float = 0.9,
    alpha: float = 0.72,
    mutation_scale: float = 8.0,
    zorder: int = 2,
) -> None:
    if float(np.linalg.norm(end - start)) <= 1e-12:
        return
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        shrinkA=8.0,
        shrinkB=8.0,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(patch)


def draw_witness_segments_1d(ax: plt.Axes, coords: np.ndarray, witness: Dict[str, Any]) -> None:
    current = witness["current_pair"]
    successor = witness["successor_pair_ordered"]
    ax.plot(coords[current, 0], [-0.46, -0.46], color="black", linewidth=2.0, solid_capstyle="round")
    ax.plot(
        coords[successor, 0],
        [-0.36, -0.36],
        color="black",
        linewidth=2.0,
        linestyle=(0, (3, 2)),
        solid_capstyle="round",
    )


def draw_witness_segments_2d(ax: plt.Axes, coords: np.ndarray, witness: Dict[str, Any]) -> None:
    current = witness["current_pair"]
    successor = witness["successor_pair_ordered"]
    ax.plot(
        coords[current, 0],
        coords[current, 1],
        color="black",
        linewidth=2.1,
        solid_capstyle="round",
        zorder=5,
    )
    ax.plot(
        coords[successor, 0],
        coords[successor, 1],
        color="black",
        linewidth=2.1,
        linestyle=(0, (3, 2)),
        solid_capstyle="round",
        zorder=6,
    )


def draw_witness_segments_3d(ax: plt.Axes, coords: np.ndarray, witness: Dict[str, Any]) -> None:
    current = witness["current_pair"]
    successor = witness["successor_pair_ordered"]
    ax.plot(
        coords[current, 0],
        coords[current, 1],
        coords[current, 2],
        color="black",
        linewidth=2.0,
        solid_capstyle="round",
        zorder=5,
    )
    ax.plot(
        coords[successor, 0],
        coords[successor, 1],
        coords[successor, 2],
        color="black",
        linewidth=2.0,
        linestyle=(0, (3, 2)),
        solid_capstyle="round",
        zorder=6,
    )


def action_edges_to_draw(transition: List[int]) -> List[Tuple[int, int]]:
    edges = [(idx, int(dst)) for idx, dst in enumerate(transition) if int(dst) != idx]
    return edges


def draw_1d_panel(ax: plt.Axes, record: Dict[str, Any]) -> None:
    coords = np.asarray(record["embedding"], dtype=float)
    transition = record["drawn_transition"]
    color = SYSTEM_COLORS[record["system"]]
    x = coords[:, 0]
    span = max(float(x.max() - x.min()), 1.0)
    pad = 0.08 * span
    ax.axhline(0.0, color="#555555", linewidth=0.7, zorder=1)
    edges = action_edges_to_draw(transition)
    for edge_idx, (src, dst) in enumerate(edges):
        height = 0.22 + 0.045 * (edge_idx % 5)
        rad = 0.35 if edge_idx % 2 == 0 else -0.35
        start = np.asarray([x[src], height])
        end = np.asarray([x[dst], height])
        add_arrow_2d(ax, start, end, color=color, rad=rad, linewidth=0.8, alpha=0.55, mutation_scale=7.5, zorder=2)
    draw_witness_segments_1d(ax, coords, record["witness"])
    ax.scatter(x, np.zeros_like(x), s=92, facecolor="white", edgecolor=color, linewidth=1.3, zorder=3)
    offsets = np.asarray([0.10, -0.16, 0.18, -0.24, 0.26, -0.32, 0.34, -0.40])
    for idx, value in enumerate(x):
        ax.text(float(value), float(offsets[idx]), str(idx), ha="center", va="center", fontsize=6.8, zorder=4)
        ax.plot([value, value], [0.02, offsets[idx] * 0.70], color="#BDBDBD", linewidth=0.32, zorder=2)
    ax.set_xlim(float(x.min() - pad), float(x.max() + pad))
    ax.set_ylim(-0.58, 0.72)
    ax.set_yticks([])
    ax.set_xlabel("coordinate 1", fontsize=6.8, labelpad=1)
    style_2d_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def draw_2d_panel(ax: plt.Axes, record: Dict[str, Any]) -> None:
    coords = np.asarray(record["embedding"], dtype=float)
    transition = record["drawn_transition"]
    color = SYSTEM_COLORS[record["system"]]
    for edge_idx, (src, dst) in enumerate(action_edges_to_draw(transition)):
        rad = 0.13 if edge_idx % 2 == 0 else -0.13
        if record["system"] != "Cycle":
            rad = 0.24 if edge_idx % 2 == 0 else -0.24
        add_arrow_2d(ax, coords[src], coords[dst], color=color, rad=rad)
    draw_witness_segments_2d(ax, coords, record["witness"])
    ax.scatter(coords[:, 0], coords[:, 1], s=112, facecolor="white", edgecolor=color, linewidth=1.5, zorder=7)
    for idx, (x, y) in enumerate(coords):
        ax.text(float(x), float(y), str(idx), ha="center", va="center", fontsize=6.8, zorder=8)
    set_axis_equal_2d(ax, coords)
    ax.set_xlabel("coordinate 1", fontsize=6.8, labelpad=1)
    ax.set_ylabel("coordinate 2", fontsize=6.8, labelpad=1)
    style_2d_axes(ax)


def draw_3d_panel(ax: plt.Axes, record: Dict[str, Any]) -> None:
    coords = np.asarray(record["embedding"], dtype=float)
    transition = record["drawn_transition"]
    color = SYSTEM_COLORS[record["system"]]
    for src, dst in action_edges_to_draw(transition):
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
            linewidth=0.95,
            alpha=0.62,
            normalize=False,
        )
    draw_witness_segments_3d(ax, coords, record["witness"])
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=58,
        facecolor="white",
        edgecolor=color,
        linewidth=1.2,
        depthshade=False,
        zorder=7,
    )
    for idx, (x, y, z) in enumerate(coords):
        ax.text(float(x), float(y), float(z), str(idx), ha="center", va="center", fontsize=6.8)
    set_axis_equal_3d(ax, coords)
    ax.view_init(elev=21, azim=-48)
    ax.set_xlabel("coord. 1", fontsize=6.5, labelpad=-2)
    ax.set_ylabel("coord. 2", fontsize=6.5, labelpad=-2)
    ax.set_zlabel("coord. 3", fontsize=6.5, labelpad=-5)
    ax.tick_params(labelsize=5, pad=-2, length=2.0, width=0.5)
    ax.grid(True, linewidth=0.25, color="#DDDDDD")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#DDDDDD")


def panel_title(record: Dict[str, Any]) -> str:
    return (
        f"gain = {float(record['exact_hard_gain']):.2f}\n"
        f"L=1 MSE = {format_mse(float(record['oracle_mse_at_budget']))}"
    )


def plot_figure(records: List[Dict[str, Any]], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    fig = plt.figure(figsize=(8.4, 8.05), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        3,
        left=0.078,
        right=0.985,
        top=0.895,
        bottom=0.112,
        wspace=0.30,
        hspace=0.52,
    )
    by_key = {(record["system"], int(record["m"])): record for record in records}

    for row_idx, (label, _) in enumerate(SYSTEMS):
        for col_idx, m in enumerate(M_VALUES):
            projection = "3d" if m == 3 else None
            ax = fig.add_subplot(grid[row_idx, col_idx], projection=projection)
            record = by_key[(label, m)]
            if m == 1:
                draw_1d_panel(ax, record)
            elif m == 2:
                draw_2d_panel(ax, record)
            elif m == 3:
                draw_3d_panel(ax, record)
            else:
                raise AssertionError(m)
            ax.set_title(panel_title(record), pad=4)

    col_labels = {
        1: "m=1: line",
        2: "m=2: plane",
        3: "m=3: three coordinates",
    }
    for col_idx, m in enumerate(M_VALUES):
        fig.text(
            0.078 + (col_idx + 0.5) * (0.985 - 0.078) / 3.0,
            0.938,
            col_labels[m],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    row_ys = [0.770, 0.502, 0.239]
    for row_idx, (label, _) in enumerate(SYSTEMS):
        fig.text(
            0.020,
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
        0.52,
        0.047,
        "Colored arrows: selected true action.  Black solid/dashed: witness pair before/after.  "
        "MSE uses gain budget L=1.",
        ha="center",
        va="center",
        fontsize=8.6,
    )
    fig.suptitle(
        "Transition arrows and constrained prediction error",
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
    output_prefix = FIGURE_DIR / "mechanism_prediction_arrows"
    plot_figure(records, output_prefix)

    data_path = output_prefix.with_name("mechanism_prediction_arrows_data.json")
    caption_path = output_prefix.with_name("mechanism_prediction_arrows_caption.txt")
    public_records = [
        {key: value for key, value in record.items() if key not in {"system_id", "selected_file"}}
        for record in records
    ]
    data = {
        "n": N,
        "dimensions": list(M_VALUES),
        "gain_budget": GAIN_BUDGET,
        "state_order": list(range(N)),
        "normalization": "center rows and scale so minimum pairwise distance is 1",
        "prediction_error": {
            "definition": (
                "mean over actions of min_Y sum_i ||Y_i - Z_T(i)||_2^2 / n "
                "subject to ||Y_i-Y_j||_2 <= L ||Z_i-Z_j||_2 for every pair"
            ),
            "budget_L": GAIN_BUDGET,
            "solver": "scipy SLSQP numerical convex oracle preview",
            "note": (
                "This is a small n=8 numerical preview. The publication E3 "
                "oracle script still uses cvxpy/SOCP when that dependency is available."
            ),
        },
        "visualization": {
            "drawn_action": (
                "Cycle uses the cyclic action; the other systems use the action "
                "from the hard-gain witness in that panel."
            ),
            "solid_black_segment": "current witness pair",
            "dashed_black_segment": "successor witness pair under the drawn action",
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
    records = build_records()
    output_paths = write_outputs(records)

    print("Selected full-run inputs, gains, L=1 oracle MSE, and drawn actions")
    for record in records:
        file_path = Path(record["selected_file"]).resolve()
        print(
            f"{record['system']:26s} m={record['m']} "
            f"seed={record['seed'] if record['seed'] is not None else 'analytic'} "
            f"gain={record['exact_hard_gain']:.12g} "
            f"L1_mse={record['oracle_mse_at_budget']:.12g} "
            f"action={record['drawn_action_index']} ({record['drawn_action_label']}) "
            f"file={file_path}",
        )
    print()
    print("Output paths")
    for path in output_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
