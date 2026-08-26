from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import pandas as pd
import torch
from torch import nn


ACTION_NAMES = ("up", "down", "left", "right")
ACTION_DELTAS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}
EPS64 = 1e-12
EPS32 = 1e-8
DEFAULT_OUTPUT_DIR = Path("outputs/learned_branching_maze_gain")
FULL_MAPS = (
    "straight_path",
    "u_path",
    "serpentine_path",
    "spiral_path",
    "path_0",
    "one_t_junction",
    "comb_4",
    "comb_8",
    "comb_12",
    "hierarchical_tree",
)
QUICK_MAPS = ("straight_path", "u_path", "path_0", "comb_4")
HARD_MAPS = (
    "straight_path",
    "u_path",
    "serpentine_path",
    "spiral_path",
    "path_0",
    "one_t_junction",
    "comb_4",
    "comb_8",
    "comb_12",
    "hierarchical_tree",
    "irregular_comb_16",
    "double_comb_12",
    "hierarchical_tree_8",
    "braided_ladder_8",
    "braided_ladder_12",
)
RESULT_COLUMNS = [
    "map_name",
    "map_group",
    "num_states",
    "branch_excess",
    "junction_count",
    "dead_end_count",
    "cycle_rank",
    "max_degree",
    "graph_diameter",
    "mean_shortest_path",
    "q95_shortest_path",
    "mean_geodesic_over_euclidean",
    "q95_geodesic_over_euclidean",
    "q99_geodesic_over_euclidean",
    "m",
    "seed",
    "horizon",
    "num_macro_actions",
    "L_budget",
    "lambda_gain",
    "detach_target",
    "dtype",
    "device",
    "macro_suite",
    "recon_mse",
    "recon_rmse",
    "decoded_next_mse",
    "latent_dyn_mse",
    "median_r_true",
    "q95_r_true",
    "q99_r_true",
    "q999_r_true",
    "max_r_true",
    "median_r_model",
    "q95_r_model",
    "q99_r_model",
    "q999_r_model",
    "max_r_model",
    "q95_ratio_deficit",
    "q99_ratio_deficit",
    "q999_ratio_deficit",
    "q95_norm_pair_error_now",
    "q99_norm_pair_error_now",
    "q999_norm_pair_error_now",
    "violation_rate_model",
    "violation_rate_true",
    "final_train_loss",
    "final_rec_loss",
    "final_dyn_loss",
    "final_next_decoded_loss",
    "final_gain_loss",
    "steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "parameter_count",
    "clipped_step_fraction",
    "checkpoint_path",
    "partial_checkpoint_path",
    "embedding_path",
    "loss_curve_path",
]
COMPLEXITY_COLUMNS = [
    "map_name",
    "map_group",
    "num_states",
    "junction_count",
    "branch_excess",
    "dead_end_count",
    "max_degree",
    "cycle_rank",
    "graph_diameter",
    "mean_shortest_path",
    "q95_shortest_path",
    "mean_geodesic_over_euclidean",
    "q95_geodesic_over_euclidean",
    "q99_geodesic_over_euclidean",
]
KEY_METRICS = [
    "recon_mse",
    "recon_rmse",
    "decoded_next_mse",
    "latent_dyn_mse",
    "median_r_true",
    "q95_r_true",
    "q99_r_true",
    "q999_r_true",
    "max_r_true",
    "median_r_model",
    "q95_r_model",
    "q99_r_model",
    "q999_r_model",
    "max_r_model",
    "q95_ratio_deficit",
    "q99_ratio_deficit",
    "q999_ratio_deficit",
    "q95_norm_pair_error_now",
    "q99_norm_pair_error_now",
    "q999_norm_pair_error_now",
    "violation_rate_model",
    "violation_rate_true",
    "final_train_loss",
    "final_rec_loss",
    "final_dyn_loss",
    "final_next_decoded_loss",
    "final_gain_loss",
]


@dataclass(frozen=True)
class RoadMap:
    name: str
    group: str
    coords: np.ndarray
    edges: Tuple[Tuple[int, int], ...]

    @property
    def num_states(self) -> int:
        return int(self.coords.shape[0])


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        if activation == "silu":
            activation_cls = nn.SiLU
        elif activation == "relu":
            activation_cls = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: List[nn.Module] = []
        dim = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(activation_cls())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_macro_actions: int,
        action_embedding_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(int(num_macro_actions), int(action_embedding_dim))
        self.mlp = MLP(
            input_dim=int(latent_dim) + int(action_embedding_dim),
            output_dim=int(latent_dim),
            hidden_dim=int(hidden_dim),
            hidden_layers=int(hidden_layers),
            activation=activation,
        )

    def forward(self, z: torch.Tensor, alpha_id: torch.Tensor) -> torch.Tensor:
        action_z = self.action_embedding(alpha_id)
        return self.mlp(torch.cat([z, action_z], dim=-1))


class LearnedMazeModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_macro_actions: int,
        hidden_dim: int,
        encoder_hidden_layers: int,
        predictor_hidden_layers: int,
        decoder_hidden_layers: int,
        action_embedding_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.encoder = MLP(
            input_dim=2,
            output_dim=int(latent_dim),
            hidden_dim=int(hidden_dim),
            hidden_layers=int(encoder_hidden_layers),
            activation=activation,
        )
        self.predictor = DynamicsPredictor(
            latent_dim=int(latent_dim),
            num_macro_actions=int(num_macro_actions),
            action_embedding_dim=int(action_embedding_dim),
            hidden_dim=int(hidden_dim),
            hidden_layers=int(predictor_hidden_layers),
            activation=activation,
        )
        self.decoder = MLP(
            input_dim=int(latent_dim),
            output_dim=2,
            hidden_dim=int(hidden_dim),
            hidden_layers=int(decoder_hidden_layers),
            activation=activation,
        )


def _stable_seed(*parts: object) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _canonical_edges(edges: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    unique = set()
    for src, dst in edges:
        if src == dst:
            raise ValueError("Self-edges are not valid road edges.")
        a, b = sorted((int(src), int(dst)))
        unique.add((a, b))
    return tuple(sorted(unique))


def _path_map(name: str, group: str, coords: Sequence[Tuple[int, int]]) -> RoadMap:
    coords_array = np.asarray(coords, dtype=np.int64)
    edges = []
    for idx in range(len(coords_array) - 1):
        delta = np.abs(coords_array[idx + 1] - coords_array[idx])
        if int(delta.sum()) != 1:
            raise ValueError(f"{name}: non-adjacent path step at index {idx}.")
        edges.append((idx, idx + 1))
    return RoadMap(name=name, group=group, coords=coords_array, edges=_canonical_edges(edges))


def _straight_path(num_states: int, name: str, group: str) -> RoadMap:
    return _path_map(name, group, [(x, 0) for x in range(num_states)])


def _u_path(num_states: int) -> RoadMap:
    width = max(4, int(round(math.sqrt(num_states))))
    width = min(width, max(1, num_states - 2))
    left = max(1, (num_states - width) // 2)
    right = max(1, num_states - width - left)
    while left + width + right > num_states:
        if width > 1:
            width -= 1
        elif right > 1:
            right -= 1
        else:
            left -= 1
    while left + width + right < num_states:
        right += 1

    coords: List[Tuple[int, int]] = []
    coords.extend((0, y) for y in range(left - 1, -1, -1))
    coords.extend((x, 0) for x in range(1, width + 1))
    coords.extend((width, y) for y in range(1, right + 1))
    return _path_map("u_path", "winding_control", coords)


def _serpentine_path(num_states: int) -> RoadMap:
    width = max(2, int(round(math.sqrt(num_states))))
    coords: List[Tuple[int, int]] = []
    y = 0
    while len(coords) < num_states:
        xs = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
        for x in xs:
            coords.append((x, y))
            if len(coords) == num_states:
                break
        y += 1
    return _path_map("serpentine_path", "winding_control", coords)


def _spiral_path(num_states: int) -> RoadMap:
    coords: List[Tuple[int, int]] = [(0, 0)]
    x, y = 0, 0
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    step_len = 1
    direction_idx = 0
    while len(coords) < num_states:
        for _ in range(2):
            dx, dy = directions[direction_idx % 4]
            for _ in range(step_len):
                if len(coords) == num_states:
                    break
                x += dx
                y += dy
                coords.append((x, y))
            direction_idx += 1
            if len(coords) == num_states:
                break
        step_len += 1
    return _path_map("spiral_path", "winding_control", coords)


def _even_positions(count: int, low: int, high: int) -> List[int]:
    if count <= 0:
        return []
    if high < low:
        raise ValueError(f"Cannot place {count} junctions in empty interval [{low}, {high}].")
    if count == 1:
        return [(low + high) // 2]
    raw = np.linspace(low, high, count)
    positions: List[int] = []
    used = set()
    for value in raw:
        candidate = int(round(float(value)))
        candidate = min(max(candidate, low), high)
        while candidate in used and candidate < high:
            candidate += 1
        while candidate in used and candidate > low:
            candidate -= 1
        if candidate in used:
            raise ValueError(f"Could not place {count} unique positions in [{low}, {high}].")
        used.add(candidate)
        positions.append(candidate)
    return sorted(positions)


def _terminal_lengths(num_states: int, fixed_states: int, terminals: int, min_len: int = 1) -> List[int]:
    if terminals <= 0:
        if fixed_states != num_states:
            raise ValueError("No terminal corridors available to adjust the state count.")
        return []
    remaining = int(num_states) - int(fixed_states)
    minimum = int(terminals) * int(min_len)
    if remaining < minimum:
        raise ValueError(
            f"Need at least {fixed_states + minimum} states for {terminals} terminal corridors, got {num_states}."
        )
    lengths = [int(min_len)] * int(terminals)
    extra = remaining - minimum
    idx = 0
    while extra > 0:
        lengths[idx % terminals] += 1
        extra -= 1
        idx += 1
    return lengths


def _comb_map(name: str, group: str, num_states: int, teeth: int, trunk_fraction: float = 0.34) -> RoadMap:
    if teeth < 1:
        raise ValueError("Comb maps need at least one tooth.")
    min_trunk = teeth + 2
    target_trunk = max(min_trunk, int(round(num_states * trunk_fraction)))
    trunk_len = min(target_trunk, num_states - teeth)
    trunk_len = max(trunk_len, min_trunk)
    if trunk_len + teeth > num_states:
        raise ValueError(f"{name}: num_states={num_states} is too small for {teeth} teeth.")

    coords: List[Tuple[int, int]] = [(x, 0) for x in range(trunk_len)]
    edges: List[Tuple[int, int]] = [(x, x + 1) for x in range(trunk_len - 1)]
    positions = _even_positions(teeth, 1, trunk_len - 2)
    lengths = _terminal_lengths(num_states, trunk_len, teeth, min_len=1)
    for tooth_idx, (x, length) in enumerate(zip(positions, lengths)):
        sign = 1 if tooth_idx % 2 == 0 else -1
        prev = x
        for step in range(1, length + 1):
            node = len(coords)
            coords.append((x, sign * step))
            edges.append((prev, node))
            prev = node
    return RoadMap(name=name, group=group, coords=np.asarray(coords, dtype=np.int64), edges=_canonical_edges(edges))


def _one_t_junction(num_states: int) -> RoadMap:
    trunk_len = min(max(5, int(round(num_states * 0.68))), num_states - 1)
    coords: List[Tuple[int, int]] = [(x, 0) for x in range(trunk_len)]
    edges: List[Tuple[int, int]] = [(x, x + 1) for x in range(trunk_len - 1)]
    attach = trunk_len // 2
    branch_len = num_states - trunk_len
    prev = attach
    for step in range(1, branch_len + 1):
        node = len(coords)
        coords.append((attach, step))
        edges.append((prev, node))
        prev = node
    return RoadMap("one_t_junction", "branching_sweep", np.asarray(coords, dtype=np.int64), _canonical_edges(edges))


def _hierarchical_tree_map(name: str, num_states: int, branch_points: int, side_len: Optional[int] = None) -> RoadMap:
    if side_len is None:
        side_len = 4 if num_states >= 96 else 2
    terminal_count = branch_points * 4
    min_trunk = branch_points * 5 + 2
    trunk_len = max(min_trunk, int(round(num_states * 0.35)))
    side_states = branch_points * 2 * side_len
    if trunk_len + side_states + terminal_count > num_states:
        while side_len > 1 and trunk_len + branch_points * 2 * side_len + terminal_count > num_states:
            side_len -= 1
        side_states = branch_points * 2 * side_len
    if trunk_len + side_states + terminal_count > num_states:
        raise ValueError(f"{name}: num_states={num_states} is too small.")

    coords: List[Tuple[int, int]] = [(0, y) for y in range(trunk_len)]
    edges: List[Tuple[int, int]] = [(y, y + 1) for y in range(trunk_len - 1)]
    centers = _even_positions(branch_points, 2, trunk_len - 3)
    terminal_lengths = _terminal_lengths(num_states, trunk_len + side_states, terminal_count, min_len=1)
    length_idx = 0
    for center_y in centers:
        trunk_state = center_y
        for side in (-1, 1):
            prev = trunk_state
            for step in range(1, side_len + 1):
                node = len(coords)
                coords.append((side * step, center_y))
                edges.append((prev, node))
                prev = node
            endpoint = prev
            endpoint_x = side * side_len
            for arm_sign in (-1, 1):
                length = terminal_lengths[length_idx]
                length_idx += 1
                prev_arm = endpoint
                for step in range(1, length + 1):
                    node = len(coords)
                    coords.append((endpoint_x, center_y + arm_sign * step))
                    edges.append((prev_arm, node))
                    prev_arm = node

    return RoadMap(
        name,
        "branching_sweep",
        np.asarray(coords, dtype=np.int64),
        _canonical_edges(edges),
    )


def _hierarchical_tree(num_states: int) -> RoadMap:
    return _hierarchical_tree_map("hierarchical_tree", num_states, branch_points=4)


def _irregular_comb_map(num_states: int, teeth: int = 16) -> RoadMap:
    name = f"irregular_comb_{teeth}"
    min_trunk = teeth + 2
    target_trunk = max(min_trunk, int(round(num_states * 0.30)))
    trunk_len = min(target_trunk, num_states - teeth)
    trunk_len = max(trunk_len, min_trunk)
    if trunk_len + teeth > num_states:
        raise ValueError(f"{name}: num_states={num_states} is too small.")

    coords: List[Tuple[int, int]] = [(x, 0) for x in range(trunk_len)]
    edges: List[Tuple[int, int]] = [(x, x + 1) for x in range(trunk_len - 1)]
    positions = _even_positions(teeth, 1, trunk_len - 2)
    rng = np.random.default_rng(_stable_seed(name, num_states, teeth))
    extra = num_states - trunk_len - teeth
    extras = rng.multinomial(extra, np.ones(teeth) / teeth) if extra > 0 else np.zeros(teeth, dtype=np.int64)
    lengths = (1 + extras).astype(np.int64).tolist()
    signs = [1, -1, 1, 1, -1, -1, 1, -1]
    for tooth_idx, (x, length) in enumerate(zip(positions, lengths)):
        sign = signs[tooth_idx % len(signs)]
        prev = x
        for step in range(1, int(length) + 1):
            node = len(coords)
            coords.append((x, sign * step))
            edges.append((prev, node))
            prev = node
    return RoadMap(name, "branching_sweep", np.asarray(coords, dtype=np.int64), _canonical_edges(edges))


def _double_comb_map(num_states: int, junctions: int = 12) -> RoadMap:
    name = f"double_comb_{junctions}"
    terminals = 2 * junctions
    min_trunk = junctions + 2
    target_trunk = max(min_trunk, int(round(num_states * 0.28)))
    trunk_len = min(target_trunk, num_states - terminals)
    trunk_len = max(trunk_len, min_trunk)
    if trunk_len + terminals > num_states:
        raise ValueError(f"{name}: num_states={num_states} is too small.")

    coords: List[Tuple[int, int]] = [(x, 0) for x in range(trunk_len)]
    edges: List[Tuple[int, int]] = [(x, x + 1) for x in range(trunk_len - 1)]
    positions = _even_positions(junctions, 1, trunk_len - 2)
    lengths = _terminal_lengths(num_states, trunk_len, terminals, min_len=1)
    length_idx = 0
    for x in positions:
        for sign in (1, -1):
            length = lengths[length_idx]
            length_idx += 1
            prev = x
            for step in range(1, int(length) + 1):
                node = len(coords)
                coords.append((x, sign * step))
                edges.append((prev, node))
                prev = node
    return RoadMap(name, "branching_sweep", np.asarray(coords, dtype=np.int64), _canonical_edges(edges))


def _braided_ladder_map(num_states: int, rungs: int = 8) -> RoadMap:
    name = f"braided_ladder_{rungs}"
    teeth = 2 * rungs
    min_trunk = rungs + 2
    target_trunk = max(min_trunk, int(round(num_states * 0.34)))
    trunk_len = min(target_trunk, (num_states - rungs - teeth) // 2)
    trunk_len = max(trunk_len, min_trunk)
    base_states = 2 * trunk_len + rungs
    if base_states + teeth > num_states:
        raise ValueError(f"{name}: num_states={num_states} is too small.")

    coords: List[Tuple[int, int]] = []
    bottom: Dict[int, int] = {}
    top: Dict[int, int] = {}
    for x in range(trunk_len):
        bottom[x] = len(coords)
        coords.append((x, 0))
    for x in range(trunk_len):
        top[x] = len(coords)
        coords.append((x, 2))

    edges: List[Tuple[int, int]] = []
    for x in range(trunk_len - 1):
        edges.append((bottom[x], bottom[x + 1]))
        edges.append((top[x], top[x + 1]))

    rung_positions = _even_positions(rungs, 1, trunk_len - 2)
    for x in rung_positions:
        mid = len(coords)
        coords.append((x, 1))
        edges.append((bottom[x], mid))
        edges.append((mid, top[x]))

    attach_positions = _even_positions(teeth, 1, trunk_len - 2)
    fixed_states = len(coords)
    lengths = _terminal_lengths(num_states, fixed_states, teeth, min_len=1)
    for tooth_idx, (x, length) in enumerate(zip(attach_positions, lengths)):
        if tooth_idx % 2 == 0:
            attach = bottom[x]
            y0 = 0
            direction = -1
        else:
            attach = top[x]
            y0 = 2
            direction = 1
        prev = attach
        for step in range(1, int(length) + 1):
            node = len(coords)
            coords.append((x, y0 + direction * step))
            edges.append((prev, node))
            prev = node
    return RoadMap(name, "weird_sweep", np.asarray(coords, dtype=np.int64), _canonical_edges(edges))


def build_maps(num_states: int, selected_names: Optional[Sequence[str]] = None) -> List[RoadMap]:
    factories = {
        "straight_path": lambda: _straight_path(num_states, "straight_path", "winding_control"),
        "u_path": lambda: _u_path(num_states),
        "serpentine_path": lambda: _serpentine_path(num_states),
        "spiral_path": lambda: _spiral_path(num_states),
        "path_0": lambda: _straight_path(num_states, "path_0", "branching_sweep"),
        "one_t_junction": lambda: _one_t_junction(num_states),
        "comb_4": lambda: _comb_map("comb_4", "branching_sweep", num_states, teeth=4),
        "comb_8": lambda: _comb_map("comb_8", "branching_sweep", num_states, teeth=8),
        "comb_12": lambda: _comb_map("comb_12", "branching_sweep", num_states, teeth=12),
        "hierarchical_tree": lambda: _hierarchical_tree(num_states),
        "irregular_comb_16": lambda: _irregular_comb_map(num_states, teeth=16),
        "double_comb_12": lambda: _double_comb_map(num_states, junctions=12),
        "hierarchical_tree_8": lambda: _hierarchical_tree_map("hierarchical_tree_8", num_states, branch_points=8),
        "braided_ladder_8": lambda: _braided_ladder_map(num_states, rungs=8),
        "braided_ladder_12": lambda: _braided_ladder_map(num_states, rungs=12),
    }
    selected = set(selected_names) if selected_names is not None else set(FULL_MAPS)
    unknown = sorted(selected - set(factories))
    if unknown:
        raise ValueError(f"Unknown map names: {unknown}. Available maps: {list(factories)}")
    maps = [factories[name]() for name in factories if name in selected]
    return maps


def adjacency_list(road_map: RoadMap) -> List[List[int]]:
    neighbors: List[List[int]] = [[] for _ in range(road_map.num_states)]
    for src, dst in road_map.edges:
        neighbors[src].append(dst)
        neighbors[dst].append(src)
    for items in neighbors:
        items.sort()
    return neighbors


def graph_degrees(road_map: RoadMap) -> np.ndarray:
    degrees = np.zeros(road_map.num_states, dtype=np.int64)
    for src, dst in road_map.edges:
        degrees[src] += 1
        degrees[dst] += 1
    return degrees


def shortest_path_matrix(road_map: RoadMap) -> np.ndarray:
    neighbors = adjacency_list(road_map)
    n = road_map.num_states
    dist = np.full((n, n), np.inf, dtype=np.float64)
    for start in range(n):
        dist[start, start] = 0.0
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in neighbors[node]:
                if not math.isfinite(float(dist[start, nxt])):
                    dist[start, nxt] = dist[start, node] + 1.0
                    queue.append(nxt)
    return dist


def compute_complexity(road_map: RoadMap) -> Dict[str, object]:
    degrees = graph_degrees(road_map)
    edge_count = len(road_map.edges)
    dist = shortest_path_matrix(road_map)
    if not np.isfinite(dist).all():
        raise ValueError(f"{road_map.name}: graph is not connected.")
    left, right = np.triu_indices(road_map.num_states, k=1)
    shortest = dist[left, right]
    euclidean = np.linalg.norm(road_map.coords[left] - road_map.coords[right], axis=1)
    ratio = shortest / np.maximum(euclidean, EPS64)
    return {
        "map_name": road_map.name,
        "map_group": road_map.group,
        "num_states": road_map.num_states,
        "junction_count": int(np.sum(degrees >= 3)),
        "branch_excess": int(np.sum(np.maximum(degrees - 2, 0))),
        "dead_end_count": int(np.sum(degrees == 1)),
        "max_degree": int(np.max(degrees)),
        "cycle_rank": int(edge_count - road_map.num_states + 1),
        "graph_diameter": float(np.max(shortest)),
        "mean_shortest_path": float(np.mean(shortest)),
        "q95_shortest_path": float(np.quantile(shortest, 0.95)),
        "mean_geodesic_over_euclidean": float(np.mean(ratio)),
        "q95_geodesic_over_euclidean": float(np.quantile(ratio, 0.95)),
        "q99_geodesic_over_euclidean": float(np.quantile(ratio, 0.99)),
    }


def validate_map(road_map: RoadMap, expected_states: int) -> None:
    if road_map.num_states != expected_states:
        raise AssertionError(f"{road_map.name}: expected {expected_states} states, got {road_map.num_states}.")
    if road_map.coords.shape != (expected_states, 2):
        raise AssertionError(f"{road_map.name}: coords must have shape ({expected_states}, 2).")
    coord_tuples = [tuple(map(int, coord)) for coord in road_map.coords]
    if len(set(coord_tuples)) != len(coord_tuples):
        raise AssertionError(f"{road_map.name}: duplicate road cells.")
    seen_edges = set()
    for src, dst in road_map.edges:
        if not (0 <= src < expected_states and 0 <= dst < expected_states):
            raise AssertionError(f"{road_map.name}: invalid edge endpoint {src}, {dst}.")
        delta = np.abs(road_map.coords[src] - road_map.coords[dst])
        if int(delta.sum()) != 1:
            raise AssertionError(f"{road_map.name}: edge {src}-{dst} is not a cardinal lattice edge.")
        edge = tuple(sorted((src, dst)))
        if edge in seen_edges:
            raise AssertionError(f"{road_map.name}: duplicate edge {edge}.")
        seen_edges.add(edge)
    dist = shortest_path_matrix(road_map)
    if not np.isfinite(dist).all():
        raise AssertionError(f"{road_map.name}: graph is not connected.")


def validate_map_family(maps: Sequence[RoadMap], num_states: int) -> Dict[str, Dict[str, object]]:
    if len({road_map.name for road_map in maps}) != len(maps):
        raise AssertionError("Map names must be unique.")
    metrics: Dict[str, Dict[str, object]] = {}
    for road_map in maps:
        validate_map(road_map, num_states)
        metrics[road_map.name] = compute_complexity(road_map)

    winding = [road_map for road_map in maps if road_map.group == "winding_control"]
    branching = [road_map for road_map in maps if road_map.group == "branching_sweep"]
    for road_map in winding:
        degrees = graph_degrees(road_map)
        if int(metrics[road_map.name]["branch_excess"]) != 0:
            raise AssertionError(f"{road_map.name}: winding controls must have branch_excess = 0.")
        if int(np.max(degrees)) > 2:
            raise AssertionError(f"{road_map.name}: winding controls must have graph degree <= 2.")

    branch_excesses = [int(metrics[road_map.name]["branch_excess"]) for road_map in branching]
    if branch_excesses != sorted(branch_excesses):
        raise AssertionError(f"Branching sweep branch_excess must be nondecreasing, got {branch_excesses}.")
    for road_map in branching:
        if int(metrics[road_map.name]["cycle_rank"]) != 0:
            raise AssertionError(f"{road_map.name}: tree-like branching map has nonzero cycle_rank.")
    return metrics


def build_transition_table(road_map: RoadMap) -> np.ndarray:
    coord_to_state = {tuple(map(int, coord)): idx for idx, coord in enumerate(road_map.coords)}
    edge_set = set(road_map.edges)
    edge_set.update((dst, src) for src, dst in road_map.edges)
    table = np.empty((len(ACTION_NAMES), road_map.num_states), dtype=np.int64)
    for action_id, action_name in enumerate(ACTION_NAMES):
        dx, dy = ACTION_DELTAS[action_name]
        for state, coord in enumerate(road_map.coords):
            next_coord = (int(coord[0]) + dx, int(coord[1]) + dy)
            neighbor = coord_to_state.get(next_coord)
            if neighbor is not None and (state, neighbor) in edge_set:
                table[action_id, state] = neighbor
            else:
                table[action_id, state] = state
    if not np.all((0 <= table) & (table < road_map.num_states)):
        raise AssertionError(f"{road_map.name}: transition table contains invalid next-state indices.")
    return table


def _repeat_to_horizon(pattern: Sequence[int], horizon: int) -> List[int]:
    if not pattern:
        raise ValueError("Cannot repeat an empty macro-action pattern.")
    return [int(pattern[step % len(pattern)]) for step in range(horizon)]


def _inverse_action(action: int) -> int:
    inverse = {
        ACTION_TO_ID["up"]: ACTION_TO_ID["down"],
        ACTION_TO_ID["down"]: ACTION_TO_ID["up"],
        ACTION_TO_ID["left"]: ACTION_TO_ID["right"],
        ACTION_TO_ID["right"]: ACTION_TO_ID["left"],
    }
    return inverse[int(action)]


def _append_unique_block(blocks: List[List[int]], seen: set[Tuple[int, ...]], block: Sequence[int]) -> None:
    key = tuple(int(action) for action in block)
    if key not in seen:
        seen.add(key)
        blocks.append(list(key))


def generate_macro_actions(
    horizon: int,
    num_macro_actions: int,
    seed: int,
    suite: str = "standard",
) -> np.ndarray:
    deterministic: List[List[int]] = []
    for action in ACTION_NAMES:
        deterministic.append([ACTION_TO_ID[action]] * horizon)
    alternating_pairs = [
        ("up", "right"),
        ("right", "up"),
        ("up", "left"),
        ("left", "up"),
        ("down", "right"),
        ("right", "down"),
        ("down", "left"),
        ("left", "down"),
        ("up", "down"),
        ("left", "right"),
        ("right", "left"),
        ("down", "up"),
    ]
    for first, second in alternating_pairs:
        deterministic.append([ACTION_TO_ID[first if step % 2 == 0 else second] for step in range(horizon)])

    if suite == "rich":
        motifs = [
            ("up", "right", "down", "left"),
            ("right", "down", "left", "up"),
            ("up", "up", "right", "down", "right", "down", "left", "left"),
            ("left", "left", "up", "right", "up", "right", "down", "down"),
            ("up", "right", "right", "down", "left", "down", "left", "up"),
            ("right", "up", "left", "up", "left", "down", "right", "down"),
        ]
        for motif in motifs:
            deterministic.append(_repeat_to_horizon([ACTION_TO_ID[action] for action in motif], horizon))

        burst_lengths = [max(1, horizon // 4), max(1, horizon // 2), max(1, (3 * horizon) // 4)]
        for first in ACTION_NAMES:
            for second in ACTION_NAMES:
                if first == second:
                    continue
                for burst_len in burst_lengths[:2]:
                    block = [ACTION_TO_ID[first]] * burst_len + [ACTION_TO_ID[second]] * (horizon - burst_len)
                    deterministic.append(block)

        for first in ACTION_NAMES:
            for second in ACTION_NAMES:
                if first == second:
                    continue
                block = [ACTION_TO_ID[first], ACTION_TO_ID[first], ACTION_TO_ID[second]]
                deterministic.append(_repeat_to_horizon(block, horizon))
    elif suite != "standard":
        raise ValueError(f"Unknown macro suite: {suite}")

    if num_macro_actions < len(deterministic):
        raise ValueError(
            f"num_macro_actions must be at least {len(deterministic)} for macro_suite={suite}."
        )

    rng = np.random.default_rng(seed)
    blocks: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()
    for block in deterministic:
        _append_unique_block(blocks, seen, block)
    while len(blocks) < num_macro_actions:
        mode = "uniform" if suite == "standard" else str(rng.choice(["uniform", "sticky", "bursty", "biased", "palindrome"]))
        if mode == "uniform":
            block = rng.integers(0, len(ACTION_NAMES), size=horizon, dtype=np.int64).tolist()
        elif mode == "sticky":
            current = int(rng.integers(0, len(ACTION_NAMES)))
            block = []
            for _ in range(horizon):
                if rng.random() < 0.18:
                    current = int(rng.integers(0, len(ACTION_NAMES)))
                block.append(current)
        elif mode == "bursty":
            block = []
            while len(block) < horizon:
                action = int(rng.integers(0, len(ACTION_NAMES)))
                run = int(rng.integers(2, max(3, horizon // 2 + 1)))
                block.extend([action] * run)
            block = block[:horizon]
        elif mode == "biased":
            probs = rng.dirichlet(np.full(len(ACTION_NAMES), 0.22))
            block = rng.choice(len(ACTION_NAMES), size=horizon, p=probs).astype(np.int64).tolist()
        else:
            half = rng.integers(0, len(ACTION_NAMES), size=(horizon + 1) // 2, dtype=np.int64).tolist()
            mirror = [_inverse_action(action) for action in reversed(half)]
            block = (half + mirror)[:horizon]
        _append_unique_block(blocks, seen, block)
    return np.asarray(blocks, dtype=np.int64)


def compose_macro_transitions(T_one: np.ndarray, macro_actions: np.ndarray) -> np.ndarray:
    n = T_one.shape[1]
    composed = np.empty((macro_actions.shape[0], n), dtype=np.int64)
    for macro_idx, block in enumerate(macro_actions):
        states = np.arange(n, dtype=np.int64)
        for action in block:
            states = T_one[int(action), states]
        composed[macro_idx] = states
    if not np.all((0 <= composed) & (composed < n)):
        raise AssertionError("Macro transition table contains invalid next-state indices.")
    return composed


def save_macro_actions(path: Path, blocks: np.ndarray, seed: int, suite: str) -> None:
    payload = {
        "seed": int(seed),
        "suite": suite,
        "horizon": int(blocks.shape[1]),
        "num_macro_actions": int(blocks.shape[0]),
        "action_names": list(ACTION_NAMES),
        "blocks": [
            {
                "macro_action": int(idx),
                "action_ids": [int(action) for action in block],
                "actions": [ACTION_NAMES[int(action)] for action in block],
            }
            for idx, block in enumerate(blocks)
        ],
    }
    write_json(path, payload)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords_f = np.asarray(coords, dtype=np.float64)
    low = coords_f.min(axis=0)
    high = coords_f.max(axis=0)
    span = high - low
    out = np.zeros_like(coords_f, dtype=np.float64)
    for axis in range(2):
        if span[axis] > 0:
            out[:, axis] = 2.0 * (coords_f[:, axis] - low[axis]) / span[axis] - 1.0
    return out


def all_unordered_pairs(num_states: int) -> Tuple[np.ndarray, np.ndarray]:
    left, right = np.triu_indices(num_states, k=1)
    return left.astype(np.int64), right.astype(np.int64)


def set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def make_generator(seed: int, device: torch.device) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def tensors_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tensors_to_cpu(item) for item in value)
    return value


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def save_partial_checkpoint(
    path: Path,
    model: LearnedMazeModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    step: int,
    loss_history: List[Dict[str, object]],
    final_losses: Dict[str, float],
    road_map: RoadMap,
    complexity: Dict[str, object],
    macro_next_np: np.ndarray,
    args: argparse.Namespace,
    dtype: torch.dtype,
    latent_dim: int,
    seed: int,
    output_dir: Path,
) -> None:
    payload = {
        "partial": True,
        "completed": False,
        "step": int(step),
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": tensors_to_cpu(optimizer.state_dict()),
        "generator_state": generator.get_state().cpu(),
        "loss_history": loss_history,
        "final_losses": final_losses,
        "model_config": {
            "latent_dim": int(latent_dim),
            "num_macro_actions": int(macro_next_np.shape[0]),
            "hidden_dim": int(args.hidden_dim),
            "encoder_hidden_layers": int(args.encoder_hidden_layers),
            "predictor_hidden_layers": int(args.predictor_hidden_layers),
            "decoder_hidden_layers": int(args.decoder_hidden_layers),
            "action_embedding_dim": int(args.action_embedding_dim),
            "activation": args.activation,
            "dtype": str(dtype).replace("torch.", ""),
        },
        "run_config": serializable_config(args, output_dir),
        "map": {
            "name": road_map.name,
            "group": road_map.group,
            "coords": road_map.coords,
            "edges": np.asarray(road_map.edges, dtype=np.int64),
            "complexity": complexity,
        },
        "macro_next": macro_next_np,
        "m": int(latent_dim),
        "seed": int(seed),
    }
    atomic_torch_save(payload, path)


def quantile_dict(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"median_{prefix}": float(np.quantile(values, 0.5)),
        f"q95_{prefix}": float(np.quantile(values, 0.95)),
        f"q99_{prefix}": float(np.quantile(values, 0.99)),
        f"q999_{prefix}": float(np.quantile(values, 0.999)),
        f"max_{prefix}": float(np.max(values)),
    }


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def scheduled_learning_rate(args: argparse.Namespace, step: int) -> float:
    base_lr = float(args.learning_rate)
    begin = args.lr_decay_begin
    if begin is None:
        return base_lr
    begin = int(begin)
    end = int(args.lr_decay_end) if args.lr_decay_end is not None else int(args.steps)
    if step <= begin:
        return base_lr
    if end <= begin:
        return base_lr * float(args.lr_decay_final_factor)
    frac = min(1.0, max(0.0, (float(step) - float(begin)) / (float(end) - float(begin))))
    return base_lr * ((1.0 - frac) + frac * float(args.lr_decay_final_factor))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def nearest_state_accuracy(decoded_next: torch.Tensor, coords_t: torch.Tensor, true_next: torch.Tensor, chunk: int = 4096) -> float:
    flat_decoded = decoded_next.reshape(-1, decoded_next.shape[-1])
    flat_true = true_next.reshape(-1)
    total = int(flat_decoded.shape[0])
    correct = 0
    for start in range(0, total, int(chunk)):
        block = flat_decoded[start : start + int(chunk)]
        distances = torch.sum(torch.square(block[:, None, :] - coords_t[None, :, :]), dim=-1)
        pred = torch.argmin(distances, dim=-1)
        correct += int(torch.sum(pred == flat_true[start : start + int(chunk)]).detach().cpu().item())
    return float(correct / max(total, 1))


def exhaustive_transition_diagnostics(
    model: LearnedMazeModel,
    coords_t: torch.Tensor,
    macro_next_np: np.ndarray,
    args: argparse.Namespace,
    eps: float,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    macro_next_t = torch.as_tensor(macro_next_np, dtype=torch.long, device=coords_t.device)
    k, n = macro_next_np.shape
    with torch.no_grad():
        z = model.encoder(coords_t)
        recon = model.decoder(z)
        predictions = []
        for alpha in range(k):
            alpha_ids = torch.full((n,), int(alpha), dtype=torch.long, device=coords_t.device)
            predictions.append(model.predictor(z, alpha_ids))
        z_pred = torch.stack(predictions, dim=0)
        z_next = z[macro_next_t]
        next_coords = coords_t[macro_next_t]
        decoded_next = model.decoder(z_pred.reshape(k * n, z.shape[1])).reshape(k, n, 2)
        err = z_pred - z_next
        pred_l2 = torch.linalg.norm(err, dim=-1)
        rec_loss = torch.sum(torch.square(recon - coords_t), dim=-1).mean()
        dyn_loss = torch.sum(torch.square(err), dim=-1).mean()
        next_loss = torch.sum(torch.square(decoded_next - next_coords), dim=-1).mean()
        decoded_next_rmse = torch.sqrt(next_loss)
        exact_acc = nearest_state_accuracy(decoded_next, coords_t, macro_next_t)

    z_np = z.detach().cpu().numpy().astype(np.float64)
    z_pred_np = z_pred.detach().cpu().numpy().astype(np.float64)
    z_next_np = z_next.detach().cpu().numpy().astype(np.float64)
    pred_l2_np = pred_l2.detach().cpu().numpy().astype(np.float64)
    latent_dim = int(z_np.shape[1])
    pair_i, pair_j = all_unordered_pairs(n)
    denom = np.linalg.norm(z_np[pair_i] - z_np[pair_j], axis=-1)
    denom = np.maximum(denom, eps)
    denom_p5 = float(np.quantile(denom, 0.05))
    keep_trimmed = denom > denom_p5

    r_true_values: List[np.ndarray] = []
    norm_pair_error_values: List[np.ndarray] = []
    norm_pair_error_trimmed_values: List[np.ndarray] = []
    top_denominator_values: List[np.ndarray] = []
    gain_values: List[np.ndarray] = []
    for alpha in range(k):
        d_true = np.linalg.norm(z_next_np[alpha, pair_i] - z_next_np[alpha, pair_j], axis=-1)
        d_pred = np.linalg.norm(z_pred_np[alpha, pair_i] - z_pred_np[alpha, pair_j], axis=-1)
        r_true_values.append(d_true / denom)
        state_error = np.linalg.norm(z_pred_np[alpha] - z_next_np[alpha], axis=-1)
        pair_error = np.sqrt((np.square(state_error[pair_i]) + np.square(state_error[pair_j])) / 2.0)
        norm_pair = pair_error / denom
        norm_pair_error_values.append(norm_pair)
        norm_pair_error_trimmed_values.append(norm_pair[keep_trimmed])
        top_threshold = float(np.quantile(norm_pair, 0.99))
        top_denominator_values.append(denom[norm_pair >= top_threshold])
        violation = np.maximum(d_pred - float(args.L_budget) * denom, 0.0)
        gain_values.append(np.square(violation) / (np.square(denom) + eps))

    r_true_all = np.concatenate(r_true_values, axis=0)
    norm_pair_all = np.concatenate(norm_pair_error_values, axis=0)
    norm_pair_trimmed_all = np.concatenate(norm_pair_error_trimmed_values, axis=0)
    top_denominator_all = np.concatenate(top_denominator_values, axis=0)
    gain_loss = float(np.mean(np.concatenate(gain_values, axis=0))) if float(args.lambda_gain) > 0.0 else 0.0
    rec_value = float(rec_loss.detach().cpu().item())
    dyn_value = float(dyn_loss.detach().cpu().item())
    next_value = float(next_loss.detach().cpu().item())
    total = (
        float(args.lambda_rec) * rec_value
        + float(args.lambda_dyn) * dyn_value
        + float(args.lambda_next) * next_value
        + float(args.lambda_gain) * gain_loss
    )

    metrics = {
        "eval_total_objective": float(total),
        "eval_rec_loss": rec_value,
        "eval_dyn_loss": dyn_value,
        "eval_next_decoded_loss": next_value,
        "eval_gain_loss": gain_loss,
        "eval_mean_raw_latent_pred_l2": float(np.mean(pred_l2_np)),
        "eval_q99_raw_latent_pred_l2": float(np.quantile(pred_l2_np, 0.99)),
        "eval_mean_raw_latent_pred_l2_over_sqrt_m": float(np.mean(pred_l2_np) / math.sqrt(latent_dim)),
        "eval_q99_raw_latent_pred_l2_over_sqrt_m": float(np.quantile(pred_l2_np, 0.99) / math.sqrt(latent_dim)),
        "eval_per_coordinate_pred_mse": float(np.mean(np.square(z_pred_np - z_next_np))),
        "eval_decoded_next_rmse": float(decoded_next_rmse.detach().cpu().item()),
        "eval_nearest_state_next_accuracy": exact_acc,
        "eval_q99_r_true": float(np.quantile(r_true_all, 0.99)),
        "eval_q99_norm_pair_error_now": float(np.quantile(norm_pair_all, 0.99)),
        "eval_q99_norm_pair_error_exclude_bottom5_denom": float(np.quantile(norm_pair_trimmed_all, 0.99)),
    }
    for prefix, values in [
        ("eval_latent_norm", np.linalg.norm(z_np, axis=-1)),
        ("eval_pair_distance", denom),
        ("eval_top1_norm_error_pair_distance", top_denominator_all),
    ]:
        metrics[f"{prefix}_min"] = float(np.min(values))
        metrics[f"{prefix}_p1"] = float(np.quantile(values, 0.01))
        metrics[f"{prefix}_p5"] = float(np.quantile(values, 0.05))
        metrics[f"{prefix}_median"] = float(np.quantile(values, 0.50))
        metrics[f"{prefix}_p99"] = float(np.quantile(values, 0.99))
        metrics[f"{prefix}_max"] = float(np.max(values))
    if was_training:
        model.train()
    return metrics


def save_checkpoint_snapshot(
    path: Path,
    model: LearnedMazeModel,
    metrics: Dict[str, float],
    final_losses: Dict[str, float],
    road_map: RoadMap,
    complexity: Dict[str, object],
    macro_next_np: np.ndarray,
    args: argparse.Namespace,
    dtype: torch.dtype,
    latent_dim: int,
    output_dir: Path,
    checkpoint_kind: str,
) -> None:
    atomic_torch_save(
        {
            "checkpoint_kind": checkpoint_kind,
            "model_state_dict": cpu_state_dict(model),
            "model_config": {
                "latent_dim": int(latent_dim),
                "num_macro_actions": int(macro_next_np.shape[0]),
                "hidden_dim": int(args.hidden_dim),
                "encoder_hidden_layers": int(args.encoder_hidden_layers),
                "predictor_hidden_layers": int(args.predictor_hidden_layers),
                "decoder_hidden_layers": int(args.decoder_hidden_layers),
                "action_embedding_dim": int(args.action_embedding_dim),
                "activation": args.activation,
                "dtype": str(dtype).replace("torch.", ""),
            },
            "run_config": serializable_config(args, output_dir),
            "metrics": metrics,
            "final_losses": final_losses,
            "map": {
                "name": road_map.name,
                "group": road_map.group,
                "coords": road_map.coords,
                "edges": np.asarray(road_map.edges, dtype=np.int64),
                "complexity": complexity,
            },
            "macro_next": macro_next_np,
        },
        path,
    )


def assert_encoder_uses_only_xy(model: LearnedMazeModel) -> None:
    first_linear: Optional[nn.Linear] = None
    for module in model.encoder.modules():
        if isinstance(module, nn.Embedding):
            raise AssertionError("The encoder must be a learned coordinate MLP, not a state lookup table.")
        if isinstance(module, nn.Linear) and first_linear is None:
            first_linear = module
    if first_linear is None or first_linear.in_features != 2:
        raise AssertionError("The encoder input must be exactly normalized (x, y).")


def evaluate_model(
    model: LearnedMazeModel,
    coords_t: torch.Tensor,
    macro_next_np: np.ndarray,
    L_budget: float,
    eps: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    macro_next_t = torch.as_tensor(macro_next_np, dtype=torch.long, device=coords_t.device)
    k, n = macro_next_np.shape
    with torch.no_grad():
        z = model.encoder(coords_t)
        recon = model.decoder(z)
        recon_mse = torch.sum(torch.square(recon - coords_t), dim=-1).mean()

        predictions = []
        for alpha in range(k):
            alpha_ids = torch.full((n,), int(alpha), dtype=torch.long, device=coords_t.device)
            predictions.append(model.predictor(z, alpha_ids))
        z_pred = torch.stack(predictions, dim=0)
        z_next = z[macro_next_t]
        next_coords = coords_t[macro_next_t]
        decoded_next = model.decoder(z_pred.reshape(k * n, z.shape[1])).reshape(k, n, 2)

        latent_dyn_mse = torch.sum(torch.square(z_pred - z_next), dim=-1).mean()
        decoded_next_mse = torch.sum(torch.square(decoded_next - next_coords), dim=-1).mean()

    z_np = z.detach().cpu().numpy().astype(np.float64)
    z_pred_np = z_pred.detach().cpu().numpy().astype(np.float64)
    z_next_np = z_next.detach().cpu().numpy().astype(np.float64)
    pair_i, pair_j = all_unordered_pairs(n)
    denom = np.linalg.norm(z_np[pair_i] - z_np[pair_j], axis=-1)
    denom = np.maximum(denom, eps)

    r_true_values: List[np.ndarray] = []
    r_model_values: List[np.ndarray] = []
    ratio_deficit_values: List[np.ndarray] = []
    norm_pair_error_values: List[np.ndarray] = []
    for alpha in range(k):
        d_true = np.linalg.norm(z_next_np[alpha, pair_i] - z_next_np[alpha, pair_j], axis=-1)
        d_model = np.linalg.norm(z_pred_np[alpha, pair_i] - z_pred_np[alpha, pair_j], axis=-1)
        r_true = d_true / denom
        r_model = d_model / denom
        state_error = np.linalg.norm(z_pred_np[alpha] - z_next_np[alpha], axis=-1)
        pair_error = np.sqrt((np.square(state_error[pair_i]) + np.square(state_error[pair_j])) / 2.0)
        r_true_values.append(r_true)
        r_model_values.append(r_model)
        ratio_deficit_values.append(np.maximum(r_true - r_model, 0.0))
        norm_pair_error_values.append(pair_error / denom)

    r_true_all = np.concatenate(r_true_values, axis=0)
    r_model_all = np.concatenate(r_model_values, axis=0)
    ratio_deficit_all = np.concatenate(ratio_deficit_values, axis=0)
    norm_pair_error_all = np.concatenate(norm_pair_error_values, axis=0)

    metrics: Dict[str, float] = {
        "recon_mse": float(recon_mse.detach().cpu().item()),
        "recon_rmse": float(math.sqrt(max(float(recon_mse.detach().cpu().item()), 0.0))),
        "decoded_next_mse": float(decoded_next_mse.detach().cpu().item()),
        "latent_dyn_mse": float(latent_dyn_mse.detach().cpu().item()),
        "violation_rate_model": float(np.mean(r_model_all > float(L_budget))),
        "violation_rate_true": float(np.mean(r_true_all > float(L_budget))),
        "q95_ratio_deficit": float(np.quantile(ratio_deficit_all, 0.95)),
        "q99_ratio_deficit": float(np.quantile(ratio_deficit_all, 0.99)),
        "q999_ratio_deficit": float(np.quantile(ratio_deficit_all, 0.999)),
        "q95_norm_pair_error_now": float(np.quantile(norm_pair_error_all, 0.95)),
        "q99_norm_pair_error_now": float(np.quantile(norm_pair_error_all, 0.99)),
        "q999_norm_pair_error_now": float(np.quantile(norm_pair_error_all, 0.999)),
    }
    metrics.update(quantile_dict(r_true_all, "r_true"))
    metrics.update(quantile_dict(r_model_all, "r_model"))
    return metrics, z_np


def train_one_model(
    road_map: RoadMap,
    complexity: Dict[str, object],
    macro_next_np: np.ndarray,
    args: argparse.Namespace,
    dtype: torch.dtype,
    output_dir: Path,
    seed: int,
    latent_dim: int,
) -> Dict[str, object]:
    run_seed = _stable_seed(args.seed_base, road_map.name, latent_dim, seed, "learned")
    set_global_seeds(run_seed)
    eps = EPS32 if dtype == torch.float32 else EPS64
    device = resolve_device(args.device)
    coords_np = normalize_coords(road_map.coords)
    coords_t = torch.as_tensor(coords_np, dtype=dtype, device=device)
    macro_next_t = torch.as_tensor(macro_next_np, dtype=torch.long, device=device)
    k, n = macro_next_np.shape
    partial_dir = output_dir / "checkpoints_partial"
    partial_path = partial_dir / f"{road_map.name}_m{latent_dim}_seed{seed}.partial.pt"
    diagnostics_dir = output_dir / "diagnostics"
    diagnostic_path = diagnostics_dir / f"{road_map.name}_m{latent_dim}_seed{seed}.csv"
    best_total_dir = output_dir / "checkpoints_best_total"
    best_dyn_dir = output_dir / "checkpoints_best_dyn"

    model = LearnedMazeModel(
        latent_dim=latent_dim,
        num_macro_actions=k,
        hidden_dim=args.hidden_dim,
        encoder_hidden_layers=args.encoder_hidden_layers,
        predictor_hidden_layers=args.predictor_hidden_layers,
        decoder_hidden_layers=args.decoder_hidden_layers,
        action_embedding_dim=args.action_embedding_dim,
        activation=args.activation,
    ).to(device=device, dtype=dtype)
    assert_encoder_uses_only_xy(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    generator = make_generator(run_seed, device)
    log_every = max(1, int(args.log_every))
    checkpoint_every = int(args.checkpoint_every)
    loss_history: List[Dict[str, object]] = []
    diagnostic_history: List[Dict[str, object]] = []
    if args.resume and diagnostic_path.exists():
        diagnostic_history = pd.read_csv(diagnostic_path).to_dict("records")
    best_total_objective = float("inf")
    best_dyn_objective = float("inf")
    clipped_steps = 0
    seen_optimizer_steps = 0
    last_grad_norm_preclip = float("nan")
    final_losses = {
        "final_train_loss": float("nan"),
        "final_rec_loss": float("nan"),
        "final_dyn_loss": float("nan"),
        "final_next_decoded_loss": float("nan"),
        "final_gain_loss": float("nan"),
    }
    start_step = 1
    if args.resume and partial_path.exists():
        partial = torch_load_checkpoint(partial_path)
        partial_step = int(partial.get("step", 0))
        if partial_step > 0:
            model.load_state_dict(partial["model_state_dict"])
            optimizer.load_state_dict(partial["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
            if "generator_state" in partial:
                generator.set_state(partial["generator_state"].cpu())
            loss_history = list(partial.get("loss_history", []))
            final_losses = dict(partial.get("final_losses", final_losses))
            start_step = min(partial_step + 1, int(args.steps) + 1)
            print(
                f"  Resuming partial checkpoint {partial_path} from step {partial_step}/{int(args.steps)}.",
                flush=True,
            )

    model.train()
    for step in range(start_step, int(args.steps) + 1):
        current_lr = scheduled_learning_rate(args, step)
        set_optimizer_lr(optimizer, current_lr)
        state_idx = torch.randint(n, (int(args.batch_size),), generator=generator, dtype=torch.long, device=device)
        alpha_idx = torch.randint(k, (int(args.batch_size),), generator=generator, dtype=torch.long, device=device)
        next_idx = macro_next_t[alpha_idx, state_idx]
        s = coords_t[state_idx]
        s_next = coords_t[next_idx]

        z = model.encoder(s)
        z_next = model.encoder(s_next)
        z_target = z_next.detach() if args.detach_target else z_next
        z_pred = model.predictor(z, alpha_idx)
        s_recon = model.decoder(z)
        s_next_decoded = model.decoder(z_pred)

        rec_loss = torch.sum(torch.square(s_recon - s), dim=-1).mean()
        dyn_loss = torch.sum(torch.square(z_pred - z_target), dim=-1).mean()
        next_loss = torch.sum(torch.square(s_next_decoded - s_next), dim=-1).mean()

        if float(args.lambda_gain) > 0.0:
            pair_alpha = torch.randint(k, (int(args.batch_size),), generator=generator, dtype=torch.long, device=device)
            pair_i = torch.randint(n, (int(args.batch_size),), generator=generator, dtype=torch.long, device=device)
            pair_j = torch.randint(n - 1, (int(args.batch_size),), generator=generator, dtype=torch.long, device=device)
            pair_j = pair_j + (pair_j >= pair_i).to(torch.long)
            z_i = model.encoder(coords_t[pair_i])
            z_j = model.encoder(coords_t[pair_j])
            pred_i = model.predictor(z_i, pair_alpha)
            pred_j = model.predictor(z_j, pair_alpha)
            d_now = torch.linalg.norm(z_i - z_j, dim=-1)
            d_pred = torch.linalg.norm(pred_i - pred_j, dim=-1)
            violation = torch.relu(d_pred - float(args.L_budget) * d_now)
            gain_loss = torch.mean(torch.square(violation) / (torch.square(d_now) + eps))
        else:
            gain_loss = torch.zeros((), dtype=dtype, device=device)

        total_loss = (
            float(args.lambda_rec) * rec_loss
            + float(args.lambda_dyn) * dyn_loss
            + float(args.lambda_next) * next_loss
            + float(args.lambda_gain) * gain_loss
        )

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"{road_map.name} m={latent_dim} seed={seed}: non-finite training loss.")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.gradient_clip_norm))
        last_grad_norm_preclip = float(grad_norm.detach().cpu().item())
        seen_optimizer_steps += 1
        if last_grad_norm_preclip > float(args.gradient_clip_norm):
            clipped_steps += 1
        optimizer.step()

        should_log_step = (
            step == 1
            or step % log_every == 0
            or step == int(args.steps)
            or (checkpoint_every > 0 and step % checkpoint_every == 0)
        )
        if should_log_step:
            row = {
                "step": int(step),
                "total_loss": float(total_loss.detach().cpu().item()),
                "rec_loss": float(rec_loss.detach().cpu().item()),
                "dyn_loss": float(dyn_loss.detach().cpu().item()),
                "next_decoded_loss": float(next_loss.detach().cpu().item()),
                "gain_loss": float(gain_loss.detach().cpu().item()),
                "learning_rate": float(current_lr),
                "grad_norm_preclip": float(last_grad_norm_preclip),
                "grad_was_clipped": bool(last_grad_norm_preclip > float(args.gradient_clip_norm)),
                "clipped_step_fraction": float(clipped_steps / max(seen_optimizer_steps, 1)),
            }
            loss_history.append(row)
            final_losses = {
                "final_train_loss": row["total_loss"],
                "final_rec_loss": row["rec_loss"],
                "final_dyn_loss": row["dyn_loss"],
                "final_next_decoded_loss": row["next_decoded_loss"],
                "final_gain_loss": row["gain_loss"],
            }
        if checkpoint_every > 0 and (step % checkpoint_every == 0 or step == int(args.steps)):
            save_partial_checkpoint(
                path=partial_path,
                model=model,
                optimizer=optimizer,
                generator=generator,
                step=step,
                loss_history=loss_history,
                final_losses=final_losses,
                road_map=road_map,
                complexity=complexity,
                macro_next_np=macro_next_np,
                args=args,
                dtype=dtype,
                latent_dim=latent_dim,
                seed=seed,
                output_dir=output_dir,
            )
        if int(args.eval_every) > 0 and (step == 1 or step % int(args.eval_every) == 0 or step == int(args.steps)):
            diag_row: Dict[str, object] = {
                "step": int(step),
                "learning_rate": float(current_lr),
                "parameter_count": int(parameter_count(model)),
                "grad_norm_preclip": float(last_grad_norm_preclip),
                "grad_was_clipped": bool(last_grad_norm_preclip > float(args.gradient_clip_norm)),
                "clipped_step_fraction": float(clipped_steps / max(seen_optimizer_steps, 1)),
            }
            diag_row.update(exhaustive_transition_diagnostics(model, coords_t, macro_next_np, args, eps))
            diagnostic_history.append(diag_row)
            atomic_to_csv(pd.DataFrame(diagnostic_history), diagnostic_path)
            if bool(args.save_exhaustive_best):
                total_objective = float(diag_row["eval_total_objective"])
                dyn_objective = float(diag_row["eval_dyn_loss"])
                if total_objective < best_total_objective:
                    best_total_objective = total_objective
                    save_checkpoint_snapshot(
                        path=best_total_dir / f"{road_map.name}_m{latent_dim}_seed{seed}.pt",
                        model=model,
                        metrics={key: value for key, value in diag_row.items() if key.startswith("eval_")},
                        final_losses=final_losses,
                        road_map=road_map,
                        complexity=complexity,
                        macro_next_np=macro_next_np,
                        args=args,
                        dtype=dtype,
                        latent_dim=latent_dim,
                        output_dir=output_dir,
                        checkpoint_kind="best_exhaustive_total",
                    )
                if dyn_objective < best_dyn_objective:
                    best_dyn_objective = dyn_objective
                    save_checkpoint_snapshot(
                        path=best_dyn_dir / f"{road_map.name}_m{latent_dim}_seed{seed}.pt",
                        model=model,
                        metrics={key: value for key, value in diag_row.items() if key.startswith("eval_")},
                        final_losses=final_losses,
                        road_map=road_map,
                        complexity=complexity,
                        macro_next_np=macro_next_np,
                        args=args,
                        dtype=dtype,
                        latent_dim=latent_dim,
                        output_dir=output_dir,
                        checkpoint_kind="best_exhaustive_dyn",
                    )

    metrics, z_np = evaluate_model(model, coords_t, macro_next_np, args.L_budget, eps)

    checkpoints_dir = output_dir / "checkpoints"
    embeddings_dir = output_dir / "embeddings"
    losses_dir = output_dir / "losses"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    losses_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{road_map.name}_m{latent_dim}_seed{seed}.pt"
    embedding_path = embeddings_dir / f"{road_map.name}_m{latent_dim}_seed{seed}.npz"
    loss_curve_path = losses_dir / f"{road_map.name}_m{latent_dim}_seed{seed}.csv"

    atomic_to_csv(pd.DataFrame(loss_history), loss_curve_path)
    np.savez_compressed(
        embedding_path,
        z=z_np,
        coords_normalized=coords_np,
        coords_lattice=road_map.coords,
        state_order=np.asarray(state_order(road_map), dtype=np.int64),
        macro_next=macro_next_np,
        map_name=np.asarray(road_map.name),
        map_group=np.asarray(road_map.group),
        m=np.asarray(int(latent_dim), dtype=np.int64),
        seed=np.asarray(int(seed), dtype=np.int64),
    )
    atomic_torch_save(
        {
            "checkpoint_kind": "final",
            "model_state_dict": cpu_state_dict(model),
            "model_config": {
                "latent_dim": int(latent_dim),
                "num_macro_actions": int(k),
                "hidden_dim": int(args.hidden_dim),
                "encoder_hidden_layers": int(args.encoder_hidden_layers),
                "predictor_hidden_layers": int(args.predictor_hidden_layers),
                "decoder_hidden_layers": int(args.decoder_hidden_layers),
                "action_embedding_dim": int(args.action_embedding_dim),
                "activation": args.activation,
                "dtype": str(dtype).replace("torch.", ""),
            },
            "run_config": serializable_config(args, output_dir),
            "metrics": metrics,
            "final_losses": final_losses,
            "map": {
                "name": road_map.name,
                "group": road_map.group,
                "coords": road_map.coords,
                "edges": np.asarray(road_map.edges, dtype=np.int64),
                "complexity": complexity,
            },
            "macro_next": macro_next_np,
            "partial_checkpoint_path": str(partial_path),
        },
        checkpoint_path,
    )

    row: Dict[str, object] = {
        "map_name": road_map.name,
        "map_group": road_map.group,
        "num_states": int(road_map.num_states),
        "m": int(latent_dim),
        "seed": int(seed),
        "horizon": int(args.horizon),
        "num_macro_actions": int(k),
        "L_budget": float(args.L_budget),
        "lambda_gain": float(args.lambda_gain),
        "detach_target": bool(args.detach_target),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "macro_suite": args.macro_suite,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "parameter_count": int(parameter_count(model)),
        "clipped_step_fraction": float(clipped_steps / max(seen_optimizer_steps, 1)),
        "checkpoint_path": str(checkpoint_path),
        "partial_checkpoint_path": str(partial_path),
        "embedding_path": str(embedding_path),
        "loss_curve_path": str(loss_curve_path),
    }
    row.update(complexity)
    row.update(metrics)
    row.update(final_losses)
    return row


def state_order(road_map: RoadMap) -> List[int]:
    degrees = graph_degrees(road_map)
    neighbors = adjacency_list(road_map)
    if int(np.max(degrees)) <= 2:
        endpoints = np.where(degrees <= 1)[0]
        if endpoints.size:
            endpoint_coords = road_map.coords[endpoints]
            start = int(endpoints[np.lexsort((endpoint_coords[:, 1], endpoint_coords[:, 0]))][0])
        else:
            start = 0
        order = [start]
        prev = -1
        current = start
        while len(order) < road_map.num_states:
            candidates = [nxt for nxt in neighbors[current] if nxt != prev]
            if not candidates:
                break
            nxt = candidates[0]
            order.append(nxt)
            prev, current = current, nxt
        if len(order) == road_map.num_states:
            return order

    start = int(np.lexsort((road_map.coords[:, 1], road_map.coords[:, 0]))[0])
    order = []
    visited = {start}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in neighbors[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    if len(order) != road_map.num_states:
        raise AssertionError(f"{road_map.name}: BFS state order did not visit all states.")
    return order


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, torch.dtype):
            return str(value)
        return str(value)

    with path.open("w") as file:
        json.dump(payload, file, indent=2, default=default)


def ordered_columns(df: pd.DataFrame, preferred: Sequence[str]) -> List[str]:
    preferred_existing = [col for col in preferred if col in df.columns]
    rest = [col for col in df.columns if col not in preferred_existing]
    return preferred_existing + rest


def atomic_to_csv(df: pd.DataFrame, path: Path, columns: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if columns is None:
        df.to_csv(tmp_path, index=False)
    else:
        df.to_csv(tmp_path, index=False, columns=list(columns))
    tmp_path.replace(path)


def dataframe_with_order(rows: List[Dict[str, object]], map_order: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {name: idx for idx, name in enumerate(map_order)}
    df["_map_order"] = df["map_name"].map(order)
    sort_cols = [col for col in ["_map_order", "m", "seed"] if col in df.columns]
    df = df.sort_values(sort_cols).drop(columns=["_map_order"])
    return df.reset_index(drop=True)


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "map_name",
        "map_group",
        "num_states",
        "branch_excess",
        "junction_count",
        "dead_end_count",
        "cycle_rank",
        "max_degree",
        "m",
        "L_budget",
        "lambda_gain",
    ]
    rows: List[Dict[str, object]] = []
    for group_values, group in results_df.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, group_values))
        for metric in KEY_METRICS:
            if metric in group.columns:
                row[f"mean_{metric}"] = float(group[metric].mean())
                row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        best_idx = group["decoded_next_mse"].astype(float).idxmin()
        sorted_by_q99 = group.sort_values(["q99_r_true", "seed"]).reset_index(drop=True)
        median_idx = len(sorted_by_q99) // 2
        row["best_seed_by_decoded_next_mse"] = int(results_df.loc[best_idx, "seed"])
        row["median_seed_by_q99_r_true"] = int(sorted_by_q99.loc[median_idx, "seed"])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_required_dimensions(summary_df: pd.DataFrame, budgets: Sequence[float], m_values: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    branch_df = summary_df[summary_df["map_group"] != "winding_control"].copy()
    if branch_df.empty:
        return pd.DataFrame(rows)
    max_m = max(int(value) for value in m_values)
    sentinel = float(max_m) * 1.15
    for budget in budgets:
        for map_name, group in branch_df.groupby("map_name", sort=False):
            group = group.sort_values("m")
            satisfying = group[group["mean_q99_r_true"] <= float(budget)]
            first = group.iloc[0]
            if satisfying.empty:
                required_dim: object = ""
                label = f">{max_m}"
                plot_dim = sentinel
                reached = False
            else:
                required_dim = int(satisfying.iloc[0]["m"])
                label = str(required_dim)
                plot_dim = float(required_dim)
                reached = True
            rows.append(
                {
                    "gain_budget": float(budget),
                    "map_name": map_name,
                    "branch_excess": int(first["branch_excess"]),
                    "junction_count": int(first["junction_count"]),
                    "required_dimension": required_dim,
                    "required_dimension_label": label,
                    "required_dimension_for_plot": plot_dim,
                    "reached_budget": bool(reached),
                }
            )
    return pd.DataFrame(rows)


def run_key(map_name: object, latent_dim: object, seed: object) -> Tuple[str, int, int]:
    return (str(map_name), int(latent_dim), int(seed))


def completed_run_keys(rows: Sequence[Dict[str, object]]) -> set[Tuple[str, int, int]]:
    keys = set()
    for row in rows:
        if "map_name" in row and "m" in row and "seed" in row:
            keys.add(run_key(row["map_name"], row["m"], row["seed"]))
    return keys


def parse_checkpoint_key(path: Path) -> Optional[Tuple[str, int, int]]:
    stem = path.stem
    if "_m" not in stem or "_seed" not in stem:
        return None
    map_name, rest = stem.rsplit("_m", 1)
    dim_text, seed_text = rest.split("_seed", 1)
    try:
        return run_key(map_name, int(dim_text), int(seed_text))
    except ValueError:
        return None


def torch_load_checkpoint(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def row_from_checkpoint(path: Path, fallback_args: argparse.Namespace) -> Optional[Dict[str, object]]:
    key = parse_checkpoint_key(path)
    if key is None:
        return None
    map_name, latent_dim, seed = key
    payload = torch_load_checkpoint(path)
    map_payload = payload.get("map", {})
    complexity = dict(map_payload.get("complexity", {}))
    metrics = dict(payload.get("metrics", {}))
    final_losses = dict(payload.get("final_losses", {}))
    run_config = dict(payload.get("run_config", {}))
    model_config = dict(payload.get("model_config", {}))
    coords = map_payload.get("coords", np.empty((0, 2), dtype=np.int64))
    embedding_path = path.parent.parent / "embeddings" / f"{map_name}_m{latent_dim}_seed{seed}.npz"
    loss_curve_path = path.parent.parent / "losses" / f"{map_name}_m{latent_dim}_seed{seed}.csv"
    partial_checkpoint_path = path.parent.parent / "checkpoints_partial" / f"{map_name}_m{latent_dim}_seed{seed}.partial.pt"
    row: Dict[str, object] = {
        "map_name": map_name,
        "map_group": map_payload.get("group", complexity.get("map_group", "")),
        "num_states": int(complexity.get("num_states", len(coords))),
        "m": int(latent_dim),
        "seed": int(seed),
        "horizon": int(run_config.get("horizon", fallback_args.horizon)),
        "num_macro_actions": int(run_config.get("num_macro_actions", fallback_args.num_macro_actions)),
        "L_budget": float(run_config.get("L_budget", fallback_args.L_budget)),
        "lambda_gain": float(run_config.get("lambda_gain", fallback_args.lambda_gain)),
        "detach_target": bool(run_config.get("detach_target", fallback_args.detach_target)),
        "dtype": model_config.get("dtype", "float64"),
        "device": run_config.get("device", "unknown"),
        "macro_suite": run_config.get("macro_suite", "standard"),
        "steps": int(run_config.get("steps", fallback_args.steps)),
        "batch_size": int(run_config.get("batch_size", fallback_args.batch_size)),
        "learning_rate": float(run_config.get("learning_rate", fallback_args.learning_rate)),
        "weight_decay": float(run_config.get("weight_decay", fallback_args.weight_decay)),
        "checkpoint_path": str(path),
        "partial_checkpoint_path": str(partial_checkpoint_path),
        "embedding_path": str(embedding_path),
        "loss_curve_path": str(loss_curve_path),
    }
    row.update(complexity)
    row.update(metrics)
    row.update(final_losses)
    return row


def load_checkpoint_rows(
    output_dir: Path,
    wanted_keys: set[Tuple[str, int, int]],
    existing_keys: set[Tuple[str, int, int]],
    fallback_args: argparse.Namespace,
) -> List[Dict[str, object]]:
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    rows: List[Dict[str, object]] = []
    for path in sorted(checkpoint_dir.glob("*_m*_seed*.pt")):
        key = parse_checkpoint_key(path)
        if key is None or key not in wanted_keys or key in existing_keys:
            continue
        row = row_from_checkpoint(path, fallback_args)
        if row is not None:
            rows.append(row)
            existing_keys.add(key)
    return rows


def write_result_tables(
    rows: List[Dict[str, object]],
    map_order: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_df = dataframe_with_order(rows, map_order)
    results_path = output_dir / "results.csv"
    atomic_to_csv(results_df, results_path, columns=ordered_columns(results_df, RESULT_COLUMNS))
    summary_df = build_summary(results_df)
    summary_df = dataframe_with_order(summary_df.to_dict("records"), map_order)
    summary_path = output_dir / "summary.csv"
    atomic_to_csv(summary_df, summary_path)
    required_df = compute_required_dimensions(summary_df, budgets=[1.05, 1.1, 1.2, 1.5], m_values=args.m_values)
    required_path = output_dir / "required_dimension.csv"
    atomic_to_csv(required_df, required_path)
    return results_df, summary_df, required_df


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
        }
    )
    return plt


def _save_figure(fig: object, base_path: Path) -> List[Path]:
    paths = [base_path.with_suffix(".png"), base_path.with_suffix(".pdf")]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            fig.savefig(path, dpi=260)
        else:
            fig.savefig(path)
    return paths


def plot_map_gallery(maps: Sequence[RoadMap], complexity: Dict[str, Dict[str, object]], figures_dir: Path) -> List[Path]:
    plt = _matplotlib()
    group_order = []
    for road_map in maps:
        if road_map.group not in group_order:
            group_order.append(road_map.group)
    groups = [(group, [road_map for road_map in maps if road_map.group == group]) for group in group_order]
    columns = max(1, max(len(group_maps) for _group, group_maps in groups))
    fig, axes = plt.subplots(
        len(groups),
        columns,
        figsize=(2.35 * columns, 2.35 * len(groups)),
        facecolor="white",
        squeeze=False,
    )
    group_titles = {
        "winding_control": "Winding controls",
        "branching_sweep": "Tree branching sweep",
        "weird_sweep": "Weird cyclic sweep",
    }
    for row_idx, (group_name, group_maps) in enumerate(groups):
        for col_idx in range(columns):
            ax = axes[row_idx, col_idx]
            ax.set_axis_off()
            if col_idx >= len(group_maps):
                continue
            road_map = group_maps[col_idx]
            coords = road_map.coords
            for src, dst in road_map.edges:
                ax.plot(
                    [coords[src, 0], coords[dst, 0]],
                    [coords[src, 1], coords[dst, 1]],
                    color="0.35",
                    lw=1.1,
                    solid_capstyle="round",
                )
            ax.scatter(coords[:, 0], coords[:, 1], s=5, color="#2F6F9F", zorder=3)
            metrics = complexity[road_map.name]
            ax.set_title(
                f"{road_map.name}\n"
                f"b={metrics['branch_excess']} j={metrics['junction_count']} "
                f"dend={metrics['dead_end_count']} cyc={metrics['cycle_rank']}",
                fontsize=8,
            )

    all_coords = np.vstack([road_map.coords for road_map in maps])
    min_xy = all_coords.min(axis=0).astype(float)
    max_xy = all_coords.max(axis=0).astype(float)
    center = (min_xy + max_xy) / 2.0
    span = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1.0))
    pad = max(1.0, 0.04 * span)
    for ax in axes.ravel():
        ax.set_xlim(center[0] - span / 2.0 - pad, center[0] + span / 2.0 + pad)
        ax.set_ylim(center[1] - span / 2.0 - pad, center[1] + span / 2.0 + pad)
        ax.set_aspect("equal")
    for row_idx, (group_name, _group_maps) in enumerate(groups):
        axes[row_idx, 0].text(
            0.0,
            1.12,
            group_titles.get(group_name, group_name),
            transform=axes[row_idx, 0].transAxes,
            fontsize=9,
            weight="bold",
        )
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir / "learned_branching_maze_gallery")
    plt.close(fig)
    return paths


def plot_state_reconstruction(summary_df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor="white")
    for map_name, group in summary_df.groupby("map_name", sort=False):
        group = group.sort_values("m")
        label = f"{map_name} (b={int(group.iloc[0]['branch_excess'])})"
        ax.plot(group["m"], group["mean_recon_rmse"], marker="o", lw=1.4, ms=3.8, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("latent dimension m")
    ax.set_ylabel("reconstruction RMSE")
    ax.set_title("State reconstruction sanity check")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6.8, ncol=2)
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir / "learned_branching_state_reconstruction")
    plt.close(fig)
    return paths


def plot_gain_frontier(summary_df: pd.DataFrame, figures_dir: Path, L_budget: float) -> List[Path]:
    plt = _matplotlib()
    branch_df = summary_df[summary_df["map_group"] != "winding_control"].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.0), facecolor="white")
    for map_name, group in branch_df.groupby("map_name", sort=False):
        group = group.sort_values("m")
        label = f"{map_name} (b={int(group.iloc[0]['branch_excess'])})"
        line = ax.plot(group["m"], group["mean_q99_r_true"], marker="o", lw=1.6, ms=4, label=label)[0]
        ax.plot(group["m"], group["mean_q99_r_model"], ls="--", lw=1.1, color=line.get_color(), alpha=0.72)
    ax.axhline(float(L_budget), color="0.25", lw=1.0, ls=":", label=f"L_budget={float(L_budget):g}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("latent dimension m")
    ax.set_ylabel("q99 gain")
    ax.set_title("Branching dimension-gain diagnostic")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir / "learned_branching_gain_frontier")
    plt.close(fig)
    return paths


def plot_deficit_error(summary_df: pd.DataFrame, figures_dir: Path) -> List[Path]:
    plt = _matplotlib()
    branch_df = summary_df[summary_df["map_group"] != "winding_control"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), facecolor="white")
    for map_name, group in branch_df.groupby("map_name", sort=False):
        group = group.sort_values("m")
        label = f"{map_name} (b={int(group.iloc[0]['branch_excess'])})"
        axes[0].plot(group["m"], group["mean_q99_ratio_deficit"], marker="o", lw=1.4, ms=3.8, label=label)
        axes[1].plot(group["m"], group["mean_q99_norm_pair_error_now"], marker="o", lw=1.4, ms=3.8, label=label)
    axes[0].set_title("A. Ratio deficit")
    axes[0].set_ylabel("q99 ratio deficit")
    axes[1].set_title("B. Normalized pair error")
    axes[1].set_ylabel("q99 normalized pair error")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("latent dimension m")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=6.8)
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir / "learned_branching_deficit_error")
    plt.close(fig)
    return paths


def plot_required_dimension(
    required_df: pd.DataFrame,
    m_values: Sequence[int],
    figures_dir: Path,
) -> List[Path]:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(5.8, 3.7), facecolor="white")
    if not required_df.empty:
        for budget, group in required_df.groupby("gain_budget", sort=True):
            group = group.sort_values(["branch_excess", "map_name"])
            line = ax.plot(
                group["branch_excess"],
                group["required_dimension_for_plot"],
                marker="o",
                lw=1.5,
                ms=4,
                label=f"L0={float(budget):g}",
            )[0]
            missed = group[~group["reached_budget"].astype(bool)]
            if not missed.empty:
                ax.scatter(
                    missed["branch_excess"],
                    missed["required_dimension_for_plot"],
                    marker="^",
                    s=34,
                    color=line.get_color(),
                    zorder=5,
                )
    max_m = max(int(value) for value in m_values)
    sentinel = float(max_m) * 1.15
    ticks = sorted({float(value) for value in m_values} | {sentinel})
    labels = [f">{max_m}" if abs(tick - sentinel) < 1e-9 else str(int(tick)) for tick in ticks]
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("branch_excess")
    ax.set_ylabel("empirical required dimension")
    ax.set_title("Branching complexity versus required dimension")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir / "learned_branching_required_dimension")
    plt.close(fig)
    return paths


def plot_winding_controls(summary_df: pd.DataFrame, figures_dir: Path, L_budget: float) -> List[Path]:
    plt = _matplotlib()
    winding_df = summary_df[summary_df["map_group"] == "winding_control"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), facecolor="white")
    for map_name, group in winding_df.groupby("map_name", sort=False):
        group = group.sort_values("m")
        axes[0].plot(group["m"], group["mean_q99_r_true"], marker="o", lw=1.5, ms=4, label=map_name)
        axes[1].plot(group["m"], group["mean_q99_ratio_deficit"], marker="o", lw=1.5, ms=4, label=map_name)
    axes[0].axhline(float(L_budget), color="0.25", lw=1.0, ls=":", label=f"L_budget={float(L_budget):g}")
    axes[0].set_title("Winding q99 true gain")
    axes[0].set_ylabel("q99_r_true")
    axes[1].set_title("Winding q99 ratio deficit")
    axes[1].set_ylabel("q99_ratio_deficit")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("latent dimension m")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    paths = _save_figure(fig, figures_dir / "learned_winding_controls")
    plt.close(fig)
    return paths


def plot_latent_geometry(
    maps: Sequence[RoadMap],
    summary_df: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
) -> List[Path]:
    selected_maps = [
        "straight_path",
        "u_path",
        "comb_4",
        "comb_12",
        "irregular_comb_16",
        "double_comb_12",
        "braided_ladder_8",
    ]
    selected_dims = [2, 8, 32]
    map_by_name = {road_map.name: road_map for road_map in maps}
    available_dims = [dim for dim in selected_dims if dim in set(summary_df["m"].astype(int))]
    available_maps = [
        name
        for name in selected_maps
        if name in map_by_name and not summary_df[(summary_df["map_name"] == name) & (summary_df["m"].isin(available_dims))].empty
    ]
    if not available_maps or not available_dims:
        return []

    plt = _matplotlib()
    fig, axes = plt.subplots(
        len(available_maps),
        len(available_dims),
        figsize=(3.0 * len(available_dims), 2.75 * len(available_maps)),
        facecolor="white",
        squeeze=False,
    )
    for row_idx, map_name in enumerate(available_maps):
        road_map = map_by_name[map_name]
        order = np.asarray(state_order(road_map), dtype=np.int64)
        for col_idx, dim in enumerate(available_dims):
            ax = axes[row_idx, col_idx]
            selected = summary_df[(summary_df["map_name"] == map_name) & (summary_df["m"] == dim)]
            if selected.empty:
                ax.set_axis_off()
                continue
            seed = int(selected.iloc[0]["best_seed_by_decoded_next_mse"])
            embedding_path = output_dir / "embeddings" / f"{map_name}_m{dim}_seed{seed}.npz"
            if not embedding_path.exists():
                ax.set_axis_off()
                continue
            z = np.load(embedding_path)["z"].astype(np.float64)
            if dim == 2:
                color = np.empty(road_map.num_states, dtype=np.float64)
                color[order] = np.arange(road_map.num_states, dtype=np.float64)
                ax.scatter(z[:, 0], z[:, 1], c=color, cmap="viridis", s=11, linewidths=0)
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                distances = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
                ordered_distances = distances[np.ix_(order, order)]
                ax.imshow(ordered_distances, cmap="magma", interpolation="nearest", aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(f"{map_name}, m={dim}, seed={seed}", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Heatmaps show pairwise latent distances, not transition matrices; higher dimension can reorganize geometry while preserving the same (x, y) state information.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    paths = _save_figure(fig, figures_dir / "learned_branching_latent_geometry")
    plt.close(fig)
    return paths


def make_figures(
    maps: Sequence[RoadMap],
    complexity: Dict[str, Dict[str, object]],
    summary_df: pd.DataFrame,
    required_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> List[Path]:
    figures_dir = output_dir / "figures"
    figure_paths: List[Path] = []
    figure_paths.extend(plot_map_gallery(maps, complexity, figures_dir))
    if not summary_df.empty:
        figure_paths.extend(plot_state_reconstruction(summary_df, figures_dir))
        figure_paths.extend(plot_gain_frontier(summary_df, figures_dir, args.L_budget))
        figure_paths.extend(plot_deficit_error(summary_df, figures_dir))
        figure_paths.extend(plot_required_dimension(required_df, args.m_values, figures_dir))
        figure_paths.extend(plot_winding_controls(summary_df, figures_dir, args.L_budget))
        figure_paths.extend(plot_latent_geometry(maps, summary_df, output_dir, figures_dir))
    return figure_paths


def serializable_config(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["encoder_input"] = "normalized_xy_only"
    config["uses_image_input"] = False
    config["uses_d4rl"] = False
    config["uses_mujoco"] = False
    config["uses_leworldmodel"] = False
    config["uses_jepa"] = False
    config["uses_state_lookup_table_embedding"] = False
    return config


def resolve_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled learned-encoder experiment for branching maze predictive realization."
    )
    parser.add_argument("--quick", action="store_true", help="Use the requested smaller quick-mode sweep.")
    parser.add_argument("--hard", action="store_true", help="Use a larger cloud/GPU-oriented weird-transition sweep.")
    parser.add_argument("--num-states", type=int, default=None)
    parser.add_argument("--maps", nargs="+", default=None, help="Optional map-name subset.")
    parser.add_argument("--m-values", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--num-macro-actions", type=int, default=None)
    parser.add_argument("--L-budget", "--l-budget", dest="L_budget", type=float, default=1.1)
    parser.add_argument("--lambda-gain", type=float, default=0.1)
    parser.add_argument("--lambda-rec", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=1.0)
    parser.add_argument("--lambda-next", type=float, default=1.0)
    parser.add_argument("--detach-target", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--encoder-hidden-layers", type=int, default=3)
    parser.add_argument("--predictor-hidden-layers", type=int, default=3)
    parser.add_argument("--decoder-hidden-layers", type=int, default=2)
    parser.add_argument("--action-embedding-dim", type=int, default=None)
    parser.add_argument("--activation", choices=["silu", "relu"], default="silu")
    parser.add_argument("--float32", action="store_true", help="Use torch.float32 and log this choice.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or auto.")
    parser.add_argument("--macro-suite", choices=["standard", "rich"], default=None)
    parser.add_argument("--macro-seed", type=int, default=12345)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save an intra-run partial checkpoint every N training steps; 0 disables partial checkpoints.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run exhaustive full-transition diagnostics every N optimizer steps; 0 disables diagnostics.",
    )
    parser.add_argument(
        "--save-exhaustive-best",
        action="store_true",
        help="When exhaustive diagnostics are enabled, save best-total-objective and best-dynamics checkpoints.",
    )
    parser.add_argument("--lr-decay-begin", type=int, default=None)
    parser.add_argument("--lr-decay-end", type=int, default=None)
    parser.add_argument("--lr-decay-final-factor", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Load existing results.csv and skip completed map/m/seed runs.")
    parser.add_argument("--overwrite-mode-mismatch", action="store_true")
    args = parser.parse_args()

    if args.quick and args.hard:
        raise ValueError("Choose only one of --quick or --hard.")

    if args.quick:
        args.mode = "quick"
        args.num_states = 64 if args.num_states is None else args.num_states
        args.maps = list(QUICK_MAPS) if args.maps is None else args.maps
        args.m_values = [2, 4, 8, 16] if args.m_values is None else args.m_values
        args.seeds = 2 if args.seeds is None else args.seeds
        args.steps = 2000 if args.steps is None else args.steps
        args.num_macro_actions = 24 if args.num_macro_actions is None else args.num_macro_actions
        args.horizon = 6 if args.horizon is None else args.horizon
        args.batch_size = 512 if args.batch_size is None else args.batch_size
        args.hidden_dim = 128 if args.hidden_dim is None else args.hidden_dim
        args.action_embedding_dim = 32 if args.action_embedding_dim is None else args.action_embedding_dim
        args.macro_suite = "standard" if args.macro_suite is None else args.macro_suite
        args.device = "cpu" if args.device is None else args.device
        args.log_every = 100 if args.log_every is None else args.log_every
        args.checkpoint_every = 500 if args.checkpoint_every is None else args.checkpoint_every
        args.output_dir = str(DEFAULT_OUTPUT_DIR / "quick") if args.output_dir is None else args.output_dir
    elif args.hard:
        args.mode = "hard"
        args.num_states = 256 if args.num_states is None else args.num_states
        args.maps = list(HARD_MAPS) if args.maps is None else args.maps
        args.m_values = [2, 4, 8, 16, 32, 64, 128] if args.m_values is None else args.m_values
        args.seeds = 8 if args.seeds is None else args.seeds
        args.steps = 50000 if args.steps is None else args.steps
        args.num_macro_actions = 256 if args.num_macro_actions is None else args.num_macro_actions
        args.horizon = 16 if args.horizon is None else args.horizon
        args.batch_size = 2048 if args.batch_size is None else args.batch_size
        args.hidden_dim = 256 if args.hidden_dim is None else args.hidden_dim
        args.action_embedding_dim = 64 if args.action_embedding_dim is None else args.action_embedding_dim
        args.macro_suite = "rich" if args.macro_suite is None else args.macro_suite
        args.device = "auto" if args.device is None else args.device
        args.log_every = 500 if args.log_every is None else args.log_every
        args.checkpoint_every = 5000 if args.checkpoint_every is None else args.checkpoint_every
        args.output_dir = str(DEFAULT_OUTPUT_DIR / "hard") if args.output_dir is None else args.output_dir
    else:
        args.mode = "full"
        args.num_states = 128 if args.num_states is None else args.num_states
        args.maps = list(FULL_MAPS) if args.maps is None else args.maps
        args.m_values = [2, 4, 8, 16, 32, 64] if args.m_values is None else args.m_values
        args.seeds = 5 if args.seeds is None else args.seeds
        args.steps = 20000 if args.steps is None else args.steps
        args.num_macro_actions = 64 if args.num_macro_actions is None else args.num_macro_actions
        args.horizon = 8 if args.horizon is None else args.horizon
        args.batch_size = 512 if args.batch_size is None else args.batch_size
        args.hidden_dim = 128 if args.hidden_dim is None else args.hidden_dim
        args.action_embedding_dim = 32 if args.action_embedding_dim is None else args.action_embedding_dim
        args.macro_suite = "standard" if args.macro_suite is None else args.macro_suite
        args.device = "cpu" if args.device is None else args.device
        args.log_every = 100 if args.log_every is None else args.log_every
        args.checkpoint_every = 1000 if args.checkpoint_every is None else args.checkpoint_every
        args.output_dir = str(DEFAULT_OUTPUT_DIR) if args.output_dir is None else args.output_dir

    if args.num_states < 2:
        raise ValueError("--num-states must be at least 2.")
    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1.")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2.")
    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative.")
    if int(args.eval_every) < 0:
        raise ValueError("--eval-every must be non-negative.")
    if float(args.lr_decay_final_factor) <= 0.0:
        raise ValueError("--lr-decay-final-factor must be positive.")
    if args.lr_decay_begin is not None and int(args.lr_decay_begin) < 1:
        raise ValueError("--lr-decay-begin must be at least 1 when provided.")
    if args.lr_decay_end is not None and int(args.lr_decay_end) < 1:
        raise ValueError("--lr-decay-end must be at least 1 when provided.")
    if any(int(m) < 1 for m in args.m_values):
        raise ValueError("--m-values must be positive.")
    args.m_values = [int(m) for m in args.m_values]
    args.maps = list(args.maps)
    return args


def guard_output_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config_path = output_dir / "run_config.json"
    if not config_path.exists() or args.overwrite_mode_mismatch:
        return
    with config_path.open() as file:
        existing = json.load(file)
    existing_mode = str(existing.get("mode", "quick" if existing.get("quick", False) else "full"))
    if existing_mode != str(args.mode):
        raise RuntimeError(
            f"{output_dir} already contains {existing_mode} outputs, but this run is {args.mode}. "
            "Choose a different --output-dir or pass --overwrite-mode-mismatch intentionally."
        )


def print_table(title: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    print(f"\n{title}")
    if df.empty:
        print("(empty)")
        return
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 80, "display.width", 160):
        print(df.to_string(index=True))


def main() -> None:
    args = resolve_args()
    output_dir = Path(args.output_dir)
    guard_output_mode(output_dir, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.float32 else torch.float64
    if args.float32:
        print("Using torch.float32 because --float32 was requested. Main default is torch.float64.")
    else:
        print("Using torch.float64.")
    torch.set_default_dtype(dtype)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    maps = build_maps(args.num_states, args.maps)
    map_order = [road_map.name for road_map in maps]
    complexity = validate_map_family(maps, args.num_states)
    complexity_rows = [complexity[road_map.name] for road_map in maps]
    complexity_df = dataframe_with_order(complexity_rows, map_order)
    complexity_path = output_dir / "map_complexity.csv"
    atomic_to_csv(complexity_df, complexity_path, columns=ordered_columns(complexity_df, COMPLEXITY_COLUMNS))

    macro_actions = generate_macro_actions(args.horizon, args.num_macro_actions, args.macro_seed, args.macro_suite)
    macro_actions_path = output_dir / "macro_actions.json"
    save_macro_actions(macro_actions_path, macro_actions, args.macro_seed, args.macro_suite)
    macro_next_by_map = {road_map.name: compose_macro_transitions(build_transition_table(road_map), macro_actions) for road_map in maps}

    write_json(output_dir / "run_config.json", serializable_config(args, output_dir))
    print(f"Training {len(maps)} maps x {len(args.m_values)} dimensions x {args.seeds} seeds.")

    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    required_path = output_dir / "required_dimension.csv"
    rows: List[Dict[str, object]] = []
    if args.resume and results_path.exists():
        rows = pd.read_csv(results_path).to_dict("records")
        print(f"Loaded {len(rows)} existing completed rows from {results_path}.")
    completed = completed_run_keys(rows)
    wanted = {
        run_key(road_map.name, latent_dim, seed)
        for road_map in maps
        for latent_dim in args.m_values
        for seed in range(int(args.seeds))
    }
    if args.resume:
        checkpoint_rows = load_checkpoint_rows(output_dir, wanted, completed, args)
        if checkpoint_rows:
            rows.extend(checkpoint_rows)
            print(f"Recovered {len(checkpoint_rows)} completed rows from checkpoints.")
    total_runs = len(maps) * len(args.m_values) * int(args.seeds)
    run_idx = 0
    results_df = dataframe_with_order(rows, map_order) if rows else pd.DataFrame()
    summary_df = build_summary(results_df) if not results_df.empty else pd.DataFrame()
    required_df = (
        compute_required_dimensions(summary_df, budgets=[1.05, 1.1, 1.2, 1.5], m_values=args.m_values)
        if not summary_df.empty
        else pd.DataFrame()
    )
    for road_map in maps:
        for latent_dim in args.m_values:
            for seed in range(int(args.seeds)):
                run_idx += 1
                key = run_key(road_map.name, latent_dim, seed)
                if args.resume and key in completed:
                    print(
                        f"[{run_idx}/{total_runs}] {road_map.name} m={latent_dim} seed={seed} already complete; skipping.",
                        flush=True,
                    )
                    continue
                print(
                    f"[{run_idx}/{total_runs}] {road_map.name} m={latent_dim} seed={seed} "
                    f"(branch_excess={complexity[road_map.name]['branch_excess']})",
                    flush=True,
                )
                row = train_one_model(
                    road_map=road_map,
                    complexity=complexity[road_map.name],
                    macro_next_np=macro_next_by_map[road_map.name],
                    args=args,
                    dtype=dtype,
                    output_dir=output_dir,
                    seed=seed,
                    latent_dim=latent_dim,
                )
                rows.append(row)
                completed.add(key)
                results_df, summary_df, required_df = write_result_tables(rows, map_order, output_dir, args)

    if rows:
        results_df, summary_df, required_df = write_result_tables(rows, map_order, output_dir, args)

    figure_paths: List[Path] = []
    if not args.skip_figures:
        figure_paths = make_figures(maps, complexity, summary_df, required_df, args, output_dir)

    output_paths = [
        output_dir / "run_config.json",
        complexity_path,
        macro_actions_path,
        results_path,
        summary_path,
        required_path,
        output_dir / "checkpoints",
        output_dir / "checkpoints_partial",
        output_dir / "checkpoints_best_total",
        output_dir / "checkpoints_best_dyn",
        output_dir / "embeddings",
        output_dir / "losses",
        output_dir / "diagnostics",
    ] + figure_paths

    print("\nOutput paths")
    for path in output_paths:
        print(f"- {path}")

    print_table(
        "Map complexity table",
        complexity_df.set_index("map_name")[
            [
                "map_group",
                "num_states",
                "branch_excess",
                "junction_count",
                "dead_end_count",
                "cycle_rank",
                "graph_diameter",
                "mean_shortest_path",
                "q95_geodesic_over_euclidean",
            ]
        ],
    )
    q99_gain = summary_df.pivot(index="map_name", columns="m", values="mean_q99_r_true").reindex(map_order)
    print_table("Summary q99_r_true by map and dimension", q99_gain)
    q99_errors = summary_df.pivot(index="map_name", columns="m", values="mean_q99_ratio_deficit").reindex(map_order)
    print_table("Summary q99_ratio_deficit by map and dimension", q99_errors)
    q99_pair_error = summary_df.pivot(index="map_name", columns="m", values="mean_q99_norm_pair_error_now").reindex(map_order)
    print_table("Summary q99_norm_pair_error_now by map and dimension", q99_pair_error)
    recon = summary_df.pivot(index="map_name", columns="m", values="mean_recon_rmse").reindex(map_order)
    print_table("Reconstruction RMSE by map and dimension", recon)


if __name__ == "__main__":
    main()
