from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


ACTION_NAMES = ("up", "down", "left", "right")
ACTION_DELTAS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}
EPS = 1e-12
DTYPE = torch.float64
DEFAULT_OUTPUT_DIR = Path("outputs/branching_maze_gain")
RESULT_FIELDS = [
    "map_name",
    "map_group",
    "num_states",
    "branch_excess",
    "junction_count",
    "dead_end_count",
    "max_degree",
    "cycle_rank",
    "graph_diameter",
    "q95_geodesic_over_euclidean",
    "horizon",
    "num_macro_actions",
    "m",
    "seed",
    "best_hard_gain",
    "final_hard_gain",
    "L_phys",
    "median_ratio",
    "q95_ratio",
    "q99_ratio",
    "q999_ratio",
    "fraction_ratio_gt_1",
    "fraction_ratio_gt_1_05",
    "fraction_ratio_gt_1_1",
    "fraction_ratio_gt_1_2",
    "fraction_ratio_gt_1_5",
    "steps_run",
    "converged",
    "restart_count",
    "embedding_path",
]
SUMMARY_FIELDS = [
    "map_name",
    "map_group",
    "m",
    "best_gain",
    "median_gain",
    "iqr_low",
    "iqr_high",
    "std_gain",
    "best_seed",
    "L_phys",
    "branch_excess",
    "junction_count",
    "dead_end_count",
    "cycle_rank",
]
COMPLEXITY_FIELDS = [
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
    "L_phys",
]


@dataclass(frozen=True)
class RoadMap:
    name: str
    group: str
    coords: np.ndarray
    edges: Tuple[Tuple[int, int], ...]
    cyclic_control: bool = False

    @property
    def num_states(self) -> int:
        return int(self.coords.shape[0])


@dataclass(frozen=True)
class OptimizationConfig:
    steps: int
    learning_rate: float
    eval_every: int
    gradient_clip_norm: float
    training_pairs: int
    max_restarts: int
    seed_base: int


def _safe_float(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):.12g}"
        return str(float(value))
    return value


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _safe_float(row.get(key, "")) for key in fieldnames})


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def _stable_seed(*parts: object) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


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
            raise ValueError(f"{name}: non-adjacent path step at index {idx}: {coords_array[idx]} -> {coords_array[idx + 1]}")
        edges.append((idx, idx + 1))
    return RoadMap(name=name, group=group, coords=coords_array, edges=_canonical_edges(edges))


def _straight_path(num_states: int, name: str, group: str) -> RoadMap:
    return _path_map(name, group, [(x, 0) for x in range(num_states)])


def _u_path(num_states: int) -> RoadMap:
    width = max(4, int(round(math.sqrt(num_states))))
    width = min(width, max(1, num_states - 2))
    left = (num_states - width) // 2
    right = num_states - width - left
    left = max(1, left)
    right = max(1, right)
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


def _comb_map(name: str, group: str, num_states: int, teeth: int, trunk_fraction: float = 0.35) -> RoadMap:
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


def _rooms_tree_like(num_states: int, rooms: int = 7) -> RoadMap:
    terminals = 2 * rooms
    min_trunk = rooms + 2
    target_trunk = max(min_trunk, int(round(num_states * 0.34)))
    trunk_len = min(target_trunk, num_states - terminals)
    trunk_len = max(trunk_len, min_trunk)
    if trunk_len + terminals > num_states:
        raise ValueError(f"rooms_tree_like: num_states={num_states} is too small for {rooms} rooms.")

    coords: List[Tuple[int, int]] = [(x, 0) for x in range(trunk_len)]
    edges: List[Tuple[int, int]] = [(x, x + 1) for x in range(trunk_len - 1)]
    centers = _even_positions(rooms, 1, trunk_len - 2)
    lengths = _terminal_lengths(num_states, trunk_len, terminals, min_len=1)
    length_iter = iter(lengths)
    for center in centers:
        for sign in (1, -1):
            length = next(length_iter)
            prev = center
            for step in range(1, length + 1):
                node = len(coords)
                coords.append((center, sign * step))
                edges.append((prev, node))
                prev = node
    return RoadMap(
        "rooms_tree_like",
        "branching_sweep",
        np.asarray(coords, dtype=np.int64),
        _canonical_edges(edges),
    )


def build_maps(num_states: int) -> List[RoadMap]:
    maps = [
        _straight_path(num_states, "straight_path", "winding_control"),
        _u_path(num_states),
        _serpentine_path(num_states),
        _spiral_path(num_states),
        _straight_path(num_states, "path_0", "branching_sweep"),
        _one_t_junction(num_states),
        _comb_map("comb_4", "branching_sweep", num_states, teeth=4, trunk_fraction=0.34),
        _comb_map("comb_8", "branching_sweep", num_states, teeth=8, trunk_fraction=0.34),
        _comb_map("comb_12", "branching_sweep", num_states, teeth=12, trunk_fraction=0.34),
        _rooms_tree_like(num_states, rooms=7),
    ]
    return maps


def adjacency_list(road_map: RoadMap) -> List[List[int]]:
    neighbors: List[List[int]] = [[] for _ in range(road_map.num_states)]
    for src, dst in road_map.edges:
        neighbors[src].append(dst)
        neighbors[dst].append(src)
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


def compute_complexity(road_map: RoadMap, L_phys: Optional[float] = None) -> Dict[str, object]:
    degrees = graph_degrees(road_map)
    edge_count = len(road_map.edges)
    dist = shortest_path_matrix(road_map)
    connected = np.isfinite(dist).all()
    if not connected:
        raise ValueError(f"{road_map.name}: graph is not connected.")
    left, right = np.triu_indices(road_map.num_states, k=1)
    shortest = dist[left, right]
    euclidean = np.linalg.norm(road_map.coords[left] - road_map.coords[right], axis=1)
    ratio = shortest / np.maximum(euclidean, EPS)
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
        "L_phys": float("nan") if L_phys is None else float(L_phys),
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
        if not road_map.cyclic_control and int(metrics[road_map.name]["cycle_rank"]) != 0:
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


def generate_macro_actions(horizon: int, num_macro_actions: int, seed: int) -> np.ndarray:
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
    if num_macro_actions < len(deterministic):
        raise ValueError(f"num_macro_actions must be at least {len(deterministic)} to include repeated and alternating blocks.")

    rng = np.random.default_rng(seed)
    blocks = list(deterministic)
    seen = {tuple(block) for block in blocks}
    while len(blocks) < num_macro_actions:
        block = rng.integers(0, len(ACTION_NAMES), size=horizon, endpoint=False).astype(np.int64).tolist()
        key = tuple(block)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(block)
    return np.asarray(blocks, dtype=np.int64)


def save_macro_actions(path: Path, blocks: np.ndarray, seed: int) -> None:
    payload = {
        "seed": int(seed),
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
    _write_json(path, payload)


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


def all_unordered_pairs(num_states: int) -> Tuple[np.ndarray, np.ndarray]:
    left, right = np.triu_indices(num_states, k=1)
    return left.astype(np.int64), right.astype(np.int64)


def _normalize_embedding_(Z: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor, eps: float = EPS) -> bool:
    with torch.no_grad():
        Z -= Z.mean(dim=0, keepdim=True)
        distances = torch.linalg.norm(Z[pair_i] - Z[pair_j], dim=-1)
        min_dist = torch.min(distances)
        if (not torch.isfinite(min_dist)) or float(min_dist) < 1e-10:
            return False
        Z /= min_dist.clamp_min(eps)
        Z -= Z.mean(dim=0, keepdim=True)
    return True


def _assert_embedding_normalized(Z: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor, name: str) -> None:
    with torch.no_grad():
        centered_error = torch.max(torch.abs(Z.mean(dim=0)))
        distances = torch.linalg.norm(Z[pair_i] - Z[pair_j], dim=-1)
        min_dist = torch.min(distances)
    if float(centered_error) > 1e-7:
        raise AssertionError(f"{name}: optimized embedding is not centered; max mean={float(centered_error):.3g}.")
    if abs(float(min_dist) - 1.0) > 1e-6:
        raise AssertionError(f"{name}: min pairwise distance is {float(min_dist):.12g}, not 1.")


def transition_gain_stats(
    Z: torch.Tensor,
    macro_next: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    eps: float = EPS,
    macro_chunk: int = 16,
) -> Dict[str, float]:
    with torch.no_grad():
        Z = Z.to(dtype=DTYPE, device="cpu")
        macro_next = macro_next.to(device="cpu", dtype=torch.long)
        pair_i = pair_i.to(device="cpu", dtype=torch.long)
        pair_j = pair_j.to(device="cpu", dtype=torch.long)
        denom = torch.linalg.norm(Z[pair_i] - Z[pair_j], dim=-1).clamp_min(eps)
        ratios = []
        for start in range(0, macro_next.shape[0], macro_chunk):
            next_chunk = macro_next[start : start + macro_chunk]
            next_i = next_chunk[:, pair_i]
            next_j = next_chunk[:, pair_j]
            d_next = torch.linalg.norm(Z[next_i] - Z[next_j], dim=-1)
            ratios.append((d_next / denom.unsqueeze(0)).reshape(-1))
        ratio = torch.cat(ratios, dim=0)
        quantiles = torch.quantile(ratio, torch.tensor([0.5, 0.95, 0.99, 0.999], dtype=DTYPE))
        return {
            "hard_gain": float(torch.max(ratio).item()),
            "median_ratio": float(quantiles[0].item()),
            "q95_ratio": float(quantiles[1].item()),
            "q99_ratio": float(quantiles[2].item()),
            "q999_ratio": float(quantiles[3].item()),
            "fraction_ratio_gt_1": float(torch.mean((ratio > 1.0).to(DTYPE)).item()),
            "fraction_ratio_gt_1_05": float(torch.mean((ratio > 1.05).to(DTYPE)).item()),
            "fraction_ratio_gt_1_1": float(torch.mean((ratio > 1.1).to(DTYPE)).item()),
            "fraction_ratio_gt_1_2": float(torch.mean((ratio > 1.2).to(DTYPE)).item()),
            "fraction_ratio_gt_1_5": float(torch.mean((ratio > 1.5).to(DTYPE)).item()),
        }


def _validate_hard_gain_example() -> None:
    Z = torch.tensor([[0.0], [1.0], [3.0]], dtype=DTYPE)
    macro_next = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pair_i = torch.tensor([0, 0, 1], dtype=torch.long)
    pair_j = torch.tensor([1, 2, 2], dtype=torch.long)
    stats = transition_gain_stats(Z, macro_next, pair_i, pair_j)
    brute = max(2.0 / 1.0, 1.0 / 3.0, 0.0 / 2.0)
    if abs(stats["hard_gain"] - brute) > 1e-12:
        raise AssertionError(f"Hard-gain validation failed: got {stats['hard_gain']}, expected {brute}.")


def physical_coordinate_gain(
    road_map: RoadMap,
    macro_next_np: np.ndarray,
    pair_i_np: np.ndarray,
    pair_j_np: np.ndarray,
) -> float:
    coords = torch.as_tensor(road_map.coords, dtype=DTYPE)
    distances = torch.linalg.norm(coords[pair_i_np] - coords[pair_j_np], dim=-1)
    min_dist = torch.min(distances)
    if (not torch.isfinite(min_dist)) or float(min_dist) <= 0.0:
        raise AssertionError(f"{road_map.name}: invalid physical coordinate minimum pairwise distance.")
    coords = coords / min_dist
    stats = transition_gain_stats(
        coords,
        torch.as_tensor(macro_next_np, dtype=torch.long),
        torch.as_tensor(pair_i_np, dtype=torch.long),
        torch.as_tensor(pair_j_np, dtype=torch.long),
    )
    return float(stats["hard_gain"])


def _beta_for_step(step: int, steps: int) -> float:
    frac = step / max(1, steps)
    if frac < 0.30:
        return 10.0
    if frac < 0.60:
        return 30.0
    return 100.0


def _initial_embedding(num_states: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    Z = 0.05 * torch.randn((num_states, dim), dtype=DTYPE, generator=generator)
    line = torch.linspace(0.0, float(num_states - 1), num_states, dtype=DTYPE).unsqueeze(1)
    Z[:, :1] += line
    if dim > 1:
        Z[:, 1:] += 0.01 * torch.randn((num_states, dim - 1), dtype=DTYPE, generator=generator)
    return Z


def _sample_training_pairs(
    pair_i_np: np.ndarray,
    pair_j_np: np.ndarray,
    max_pairs: int,
    map_name: str,
    seed: int,
    seed_base: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_pairs <= 0 or max_pairs >= len(pair_i_np):
        return pair_i_np, pair_j_np
    rng = np.random.default_rng(_stable_seed(seed_base, map_name, seed, "training_pairs"))
    selected = np.sort(rng.choice(len(pair_i_np), size=max_pairs, replace=False))
    return pair_i_np[selected], pair_j_np[selected]


def _smooth_gain_loss(
    Z: torch.Tensor,
    macro_next: torch.Tensor,
    train_i: torch.Tensor,
    train_j: torch.Tensor,
    beta: float,
    eps: float = EPS,
) -> torch.Tensor:
    d_now = torch.linalg.norm(Z[train_i] - Z[train_j], dim=-1)
    log_now = torch.log(d_now + eps)
    next_i = macro_next[:, train_i]
    next_j = macro_next[:, train_j]
    d_next = torch.linalg.norm(Z[next_i] - Z[next_j], dim=-1)
    log_ratio = torch.log(d_next + eps) - log_now.unsqueeze(0)
    return torch.logsumexp(float(beta) * log_ratio.reshape(-1), dim=0) / float(beta)


def optimize_embedding(
    road_map: RoadMap,
    macro_next_np: np.ndarray,
    pair_i_np: np.ndarray,
    pair_j_np: np.ndarray,
    train_i_np: np.ndarray,
    train_j_np: np.ndarray,
    dim: int,
    seed: int,
    config: OptimizationConfig,
    embedding_dir: Path,
) -> Dict[str, object]:
    pair_i = torch.as_tensor(pair_i_np, dtype=torch.long)
    pair_j = torch.as_tensor(pair_j_np, dtype=torch.long)
    train_i = torch.as_tensor(train_i_np, dtype=torch.long)
    train_j = torch.as_tensor(train_j_np, dtype=torch.long)
    macro_next = torch.as_tensor(macro_next_np, dtype=torch.long)
    restart_count = 0
    steps_run = 0
    best_Z: Optional[torch.Tensor] = None
    best_hard_gain = float("inf")
    final_hard_gain = float("inf")
    completed = False

    while restart_count <= config.max_restarts:
        init_seed = _stable_seed(config.seed_base, road_map.name, dim, seed, restart_count)
        Z = _initial_embedding(road_map.num_states, dim, init_seed)
        if not _normalize_embedding_(Z, pair_i, pair_j):
            restart_count += 1
            continue
        Z.requires_grad_(True)
        optimizer = torch.optim.Adam([Z], lr=config.learning_rate)
        unstable = False
        stats = transition_gain_stats(Z.detach(), macro_next, pair_i, pair_j)
        best_hard_gain = float(stats["hard_gain"])
        best_Z = Z.detach().clone()

        for step in range(config.steps):
            beta = _beta_for_step(step, config.steps)
            optimizer.zero_grad(set_to_none=True)
            loss = _smooth_gain_loss(Z, macro_next, train_i, train_j, beta)
            if not torch.isfinite(loss):
                unstable = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Z], max_norm=config.gradient_clip_norm)
            optimizer.step()
            if not _normalize_embedding_(Z, pair_i, pair_j):
                unstable = True
                break
            steps_run = step + 1
            should_eval = (step + 1) % config.eval_every == 0 or (step + 1) == config.steps
            if should_eval:
                stats = transition_gain_stats(Z.detach(), macro_next, pair_i, pair_j)
                hard_gain = float(stats["hard_gain"])
                if hard_gain < best_hard_gain:
                    best_hard_gain = hard_gain
                    best_Z = Z.detach().clone()

        if unstable:
            restart_count += 1
            continue
        final_stats = transition_gain_stats(Z.detach(), macro_next, pair_i, pair_j)
        final_hard_gain = float(final_stats["hard_gain"])
        if final_hard_gain < best_hard_gain:
            best_hard_gain = final_hard_gain
            best_Z = Z.detach().clone()
        completed = True
        break

    if (not completed) or best_Z is None or not math.isfinite(best_hard_gain):
        raise RuntimeError(f"{road_map.name} m={dim} seed={seed}: optimization failed after {restart_count} restarts.")

    _assert_embedding_normalized(best_Z, pair_i, pair_j, f"{road_map.name} m={dim} seed={seed}")
    best_stats = transition_gain_stats(best_Z, macro_next, pair_i, pair_j)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = embedding_dir / f"{road_map.name}_m{dim}_seed{seed}.pt"
    torch.save(
        {
            "Z": best_Z.cpu(),
            "map_name": road_map.name,
            "map_group": road_map.group,
            "m": int(dim),
            "seed": int(seed),
            "best_hard_gain": float(best_stats["hard_gain"]),
            "state_coords": torch.as_tensor(road_map.coords, dtype=torch.long),
            "macro_horizon_fixed_transition": torch.as_tensor(macro_next_np, dtype=torch.long),
        },
        embedding_path,
    )
    return {
        "m": int(dim),
        "seed": int(seed),
        "best_hard_gain": float(best_stats["hard_gain"]),
        "final_hard_gain": float(final_hard_gain),
        "median_ratio": best_stats["median_ratio"],
        "q95_ratio": best_stats["q95_ratio"],
        "q99_ratio": best_stats["q99_ratio"],
        "q999_ratio": best_stats["q999_ratio"],
        "fraction_ratio_gt_1": best_stats["fraction_ratio_gt_1"],
        "fraction_ratio_gt_1_05": best_stats["fraction_ratio_gt_1_05"],
        "fraction_ratio_gt_1_1": best_stats["fraction_ratio_gt_1_1"],
        "fraction_ratio_gt_1_2": best_stats["fraction_ratio_gt_1_2"],
        "fraction_ratio_gt_1_5": best_stats["fraction_ratio_gt_1_5"],
        "steps_run": int(steps_run),
        "converged": bool(steps_run == config.steps and math.isfinite(final_hard_gain)),
        "restart_count": int(restart_count),
        "embedding_path": str(embedding_path),
    }


def summarize_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    keys = sorted({(str(row["map_name"]), int(row["m"])) for row in results})
    for map_name, dim in keys:
        group_rows = [row for row in results if str(row["map_name"]) == map_name and int(row["m"]) == dim]
        gains = np.asarray([float(row["best_hard_gain"]) for row in group_rows], dtype=np.float64)
        best_idx = int(np.argmin(gains))
        first = group_rows[0]
        summary_rows.append(
            {
                "map_name": map_name,
                "map_group": first["map_group"],
                "m": dim,
                "best_gain": float(np.min(gains)),
                "median_gain": float(np.median(gains)),
                "iqr_low": float(np.quantile(gains, 0.25)),
                "iqr_high": float(np.quantile(gains, 0.75)),
                "std_gain": float(np.std(gains, ddof=0)),
                "best_seed": int(group_rows[best_idx]["seed"]),
                "L_phys": float(first["L_phys"]),
                "branch_excess": int(first["branch_excess"]),
                "junction_count": int(first["junction_count"]),
                "dead_end_count": int(first["dead_end_count"]),
                "cycle_rank": int(first["cycle_rank"]),
            }
        )
    return summary_rows


def required_dimension_rows(
    summary_rows: List[Dict[str, object]],
    budgets: Sequence[float],
    map_group: str = "branching_sweep",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    map_names = sorted(
        {str(row["map_name"]) for row in summary_rows if str(row["map_group"]) == map_group},
        key=lambda name: (
            int(next(row["branch_excess"] for row in summary_rows if row["map_name"] == name)),
            name,
        ),
    )
    max_dim = max(int(row["m"]) for row in summary_rows)
    for budget in budgets:
        for map_name in map_names:
            entries = sorted(
                [row for row in summary_rows if row["map_name"] == map_name],
                key=lambda row: int(row["m"]),
            )
            selected = [int(row["m"]) for row in entries if float(row["best_gain"]) <= float(budget)]
            if selected:
                dim = min(selected)
                label = str(dim)
                capped = dim
                reached = True
            else:
                dim = None
                label = f">{max_dim}"
                capped = max_dim
                reached = False
            first = entries[0]
            rows.append(
                {
                    "gain_budget": float(budget),
                    "map_name": map_name,
                    "branch_excess": int(first["branch_excess"]),
                    "junction_count": int(first["junction_count"]),
                    "required_dimension": "" if dim is None else int(dim),
                    "required_dimension_label": label,
                    "required_dimension_capped": int(capped),
                    "reached_budget": bool(reached),
                }
            )
    return rows


def _matplotlib() -> object:
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


def _set_same_map_scale(axes: Sequence[object], maps: Sequence[RoadMap]) -> None:
    all_coords = np.vstack([road_map.coords for road_map in maps])
    min_xy = all_coords.min(axis=0).astype(float)
    max_xy = all_coords.max(axis=0).astype(float)
    center = (min_xy + max_xy) / 2.0
    span = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], 1.0))
    pad = max(1.0, 0.04 * span)
    for ax in axes:
        ax.set_xlim(center[0] - span / 2.0 - pad, center[0] + span / 2.0 + pad)
        ax.set_ylim(center[1] - span / 2.0 - pad, center[1] + span / 2.0 + pad)
        ax.set_aspect("equal")


def plot_map_gallery(
    maps: Sequence[RoadMap],
    complexity: Dict[str, Dict[str, object]],
    output_dir: Path,
) -> List[Path]:
    plt = _matplotlib()
    winding = [road_map for road_map in maps if road_map.group == "winding_control"]
    branching = [road_map for road_map in maps if road_map.group == "branching_sweep"]
    columns = max(len(winding), len(branching))
    fig, axes = plt.subplots(2, columns, figsize=(2.6 * columns, 5.2), facecolor="white")
    axes_array = np.asarray(axes).reshape(2, columns)
    for row_idx, group_maps in enumerate((winding, branching)):
        for col_idx in range(columns):
            ax = axes_array[row_idx, col_idx]
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
                    lw=1.2,
                    solid_capstyle="round",
                )
            ax.scatter(coords[:, 0], coords[:, 1], s=5, color="#2F6F9F", zorder=3)
            metrics = complexity[road_map.name]
            ax.set_title(
                f"{road_map.name}\n"
                f"b={metrics['branch_excess']} j={metrics['junction_count']} "
                f"dend={metrics['dead_end_count']} cyc={metrics['cycle_rank']}\n"
                f"L_phys={float(metrics['L_phys']):.3g}",
                fontsize=8,
            )
    _set_same_map_scale(axes_array.ravel().tolist(), maps)
    fig.suptitle("Controlled road graphs: winding controls vs branching sweep", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    paths = [output_dir / "map_gallery.png", output_dir / "map_gallery.pdf"]
    for path in paths:
        fig.savefig(path, dpi=260 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def _rows_by_map(summary_rows: List[Dict[str, object]], group: Optional[str] = None) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        if group is not None and str(row["map_group"]) != group:
            continue
        out.setdefault(str(row["map_name"]), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda item: int(item["m"]))
    return out


def plot_winding_frontier(summary_rows: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    plt = _matplotlib()
    rows_by_map = _rows_by_map(summary_rows, "winding_control")
    fig, ax = plt.subplots(figsize=(5.8, 3.4), facecolor="white")
    for map_name in ["straight_path", "u_path", "serpentine_path", "spiral_path"]:
        rows = rows_by_map.get(map_name, [])
        if not rows:
            continue
        xs = [int(row["m"]) for row in rows]
        ys = [float(row["best_gain"]) for row in rows]
        ax.plot(xs, ys, marker="o", lw=1.6, ms=4, label=map_name)
    ax.axhline(1.0, color="0.45", lw=0.9, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("latent dimension m")
    ax.set_ylabel("best-found required gain")
    ax.set_title("Winding-control optimized dimension-gain frontier")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    paths = [output_dir / "winding_control_frontier.png", output_dir / "winding_control_frontier.pdf"]
    for path in paths:
        fig.savefig(path, dpi=260 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def plot_branching_frontier(summary_rows: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    plt = _matplotlib()
    rows_by_map = _rows_by_map(summary_rows, "branching_sweep")
    ordered = sorted(rows_by_map, key=lambda name: (int(rows_by_map[name][0]["branch_excess"]), name))
    fig, ax = plt.subplots(figsize=(6.4, 3.7), facecolor="white")
    for map_name in ordered:
        rows = rows_by_map[map_name]
        xs = np.asarray([int(row["m"]) for row in rows], dtype=np.float64)
        ys = np.asarray([float(row["best_gain"]) for row in rows], dtype=np.float64)
        low = np.asarray([float(row["iqr_low"]) for row in rows], dtype=np.float64)
        high = np.asarray([float(row["iqr_high"]) for row in rows], dtype=np.float64)
        label = f"{map_name} (b={int(rows[0]['branch_excess'])})"
        line = ax.plot(xs, ys, marker="o", lw=1.5, ms=4, label=label)[0]
        if np.any(high > low):
            ax.fill_between(xs, low, high, color=line.get_color(), alpha=0.12, linewidth=0)
    for level, alpha in [(1.0, 0.65), (1.1, 0.35), (1.2, 0.35)]:
        ax.axhline(level, color="0.35", lw=0.85, ls="--", alpha=alpha)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("latent dimension m")
    ax.set_ylabel("best-found required gain")
    ax.set_title("Branching sweep optimized dimension-gain frontier")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    paths = [output_dir / "branching_frontier.png", output_dir / "branching_frontier.pdf"]
    for path in paths:
        fig.savefig(path, dpi=260 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def plot_required_dimension(required_rows: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(5.8, 3.5), facecolor="white")
    budgets = sorted({float(row["gain_budget"]) for row in required_rows})
    for budget in budgets:
        rows = sorted(
            [row for row in required_rows if float(row["gain_budget"]) == budget],
            key=lambda row: (int(row["branch_excess"]), str(row["map_name"])),
        )
        xs = [int(row["branch_excess"]) for row in rows]
        ys = [int(row["required_dimension_capped"]) for row in rows]
        ax.plot(xs, ys, marker="o", lw=1.4, ms=4, label=f"L0={budget:g}")
        capped_x = [int(row["branch_excess"]) for row in rows if not bool(row["reached_budget"])]
        capped_y = [int(row["required_dimension_capped"]) for row in rows if not bool(row["reached_budget"])]
        if capped_x:
            ax.scatter(capped_x, capped_y, marker="^", s=35, color=ax.lines[-1].get_color(), zorder=5)
    ax.set_xlabel("branch_excess")
    ax.set_ylabel("empirical required dimension")
    ax.set_title("Dimension needed to meet a gain budget")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = [output_dir / "branching_required_dimension.png", output_dir / "branching_required_dimension.pdf"]
    for path in paths:
        fig.savefig(path, dpi=260 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def plot_physical_gain(complexity_rows: List[Dict[str, object]], output_dir: Path) -> List[Path]:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(5.4, 3.4), facecolor="white")
    groups = [
        ("winding_control", "Winding controls", "o", "#4E79A7"),
        ("branching_sweep", "Branching sweep", "s", "#E15759"),
    ]
    for group, label, marker, color in groups:
        rows = [row for row in complexity_rows if row["map_group"] == group]
        xs = [int(row["branch_excess"]) for row in rows]
        ys = [float(row["L_phys"]) for row in rows]
        ax.scatter(xs, ys, marker=marker, s=42, color=color, label=label, alpha=0.9)
        for row in rows:
            ax.annotate(str(row["map_name"]), (int(row["branch_excess"]), float(row["L_phys"])), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("branch_excess")
    ax.set_ylabel("physical-coordinate hard gain")
    ax.set_title("Raw physical-coordinate predictive gain")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = [output_dir / "physical_coordinate_gain.png", output_dir / "physical_coordinate_gain.pdf"]
    for path in paths:
        fig.savefig(path, dpi=260 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def state_order(road_map: RoadMap) -> List[int]:
    degrees = graph_degrees(road_map)
    neighbors = adjacency_list(road_map)
    if int(np.max(degrees)) <= 2:
        endpoints = np.where(degrees <= 1)[0]
        start = int(endpoints[np.lexsort((road_map.coords[endpoints, 1], road_map.coords[endpoints, 0]))][0]) if endpoints.size else 0
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
    visited = {start}
    order = []
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        sorted_neighbors = sorted(neighbors[node], key=lambda idx: (int(road_map.coords[idx, 0]), int(road_map.coords[idx, 1]), idx))
        for nxt in sorted_neighbors:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return order


def _load_embedding(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload["Z"].to(dtype=DTYPE)


def plot_distance_heatmaps(
    maps_by_name: Dict[str, RoadMap],
    summary_rows: List[Dict[str, object]],
    output_dir: Path,
) -> List[Path]:
    selected_maps = ["straight_path", "u_path", "comb_4", "comb_12"]
    selected_dims = [1, 2, 8, 32]
    entries = []
    for map_name in selected_maps:
        for dim in selected_dims:
            matches = [row for row in summary_rows if row["map_name"] == map_name and int(row["m"]) == dim]
            if matches:
                entries.append(matches[0])
    if not entries:
        return []

    plt = _matplotlib()
    rows = len(selected_maps)
    cols = len(selected_dims)
    fig, axes = plt.subplots(rows, cols, figsize=(2.35 * cols, 2.15 * rows), facecolor="white", squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for row in entries:
        map_name = str(row["map_name"])
        dim = int(row["m"])
        r = selected_maps.index(map_name)
        c = selected_dims.index(dim)
        ax = axes[r, c]
        road_map = maps_by_name[map_name]
        best_seed = int(row["best_seed"])
        embedding_path = output_dir / "embeddings" / f"{map_name}_m{dim}_seed{best_seed}.pt"
        Z = _load_embedding(embedding_path)
        order = state_order(road_map)
        D = torch.cdist(Z[order], Z[order], p=2).numpy()
        ax.imshow(D, cmap="magma", aspect="auto", interpolation="nearest")
        ax.set_title(f"{map_name}\nm={dim}, seed={best_seed}", fontsize=8)
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Pairwise latent distances, not transition matrices. Extra dimensions let branching constraints reorganize at lower gain.",
        y=0.995,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    paths = [output_dir / "distance_heatmaps.png", output_dir / "distance_heatmaps.pdf"]
    for path in paths:
        fig.savefig(path, dpi=260 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def make_plots(
    maps: Sequence[RoadMap],
    complexity_rows: List[Dict[str, object]],
    complexity_by_name: Dict[str, Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    required_rows: List[Dict[str, object]],
    output_dir: Path,
) -> List[Path]:
    paths: List[Path] = []
    maps_by_name = {road_map.name: road_map for road_map in maps}
    paths.extend(plot_map_gallery(maps, complexity_by_name, output_dir))
    paths.extend(plot_winding_frontier(summary_rows, output_dir))
    paths.extend(plot_branching_frontier(summary_rows, output_dir))
    paths.extend(plot_required_dimension(required_rows, output_dir))
    paths.extend(plot_physical_gain(complexity_rows, output_dir))
    paths.extend(plot_distance_heatmaps(maps_by_name, summary_rows, output_dir))
    return paths


def parse_m_values(value: Optional[str], quick: bool) -> List[int]:
    if value:
        out = [int(item) for item in value.replace(",", " ").split()]
    else:
        out = [1, 2, 4, 8, 16] if quick else [1, 2, 4, 8, 16, 32]
    if not out or any(dim <= 0 for dim in out):
        raise ValueError("m values must be positive integers.")
    return sorted(set(out))


def parse_gain_budgets(value: str) -> List[float]:
    budgets = [float(item) for item in value.replace(",", " ").split()]
    if not budgets:
        raise ValueError("At least one gain budget is required.")
    return budgets


def resolve_config(args: argparse.Namespace) -> Dict[str, object]:
    quick = bool(args.quick)
    num_states = args.num_states if args.num_states is not None else (64 if quick else 128)
    seeds = args.seeds if args.seeds is not None else (2 if quick else 5)
    steps = args.steps if args.steps is not None else (1500 if quick else 10000)
    horizon = args.horizon if args.horizon is not None else 8
    num_macro_actions = args.num_macro_actions if args.num_macro_actions is not None else (24 if quick else 64)
    m_values = parse_m_values(args.m_values, quick)
    training_pairs = args.training_pairs if args.training_pairs is not None else 0
    output_dir = Path(args.output_dir)
    if num_states < 32:
        raise ValueError("num_states must be at least 32 for the requested branching map family.")
    if seeds <= 0:
        raise ValueError("seeds must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    return {
        "quick": quick,
        "num_states": int(num_states),
        "seeds": int(seeds),
        "steps": int(steps),
        "horizon": int(horizon),
        "num_macro_actions": int(num_macro_actions),
        "m_values": m_values,
        "learning_rate": float(args.learning_rate),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "eval_every": int(args.eval_every),
        "training_pairs": int(training_pairs),
        "output_dir": output_dir,
        "macro_seed": int(args.macro_seed),
        "seed_base": int(args.seed_base),
        "gain_budgets": parse_gain_budgets(args.gain_budgets),
        "max_restarts": int(args.max_restarts),
        "make_plots": bool(args.plots),
    }


def validate_output_config(output_dir: Path, run_config: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    comparable_keys = [
        "quick",
        "num_states",
        "seeds",
        "steps",
        "horizon",
        "num_macro_actions",
        "m_values",
        "learning_rate",
        "gradient_clip_norm",
        "training_pairs",
        "macro_seed",
        "seed_base",
    ]
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        mismatches = [key for key in comparable_keys if existing.get(key) != run_config.get(key)]
        if mismatches:
            raise RuntimeError(
                f"{output_dir} already has outputs from a different run configuration ({', '.join(mismatches)} differ). "
                "Use a fresh --output-dir so quick/smoke and full outputs are not mixed."
            )
    _write_json(config_path, {key: value if not isinstance(value, Path) else str(value) for key, value in run_config.items()})


def print_complexity_table(complexity_rows: List[Dict[str, object]]) -> None:
    print("\nMap complexity and physical-coordinate gain")
    print("map_name             group              branch_excess  junctions  L_phys")
    for row in complexity_rows:
        print(
            f"{str(row['map_name']):20s} {str(row['map_group']):18s} "
            f"{int(row['branch_excess']):13d} {int(row['junction_count']):10d} {float(row['L_phys']):8.4g}"
        )


def print_best_gain_table(summary_rows: List[Dict[str, object]]) -> None:
    print("\nBest-found gain by map and dimension")
    map_names = sorted({str(row["map_name"]) for row in summary_rows}, key=lambda name: (str(next(row["map_group"] for row in summary_rows if row["map_name"] == name)), int(next(row["branch_excess"] for row in summary_rows if row["map_name"] == name)), name))
    dims = sorted({int(row["m"]) for row in summary_rows})
    header = "map_name".ljust(20) + "".join(f"m={dim}".rjust(12) for dim in dims)
    print(header)
    for map_name in map_names:
        cells = []
        for dim in dims:
            match = next((row for row in summary_rows if row["map_name"] == map_name and int(row["m"]) == dim), None)
            cells.append("".rjust(12) if match is None else f"{float(match['best_gain']):12.4g}")
        print(map_name.ljust(20) + "".join(cells))


def print_required_dimensions(required_rows: List[Dict[str, object]]) -> None:
    print("\nEmpirical required dimension by gain budget")
    budgets = sorted({float(row["gain_budget"]) for row in required_rows})
    map_names = sorted({str(row["map_name"]) for row in required_rows}, key=lambda name: (int(next(row["branch_excess"] for row in required_rows if row["map_name"] == name)), name))
    header = "map_name".ljust(20) + "branch_excess".rjust(15) + "".join(f"L0={budget:g}".rjust(12) for budget in budgets)
    print(header)
    for map_name in map_names:
        bx = int(next(row["branch_excess"] for row in required_rows if row["map_name"] == map_name))
        cells = []
        for budget in budgets:
            row = next(row for row in required_rows if row["map_name"] == map_name and float(row["gain_budget"]) == budget)
            cells.append(str(row["required_dimension_label"]).rjust(12))
        print(map_name.ljust(20) + f"{bx:15d}" + "".join(cells))


def print_output_paths(output_dir: Path, plot_paths: Sequence[Path]) -> None:
    paths = [
        output_dir / "run_config.json",
        output_dir / "macro_actions.json",
        output_dir / "map_complexity.csv",
        output_dir / "results.csv",
        output_dir / "summary.csv",
        output_dir / "required_dimension.csv",
    ]
    paths.extend(plot_paths)
    paths.append(output_dir / "embeddings")
    print("\nOutput paths")
    for path in paths:
        print(f"  {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled branching-maze dimension/gain experiment for predictive-realization. "
            "Uses fixed transition tables and float64 CPU latent-coordinate optimization."
        )
    )
    parser.add_argument("--quick", action="store_true", help="Use quick defaults: N=64, m=1,2,4,8,16, seeds=2, steps=1500, K=24.")
    parser.add_argument("--num-states", type=int, default=None, help="Number of road states per map.")
    parser.add_argument("--seeds", type=int, default=None, help="Number of optimization seeds per map and dimension.")
    parser.add_argument("--steps", type=int, default=None, help="Adam steps per seed.")
    parser.add_argument("--horizon", type=int, default=None, help="Fixed macro-action horizon H.")
    parser.add_argument("--num-macro-actions", type=int, default=None, help="Number of fixed macro-action blocks K.")
    parser.add_argument("--m-values", default=None, help="Comma/space-separated latent dimensions.")
    parser.add_argument("--training-pairs", type=int, default=None, help="Sample this many unordered pairs for optimization; 0 uses all pairs.")
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--macro-seed", type=int, default=1729)
    parser.add_argument("--seed-base", type=int, default=20260729)
    parser.add_argument("--gain-budgets", default="1.05,1.1,1.2,1.5")
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--plots", dest="plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_config = resolve_config(args)
    output_dir = Path(run_config["output_dir"])
    validate_output_config(output_dir, run_config)
    _validate_hard_gain_example()

    num_states = int(run_config["num_states"])
    m_values = list(run_config["m_values"])
    seeds = list(range(int(run_config["seeds"])))
    maps = build_maps(num_states)
    validate_map_family(maps, num_states)
    pair_i_np, pair_j_np = all_unordered_pairs(num_states)
    macro_actions = generate_macro_actions(
        horizon=int(run_config["horizon"]),
        num_macro_actions=int(run_config["num_macro_actions"]),
        seed=int(run_config["macro_seed"]),
    )
    save_macro_actions(output_dir / "macro_actions.json", macro_actions, int(run_config["macro_seed"]))

    macro_transitions: Dict[str, np.ndarray] = {}
    complexity_by_name: Dict[str, Dict[str, object]] = {}
    complexity_rows: List[Dict[str, object]] = []
    for road_map in maps:
        T_one = build_transition_table(road_map)
        macro_next = compose_macro_transitions(T_one, macro_actions)
        macro_transitions[road_map.name] = macro_next
        L_phys = physical_coordinate_gain(road_map, macro_next, pair_i_np, pair_j_np)
        row = compute_complexity(road_map, L_phys=L_phys)
        complexity_by_name[road_map.name] = row
        complexity_rows.append(row)

    _write_csv(output_dir / "map_complexity.csv", complexity_rows, COMPLEXITY_FIELDS)

    opt_config = OptimizationConfig(
        steps=int(run_config["steps"]),
        learning_rate=float(run_config["learning_rate"]),
        eval_every=int(run_config["eval_every"]),
        gradient_clip_norm=float(run_config["gradient_clip_norm"]),
        training_pairs=int(run_config["training_pairs"]),
        max_restarts=int(run_config["max_restarts"]),
        seed_base=int(run_config["seed_base"]),
    )

    results: List[Dict[str, object]] = []
    embedding_dir = output_dir / "embeddings"
    total_runs = len(maps) * len(m_values) * len(seeds)
    run_idx = 0
    for road_map in maps:
        metrics = complexity_by_name[road_map.name]
        macro_next = macro_transitions[road_map.name]
        for seed in seeds:
            train_i_np, train_j_np = _sample_training_pairs(
                pair_i_np,
                pair_j_np,
                opt_config.training_pairs,
                road_map.name,
                seed,
                opt_config.seed_base,
            )
            for dim in m_values:
                run_idx += 1
                print(
                    f"[branching_maze] {run_idx}/{total_runs}: map={road_map.name} m={dim} seed={seed}",
                    flush=True,
                )
                opt_row = optimize_embedding(
                    road_map=road_map,
                    macro_next_np=macro_next,
                    pair_i_np=pair_i_np,
                    pair_j_np=pair_j_np,
                    train_i_np=train_i_np,
                    train_j_np=train_j_np,
                    dim=dim,
                    seed=seed,
                    config=opt_config,
                    embedding_dir=embedding_dir,
                )
                result_row = {
                    "map_name": road_map.name,
                    "map_group": road_map.group,
                    "num_states": num_states,
                    "branch_excess": int(metrics["branch_excess"]),
                    "junction_count": int(metrics["junction_count"]),
                    "dead_end_count": int(metrics["dead_end_count"]),
                    "max_degree": int(metrics["max_degree"]),
                    "cycle_rank": int(metrics["cycle_rank"]),
                    "graph_diameter": float(metrics["graph_diameter"]),
                    "q95_geodesic_over_euclidean": float(metrics["q95_geodesic_over_euclidean"]),
                    "horizon": int(run_config["horizon"]),
                    "num_macro_actions": int(run_config["num_macro_actions"]),
                    "L_phys": float(metrics["L_phys"]),
                    **opt_row,
                }
                results.append(result_row)
                _write_csv(output_dir / "results.csv", results, RESULT_FIELDS)

    summary_rows = summarize_results(results)
    _write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    required_rows = required_dimension_rows(summary_rows, budgets=list(run_config["gain_budgets"]))
    _write_csv(
        output_dir / "required_dimension.csv",
        required_rows,
        [
            "gain_budget",
            "map_name",
            "branch_excess",
            "junction_count",
            "required_dimension",
            "required_dimension_label",
            "required_dimension_capped",
            "reached_budget",
        ],
    )

    plot_paths: List[Path] = []
    if bool(run_config["make_plots"]):
        plot_paths = make_plots(maps, complexity_rows, complexity_by_name, summary_rows, required_rows, output_dir)

    print_output_paths(output_dir, plot_paths)
    print_complexity_table(complexity_rows)
    print_best_gain_table(summary_rows)
    print_required_dimensions(required_rows)
    print(
        "\nInterpretation note: optimized gains are best-found empirical upper bounds from this "
        "coordinate search, not certified global optima.",
        flush=True,
    )


if __name__ == "__main__":
    main()
