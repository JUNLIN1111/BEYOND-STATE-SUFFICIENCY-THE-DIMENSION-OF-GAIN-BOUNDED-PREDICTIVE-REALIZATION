#!/usr/bin/env python3
"""Standalone direct-realization experiment for the 8x8 maze study.

Copy this one file to another machine/project and run it with a Python
environment that has PyTorch installed. Matplotlib is optional unless plots are
enabled.

Examples:
    python direct_maze_realization.py --self-test
    python direct_maze_realization.py --pilot --device cpu
    python direct_maze_realization.py --random-restarts 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import torch


N = 8
NUM_STATES = N * N
NUM_PAIRS = NUM_STATES * (NUM_STATES - 1) // 2
ACTION_DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
ACTION_ORDER = ("U", "D", "L", "R")
NUM_CONSTRAINTS = len(ACTION_ORDER) * NUM_PAIRS
START = (0, 0)
GOAL = (7, 7)
MAZE_NAMES = ("A", "B", "C")

A_WALLS = frozenset(
    {
        ("V", 1, 3),
        ("V", 2, 3),
        ("V", 3, 3),
        ("H", 4, 4),
        ("H", 4, 5),
        ("H", 4, 6),
    }
)

B_WALLS = frozenset(
    {
        ("V", 0, 1),
        ("V", 1, 1),
        ("V", 1, 2),
        ("V", 1, 4),
        ("V", 2, 1),
        ("V", 3, 1),
        ("V", 3, 5),
        ("V", 4, 1),
        ("V", 4, 5),
        ("V", 5, 5),
        ("V", 6, 5),
        ("V", 7, 4),
        ("H", 0, 1),
        ("H", 0, 2),
        ("H", 1, 3),
        ("H", 1, 4),
        ("H", 1, 5),
        ("H", 2, 3),
        ("H", 4, 5),
        ("H", 4, 7),
        ("H", 5, 1),
        ("H", 5, 2),
        ("H", 5, 3),
        ("H", 6, 5),
    }
)

C_WALLS = frozenset(
    {
        ("H", 0, 0),
        ("H", 1, 1),
        ("H", 1, 4),
        ("H", 1, 5),
        ("H", 1, 7),
        ("H", 2, 5),
        ("H", 3, 1),
        ("H", 3, 4),
        ("H", 3, 5),
        ("H", 4, 5),
        ("H", 5, 1),
        ("H", 5, 3),
        ("H", 5, 4),
        ("H", 6, 1),
        ("H", 6, 3),
        ("H", 6, 5),
        ("H", 6, 6),
        ("V", 0, 1),
        ("V", 0, 2),
        ("V", 0, 3),
        ("V", 0, 4),
        ("V", 0, 5),
        ("V", 0, 6),
        ("V", 1, 1),
        ("V", 2, 0),
        ("V", 2, 5),
        ("V", 3, 0),
        ("V", 3, 6),
        ("V", 4, 1),
        ("V", 4, 2),
        ("V", 4, 3),
        ("V", 4, 5),
        ("V", 4, 6),
        ("V", 5, 0),
        ("V", 5, 6),
        ("V", 6, 3),
        ("V", 6, 4),
        ("V", 6, 5),
        ("V", 6, 6),
        ("V", 7, 1),
        ("V", 7, 6),
    }
)

WALLS_BY_MAZE = {"A": A_WALLS, "B": B_WALLS, "C": C_WALLS}
EXPECTED_STATS = {
    "A": {"barriers": 6, "reachable": 64, "shortest_path": 14, "graph_diameter": 14},
    "B": {"barriers": 24, "reachable": 64, "shortest_path": 20, "graph_diameter": 21},
    "C": {"barriers": 41, "reachable": 64, "shortest_path": 22, "graph_diameter": 22},
}
CERTIFIED_GAIN_ONE_LOWER_BOUNDS = {"A": 29, "B": 47, "C": 52}
DIMENSIONS = [2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 28, 29, 30, 40, 46, 47, 48, 51, 52, 53, 63]
DEFAULT_DIMENSION_ARG = ",".join(str(d) for d in DIMENSIONS)
TAU_SCHEDULE = [0.10, 0.03, 0.01, 0.003]
CHECKPOINT_VERSION = "maze_frontier_v2"
OBJECTIVE_VERSION = "direct_logmeanexp_squared_dist_v1"
NORMALIZATION_METHOD = "center_then_global_rms"
CONTINUATION_METHOD_VERSION = "baseline_eval_plus_jittered_v1"
RAW_FIELDS = [
    "maze",
    "dimension",
    "seed",
    "initialization_type",
    "checkpoint_version",
    "hard_gain",
    "best_step",
    "best_tau",
    "best_hard_gain",
    "surrogate_loss_at_best",
    "min_pair_distance",
    "max_pair_distance",
    "aspect_ratio",
    "rms_norm",
    "num_steps",
    "status",
]
SUMMARY_FIELDS = [
    "maze",
    "dimension",
    "best_hard_gain",
    "best_initialization_type",
    "median_random_restart_hard_gain",
    "q25_random_restart_hard_gain",
    "q75_random_restart_hard_gain",
    "best_min_distance",
    "best_aspect_ratio",
    "best_status",
]
TRACE_FIELDS = [
    "maze",
    "dimension",
    "seed",
    "initialization_type",
    "step",
    "tau",
    "surrogate_loss",
    "hard_gain",
    "min_pair_distance",
    "max_pair_distance",
    "aspect_ratio",
    "rms_norm",
    "status",
]


@dataclass(frozen=True)
class Config:
    mazes: tuple[str, ...] = MAZE_NAMES
    dimensions: tuple[int, ...] = tuple(DIMENSIONS)
    random_restarts: int = 10
    steps_per_stage: int = 2500
    tau_schedule: tuple[float, ...] = tuple(TAU_SCHEDULE)
    lr: float = 0.05
    continuation_jitter_std: float = 1e-3
    eval_every: int = 100
    base_seed: int = 0
    floor: float = 1e-300
    degeneracy_tol: float = 1e-12
    below_one_tol: float = 1e-6
    allow_below_one: bool = False
    device: str = "auto"
    output_dir: str = "outputs/direct_maze_realization"
    make_plots: bool = True
    resume: bool = True


@dataclass
class GainResult:
    hard_gain: float
    min_pair_distance: float
    max_pair_distance: float
    aspect_ratio: float
    rms_norm: float
    degenerate: bool
    num_constraints: int = NUM_CONSTRAINTS
    num_zero_successor_constraints: int = 0


def states() -> list[tuple[int, int]]:
    return [(r, c) for r in range(N) for c in range(N)]


def state_id(s: tuple[int, int]) -> int:
    return N * s[0] + s[1]


def in_bounds(s: tuple[int, int]) -> bool:
    return 0 <= s[0] < N and 0 <= s[1] < N


def canon(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
        raise ValueError(f"blocked edge must connect neighboring cells: {a}, {b}")
    if not in_bounds(a) or not in_bounds(b):
        raise ValueError(f"blocked edge must be internal: {a}, {b}")
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def barrier_to_edge(barrier: tuple[str, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    kind, r, c = barrier
    if kind == "V":
        return canon((r, c), (r, c + 1))
    if kind == "H":
        return canon((r, c), (r + 1, c))
    raise ValueError(f"unknown wall kind: {kind}")


def action_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    delta = (b[0] - a[0], b[1] - a[1])
    for action, action_delta in ACTION_DELTAS.items():
        if action_delta == delta:
            return action
    raise AssertionError(f"states are not adjacent: {a}, {b}")


def blocked_edges(maze_name: str) -> frozenset[tuple[tuple[int, int], tuple[int, int]]]:
    return frozenset(barrier_to_edge(wall) for wall in WALLS_BY_MAZE[maze_name])


def transition(maze_name: str, s: tuple[int, int], action: str) -> tuple[int, int]:
    dr, dc = ACTION_DELTAS[action]
    nxt = (s[0] + dr, s[1] + dc)
    if not in_bounds(nxt):
        return s
    if canon(s, nxt) in blocked_edges(maze_name):
        return s
    return nxt


def graph_neighbors(maze_name: str, s: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for action in ACTION_ORDER:
        nxt = transition(maze_name, s, action)
        if nxt != s:
            out.append(nxt)
    return out


def reachable_count(maze_name: str) -> int:
    seen = {START}
    queue = deque([START])
    while queue:
        cur = queue.popleft()
        for nxt in graph_neighbors(maze_name, cur):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def shortest_path_length(maze_name: str, start: tuple[int, int] = START, goal: tuple[int, int] = GOAL) -> int:
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cur, dist = queue.popleft()
        if cur == goal:
            return dist
        for nxt in graph_neighbors(maze_name, cur):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    raise AssertionError(f"{maze_name}: goal not reachable")


def graph_diameter(maze_name: str) -> int:
    max_dist = 0
    for root in states():
        seen = {root}
        queue = deque([(root, 0)])
        while queue:
            cur, dist = queue.popleft()
            max_dist = max(max_dist, dist)
            for nxt in graph_neighbors(maze_name, cur):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        if len(seen) != NUM_STATES:
            raise AssertionError(f"{maze_name}: graph is disconnected from {root}")
    return max_dist


def build_transition_table(maze_name: str) -> torch.Tensor:
    table = torch.empty((len(ACTION_ORDER), NUM_STATES), dtype=torch.long)
    for action_idx, action in enumerate(ACTION_ORDER):
        for s in states():
            table[action_idx, state_id(s)] = state_id(transition(maze_name, s, action))
    validate_maze(maze_name, table)
    return table


def validate_maze(maze_name: str, table: torch.Tensor | None = None) -> dict[str, int]:
    if maze_name not in WALLS_BY_MAZE:
        raise ValueError(f"unknown maze: {maze_name}")
    edges = blocked_edges(maze_name)
    if len(edges) != len(WALLS_BY_MAZE[maze_name]):
        raise AssertionError(f"{maze_name}: wall encoding produced duplicate edges")
    if table is not None:
        if tuple(table.shape) != (4, 64) or table.dtype != torch.long:
            raise AssertionError(f"{maze_name}: transition table must be int64 [4,64]")
        if int(table.min()) < 0 or int(table.max()) >= NUM_STATES:
            raise AssertionError(f"{maze_name}: invalid successor index")
        for s, action in [((0, 0), "U"), ((0, 0), "L"), ((7, 7), "D"), ((7, 7), "R")]:
            if int(table[ACTION_ORDER.index(action), state_id(s)]) != state_id(s):
                raise AssertionError(f"{maze_name}: boundary move {s},{action} must self-loop")
    for wall in WALLS_BY_MAZE[maze_name]:
        a, b = barrier_to_edge(wall)
        if b in graph_neighbors(maze_name, a) or a in graph_neighbors(maze_name, b):
            raise AssertionError(f"{maze_name}: wall did not block both directions: {wall}")
        if table is not None:
            for src, dst in ((a, b), (b, a)):
                action = action_between(src, dst)
                if int(table[ACTION_ORDER.index(action), state_id(src)]) != state_id(src):
                    raise AssertionError(f"{maze_name}: blocked move {src},{action} must self-loop")
    got = {
        "barriers": len(WALLS_BY_MAZE[maze_name]),
        "reachable": reachable_count(maze_name),
        "shortest_path": shortest_path_length(maze_name),
        "graph_diameter": graph_diameter(maze_name),
    }
    for key, expected in EXPECTED_STATS[maze_name].items():
        if got[key] != expected:
            raise AssertionError(f"{maze_name} {key}: expected {expected}, got {got[key]}")
    return got


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def unordered_pair_indices(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    pair_i, pair_j = torch.triu_indices(NUM_STATES, NUM_STATES, offset=1, device=device)
    return pair_i.long(), pair_j.long()


def normalize_gauge_(z: torch.Tensor, floor: float = 1e-300) -> torch.Tensor:
    with torch.no_grad():
        z.sub_(z.mean(dim=0, keepdim=True))
        rms = torch.sqrt(torch.mean(torch.sum(z * z, dim=1)))
        if not torch.isfinite(rms) or float(rms) <= floor:
            raise FloatingPointError("cannot normalize zero or non-finite embedding")
        z.div_(rms)
    return z


def random_embedding(dimension: int, seed: int, device: torch.device, floor: float = 1e-300) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    z = torch.randn((NUM_STATES, dimension), dtype=torch.float64, device=device, generator=gen)
    return normalize_gauge_(z, floor=floor)


def regular_simplex(device: torch.device | None = None) -> torch.Tensor:
    eye = torch.eye(NUM_STATES, dtype=torch.float64, device=device)
    z = eye - torch.full((NUM_STATES, NUM_STATES), 1.0 / NUM_STATES, dtype=torch.float64, device=device)
    q, _ = torch.linalg.qr(z.T[:, :-1], mode="reduced")
    return normalize_gauge_(z @ q)


def pair_diagnostics(z: torch.Tensor, degeneracy_tol: float = 1e-12) -> GainResult:
    z = z.detach().to(dtype=torch.float64)
    if tuple(z.shape[:1]) != (NUM_STATES,):
        raise AssertionError("embedding must have one row per state")
    if not torch.isfinite(z).all():
        return GainResult(float("inf"), float("nan"), float("nan"), float("inf"), float("nan"), True)
    dists = torch.pdist(z, p=2)
    min_dist = float(dists.min().item())
    max_dist = float(dists.max().item())
    rms = float(torch.sqrt(torch.mean(torch.sum(z * z, dim=1))).item())
    degenerate = min_dist <= degeneracy_tol or not math.isfinite(min_dist)
    aspect = max_dist / min_dist if min_dist > 0.0 else float("inf")
    return GainResult(float("nan"), min_dist, max_dist, aspect, rms, degenerate)


def exact_hard_gain(z: torch.Tensor, transitions: torch.Tensor, degeneracy_tol: float = 1e-12) -> GainResult:
    """Recompute all 8064 primitive-action ratios from a frozen embedding."""

    with torch.no_grad():
        z = z.detach().to(dtype=torch.float64)
        transitions = transitions.to(device=z.device, dtype=torch.long)
        result = pair_diagnostics(z, degeneracy_tol=degeneracy_tol)
        if result.degenerate:
            result.hard_gain = float("inf")
            return result
        pair_i, pair_j = unordered_pair_indices(z.device)
        denom = torch.linalg.norm(z[pair_i] - z[pair_j], dim=1)
        ratios = []
        zero_successors = 0
        for action_idx in range(len(ACTION_ORDER)):
            succ_i = transitions[action_idx, pair_i]
            succ_j = transitions[action_idx, pair_j]
            coincident = succ_i == succ_j
            zero_successors += int(coincident.sum().item())
            action_ratios = torch.zeros_like(denom)
            valid = ~coincident
            action_ratios[valid] = torch.linalg.norm(z[succ_i[valid]] - z[succ_j[valid]], dim=1) / denom[valid]
            ratios.append(action_ratios)
        all_ratios = torch.cat(ratios)
        result.hard_gain = float(all_ratios.max().item())
        result.num_constraints = int(all_ratios.numel())
        result.num_zero_successor_constraints = zero_successors
        if result.num_constraints != NUM_CONSTRAINTS:
            raise AssertionError(f"expected {NUM_CONSTRAINTS} constraints, got {result.num_constraints}")
        return result


def surrogate_loss(z: torch.Tensor, transitions: torch.Tensor, tau: float, floor: float = 1e-300) -> torch.Tensor:
    """Smooth log-mean-exp max over non-coincident successor constraints."""

    transitions = transitions.to(device=z.device, dtype=torch.long)
    pair_i, pair_j = unordered_pair_indices(z.device)
    base_d2 = torch.sum((z[pair_i] - z[pair_j]) ** 2, dim=1).clamp_min(floor)
    ell_parts = []
    for action_idx in range(len(ACTION_ORDER)):
        succ_i = transitions[action_idx, pair_i]
        succ_j = transitions[action_idx, pair_j]
        mask = succ_i != succ_j
        succ_d2 = torch.sum((z[succ_i[mask]] - z[succ_j[mask]]) ** 2, dim=1).clamp_min(floor)
        ell_parts.append(0.5 * (torch.log(succ_d2) - torch.log(base_d2[mask])))
    ell = torch.cat(ell_parts)
    return tau * (torch.logsumexp(ell / tau, dim=0) - math.log(ell.numel()))


def status_from_gain(result: GainResult, below_one_tol: float) -> str:
    if result.degenerate or not math.isfinite(result.hard_gain):
        return "numerically_degenerate"
    if result.hard_gain < 1.0 - below_one_tol:
        return "below_one_audit"
    return "ok"


def checkpoint_metadata(maze_name: str, dimension: int, config: Config) -> dict[str, object]:
    return {
        "maze": maze_name,
        "dimension": dimension,
        "checkpoint_version": CHECKPOINT_VERSION,
        "action_family": list(ACTION_ORDER),
        "dtype": "torch.float64",
        "tau_schedule": list(config.tau_schedule),
        "steps_per_stage": config.steps_per_stage,
        "learning_rate": config.lr,
        "random_restarts": config.random_restarts,
        "base_seed": config.base_seed,
        "normalization_method": NORMALIZATION_METHOD,
        "degeneracy_tolerance": config.degeneracy_tol,
        "objective_surrogate_version": OBJECTIVE_VERSION,
        "continuation_method_version": CONTINUATION_METHOD_VERSION,
        "continuation_jitter_std": config.continuation_jitter_std,
    }


def checkpoint_is_compatible(
    ckpt: dict[str, object],
    maze_name: str,
    dimension: int,
    config: Config,
) -> tuple[bool, str]:
    metadata = ckpt.get("metadata")
    if not isinstance(metadata, dict):
        return False, "missing metadata"
    expected = checkpoint_metadata(maze_name, dimension, config)
    for key, expected_value in expected.items():
        got = metadata.get(key)
        if got != expected_value:
            return False, f"{key} mismatch: expected {expected_value!r}, got {got!r}"
    return True, "ok"


def optimize_embedding(
    initial_z: torch.Tensor,
    transitions: torch.Tensor,
    config: Config,
    seed: int,
    initialization_type: str,
) -> tuple[torch.Tensor, dict[str, object], list[dict[str, object]]]:
    if seed >= 0:
        torch.manual_seed(seed)
        if initial_z.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    z = torch.nn.Parameter(initial_z.detach().clone().to(dtype=torch.float64))
    normalize_gauge_(z.data, floor=config.floor)
    transitions = transitions.to(device=z.device, dtype=torch.long)
    opt = torch.optim.Adam([z], lr=config.lr)
    total_steps = config.steps_per_stage * len(config.tau_schedule)

    best_z = z.detach().clone()
    best_result = exact_hard_gain(z, transitions, config.degeneracy_tol)
    best_step = 0
    best_tau = float(config.tau_schedule[0])
    best_surrogate = float(surrogate_loss(z, transitions, best_tau, config.floor).detach().cpu())
    ever_degenerate = best_result.degenerate
    ever_below_one = best_result.hard_gain < 1.0 - config.below_one_tol
    traces: list[dict[str, object]] = []

    def record(step: int, tau: float, loss_value: float, result: GainResult) -> None:
        traces.append(
            {
                "step": step,
                "tau": tau,
                "surrogate_loss": loss_value,
                "hard_gain": result.hard_gain,
                "min_pair_distance": result.min_pair_distance,
                "max_pair_distance": result.max_pair_distance,
                "aspect_ratio": result.aspect_ratio,
                "rms_norm": result.rms_norm,
                "status": status_from_gain(result, config.below_one_tol),
            }
        )

    record(0, best_tau, best_surrogate, best_result)
    step = 0
    for tau in config.tau_schedule:
        for _ in range(config.steps_per_stage):
            step += 1
            opt.zero_grad(set_to_none=True)
            loss = surrogate_loss(z, transitions, tau, config.floor)
            loss.backward()
            opt.step()
            normalize_gauge_(z.data, floor=config.floor)
            if step % config.eval_every == 0 or step == total_steps:
                result = exact_hard_gain(z, transitions, config.degeneracy_tol)
                loss_value = float(loss.detach().cpu())
                record(step, float(tau), loss_value, result)
                ever_degenerate = ever_degenerate or result.degenerate
                ever_below_one = ever_below_one or result.hard_gain < 1.0 - config.below_one_tol
                if result.hard_gain < best_result.hard_gain:
                    best_z = z.detach().clone()
                    best_result = result
                    best_step = step
                    best_tau = float(tau)
                    best_surrogate = loss_value

    status = status_from_gain(best_result, config.below_one_tol)
    if ever_degenerate:
        status = "numerically_degenerate"
    elif ever_below_one:
        status = "below_one_audit"
    row = {
        "seed": seed,
        "initialization_type": initialization_type,
        "checkpoint_version": CHECKPOINT_VERSION,
        "hard_gain": best_result.hard_gain,
        "best_step": best_step,
        "best_tau": best_tau,
        "best_hard_gain": best_result.hard_gain,
        "surrogate_loss_at_best": best_surrogate,
        "min_pair_distance": best_result.min_pair_distance,
        "max_pair_distance": best_result.max_pair_distance,
        "aspect_ratio": best_result.aspect_ratio,
        "rms_norm": best_result.rms_norm,
        "num_steps": total_steps,
        "status": status,
    }
    return best_z.detach().clone(), row, traces


def evaluate_candidate(
    z: torch.Tensor,
    transitions: torch.Tensor,
    config: Config,
    seed: int,
    initialization_type: str,
) -> tuple[torch.Tensor, dict[str, object], list[dict[str, object]]]:
    z = normalize_gauge_(z.detach().clone().to(dtype=torch.float64), floor=config.floor)
    transitions = transitions.to(device=z.device)
    result = exact_hard_gain(z, transitions, config.degeneracy_tol)
    loss = float(surrogate_loss(z, transitions, config.tau_schedule[-1], config.floor).detach().cpu())
    status = status_from_gain(result, config.below_one_tol)
    row = {
        "seed": seed,
        "initialization_type": initialization_type,
        "checkpoint_version": CHECKPOINT_VERSION,
        "hard_gain": result.hard_gain,
        "best_step": 0,
        "best_tau": config.tau_schedule[-1],
        "best_hard_gain": result.hard_gain,
        "surrogate_loss_at_best": loss,
        "min_pair_distance": result.min_pair_distance,
        "max_pair_distance": result.max_pair_distance,
        "aspect_ratio": result.aspect_ratio,
        "rms_norm": result.rms_norm,
        "num_steps": 0,
        "status": status,
    }
    trace = {
        "step": 0,
        "tau": config.tau_schedule[-1],
        "surrogate_loss": loss,
        "hard_gain": result.hard_gain,
        "min_pair_distance": result.min_pair_distance,
        "max_pair_distance": result.max_pair_distance,
        "aspect_ratio": result.aspect_ratio,
        "rms_norm": result.rms_norm,
        "status": status,
    }
    return z, row, [trace]


def pad_embedding(z: torch.Tensor, high_dimension: int) -> torch.Tensor:
    if z.shape[1] > high_dimension:
        raise ValueError("cannot pad to a lower dimension")
    zeros = torch.zeros((z.shape[0], high_dimension - z.shape[1]), dtype=z.dtype, device=z.device)
    return torch.cat([z.detach(), zeros], dim=1)


def jittered_continuation_embedding(
    z: torch.Tensor,
    high_dimension: int,
    jitter_std: float,
    seed: int,
    device: torch.device,
    floor: float = 1e-300,
) -> torch.Tensor:
    if z.shape[1] >= high_dimension:
        raise ValueError("jittered continuation requires a strictly higher dimension")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    base = z.detach().to(device=device, dtype=torch.float64)
    extra = jitter_std * torch.randn(
        (z.shape[0], high_dimension - z.shape[1]),
        dtype=torch.float64,
        device=device,
        generator=gen,
    )
    return normalize_gauge_(torch.cat([base, extra], dim=1), floor=floor)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def save_rows(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def best_z_path(output_dir: Path, maze_name: str, dimension: int) -> Path:
    return output_dir / f"best_Z_maze_{maze_name}_m{dimension}.pt"


def verified_checkpoint_row(
    path: Path,
    maze_name: str,
    dimension: int,
    transitions: torch.Tensor,
    config: Config,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, object] | None, str]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        return None, None, "checkpoint is not a dictionary"
    compatible, reason = checkpoint_is_compatible(ckpt, maze_name, dimension, config)
    if not compatible:
        return None, None, reason
    if "Z" not in ckpt:
        raise AssertionError(f"{path}: missing Z")
    z = ckpt["Z"].detach().to(device=device, dtype=torch.float64)
    if z.shape != (NUM_STATES, dimension):
        raise AssertionError(f"{path}: expected Z shape {(NUM_STATES, dimension)}, got {tuple(z.shape)}")
    result = exact_hard_gain(z, transitions, config.degeneracy_tol)
    loss = float(surrogate_loss(z, transitions, config.tau_schedule[-1], config.floor).detach().cpu())
    status = status_from_gain(result, config.below_one_tol)
    row = {
        "maze": maze_name,
        "dimension": dimension,
        "seed": -3,
        "initialization_type": "resume_existing_best",
        "checkpoint_version": CHECKPOINT_VERSION,
        "hard_gain": result.hard_gain,
        "best_step": 0,
        "best_tau": config.tau_schedule[-1],
        "best_hard_gain": result.hard_gain,
        "surrogate_loss_at_best": loss,
        "min_pair_distance": result.min_pair_distance,
        "max_pair_distance": result.max_pair_distance,
        "aspect_ratio": result.aspect_ratio,
        "rms_norm": result.rms_norm,
        "num_steps": 0,
        "status": status,
    }
    return z.detach().cpu(), row, "ok"


def flush_outputs(
    output_dir: Path,
    raw_rows: list[dict[str, object]],
    trace_rows: list[dict[str, object]],
    config: Config,
) -> None:
    raw_csv = output_dir / "raw_restart_results.csv"
    summary_csv = output_dir / "summary.csv"
    trace_csv = output_dir / "training_traces.csv"
    save_rows(raw_csv, raw_rows, RAW_FIELDS)
    save_rows(summary_csv, build_summary(raw_rows), SUMMARY_FIELDS)
    save_rows(trace_csv, trace_rows, TRACE_FIELDS)
    (output_dir / "random_seeds.json").write_text(
        json.dumps({"base_seed": config.base_seed, "random_restart_seeds": [config.base_seed + i for i in range(config.random_restarts)]}, indent=2)
    )


def prune_incompatible_resume_rows(
    output_dir: Path,
    raw_rows: list[dict[str, object]],
    trace_rows: list[dict[str, object]],
    config: Config,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    compatible_keys: set[tuple[str, int]] = set()
    candidate_keys = {
        (str(row.get("maze")), int(row.get("dimension", -1)))
        for row in raw_rows
        if row.get("checkpoint_version") == CHECKPOINT_VERSION
    }
    for maze_name, dimension in sorted(candidate_keys):
        if maze_name not in config.mazes or dimension not in config.dimensions:
            continue
        path = best_z_path(output_dir, maze_name, dimension)
        if not path.exists():
            continue
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and checkpoint_is_compatible(ckpt, maze_name, dimension, config)[0]:
            compatible_keys.add((maze_name, dimension))
    pruned_raw = [
        row
        for row in raw_rows
        if (str(row.get("maze")), int(row.get("dimension", -1))) in compatible_keys
    ]
    pruned_trace = [
        row
        for row in trace_rows
        if (str(row.get("maze")), int(row.get("dimension", -1))) in compatible_keys
    ]
    dropped = len(raw_rows) - len(pruned_raw)
    if dropped:
        print(f"resume: dropped {dropped} stale/incompatible raw CSV rows")
    return pruned_raw, pruned_trace


def build_summary(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["maze"]), int(row["dimension"])), []).append(row)
    out = []
    for (maze_name, dimension), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        ok_rows = [r for r in rows if r["status"] == "ok" and math.isfinite(float(r["best_hard_gain"]))]
        best = min(ok_rows or rows, key=lambda r: float(r["best_hard_gain"]))
        random_values = sorted(
            float(r["best_hard_gain"])
            for r in rows
            if r["initialization_type"] == "random" and r["status"] == "ok" and math.isfinite(float(r["best_hard_gain"]))
        )
        out.append(
            {
                "maze": maze_name,
                "dimension": dimension,
                "best_hard_gain": best["best_hard_gain"],
                "best_initialization_type": best["initialization_type"],
                "median_random_restart_hard_gain": median(random_values) if random_values else float("nan"),
                "q25_random_restart_hard_gain": quantile(random_values, 0.25),
                "q75_random_restart_hard_gain": quantile(random_values, 0.75),
                "best_min_distance": best["min_pair_distance"],
                "best_aspect_ratio": best["aspect_ratio"],
                "best_status": best["status"],
            }
        )
    return out


def candidate_table_text(rows: list[dict[str, object]]) -> str:
    lines = ["initialization_type | seed | hard_gain | status | aspect_ratio"]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("initialization_type")),
                    str(row.get("seed")),
                    str(row.get("best_hard_gain", row.get("hard_gain"))),
                    str(row.get("status")),
                    str(row.get("aspect_ratio")),
                ]
            )
        )
    return "\n".join(lines)


def load_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except ValueError:
        return float("nan")


def make_plots(summary_csv: Path, trace_csv: Path, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    colors = {"A": "#1b9e77", "B": "#d95f02", "C": "#7570b3"}

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for idx, maze_name in enumerate(MAZE_NAMES):
        maze_rows = sorted([r for r in rows if r["maze"] == maze_name], key=lambda r: int(r["dimension"]))
        if not maze_rows:
            continue
        xs = [int(r["dimension"]) for r in maze_rows]
        best = [load_float(r, "best_hard_gain") for r in maze_rows]
        med = [load_float(r, "median_random_restart_hard_gain") for r in maze_rows]
        q25 = [load_float(r, "q25_random_restart_hard_gain") for r in maze_rows]
        q75 = [load_float(r, "q75_random_restart_hard_gain") for r in maze_rows]
        color = colors[maze_name]
        ax.plot(xs, best, marker="o", linewidth=2.0, color=color, label=f"Maze {maze_name} best")
        if any(math.isfinite(v) for v in med):
            ax.plot(xs, med, linestyle="--", linewidth=1.2, color=color, alpha=0.8)
            ax.fill_between(xs, q25, q75, color=color, alpha=0.13, linewidth=0)
        bound = CERTIFIED_GAIN_ONE_LOWER_BOUNDS[maze_name]
        ax.axvspan(min(xs) - 0.5, bound - 0.5, ymin=0.02 + idx * 0.04, ymax=0.045 + idx * 0.04, color=color, alpha=0.35)
        ax.axvline(bound, color=color, linestyle=":", linewidth=1.0, alpha=0.75)
    ax.axhline(1.0, color="black", linewidth=1.0, label="gain = 1")
    ax.set_xlabel("ambient dimension m")
    ax.set_ylabel("exact hard gain")
    ax.set_title("Constructive direct-realization frontier")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.01, 0.98, "Bottom bars mark certified infeasible regions m < bound.", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "dimension_gain.pdf")
    fig.savefig(output_dir / "dimension_gain.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for maze_name in MAZE_NAMES:
        maze_rows = sorted([r for r in rows if r["maze"] == maze_name], key=lambda r: int(r["dimension"]))
        if maze_rows:
            ax.plot(
                [int(r["dimension"]) for r in maze_rows],
                [load_float(r, "best_aspect_ratio") for r in maze_rows],
                marker="o",
                linewidth=2.0,
                color=colors[maze_name],
                label=f"Maze {maze_name}",
            )
    ax.set_xlabel("ambient dimension m")
    ax.set_ylabel("aspect ratio")
    ax.set_yscale("log")
    ax.set_title("Embedding degeneration diagnostic")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "aspect_ratio.pdf")
    fig.savefig(output_dir / "aspect_ratio.png", dpi=300)
    plt.close(fig)

    if not trace_csv.exists():
        return
    with trace_csv.open(newline="") as f:
        traces = list(csv.DictReader(f))
    targets = {"A": 28, "B": 46, "C": 51}
    selected = [r for r in traces if int(r["dimension"]) == targets.get(r["maze"], -1)]
    if not selected:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for maze_name, dimension in targets.items():
        maze_rows = sorted(
            [r for r in selected if r["maze"] == maze_name and r["initialization_type"] == "random" and int(r["seed"]) == 0],
            key=lambda r: int(r["step"]),
        )
        if maze_rows:
            xs = [int(r["step"]) for r in maze_rows]
            axes[0].plot(xs, [load_float(r, "hard_gain") for r in maze_rows], color=colors[maze_name], label=f"{maze_name}, m={dimension}")
            axes[1].plot(xs, [load_float(r, "aspect_ratio") for r in maze_rows], color=colors[maze_name])
    axes[0].axhline(1.0, color="black", linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("exact hard gain")
    axes[1].set_ylabel("aspect ratio")
    axes[1].set_xlabel("optimization step")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_training_traces.pdf")
    fig.savefig(output_dir / "boundary_training_traces.png", dpi=300)
    plt.close(fig)


def final_validate_and_print(
    output_dir: Path,
    summary_rows: list[dict[str, object]],
    transitions_by_maze: dict[str, torch.Tensor],
    config: Config,
    device: torch.device,
) -> None:
    simplex = regular_simplex(device=device)
    assert_simplex(simplex, transitions_by_maze, config)
    print("\nmaze | dimension | best_gain | winner_type | random_median | aspect_ratio")
    print("-----|-----------|-----------|-------------|---------------|-------------")
    for maze_name in config.mazes:
        maze_rows = sorted(
            [row for row in summary_rows if str(row.get("maze")) == maze_name],
            key=lambda row: int(row["dimension"]),
        )
        prev_gain: float | None = None
        prev_z: torch.Tensor | None = None
        transitions = transitions_by_maze[maze_name].to(device=device)
        for row in maze_rows:
            dimension = int(row["dimension"])
            gain = float(row["best_hard_gain"])
            if not math.isfinite(gain):
                raise AssertionError(f"{maze_name} m={dimension}: non-finite best hard gain")
            if gain < 1.0 - config.below_one_tol and not config.allow_below_one:
                raise AssertionError(f"{maze_name} m={dimension}: materially sub-one gain requires audit")
            if prev_gain is not None and gain > prev_gain + 1e-10:
                raise AssertionError(f"{maze_name} m={dimension}: non-monotone summary gain {gain} > {prev_gain}")
            ckpt_path = best_z_path(output_dir, maze_name, dimension)
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=device)
            if not isinstance(ckpt, dict):
                raise AssertionError(f"{maze_name} m={dimension}: winning checkpoint is not a dictionary")
            compatible, reason = checkpoint_is_compatible(ckpt, maze_name, dimension, config)
            if not compatible:
                raise AssertionError(f"{maze_name} m={dimension}: incompatible winning checkpoint: {reason}")
            z = ckpt["Z"].detach().to(device=device, dtype=torch.float64)
            result = exact_hard_gain(z, transitions, config.degeneracy_tol)
            if result.degenerate:
                raise AssertionError(f"{maze_name} m={dimension}: winning embedding is not injective above tolerance")
            if abs(result.hard_gain - gain) > 1e-8:
                raise AssertionError(f"{maze_name} m={dimension}: checkpoint gain {result.hard_gain} != summary {gain}")
            if prev_z is not None and prev_gain is not None:
                padded = pad_embedding(prev_z, dimension)
                padded_gain = exact_hard_gain(padded, transitions, config.degeneracy_tol).hard_gain
                if abs(padded_gain - prev_gain) > 1e-10:
                    raise AssertionError(
                        f"{maze_name} m={dimension}: zero-padding changed previous gain "
                        f"from {prev_gain} to {padded_gain}"
                    )
            if dimension == 63 and abs(gain - 1.0) > 1e-10:
                raise AssertionError(f"{maze_name} m=63: analytic simplex best should be 1, got {gain}")
            print(
                f"{maze_name} | {dimension} | {gain:.12g} | "
                f"{row['best_initialization_type']} | {float(row['median_random_restart_hard_gain']):.12g} | "
                f"{float(row['best_aspect_ratio']):.12g}"
            )
            prev_gain = gain
            prev_z = z
    print("\nfinal validation passed")


def assert_simplex(simplex: torch.Tensor, transitions_by_maze: dict[str, torch.Tensor], config: Config) -> None:
    if simplex.shape != (64, 63):
        raise AssertionError("simplex must have shape [64,63]")
    dists = torch.pdist(simplex, p=2)
    if float((dists.max() - dists.min()).abs().item()) > 1e-10:
        raise AssertionError("simplex pair distances are not equal")
    for maze_name, transitions in transitions_by_maze.items():
        gain = exact_hard_gain(simplex, transitions.to(device=simplex.device), config.degeneracy_tol).hard_gain
        if abs(gain - 1.0) > 1e-10:
            raise AssertionError(f"{maze_name}: simplex hard gain must be 1, got {gain}")


def run_sweep(config: Config) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    random.seed(config.base_seed)
    torch.manual_seed(config.base_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.base_seed)

    transitions_by_maze = {maze: build_transition_table(maze) for maze in config.mazes}
    simplex = regular_simplex(device=device)
    assert_simplex(simplex, transitions_by_maze, config)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "num_states": NUM_STATES,
                "actions": list(ACTION_ORDER),
                "primitive_constraints": NUM_CONSTRAINTS,
                "certified_gain_one_lower_bounds": CERTIFIED_GAIN_ONE_LOWER_BOUNDS,
                "guardrails": [
                    "Results are constructive upper bounds, not global optimality certificates.",
                    "29, 47, and 52 are lower bounds, not exact crossing dimensions.",
                    "Start/goal path lengths are sanity checks only.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    raw_csv = output_dir / "raw_restart_results.csv"
    trace_csv = output_dir / "training_traces.csv"
    summary_csv = output_dir / "summary.csv"
    raw_rows = load_rows(raw_csv) if config.resume else []
    trace_rows = load_rows(trace_csv) if config.resume else []
    if config.resume and raw_rows:
        raw_rows, trace_rows = prune_incompatible_resume_rows(output_dir, raw_rows, trace_rows, config, device)
    best_by_maze: dict[str, torch.Tensor | None] = {maze: None for maze in config.mazes}
    best_dim_by_maze: dict[str, int | None] = {maze: None for maze in config.mazes}
    previous_best_gain_by_maze: dict[str, float | None] = {maze: None for maze in config.mazes}

    for maze_name in config.mazes:
        transitions = transitions_by_maze[maze_name].to(device=device)
        for dimension in sorted(config.dimensions):
            ckpt_path = best_z_path(output_dir, maze_name, dimension)
            existing_rows = [
                row
                for row in raw_rows
                if str(row.get("maze")) == maze_name and int(row.get("dimension", -1)) == dimension
            ]
            if config.resume and ckpt_path.exists():
                resumed_z, checkpoint_row, reason = verified_checkpoint_row(
                    ckpt_path,
                    maze_name,
                    dimension,
                    transitions,
                    config,
                    device,
                )
                existing_rows_compatible = existing_rows and all(
                    row.get("checkpoint_version") == CHECKPOINT_VERSION for row in existing_rows
                )
                if resumed_z is not None and checkpoint_row is not None and existing_rows_compatible:
                    gain = float(checkpoint_row["best_hard_gain"])
                    previous_gain = previous_best_gain_by_maze[maze_name]
                    if previous_gain is not None and gain > previous_gain + 1e-10:
                        raise RuntimeError(
                            f"{maze_name} m={dimension}: resumed checkpoint violates monotone frontier "
                            f"({gain} > {previous_gain})"
                        )
                    best_by_maze[maze_name] = resumed_z
                    best_dim_by_maze[maze_name] = dimension
                    previous_best_gain_by_maze[maze_name] = gain
                    print(f"{maze_name} m={dimension}: resume skip, verified hard gain {gain:.12g}")
                    continue
                if resumed_z is not None and checkpoint_row is not None and not existing_rows_compatible:
                    raw_rows = [
                        row
                        for row in raw_rows
                        if not (str(row.get("maze")) == maze_name and int(row.get("dimension", -1)) == dimension)
                    ]
                    trace_rows = [
                        row
                        for row in trace_rows
                        if not (str(row.get("maze")) == maze_name and int(row.get("dimension", -1)) == dimension)
                    ]
                    raw_rows.append(checkpoint_row)
                    flush_outputs(output_dir, raw_rows, trace_rows, config)
                    gain = float(checkpoint_row["best_hard_gain"])
                    previous_gain = previous_best_gain_by_maze[maze_name]
                    if previous_gain is not None and gain > previous_gain + 1e-10:
                        raise RuntimeError(
                            f"{maze_name} m={dimension}: resumed checkpoint violates monotone frontier "
                            f"({gain} > {previous_gain})"
                        )
                    best_by_maze[maze_name] = resumed_z
                    best_dim_by_maze[maze_name] = dimension
                    previous_best_gain_by_maze[maze_name] = gain
                    print(f"{maze_name} m={dimension}: resume skip, verified hard gain {gain:.12g}")
                    continue
                print(f"{maze_name} m={dimension}: checkpoint incompatible -> recomputing ({reason})")

            if existing_rows:
                raw_rows = [
                    row
                    for row in raw_rows
                    if not (str(row.get("maze")) == maze_name and int(row.get("dimension", -1)) == dimension)
                ]
                trace_rows = [
                    row
                    for row in trace_rows
                    if not (str(row.get("maze")) == maze_name and int(row.get("dimension", -1)) == dimension)
                ]

            candidates: list[tuple[torch.Tensor, dict[str, object], list[dict[str, object]]]] = []
            prev_z = best_by_maze[maze_name]
            prev_dim = best_dim_by_maze[maze_name]
            if prev_z is not None and prev_dim is not None and dimension > prev_dim:
                padded = pad_embedding(prev_z.to(device=device), dimension)
                before = exact_hard_gain(prev_z.to(device=device), transitions, config.degeneracy_tol).hard_gain
                after = exact_hard_gain(padded, transitions, config.degeneracy_tol).hard_gain
                if abs(before - after) > 1e-10:
                    raise AssertionError(f"{maze_name}: padding changed gain from {before} to {after}")
                candidates.append(evaluate_candidate(padded, transitions, config, seed=-1, initialization_type="continuation_padded_baseline"))
                jitter_seed = config.base_seed + 100000 + 1000 * MAZE_NAMES.index(maze_name) + dimension
                jittered = jittered_continuation_embedding(
                    prev_z,
                    dimension,
                    jitter_std=config.continuation_jitter_std,
                    seed=jitter_seed,
                    device=device,
                    floor=config.floor,
                )
                candidates.append(optimize_embedding(jittered, transitions, config, seed=jitter_seed, initialization_type="continuation_jittered"))
            if dimension == 63:
                candidates.append(evaluate_candidate(simplex, transitions, config, seed=-2, initialization_type="simplex_analytic"))
            for restart_idx in range(config.random_restarts):
                seed = config.base_seed + restart_idx
                z0 = random_embedding(dimension, seed, device, floor=config.floor)
                candidates.append(optimize_embedding(z0, transitions, config, seed=seed, initialization_type="random"))
            if not candidates:
                raise ValueError(f"{maze_name} m={dimension}: no candidates to evaluate")

            best_z: torch.Tensor | None = None
            best_gain = float("inf")
            for z_best, row, traces in candidates:
                row = dict(row, maze=maze_name, dimension=dimension)
                raw_rows.append(row)
                for trace in traces:
                    trace_rows.append(dict(trace, maze=maze_name, dimension=dimension, seed=row["seed"], initialization_type=row["initialization_type"]))
                if row["status"] == "below_one_audit" and not config.allow_below_one:
                    raise RuntimeError(f"{maze_name} m={dimension}: hard gain below 1 beyond tolerance; audit before accepting")
                gain = float(row["best_hard_gain"])
                if row["status"] == "ok" and gain < best_gain:
                    best_gain = gain
                    best_z = z_best.detach().cpu()
            previous_gain = previous_best_gain_by_maze[maze_name]
            if previous_gain is not None and best_gain > previous_gain + 1e-10:
                dimension_rows = [
                    row
                    for row in raw_rows
                    if str(row.get("maze")) == maze_name and int(row.get("dimension", -1)) == dimension
                ]
                raise RuntimeError(
                    f"{maze_name} m={dimension}: best constructive frontier is not monotone "
                    f"({best_gain} > {previous_gain}). Candidate table:\n{candidate_table_text(dimension_rows)}"
                )
            if best_z is None:
                best_z = min(candidates, key=lambda item: float(item[1]["best_hard_gain"]))[0].detach().cpu()
            best_by_maze[maze_name] = best_z
            best_dim_by_maze[maze_name] = dimension
            previous_best_gain_by_maze[maze_name] = best_gain
            torch.save(
                {
                    "Z": best_z,
                    "maze": maze_name,
                    "dimension": dimension,
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "metadata": checkpoint_metadata(maze_name, dimension, config),
                    "action_order": ACTION_ORDER,
                    "transitions": transitions.cpu(),
                    "config": asdict(config),
                },
                ckpt_path,
            )
            print(f"{maze_name} m={dimension}: best hard gain {best_gain:.12g}")
            flush_outputs(output_dir, raw_rows, trace_rows, config)

    flush_outputs(output_dir, raw_rows, trace_rows, config)
    final_validate_and_print(output_dir, build_summary(raw_rows), transitions_by_maze, config, device)
    if config.make_plots:
        mpl_config = output_dir / "mplconfig"
        mpl_config.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        make_plots(summary_csv, trace_csv, output_dir)


def self_test() -> None:
    device = torch.device("cpu")
    config = Config(random_restarts=0, steps_per_stage=0, make_plots=False)
    transitions_by_maze = {maze: build_transition_table(maze) for maze in MAZE_NAMES}
    for maze_name, transitions in transitions_by_maze.items():
        stats = validate_maze(maze_name, transitions)
        assert stats == EXPECTED_STATS[maze_name]
        z = random_embedding(2, seed=0, device=device)
        result = exact_hard_gain(z, transitions)
        assert result.num_constraints == 8064
        assert result.num_zero_successor_constraints > 0
        assert torch.isfinite(surrogate_loss(z, transitions, tau=0.01))
    simplex = regular_simplex(device=device)
    assert_simplex(simplex, transitions_by_maze, config)
    z2 = random_embedding(2, seed=123, device=device)
    z8 = pad_embedding(z2, 8)
    g2 = exact_hard_gain(z2, transitions_by_maze["C"]).hard_gain
    g8 = exact_hard_gain(z8, transitions_by_maze["C"]).hard_gain
    assert math.isclose(g2, g8, rel_tol=0.0, abs_tol=1e-12)
    zj = jittered_continuation_embedding(z2, 8, jitter_std=1e-3, seed=999, device=device)
    assert zj.shape == (64, 8)
    assert torch.linalg.norm(zj[:, 2:]).item() > 0.0
    for dimension in [3, 5, 6, 12, 20, 28, 29, 30, 46, 47, 48, 51, 52, 53, 63]:
        assert dimension in DIMENSIONS
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mazes", nargs="+", default=list(MAZE_NAMES), choices=list(MAZE_NAMES))
    parser.add_argument("--dimensions", default=DEFAULT_DIMENSION_ARG)
    parser.add_argument("--include-m1", action="store_true")
    parser.add_argument("--random-restarts", type=int, default=10)
    parser.add_argument("--steps-per-stage", type=int, default=2500)
    parser.add_argument("--tau-schedule", default=",".join(str(t) for t in TAU_SCHEDULE))
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--continuation-jitter-std", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--floor", type=float, default=1e-300)
    parser.add_argument("--degeneracy-tol", type=float, default=1e-12)
    parser.add_argument("--below-one-tol", type=float, default=1e-6)
    parser.add_argument("--allow-below-one", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/direct_maze_realization")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints/CSVs and overwrite outputs from scratch.")
    parser.add_argument("--pilot", action="store_true", help="One restart, short optimization, selected dimensions.")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    dimensions = list(parse_ints(args.dimensions))
    if args.include_m1 and 1 not in dimensions:
        dimensions.insert(0, 1)
    random_restarts = args.random_restarts
    steps_per_stage = args.steps_per_stage
    if args.pilot:
        random_restarts = min(random_restarts, 1)
        steps_per_stage = min(steps_per_stage, 50)
        if args.dimensions == DEFAULT_DIMENSION_ARG:
            dimensions = [2, 3, 4, 6, 8, 28, 29, 46, 47, 51, 52, 63]
    return Config(
        mazes=tuple(args.mazes),
        dimensions=tuple(sorted(set(dimensions))),
        random_restarts=random_restarts,
        steps_per_stage=steps_per_stage,
        tau_schedule=parse_floats(args.tau_schedule),
        lr=args.lr,
        continuation_jitter_std=args.continuation_jitter_std,
        eval_every=args.eval_every,
        base_seed=args.base_seed,
        floor=args.floor,
        degeneracy_tol=args.degeneracy_tol,
        below_one_tol=args.below_one_tol,
        allow_below_one=args.allow_below_one,
        device=args.device,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        resume=not args.no_resume,
    )


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.self_test:
        self_test()
    else:
        run_sweep(config_from_args(args))


if __name__ == "__main__":
    main()
