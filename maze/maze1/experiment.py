#!/usr/bin/env python3
"""Single-file 8x8 controlled-maze latent-dimension experiment.

In the Apptainer container, run with the conda environment, for example:

    conda run -n learned-maze-gpu python experiment.py build
    conda run -n learned-maze-gpu python experiment.py pilot --steps 100000 --save-every 1000
    conda run -n learned-maze-gpu python experiment.py sweep --steps 100000 --save-every 1000 --confirm-maps

If `conda run` is slow in the container, activate once instead:

    conda activate learned-maze-gpu

Training objective:
    L = L_state + L_dyn_h5

Gain, rollouts, and planning are evaluation-only diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import zlib
from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

N = 8
START = (0, 0)
GOAL = (7, 7)
ACTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
MAZES = ["A", "B", "C"]
LATENT_DIMS = [2, 4, 8, 16, 32]
SEEDS = [0, 1, 2, 3, 4]
GAIN_HORIZONS = [1, 2, 5]
ROLLOUT_HORIZONS = [1, 5, 10, 15, 20, 25]
N_ROLLOUTS = 2000
N_PLAN_PAIRS = 300
H_PLAN = 25
BEAM_WIDTH = 256
TRAIN_PREDICTION_HORIZON = 5

State = tuple[int, int]
Barrier = tuple[str, int, int]
Edge = tuple[State, State]

A_WALLS: frozenset[Barrier] = frozenset(
    {
        ("V", 1, 3),
        ("V", 2, 3),
        ("V", 3, 3),
        ("H", 4, 4),
        ("H", 4, 5),
        ("H", 4, 6),
    }
)

B_WALLS: frozenset[Barrier] = frozenset(
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

C_WALLS: frozenset[Barrier] = frozenset(
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

WALLS = {"A": A_WALLS, "B": B_WALLS, "C": C_WALLS}
EXPECTED = {
    "A": {"barriers": 6, "states": 64, "reachable": 64, "shortest_path": 14},
    "B": {"barriers": 24, "states": 64, "reachable": 64, "shortest_path": 20},
    "C": {"barriers": 41, "states": 64, "reachable": 64, "shortest_path": 22},
}


@dataclass(frozen=True)
class Maze:
    name: str
    barriers: frozenset[Barrier]
    blocked_edges: frozenset[Edge]
    start: State = START
    goal: State = GOAL


@dataclass(frozen=True)
class TrainConfig:
    maze: str
    latent_dim: int
    hidden_dim: int = 128
    steps: int = 100000
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    lambda_dyn: float = 1.0
    train_prediction_horizon: int = TRAIN_PREDICTION_HORIZON
    use_latent_regularization: bool = False
    latent_regularization_type: str = "vicreg"
    latent_var_weight: float = 0.0
    latent_cov_weight: float = 0.0
    latent_std_target: float = 1.0
    sigreg_weight: float = 0.0
    sigreg_mode: str = "sliced_w2"
    sigreg_slices: int = 32
    sigreg_t_points: int = 16
    sigreg_t_max: float = 3.0
    lip_weight: float = 0.0
    lip_target: float = 1.0
    lip_top_fraction: float = 0.05
    lip_blocks_per_step: int = 8
    lip_eps: float = 1e-6
    seed: int = 0
    save_every: int = 1000
    resume: bool = True
    eval_rollouts: int = N_ROLLOUTS
    eval_plan_pairs: int = N_PLAN_PAIRS
    beam_width: int = BEAM_WIDTH
    plan_horizon: int = H_PLAN
    use_decoder: bool = True


class WorldModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        action_block_horizon: int = TRAIN_PREDICTION_HORIZON,
        use_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.action_block_horizon = action_block_horizon
        self.action_block_dim = len(ACTIONS) * action_block_horizon
        self.use_decoder = use_decoder
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + self.action_block_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = (
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            if use_decoder
            else None
        )

    def encode(self, xy: torch.Tensor) -> torch.Tensor:
        return self.encoder(xy)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("decoder is disabled for this model")
        return self.decoder(z)

    def predict_latent(self, z: torch.Tensor, action_block_onehot: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([z, action_block_onehot], dim=-1))


def states() -> list[State]:
    return [(r, c) for r in range(N) for c in range(N)]


def sid(s: State) -> int:
    return s[0] * N + s[1]


def in_bounds(s: State) -> bool:
    return 0 <= s[0] < N and 0 <= s[1] < N


def canon(a: State, b: State) -> Edge:
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
        raise ValueError(f"edge must connect neighboring cells: {a}, {b}")
    if not in_bounds(a) or not in_bounds(b):
        raise ValueError(f"edge must be internal: {a}, {b}")
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def barrier_edge(barrier: Barrier) -> Edge:
    kind, r, c = barrier
    if kind == "V":
        return canon((r, c), (r, c + 1))
    if kind == "H":
        return canon((r, c), (r + 1, c))
    raise ValueError(f"bad barrier kind: {kind}")


def build_maze(name: str) -> Maze:
    if name not in WALLS:
        raise ValueError(f"unknown maze: {name}")
    barriers = WALLS[name]
    maze = Maze(name, barriers, frozenset(barrier_edge(b) for b in barriers))
    validate_maze(maze)
    return maze


def t1(maze: Maze, s: State, action: str) -> State:
    dr, dc = ACTIONS[action]
    nxt = (s[0] + dr, s[1] + dc)
    if not in_bounds(nxt):
        return s
    if canon(s, nxt) in maze.blocked_edges:
        return s
    return nxt


def apply_actions(maze: Maze, start: State, actions: list[str] | tuple[str, ...]) -> State:
    s = start
    for action in actions:
        s = t1(maze, s, action)
    return s


def neighbors(maze: Maze, s: State) -> list[State]:
    return [t1(maze, s, a) for a in ACTIONS if t1(maze, s, a) != s]


def shortest_actions(maze: Maze, start: State, goal: State) -> list[str]:
    q = deque([start])
    parent: dict[State, tuple[State, str] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for action in ACTIONS:
            nb = t1(maze, cur, action)
            if nb == cur or nb in parent:
                continue
            parent[nb] = (cur, action)
            q.append(nb)
    if goal not in parent:
        raise ValueError(f"unreachable goal: {start} -> {goal}")
    out: list[str] = []
    cur = goal
    while parent[cur] is not None:
        prev, action = parent[cur]  # type: ignore[misc]
        out.append(action)
        cur = prev
    out.reverse()
    return out


def all_pair_distances(maze: Maze) -> dict[tuple[State, State], int]:
    d: dict[tuple[State, State], int] = {}
    for root in states():
        q = deque([(root, 0)])
        seen = {root}
        while q:
            cur, dist = q.popleft()
            d[(root, cur)] = dist
            for nb in neighbors(maze, cur):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, dist + 1))
    return d


def maze_summary(maze: Maze) -> dict[str, int | float | str]:
    d = all_pair_distances(maze)
    reachable = len({dst for (src, dst), _ in d.items() if src == maze.start})
    nonzero = [v for (a, b), v in d.items() if a != b]
    return {
        "maze": maze.name,
        "states": len(states()),
        "actions": len(ACTIONS),
        "training_transitions": len(states()) * len(ACTIONS),
        "barriers": len(maze.barriers),
        "connected_component_size": reachable,
        "reachable": reachable,
        "degree1_dead_ends": sum(len(neighbors(maze, s)) == 1 for s in states()),
        "start_goal_shortest_path": len(shortest_actions(maze, maze.start, maze.goal)),
        "graph_diameter": max(nonzero),
        "mean_shortest_path_distance": mean(nonzero),
    }


def validate_maze(maze: Maze) -> None:
    got = maze_summary_no_validate(maze)
    for key, expected in EXPECTED[maze.name].items():
        if got[key] != expected:
            raise AssertionError(f"{maze.name} {key}: expected {expected}, got {got[key]}")
    if len(maze.barriers) != len(maze.blocked_edges):
        raise AssertionError(f"{maze.name}: duplicate barrier edge")
    if maze.name == "C":
        if ("V", 7, 6) not in maze.barriers:
            raise AssertionError("Maze C must contain ('V', 7, 6)")
        if ("V", 7, 2) in maze.barriers:
            raise AssertionError("Maze C must not contain ('V', 7, 2)")
    for barrier in maze.barriers:
        a, b = barrier_edge(barrier)
        if b in neighbors(maze, a) or a in neighbors(maze, b):
            raise AssertionError(f"{maze.name}: barrier not symmetric: {barrier}")
    for state, action in [((0, 0), "U"), ((0, 0), "L"), ((7, 7), "D"), ((7, 7), "R")]:
        if t1(maze, state, action) != state:
            raise AssertionError(f"{maze.name}: boundary collision failed")


def maze_summary_no_validate(maze: Maze) -> dict[str, int]:
    seen = {maze.start}
    q = deque([maze.start])
    while q:
        cur = q.popleft()
        for nb in neighbors(maze, cur):
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return {
        "states": len(states()),
        "barriers": len(maze.barriers),
        "reachable": len(seen),
        "shortest_path": shortest_len_no_validate(maze, maze.start, maze.goal),
    }


def shortest_len_no_validate(maze: Maze, start: State, goal: State) -> int:
    q = deque([(start, 0)])
    seen = {start}
    while q:
        cur, dist = q.popleft()
        if cur == goal:
            return dist
        for nb in neighbors(maze, cur):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, dist + 1))
    raise ValueError("goal unreachable")


def normalize(points: list[State]) -> torch.Tensor:
    x = torch.tensor(points, dtype=torch.float32)
    return (x / float(N - 1)) * 2.0 - 1.0


def to_cells(xy: torch.Tensor) -> torch.Tensor:
    return ((xy + 1.0) / 2.0) * float(N - 1)


def onehot(idx: torch.Tensor) -> torch.Tensor:
    return F.one_hot(idx, num_classes=len(ACTIONS)).float()


def block_onehot(action_seq: torch.Tensor) -> torch.Tensor:
    if action_seq.ndim != 2:
        raise ValueError(f"action_seq must have shape [batch, horizon], got {tuple(action_seq.shape)}")
    return onehot(action_seq).reshape(action_seq.shape[0], action_seq.shape[1] * len(ACTIONS))


def dataset(maze: Maze, device: torch.device, horizon: int = TRAIN_PREDICTION_HORIZON) -> dict[str, object]:
    src, act_seq, tgt = [], [], []
    action_names = list(ACTIONS)
    block_seqs = list(product(range(len(action_names)), repeat=horizon))
    for s in states():
        for seq in block_seqs:
            actions = [action_names[i] for i in seq]
            src.append(s)
            act_seq.append(seq)
            tgt.append(apply_actions(maze, s, actions))
    target_rows = []
    for seq in block_seqs:
        actions = [action_names[i] for i in seq]
        target_rows.append([sid(apply_actions(maze, s, actions)) for s in states()])
    a_seq = torch.tensor(act_seq, dtype=torch.long, device=device)
    return {
        "src": src,
        "tgt": tgt,
        "x": normalize(src).to(device),
        "y": normalize(tgt).to(device),
        "a_seq": a_seq,
        "block_seqs": torch.tensor(block_seqs, dtype=torch.long, device=device),
        "block_target_ids": torch.tensor(target_rows, dtype=torch.long, device=device),
        "all_xy": normalize(states()).to(device),
        "all_cells": torch.tensor(states(), dtype=torch.float32, device=device),
    }


def block_predict_latent(model: WorldModel, z: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
    return model.predict_latent(z, block_onehot(action_seq).to(z.device))


def latent_nearest_cells(z_query: torch.Tensor, z_all: torch.Tensor, all_cells: torch.Tensor) -> torch.Tensor:
    idx = torch.cdist(z_query, z_all).argmin(dim=1)
    return all_cells[idx]


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weight_norm(module: nn.Module) -> float:
    with torch.no_grad():
        return math.sqrt(sum(float((p.detach() ** 2).sum().cpu()) for p in module.parameters()))


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != x.shape[1]:
        raise ValueError(f"expected square matrix, got {tuple(x.shape)}")
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def latent_regularization_losses(
    z: torch.Tensor,
    std_target: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style variance/covariance losses on predictor-input latents."""

    if z.ndim != 2:
        raise ValueError(f"z must have shape [batch, latent_dim], got {tuple(z.shape)}")
    batch_size, latent_dim = z.shape
    if batch_size < 2:
        zero = z.new_tensor(0.0)
        return zero, zero

    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = torch.relu(std_target - std).mean()

    centered = z - z.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (batch_size - 1)
    if latent_dim <= 1:
        cov_loss = z.new_tensor(0.0)
    else:
        cov_loss = off_diagonal(cov).pow(2).sum() / latent_dim
    return var_loss, cov_loss


def sigreg_loss(
    z: torch.Tensor,
    mode: str = "sliced_w2",
    num_slices: int = 32,
    num_t_points: int = 16,
    t_max: float = 3.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Sliced SIGReg loss matching latent projections to N(0, 1)."""

    if z.ndim != 2:
        raise ValueError(f"z must have shape [batch, latent_dim], got {tuple(z.shape)}")
    batch_size, latent_dim = z.shape
    if batch_size < 2 or latent_dim < 1:
        return z.new_tensor(0.0)

    centered = z - z.mean(dim=0, keepdim=True)
    directions = torch.randn(num_slices, latent_dim, device=z.device, dtype=z.dtype)
    directions = directions / (directions.norm(dim=1, keepdim=True) + eps)
    projected = centered @ directions.T

    if mode == "sliced_w2":
        projected = torch.sort(projected, dim=0).values
        q = (torch.arange(batch_size, device=z.device, dtype=z.dtype) + 0.5) / batch_size
        target = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
        target = target[:, None]
        return (projected - target).square().mean()

    if mode != "ecf":
        raise ValueError(f"unknown sigreg_mode: {mode}")

    t = torch.linspace(0.0, t_max, num_t_points, device=z.device, dtype=z.dtype)
    phases = projected.unsqueeze(-1) * t
    emp_real = torch.cos(phases).mean(dim=0)
    emp_imag = torch.sin(phases).mean(dim=0)
    target_real = torch.exp(-0.5 * t.square()).unsqueeze(0)
    return (emp_real - target_real).square().mean() + emp_imag.square().mean()


def lipschitz_ratios_from_latents(
    z_all: torch.Tensor,
    target_ids: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return latent gain ratios for one or more action blocks."""

    if z_all.ndim != 2:
        raise ValueError(f"z_all must have shape [states, latent_dim], got {tuple(z_all.shape)}")
    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)
    if target_ids.ndim != 2 or target_ids.shape[1] != z_all.shape[0]:
        raise ValueError(
            "target_ids must have shape [blocks, states], "
            f"got {tuple(target_ids.shape)} for {z_all.shape[0]} states"
        )

    pair_i, pair_j = torch.triu_indices(z_all.shape[0], z_all.shape[0], offset=1, device=z_all.device)
    before = torch.cdist(z_all, z_all)[pair_i, pair_j]
    valid = before > eps
    if not bool(valid.any()):
        return z_all.new_zeros((0,))

    z_after = z_all[target_ids]
    after = torch.cdist(z_after, z_after)[:, pair_i, pair_j]
    return (after[:, valid] / before[valid].unsqueeze(0)).reshape(-1)


def lipschitz_regularization_from_ratios(
    ratios: torch.Tensor,
    target: float = 1.0,
    top_fraction: float = 0.05,
) -> torch.Tensor:
    """Penalize high latent gain ratios."""

    if ratios.numel() == 0:
        return ratios.new_tensor(0.0)
    values = ratios
    if 0.0 < top_fraction < 1.0:
        k = max(1, math.ceil(values.numel() * top_fraction))
        values = torch.topk(values, k).values
    excess = torch.relu(values - target)
    return excess.square().mean()


def lipschitz_regularization_objective(
    model: WorldModel,
    data: dict[str, object],
    config: TrainConfig,
    blocks_per_step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw lip loss plus ratio summary stats and weighted regularization."""

    all_xy = data["all_xy"]  # type: ignore[assignment]
    block_target_ids = data["block_target_ids"]  # type: ignore[assignment]
    z_all = model.encode(all_xy)
    n_blocks = block_target_ids.shape[0]
    if blocks_per_step is not None and 0 < blocks_per_step < n_blocks:
        idx = torch.randint(0, n_blocks, (blocks_per_step,), device=z_all.device)
        target_ids = block_target_ids[idx]
    else:
        target_ids = block_target_ids

    ratios = lipschitz_ratios_from_latents(
        z_all,
        target_ids,
        eps=config.lip_eps,
    )
    lip_loss = lipschitz_regularization_from_ratios(
        ratios,
        target=config.lip_target,
        top_fraction=config.lip_top_fraction,
    )
    if ratios.numel() == 0:
        zero = z_all.new_tensor(0.0)
        return lip_loss, zero, zero, zero, config.lip_weight * lip_loss
    return (
        lip_loss,
        ratios.mean(),
        torch.quantile(ratios, 0.95),
        ratios.max(),
        config.lip_weight * lip_loss,
    )


def latent_regularization_objective(
    z: torch.Tensor,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return var, cov, sigreg, and weighted total regularization losses."""

    zero = z.new_tensor(0.0)
    if not config.use_latent_regularization:
        return zero, zero, zero, zero

    if config.latent_regularization_type == "vicreg":
        var_loss, cov_loss = latent_regularization_losses(
            z,
            std_target=config.latent_std_target,
        )
        sig_loss = zero
        reg = config.latent_var_weight * var_loss + config.latent_cov_weight * cov_loss
        return var_loss, cov_loss, sig_loss, reg

    if config.latent_regularization_type == "sigreg":
        sig_loss = sigreg_loss(
            z,
            mode=config.sigreg_mode,
            num_slices=config.sigreg_slices,
            num_t_points=config.sigreg_t_points,
            t_max=config.sigreg_t_max,
        )
        return zero, zero, sig_loss, config.sigreg_weight * sig_loss

    if config.latent_regularization_type == "lip":
        return zero, zero, zero, zero

    raise ValueError(f"unknown latent_regularization_type: {config.latent_regularization_type}")


def latent_regularization_metrics(
    z: torch.Tensor,
    config: TrainConfig,
) -> dict[str, float]:
    var_loss, cov_loss, sig_loss, reg_loss = latent_regularization_objective(z, config)
    return {
        "eval_latent_var_loss": float(var_loss),
        "eval_latent_cov_loss": float(cov_loss),
        "eval_latent_sigreg_loss": float(sig_loss),
        "eval_latent_reg_loss": float(reg_loss),
    }


def latent_covariance_diagnostics(z: torch.Tensor, run_dir: Path) -> dict[str, float | int | str]:
    """Save covariance eigenvalues and return scalar effective-dimension diagnostics."""

    z = z.detach()
    centered = z - z.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
    total = eig.sum()
    if float(total) <= 0:
        effective_dim = 0.0
        entropy_effective_dim = 0.0
        stable_rank = 0.0
        numerical_rank = 0
    else:
        effective_dim = float(total.square() / (eig.square().sum() + 1e-30))
        p = eig / total
        nz = p[p > 0]
        entropy_effective_dim = float(torch.exp(-(nz * torch.log(nz)).sum()))
        stable_rank = float(total / (eig.max() + 1e-30))
        numerical_rank = int((eig > eig.max() * 1e-6).sum())

    path = run_dir / "latent_cov_eigenvalues.json"
    path.write_text(json.dumps([float(x) for x in eig.cpu()], indent=2) + "\n")
    return {
        "latent_effective_dim_pr": effective_dim,
        "latent_effective_dim_entropy": entropy_effective_dim,
        "latent_stable_rank": stable_rank,
        "latent_numerical_rank": numerical_rank,
        "latent_cov_eig_top1": float(eig[0]) if eig.numel() else 0.0,
        "latent_cov_eig_top2": float(eig[1]) if eig.numel() > 1 else 0.0,
        "latent_cov_eig_top3": float(eig[2]) if eig.numel() > 2 else 0.0,
        "latent_cov_eigvals_path": str(path),
    }


def losses(model: WorldModel, data: dict[str, object], config: TrainConfig) -> dict[str, float]:
    x = data["x"]  # type: ignore[assignment]
    y = data["y"]  # type: ignore[assignment]
    a_seq = data["a_seq"]  # type: ignore[assignment]
    with torch.no_grad():
        z = model.encode(x)
        z_next = model.encode(y)
        z_hat = block_predict_latent(model, z, a_seq)
        state = F.mse_loss(model.decode(z), x) if model.use_decoder else x.new_tensor(0.0)
        dyn = F.mse_loss(z_hat, z_next)
        var_loss, cov_loss, sig_loss, reg = latent_regularization_objective(z, config)
        if config.use_latent_regularization and config.latent_regularization_type == "lip":
            lip_loss, lip_mean, lip_p95, lip_max, lip_reg = lipschitz_regularization_objective(
                model,
                data,
                config,
                blocks_per_step=None,
            )
            reg = reg + lip_reg
        else:
            lip_loss = x.new_tensor(0.0)
            lip_mean = x.new_tensor(0.0)
            lip_p95 = x.new_tensor(0.0)
            lip_max = x.new_tensor(0.0)
            lip_reg = x.new_tensor(0.0)
    return {
        "train_total_loss": float(state + config.lambda_dyn * dyn + reg),
        "train_state_loss": float(state),
        "train_dyn_h5_loss": float(dyn),
        "train_dyn_loss": float(dyn),
        "latent_var_loss": float(var_loss),
        "latent_cov_loss": float(cov_loss),
        "latent_sigreg_loss": float(sig_loss),
        "latent_lip_loss": float(lip_loss),
        "latent_lip_mean": float(lip_mean),
        "latent_lip_p95": float(lip_p95),
        "latent_lip_max": float(lip_max),
        "latent_lip_reg_loss": float(lip_reg),
        "latent_reg_loss": float(reg),
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = round(q * (len(sorted_values) - 1))
    return sorted_values[max(0, min(len(sorted_values) - 1, idx))]


def raw_coord_gain(maze: Maze, horizon: int = 5) -> dict[str, float | int]:
    local_pairs = [(a, b) for a, b in combinations(states(), 2) if 0 < math.dist(a, b) <= 1.5]
    ratios = []
    for seq in product(ACTIONS, repeat=horizon):
        targets = {s: apply_actions(maze, s, seq) for s in states()}
        for a, b in local_pairs:
            ratios.append(math.dist(targets[a], targets[b]) / math.dist(a, b))
    ratios.sort()
    return {
        "horizon": horizon,
        "action_sequences": len(ACTIONS) ** horizon,
        "local_unordered_pairs": len(local_pairs),
        "p95": percentile(ratios, 0.95),
        "max": ratios[-1],
    }


def composed_table(maze: Maze, horizon: int, device: torch.device) -> torch.Tensor:
    seqs = list(product(ACTIONS, repeat=horizon))
    rows = []
    for s in states():
        rows.append([sid(apply_actions(maze, s, list(seq))) for seq in seqs])
    return torch.tensor(rows, dtype=torch.long, device=device)


@torch.no_grad()
def gain_metrics(model: WorldModel, maze: Maze, device: torch.device, run_dir: Path) -> dict[str, float]:
    z = model.encode(normalize(states()).to(device))
    pair_i, pair_j = torch.triu_indices(len(states()), len(states()), offset=1, device=device)
    denom = torch.cdist(z, z)[pair_i, pair_j]
    valid = denom > 1e-12
    out: dict[str, float] = {}
    full = {}
    out["gain_valid_pair_count"] = int(valid.sum())
    out["gain_total_pair_count"] = int(valid.numel())
    for h in GAIN_HORIZONS:
        table = composed_table(maze, h, device)
        ratios = []
        for col in range(table.shape[1]):
            targets = table[:, col]
            after = torch.cdist(z[targets], z[targets])[pair_i, pair_j]
            ratios.append(after[valid] / denom[valid])
        joined = torch.cat(ratios)
        full[f"Lambda{h}_ratios"] = joined.cpu()
        if joined.numel() == 0:
            out[f"Lambda{h}_max"] = float("nan")
            out[f"Lambda{h}_p95"] = float("nan")
            if h in (1, 5):
                out[f"Lambda{h}_p99"] = float("nan")
                out[f"Lambda{h}_p90"] = float("nan")
            if h == 1:
                out["Lambda1_median"] = float("nan")
            continue
        out[f"Lambda{h}_max"] = float(joined.max())
        out[f"Lambda{h}_p95"] = float(torch.quantile(joined, 0.95))
        if h in (1, 5):
            out[f"Lambda{h}_p99"] = float(torch.quantile(joined, 0.99))
            out[f"Lambda{h}_p90"] = float(torch.quantile(joined, 0.90))
        if h == 1:
            out["Lambda1_median"] = float(torch.quantile(joined, 0.50))
    torch.save(full, run_dir / "gain_distributions.pt")
    return out


def planning_pairs(maze: Maze, n: int) -> list[dict[str, object]]:
    rng = random.Random(8147 + MAZES.index(maze.name))
    by_bin = {"short": [], "medium": [], "long": []}
    for (src, dst), dist in all_pair_distances(maze).items():
        if src == dst:
            continue
        if 5 <= dist <= 9:
            by_bin["short"].append({"start": src, "goal": dst, "distance": dist, "bin": "short"})
        elif 10 <= dist <= 16:
            by_bin["medium"].append({"start": src, "goal": dst, "distance": dist, "bin": "medium"})
        elif 17 <= dist <= 25:
            by_bin["long"].append({"start": src, "goal": dst, "distance": dist, "bin": "long"})
    selected = []
    for key in ["short", "medium", "long"]:
        rng.shuffle(by_bin[key])
        selected.extend(by_bin[key][: max(1, n // 3)])
    rest = [r for rows in by_bin.values() for r in rows if r not in selected]
    rng.shuffle(rest)
    selected.extend(rest[: max(0, n - len(selected))])
    return selected[:n]


def rollout_cases(maze: Maze, n: int, horizon: int = H_PLAN) -> list[dict[str, object]]:
    rng = random.Random(1729 + MAZES.index(maze.name))
    action_names = list(ACTIONS)
    cases = []
    for _ in range(n // 2):
        cases.append({"start": rng.choice(states()), "actions": [rng.choice(action_names) for _ in range(horizon)]})
    pairs = planning_pairs(maze, max(1, n // 4))
    for row in pairs:
        path = shortest_actions(maze, row["start"], row["goal"])  # type: ignore[arg-type]
        extra = [rng.choice(action_names) for _ in range(max(0, horizon - len(path)))]
        cases.append({"start": row["start"], "actions": (path + extra)[:horizon]})
    wall_hits = [(s, a) for s in states() for a in ACTIONS if t1(maze, s, a) == s]
    while len(cases) < n:
        s, action = rng.choice(wall_hits)
        cases.append({"start": s, "actions": [action if i % 3 == 0 else rng.choice(action_names) for i in range(horizon)]})
    return cases[:n]


def action_indices(actions: list[str] | tuple[str, ...], device: torch.device) -> torch.Tensor:
    action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
    return torch.tensor([[action_to_idx[a] for a in actions]], dtype=torch.long, device=device)


@torch.no_grad()
def rollout_metrics(model: WorldModel, maze: Maze, device: torch.device, n: int, run_dir: Path) -> dict[str, float]:
    cases = rollout_cases(maze, n)
    (run_dir / "rollout_cases.json").write_text(json.dumps(serialize_cases(cases), indent=2) + "\n")
    block_h = model.action_block_horizon
    eval_horizons = [h for h in ROLLOUT_HORIZONS if h % block_h == 0]
    errors = {h: [] for h in eval_horizons}
    exact = {h: [] for h in eval_horizons}
    mean_err = {h: [] for h in eval_horizons}
    cum_err = {h: [] for h in eval_horizons}
    all_xy = normalize(states()).to(device)
    all_cells = torch.tensor(states(), dtype=torch.float32, device=device)
    z_all = model.encode(all_xy)
    for case in cases:
        true = case["start"]  # type: ignore[assignment]
        z = model.encode(normalize([true]).to(device))
        step_errors = []
        actions = case["actions"]  # type: ignore[assignment]
        for start_idx in range(0, len(actions), block_h):
            block = actions[start_idx : start_idx + block_h]
            if len(block) < block_h:
                break
            z = block_predict_latent(model, z, action_indices(block, device))
            for action in block:
                true = t1(maze, true, action)
            if model.use_decoder:
                pred_cell = to_cells(model.decode(z))[0]
            else:
                pred_cell = latent_nearest_cells(z, z_all, all_cells)[0]
            true_cell = torch.tensor(true, dtype=torch.float32, device=device)
            e = float(torch.linalg.vector_norm(pred_cell - true_cell))
            step_errors.append(e)
            t = start_idx + block_h
            if t in ROLLOUT_HORIZONS:
                errors[t].append(e)
                exact[t].append(float(torch.all(torch.round(pred_cell).clamp(0, N - 1) == true_cell)))
                mean_err[t].append(mean(step_errors))
                cum_err[t].append(sum(step_errors))
    out = {}
    for h in eval_horizons:
        out[f"rollout_{h}_terminal_error"] = mean(errors[h])
        out[f"rollout_{h}_exact_accuracy"] = mean(exact[h])
        out[f"rollout_{h}_mean_trajectory_error"] = mean(mean_err[h])
        out[f"rollout_{h}_cumulative_trajectory_error"] = mean(cum_err[h])
    return out


def serialize_cases(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for case in cases:
        out.append({"start": list(case["start"]), "actions": case["actions"]})  # type: ignore[arg-type]
    return out


@torch.no_grad()
def beam_plan(model: WorldModel, start: State, goal: State, device: torch.device, width: int, horizon: int) -> list[str]:
    action_names = list(ACTIONS)
    block_h = model.action_block_horizon
    block_choices = list(product(action_names, repeat=block_h))
    goal_cell = torch.tensor(goal, dtype=torch.float32, device=device)
    beams = [(0.0, model.encode(normalize([start]).to(device)), [])]
    best = (float("inf"), [])
    for depth in range(1, horizon // block_h + 1):
        z_batch = torch.cat([b[1] for b in beams], dim=0)
        seqs = [b[2] for b in beams]
        z_rep = z_batch.repeat_interleave(len(block_choices), dim=0)
        blocks = [
            [action_names.index(a) for a in block]
            for _ in beams
            for block in block_choices
        ]
        block_idx = torch.tensor(blocks, dtype=torch.long, device=device)
        z_next = block_predict_latent(model, z_rep, block_idx)
        decoded = to_cells(model.decode(z_next))
        primitive_depth = depth * block_h
        scores = torch.abs(decoded - goal_cell).sum(dim=1) + 0.01 * primitive_depth
        vals, inds = torch.topk(scores, k=min(width, scores.numel()), largest=False)
        new_beams = []
        for val, flat in zip(vals.tolist(), inds.tolist()):
            parent = flat // len(block_choices)
            block = list(block_choices[flat % len(block_choices)])
            seq = seqs[parent] + block
            new_beams.append((val, z_next[flat : flat + 1], seq))
            if val < best[0]:
                best = (val, seq)
        beams = new_beams
        if best[0] <= 0.5 + 0.01 * primitive_depth:
            break
    return best[1]


@torch.no_grad()
def beam_plan_latent(
    model: WorldModel,
    start: State,
    goal: State,
    device: torch.device,
    width: int,
    horizon: int,
) -> list[str]:
    action_names = list(ACTIONS)
    block_h = model.action_block_horizon
    block_choices = list(product(action_names, repeat=block_h))
    z_goal = model.encode(normalize([goal]).to(device))
    beams = [(0.0, model.encode(normalize([start]).to(device)), [])]
    best = (float("inf"), [])
    for depth in range(1, horizon // block_h + 1):
        z_batch = torch.cat([b[1] for b in beams], dim=0)
        seqs = [b[2] for b in beams]
        z_rep = z_batch.repeat_interleave(len(block_choices), dim=0)
        blocks = [
            [action_names.index(a) for a in block]
            for _ in beams
            for block in block_choices
        ]
        block_idx = torch.tensor(blocks, dtype=torch.long, device=device)
        z_next = block_predict_latent(model, z_rep, block_idx)
        primitive_depth = depth * block_h
        scores = torch.linalg.vector_norm(z_next - z_goal, dim=1) + 0.01 * primitive_depth
        vals, inds = torch.topk(scores, k=min(width, scores.numel()), largest=False)
        new_beams = []
        for val, flat in zip(vals.tolist(), inds.tolist()):
            parent = flat // len(block_choices)
            block = list(block_choices[flat % len(block_choices)])
            seq = seqs[parent] + block
            new_beams.append((val, z_next[flat : flat + 1], seq))
            if val < best[0]:
                best = (val, seq)
        beams = new_beams
    return best[1]


@torch.no_grad()
def plan_metrics(model: WorldModel, maze: Maze, device: torch.device, n: int, width: int, horizon: int, run_dir: Path) -> dict[str, float]:
    pairs = planning_pairs(maze, n)
    (run_dir / "planning_pairs.json").write_text(json.dumps(serialize_pairs(pairs), indent=2) + "\n")
    success, by_bin, by_dist = [], {"short": [], "medium": [], "long": []}, {}
    for row in pairs:
        start = row["start"]  # type: ignore[assignment]
        goal = row["goal"]  # type: ignore[assignment]
        if model.use_decoder:
            seq = beam_plan(model, start, goal, device, width, horizon)
        else:
            seq = beam_plan_latent(model, start, goal, device, width, horizon)
        cur = start
        reached = cur == goal
        for action in seq:
            cur = t1(maze, cur, action)
            reached = reached or cur == goal
        s = float(reached)
        success.append(s)
        by_bin[row["bin"]].append(s)  # type: ignore[index]
        by_dist.setdefault(int(row["distance"]), []).append(s)
    out = {
        "planning_success_rate": mean(success),
        "planning_short_SR": mean(by_bin["short"]) if by_bin["short"] else float("nan"),
        "planning_medium_SR": mean(by_bin["medium"]) if by_bin["medium"] else float("nan"),
        "planning_long_SR": mean(by_bin["long"]) if by_bin["long"] else float("nan"),
    }
    for dist, vals in sorted(by_dist.items()):
        out[f"planning_dist_{dist}_SR"] = mean(vals)
    return out


@torch.no_grad()
def latent_plan_metrics(
    model: WorldModel,
    maze: Maze,
    device: torch.device,
    n: int,
    width: int,
    horizon: int,
    run_dir: Path,
) -> dict[str, float]:
    pairs = planning_pairs(maze, n)
    (run_dir / "latent_planning_pairs.json").write_text(json.dumps(serialize_pairs(pairs), indent=2) + "\n")
    success, by_bin, by_dist = [], {"short": [], "medium": [], "long": []}, {}
    for row in pairs:
        start = row["start"]  # type: ignore[assignment]
        goal = row["goal"]  # type: ignore[assignment]
        seq = beam_plan_latent(model, start, goal, device, width, horizon)
        cur = start
        reached = cur == goal
        for action in seq:
            cur = t1(maze, cur, action)
            reached = reached or cur == goal
        s = float(reached)
        success.append(s)
        by_bin[row["bin"]].append(s)  # type: ignore[index]
        by_dist.setdefault(int(row["distance"]), []).append(s)
    out = {
        "latent_planning_success_rate": mean(success),
        "latent_planning_short_SR": mean(by_bin["short"]) if by_bin["short"] else float("nan"),
        "latent_planning_medium_SR": mean(by_bin["medium"]) if by_bin["medium"] else float("nan"),
        "latent_planning_long_SR": mean(by_bin["long"]) if by_bin["long"] else float("nan"),
    }
    for dist, vals in sorted(by_dist.items()):
        out[f"latent_planning_dist_{dist}_SR"] = mean(vals)
    return out


def serialize_pairs(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "start": list(row["start"]),  # type: ignore[arg-type]
            "goal": list(row["goal"]),  # type: ignore[arg-type]
            "distance": row["distance"],
            "bin": row["bin"],
        }
        for row in pairs
    ]


@torch.no_grad()
def eval_model(model: WorldModel, config: TrainConfig, data: dict[str, object], run_dir: Path) -> dict[str, float | int | str]:
    device = next(model.parameters()).device
    maze = build_maze(config.maze)
    x = data["x"]  # type: ignore[assignment]
    y = data["y"]  # type: ignore[assignment]
    a_seq = data["a_seq"]  # type: ignore[assignment]
    all_xy = data["all_xy"]  # type: ignore[assignment]
    all_cells = data["all_cells"]  # type: ignore[assignment]

    z_all = model.encode(all_xy)
    if model.use_decoder:
        decoded = model.decode(z_all)
        decoded_cells = to_cells(decoded)
        nearest = torch.round(decoded_cells).clamp(0, N - 1)
        state_coordinate_mse = float(F.mse_loss(decoded, all_xy))
        state_readout = "decoder"
    else:
        decoded = None
        nearest = latent_nearest_cells(z_all, z_all, all_cells)
        state_coordinate_mse = float("nan")
        state_readout = "latent_nearest"
    pair_i, pair_j = torch.triu_indices(len(states()), len(states()), offset=1, device=device)
    pair_d = torch.cdist(z_all, z_all)[pair_i, pair_j]

    z = model.encode(x)
    z_next = model.encode(y)
    z_hat = block_predict_latent(model, z, a_seq)
    true_h_cells = to_cells(y)
    if model.use_decoder:
        pred_h = model.decode(z_hat)
        pred_h_cells = to_cells(pred_h)
        pred_h_nearest = torch.round(pred_h_cells).clamp(0, N - 1)
        h5_coordinate_mse = float(F.mse_loss(pred_h, y))
    else:
        pred_h_cells = latent_nearest_cells(z_hat, z_all, all_cells)
        pred_h_nearest = pred_h_cells
        pred_h_normalized = (pred_h_cells / float(N - 1)) * 2.0 - 1.0
        h5_coordinate_mse = float(F.mse_loss(pred_h_normalized, y))
    h_euclid = torch.linalg.vector_norm(pred_h_cells - true_h_cells, dim=1)

    out: dict[str, float | int | str] = {
        "maze": config.maze,
        "latent_dim": config.latent_dim,
        "seed": config.seed,
        "parameter_count": param_count(model),
        "use_decoder": int(model.use_decoder),
        "state_readout": state_readout,
        **losses(model, data, config),
        **latent_regularization_metrics(z_all, config),
        **latent_covariance_diagnostics(z_all, run_dir),
        "latent_mean_norm": float(torch.linalg.vector_norm(z_all, dim=1).mean()),
        "latent_rms": float(torch.sqrt(torch.mean(z_all**2))),
        "decoder_weight_norm": weight_norm(model.decoder) if model.decoder is not None else 0.0,
        "predictor_weight_norm": weight_norm(model.predictor),
        "state_coordinate_mse": state_coordinate_mse,
        "state_exact_accuracy": float((nearest == all_cells).all(dim=1).float().mean()),
        "min_pairwise_latent_distance": float(pair_d.min()),
        "p01_pairwise_latent_distance": float(torch.quantile(pair_d, 0.01)),
        "p05_pairwise_latent_distance": float(torch.quantile(pair_d, 0.05)),
        "median_pairwise_latent_distance": float(torch.quantile(pair_d, 0.50)),
        "train_prediction_horizon": config.train_prediction_horizon,
        "use_latent_regularization": int(config.use_latent_regularization),
        "latent_regularization_type": config.latent_regularization_type,
        "latent_var_weight": config.latent_var_weight,
        "latent_cov_weight": config.latent_cov_weight,
        "latent_std_target": config.latent_std_target,
        "sigreg_weight": config.sigreg_weight,
        "sigreg_mode": config.sigreg_mode,
        "sigreg_slices": config.sigreg_slices,
        "sigreg_t_points": config.sigreg_t_points,
        "sigreg_t_max": config.sigreg_t_max,
        "lip_weight": config.lip_weight,
        "lip_target": config.lip_target,
        "lip_top_fraction": config.lip_top_fraction,
        "lip_blocks_per_step": config.lip_blocks_per_step,
        "lip_eps": config.lip_eps,
        "h5_coordinate_MSE": h5_coordinate_mse,
        "h5_state_error": float(h_euclid.mean()),
        "h5_euclidean_error": float(h_euclid.mean()),
        "h5_manhattan_error": float(torch.abs(pred_h_cells - true_h_cells).sum(dim=1).mean()),
        "h5_exact_accuracy": float((pred_h_nearest == true_h_cells).all(dim=1).float().mean()),
        "latent_dynamics_mse": float(F.mse_loss(z_hat, z_next)),
    }
    out.update(gain_metrics(model, maze, device, run_dir))
    out.update(rollout_metrics(model, maze, device, config.eval_rollouts, run_dir))
    out.update(plan_metrics(model, maze, device, config.eval_plan_pairs, config.beam_width, config.plan_horizon, run_dir))
    out.update(latent_plan_metrics(model, maze, device, config.eval_plan_pairs, config.beam_width, config.plan_horizon, run_dir))
    return out


def train(config: TrainConfig) -> dict[str, object]:
    if config.lambda_dyn != 1.0:
        raise ValueError("lambda_dyn is fixed to 1.0")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze = build_maze(config.maze)
    data = dataset(maze, device, horizon=config.train_prediction_horizon)
    model = WorldModel(
        config.latent_dim,
        config.hidden_dim,
        config.train_prediction_horizon,
        use_decoder=config.use_decoder,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1.0)

    run_name = (
        f"maze_{config.maze}_m{config.latent_dim}_seed{config.seed}"
        f"_h{config.hidden_dim}_blockpred{config.train_prediction_horizon}"
    )
    if not config.use_decoder:
        run_name += "_nodecoder"
    if config.use_latent_regularization:
        if config.latent_regularization_type == "vicreg":
            run_name += (
                f"_latreg_vicreg_var{config.latent_var_weight:g}"
                f"_cov{config.latent_cov_weight:g}"
                f"_std{config.latent_std_target:g}"
            )
        elif config.latent_regularization_type == "sigreg":
            run_name += (
                f"_latreg_sigreg_{config.sigreg_mode}"
                f"_w{config.sigreg_weight:g}"
                f"_s{config.sigreg_slices}"
                f"_t{config.sigreg_t_points}"
                f"_tm{config.sigreg_t_max:g}"
            )
        elif config.latent_regularization_type == "lip":
            run_name += (
                f"_latreg_lip_w{config.lip_weight:g}"
                f"_target{config.lip_target:g}"
                f"_top{config.lip_top_fraction:g}"
                f"_blocks{config.lip_blocks_per_step}"
            )
        else:
            raise ValueError(f"unknown latent_regularization_type: {config.latent_regularization_type}")
    run_dir = OUT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "latest.pt"
    start, curves = 0, []
    if config.resume and ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        sched.load_state_dict(payload["scheduler"])
        start = int(payload["step"])
        curves = list(payload.get("curves", []))

    x = data["x"]  # type: ignore[assignment]
    y = data["y"]  # type: ignore[assignment]
    a_seq = data["a_seq"]  # type: ignore[assignment]
    n = x.shape[0]
    print(f"run={run_name} device={device} params={param_count(model)} start={start}")
    for step in range(start + 1, config.steps + 1):
        idx = torch.randint(0, n, (config.batch_size,), device=device)
        xb, yb, ab = x[idx], y[idx], a_seq[idx]
        z = model.encode(xb)
        z_next = model.encode(yb)
        z_hat = block_predict_latent(model, z, ab)
        loss_state = F.mse_loss(model.decode(z), xb) if model.use_decoder else xb.new_tensor(0.0)
        loss_dyn = F.mse_loss(z_hat, z_next)
        _, _, _, loss_reg = latent_regularization_objective(z, config)
        if config.use_latent_regularization and config.latent_regularization_type == "lip":
            _, _, _, _, lip_reg = lipschitz_regularization_objective(
                model,
                data,
                config,
                blocks_per_step=config.lip_blocks_per_step,
            )
            loss_reg = loss_reg + lip_reg
        loss = loss_state + config.lambda_dyn * loss_dyn + loss_reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % config.save_every == 0 or step == config.steps:
            cur_losses = losses(model, data, config)
            curves.append({"step": step, **cur_losses})
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "config": asdict(config),
                    "curves": curves,
                    "note": "No stop-gradient is used in L_dyn.",
                },
                ckpt,
            )
            print(
                f"step {step}/{config.steps} "
                f"L={cur_losses['train_total_loss']:.8f} "
                f"L_state={cur_losses['train_state_loss']:.8f} "
                f"L_dyn={cur_losses['train_dyn_h5_loss']:.8f} "
                f"L_sigreg={cur_losses['latent_sigreg_loss']:.8f} "
                f"L_lip={cur_losses['latent_lip_loss']:.8f} "
                f"L_reg={cur_losses['latent_reg_loss']:.8f}"
            )

    metrics = eval_model(model, config, data, run_dir)
    result = {
        "config": asdict(config),
        "maze_summary": maze_summary(maze),
        "metrics": metrics,
        "checkpoint": str(ckpt),
        "objective": (
            f"L_state + L_dyn_h{config.train_prediction_horizon}, lambda_dyn=1.0; "
            "predictor receives the full action block once; no intermediate, gain, or planning losses. "
            "Optional VICReg, SIGReg, or latent Lipschitz regularization is applied only when enabled. "
            "With --no-decoder, no decoder module is created and L_state is fixed to zero."
        ),
        "stop_gradient": "No stop-gradient is used for the encoded h-step target.",
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "config": asdict(config), "metrics": metrics}, run_dir / "final.pt")
    print(json.dumps(result, indent=2))
    return result


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, diagnostics = [], []
    for name in MAZES:
        maze = build_maze(name)
        summaries.append(maze_summary(maze))
        diagnostics.append({"maze": name, "raw_coordinate_gain_h5_local": raw_coord_gain(maze, 5)})
        write_csv(
            OUT / f"maze_{name}_transitions.csv",
            [
                {
                    "maze": name,
                    "state_id": sid(s),
                    "row": s[0],
                    "col": s[1],
                    "action": a,
                    "next_id": sid(t1(maze, s, a)),
                    "next_row": t1(maze, s, a)[0],
                    "next_col": t1(maze, s, a)[1],
                }
                for s in states()
                for a in ACTIONS
            ],
        )
        (OUT / f"maze_{name}_barriers.json").write_text(json.dumps(sorted(maze.barriers), indent=2) + "\n")
        write_png(OUT / f"maze_{name}.png", draw_maze(maze))
    write_png(OUT / "maze_overview.png", concat_images([draw_maze(build_maze(name)) for name in MAZES]))
    (OUT / "maze_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "diagnostics": diagnostics}, indent=2))


def draw_maze(maze: Maze) -> list[bytearray]:
    cell, margin = 36, 22
    w = h = N * cell + 2 * margin
    img = [bytearray((255, 255, 255) * w) for _ in range(h)]

    def px(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < w and 0 <= y < h:
            img[y][3 * x : 3 * x + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thick: int = 1) -> None:
        if x0 == x1:
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for dx in range(-(thick // 2), thick // 2 + 1):
                    px(x0 + dx, y, color)
        if y0 == y1:
            for x in range(min(x0, x1), max(x0, x1) + 1):
                for dy in range(-(thick // 2), thick // 2 + 1):
                    px(x, y0 + dy, color)

    for i in range(N + 1):
        line(margin + i * cell, margin, margin + i * cell, margin + N * cell, (220, 220, 220))
        line(margin, margin + i * cell, margin + N * cell, margin + i * cell, (220, 220, 220))
    for kind, r, c in maze.barriers:
        if kind == "V":
            line(margin + (c + 1) * cell, margin + r * cell, margin + (c + 1) * cell, margin + (r + 1) * cell, (0, 0, 0), 5)
        else:
            line(margin + c * cell, margin + (r + 1) * cell, margin + (c + 1) * cell, margin + (r + 1) * cell, (0, 0, 0), 5)
    for state, color in [(maze.start, (37, 99, 235)), (maze.goal, (220, 38, 38))]:
        cy, cx = margin + state[0] * cell + cell // 2, margin + state[1] * cell + cell // 2
        for y in range(cy - 8, cy + 9):
            for x in range(cx - 8, cx + 9):
                px(x, y, color)
    return img


def concat_images(images: list[list[bytearray]]) -> list[bytearray]:
    gap = 16
    h = max(len(img) for img in images)
    w = sum(len(img[0]) // 3 for img in images) + gap * (len(images) - 1)
    out = [bytearray((255, 255, 255) * w) for _ in range(h)]
    xoff = 0
    for img in images:
        iw = len(img[0]) // 3
        for y, row in enumerate(img):
            out[y][xoff * 3 : (xoff + iw) * 3] = row
        xoff += iw + gap
    return out


def write_png(path: Path, pixels: list[bytearray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(pixels)
    width = len(pixels[0]) // 3
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_many(label: str, mazes: list[str], dims: list[int], seeds: list[int], args) -> None:
    build()
    if args.no_decoder:
        label = f"{label}_nodecoder"
    if args.use_latent_regularization:
        if args.latent_regularization_type == "vicreg":
            label = (
                f"{label}_latreg_vicreg_var{args.latent_var_weight:g}"
                f"_cov{args.latent_cov_weight:g}"
                f"_std{args.latent_std_target:g}"
            )
        elif args.latent_regularization_type == "sigreg":
            label = (
                f"{label}_latreg_sigreg_{args.sigreg_mode}"
                f"_w{args.sigreg_weight:g}"
                f"_s{args.sigreg_slices}"
                f"_t{args.sigreg_t_points}"
                f"_tm{args.sigreg_t_max:g}"
            )
        elif args.latent_regularization_type == "lip":
            label = (
                f"{label}_latreg_lip_w{args.lip_weight:g}"
                f"_target{args.lip_target:g}"
                f"_top{args.lip_top_fraction:g}"
                f"_blocks{args.lip_blocks_per_step}"
            )
        else:
            raise ValueError(f"unknown latent_regularization_type: {args.latent_regularization_type}")
    results = []
    total = len(mazes) * len(dims) * len(seeds)
    done = 0
    for maze in mazes:
        for dim in dims:
            for seed in seeds:
                done += 1
                print(f"\n=== {label} {done}/{total}: maze={maze} m={dim} seed={seed} ===")
                results.append(
                    train(
                        TrainConfig(
                            maze=maze,
                            latent_dim=dim,
                            steps=args.steps,
                            seed=seed,
                            save_every=args.save_every,
                            resume=not args.no_resume,
                            eval_rollouts=args.eval_rollouts,
                            eval_plan_pairs=args.eval_plan_pairs,
                            train_prediction_horizon=args.train_prediction_horizon,
                            use_latent_regularization=args.use_latent_regularization,
                            latent_regularization_type=args.latent_regularization_type,
                            latent_var_weight=args.latent_var_weight,
                            latent_cov_weight=args.latent_cov_weight,
                            latent_std_target=args.latent_std_target,
                            sigreg_weight=args.sigreg_weight,
                            sigreg_mode=args.sigreg_mode,
                            sigreg_slices=args.sigreg_slices,
                            sigreg_t_points=args.sigreg_t_points,
                            sigreg_t_max=args.sigreg_t_max,
                            lip_weight=args.lip_weight,
                            lip_target=args.lip_target,
                            lip_top_fraction=args.lip_top_fraction,
                            lip_blocks_per_step=args.lip_blocks_per_step,
                            lip_eps=args.lip_eps,
                            use_decoder=not args.no_decoder,
                        )
                    )
                )
    report_dir = OUT / label
    report_dir.mkdir(parents=True, exist_ok=True)
    raw = report_dir / "raw_metrics.csv"
    write_metrics_csv(results, raw)
    aggregate(raw, report_dir / "aggregate_metrics.csv")
    plot(raw, report_dir / "plots")
    (report_dir / "report.json").write_text(json.dumps({"label": label, "results": results}, indent=2) + "\n")


def write_metrics_csv(results: list[dict[str, object]], path: Path) -> None:
    rows = [r["metrics"] for r in results]
    keys = sorted({k for row in rows for k in row})  # type: ignore[union-attr]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore[arg-type]


def aggregate(raw: Path, out: Path) -> None:
    with raw.open() as f:
        rows = list(csv.DictReader(f))
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["maze"], row["latent_dim"]), []).append(row)
    agg = []
    for (maze, dim), group in sorted(groups.items()):
        row: dict[str, object] = {"maze": maze, "latent_dim": dim, "n": len(group)}
        for key in group[0]:
            if key in {"maze", "latent_dim", "seed"} or group[0][key] == "":
                continue
            try:
                vals = [float(g[key]) for g in group if g[key] != ""]
            except ValueError:
                continue
            if vals:
                sd = pstdev(vals) if len(vals) > 1 else 0.0
                row[f"{key}_mean"] = mean(vals)
                row[f"{key}_std"] = sd
                row[f"{key}_ci95"] = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        agg.append(row)
    keys = sorted({k for row in agg for k in row})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(agg)


def plot(raw: Path, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plots_skipped.txt").write_text(f"{exc}\n")
        return
    with raw.open() as f:
        rows = list(csv.DictReader(f))
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("state_accuracy", "state_exact_accuracy", "exact state decoding accuracy"),
        ("Lambda1_p95", "Lambda1_p95", "Lambda1 p95"),
        ("Lambda5_p95", "Lambda5_p95", "Lambda5 p95"),
        ("h5_block_error", "h5_state_error", "5-step block prediction error"),
        ("planning_success", "planning_success_rate", "planning success rate"),
    ]
    for filename, metric, ylabel in specs:
        plt.figure(figsize=(5.5, 3.6))
        for maze in MAZES:
            xs, ys = [], []
            for dim in LATENT_DIMS:
                vals = [float(r[metric]) for r in rows if r["maze"] == maze and int(r["latent_dim"]) == dim]
                if vals:
                    xs.append(dim)
                    ys.append(mean(vals))
            if xs:
                plt.plot(xs, ys, marker="o", label=f"Maze {maze}")
        plt.xlabel("latent dimension")
        plt.ylabel(ylabel)
        plt.xscale("log", base=2)
        plt.xticks(LATENT_DIMS, [str(d) for d in LATENT_DIMS])
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{filename}.png", dpi=200)
        plt.savefig(out_dir / f"{filename}.pdf")
        plt.close()


def add_latent_regularization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-latent-regularization",
        action="store_true",
        help="Enable optional latent anti-collapse regularization on z=encoder(s_t).",
    )
    parser.add_argument(
        "--latent-regularization-type",
        choices=["vicreg", "sigreg", "lip"],
        default="vicreg",
    )
    parser.add_argument("--latent-var-weight", type=float, default=0.0)
    parser.add_argument("--latent-cov-weight", type=float, default=0.0)
    parser.add_argument("--latent-std-target", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-mode", choices=["sliced_w2", "ecf"], default="sliced_w2")
    parser.add_argument("--sigreg-slices", type=int, default=32)
    parser.add_argument("--sigreg-t-points", type=int, default=16)
    parser.add_argument("--sigreg-t-max", type=float, default=3.0)
    parser.add_argument("--lip-weight", "--gain-loss-weight", dest="lip_weight", type=float, default=0.0)
    parser.add_argument("--lip-target", type=float, default=1.0)
    parser.add_argument("--lip-top-fraction", type=float, default=0.05)
    parser.add_argument("--lip-blocks-per-step", type=int, default=8)
    parser.add_argument("--lip-eps", type=float, default=1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("params")

    train_p = sub.add_parser("train")
    train_p.add_argument("--maze", choices=MAZES, required=True)
    train_p.add_argument("--latent-dim", type=int, required=True)
    train_p.add_argument("--hidden-dim", type=int, default=128)
    train_p.add_argument("--steps", type=int, default=100000)
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--save-every", type=int, default=1000)
    train_p.add_argument("--no-resume", action="store_true")
    train_p.add_argument("--eval-rollouts", type=int, default=N_ROLLOUTS)
    train_p.add_argument("--eval-plan-pairs", type=int, default=N_PLAN_PAIRS)
    train_p.add_argument("--train-prediction-horizon", type=int, default=TRAIN_PREDICTION_HORIZON)
    train_p.add_argument("--no-decoder", action="store_true")
    add_latent_regularization_args(train_p)

    for name in ["pilot", "sweep"]:
        p = sub.add_parser(name)
        p.add_argument("--steps", type=int, default=100000)
        p.add_argument("--save-every", type=int, default=1000)
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("--confirm-maps", action="store_true")
        p.add_argument("--eval-rollouts", type=int, default=N_ROLLOUTS)
        p.add_argument("--eval-plan-pairs", type=int, default=N_PLAN_PAIRS)
        p.add_argument("--train-prediction-horizon", type=int, default=TRAIN_PREDICTION_HORIZON)
        p.add_argument("--no-decoder", action="store_true")
        add_latent_regularization_args(p)

    wide_p = sub.add_parser("wide-control")
    wide_p.add_argument("--steps", type=int, default=100000)
    wide_p.add_argument("--seed", type=int, default=0)
    wide_p.add_argument("--hidden-dim", type=int, default=132)
    wide_p.add_argument("--save-every", type=int, default=1000)
    wide_p.add_argument("--no-resume", action="store_true")
    wide_p.add_argument("--train-prediction-horizon", type=int, default=TRAIN_PREDICTION_HORIZON)
    wide_p.add_argument("--no-decoder", action="store_true")
    add_latent_regularization_args(wide_p)

    args = parser.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "params":
        for dim in LATENT_DIMS:
            model = WorldModel(dim, 128, TRAIN_PREDICTION_HORIZON)
            print(f"m={dim} hidden=128 block={TRAIN_PREDICTION_HORIZON} params={param_count(model)}")
        wide = WorldModel(2, 132, TRAIN_PREDICTION_HORIZON)
        print(f"m=2 hidden=132 block={TRAIN_PREDICTION_HORIZON} params={param_count(wide)}")
    elif args.cmd == "train":
        train(
            TrainConfig(
                maze=args.maze,
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                steps=args.steps,
                seed=args.seed,
                save_every=args.save_every,
                resume=not args.no_resume,
                eval_rollouts=args.eval_rollouts,
                eval_plan_pairs=args.eval_plan_pairs,
                train_prediction_horizon=args.train_prediction_horizon,
                use_latent_regularization=args.use_latent_regularization,
                latent_regularization_type=args.latent_regularization_type,
                latent_var_weight=args.latent_var_weight,
                latent_cov_weight=args.latent_cov_weight,
                latent_std_target=args.latent_std_target,
                sigreg_weight=args.sigreg_weight,
                sigreg_mode=args.sigreg_mode,
                sigreg_slices=args.sigreg_slices,
                sigreg_t_points=args.sigreg_t_points,
                sigreg_t_max=args.sigreg_t_max,
                lip_weight=args.lip_weight,
                lip_target=args.lip_target,
                lip_top_fraction=args.lip_top_fraction,
                lip_blocks_per_step=args.lip_blocks_per_step,
                lip_eps=args.lip_eps,
                use_decoder=not args.no_decoder,
            )
        )
    elif args.cmd == "pilot":
        run_many("pilot", ["C"], [2, 8], [0], args)
    elif args.cmd == "sweep":
        if not args.confirm_maps:
            raise SystemExit("Run build and inspect maze1/outputs/maze_*.png, then rerun sweep with --confirm-maps.")
        run_many("main_sweep", MAZES, LATENT_DIMS, SEEDS, args)
    elif args.cmd == "wide-control":
        train(
            TrainConfig(
                maze="C",
                latent_dim=2,
                hidden_dim=args.hidden_dim,
                steps=args.steps,
                seed=args.seed,
                save_every=args.save_every,
                resume=not args.no_resume,
                train_prediction_horizon=args.train_prediction_horizon,
                use_latent_regularization=args.use_latent_regularization,
                latent_regularization_type=args.latent_regularization_type,
                latent_var_weight=args.latent_var_weight,
                latent_cov_weight=args.latent_cov_weight,
                latent_std_target=args.latent_std_target,
                sigreg_weight=args.sigreg_weight,
                sigreg_mode=args.sigreg_mode,
                sigreg_slices=args.sigreg_slices,
                sigreg_t_points=args.sigreg_t_points,
                sigreg_t_max=args.sigreg_t_max,
                lip_weight=args.lip_weight,
                lip_target=args.lip_target,
                lip_top_fraction=args.lip_top_fraction,
                lip_blocks_per_step=args.lip_blocks_per_step,
                lip_eps=args.lip_eps,
                use_decoder=not args.no_decoder,
            )
        )


if __name__ == "__main__":
    main()
