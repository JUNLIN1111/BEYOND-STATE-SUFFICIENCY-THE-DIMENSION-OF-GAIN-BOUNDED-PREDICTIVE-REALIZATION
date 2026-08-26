"""CPU experiments for finite-system dimension-gain geometry.

This standalone script directly optimizes finite latent embeddings.  It does
not train an encoder, predictor, world model, policy, or RL benchmark.

For a deterministic controlled transition table T[action, state], and an
injective embedding Z in R^{n x m}, the represented-state transition gain is

    Lambda_T(Z) = max_a max_{i != j}
        ||z_{T_a(i)} - z_{T_a(j)}||_2 / ||z_i - z_j||_2.

The experiments report exact hard gains.  The optimized values are empirical
feasible upper bounds on L_m^*(T), never lower-bound certificates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DTYPE = torch.float64
CPU = torch.device("cpu")
COLLISION_TOL = 1e-10
FAST_AGREE_ATOL = 1e-10
FEASIBILITY_TOL_DEFAULT = 1e-3

SYSTEM_ADJACENT = "AdjacentSwap"
SYSTEM_CYCLE = "Cycle"
SYSTEM_ALLSWAP = "AllSwap"

E1_N_DEFAULT = tuple(range(4, 11))
E1_N_QUICK = (4, 6)
E2_N_DEFAULT = (8, 16, 32, 64, 128)
E2_N_QUICK = (8, 16)
E2_M_DEFAULT = (1, 2, 3, 4, 6, 8)
E2_M_QUICK = (1, 2, 4)
PLOT_E1_N_PREFERRED = (4, 6, 8, 10)


@dataclass(frozen=True)
class PreparedSystem:
    """Transition table plus cached pair indices for optimization."""

    name: str
    n: int
    transitions_np: Optional[np.ndarray]
    transitions: Optional[torovch.Tensor]
    pair_i: torch.Tensor
    pair_j: torch.Tensor
    succ_i: Optional[torch.Tensor]
    succ_j: Optional[torch.Tensor]


@dataclass(frozen=True)
class OptimizeConfig:
    """Hyperparameters for direct embedding optimization."""

    steps: int
    seeds: int
    lr: float
    beta_values: Tuple[float, float, float]
    beta_boundaries: Tuple[float, float]
    eps: float
    clip_norm: float
    patience: int
    improvement_tol: float
    exact_tolerance: float


@dataclass(frozen=True)
class OptimizationOutcome:
    """A single optimized embedding and its run metadata."""

    row: Dict[str, Any]
    initial_z: torch.Tensor
    final_z: torch.Tensor
    best_z: torch.Tensor


def adjacent_swap_system(n: int) -> np.ndarray:
    """Return AdjacentSwapSystem(n) as an integer array of shape (n-1, n)."""

    if n < 2:
        raise ValueError("AdjacentSwapSystem requires n >= 2")
    identity = np.arange(n, dtype=np.int64)
    transitions = np.empty((n - 1, n), dtype=np.int64)
    for k in range(n - 1):
        row = identity.copy()
        row[k], row[k + 1] = row[k + 1], row[k]
        transitions[k] = row
    return transitions


def cycle_system(n: int) -> np.ndarray:
    """Return CycleSystem(n) as an integer array of shape (1, n)."""

    if n < 2:
        raise ValueError("CycleSystem requires n >= 2")
    return ((np.arange(n, dtype=np.int64) + 1) % n).reshape(1, n)


def allswap_system(n: int) -> np.ndarray:
    """Return AllSwapSystem(n), one transposition for every unordered pair."""

    if n < 2:
        raise ValueError("AllSwapSystem requires n >= 2")
    identity = np.arange(n, dtype=np.int64)
    rows: List[np.ndarray] = []
    for p in range(n):
        for q in range(p + 1, n):
            row = identity.copy()
            row[p], row[q] = row[q], row[p]
            rows.append(row)
    return np.stack(rows, axis=0)


def transition_system(name: str, n: int) -> np.ndarray:
    """Build one of the named transition systems."""

    if name == SYSTEM_ADJACENT:
        return adjacent_swap_system(n)
    if name == SYSTEM_CYCLE:
        return cycle_system(n)
    if name == SYSTEM_ALLSWAP:
        return allswap_system(n)
    raise ValueError(f"Unknown system {name!r}")


def unordered_pair_indices_np(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return every unordered pair i < j exactly once."""

    pair_i: List[int] = []
    pair_j: List[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_i.append(i)
            pair_j.append(j)
    return np.asarray(pair_i, dtype=np.int64), np.asarray(pair_j, dtype=np.int64)


def unordered_pair_indices_torch(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch version of unordered_pair_indices_np on CPU."""

    pair_i_np, pair_j_np = unordered_pair_indices_np(n)
    return (
        torch.as_tensor(pair_i_np, dtype=torch.long, device=CPU),
        torch.as_tensor(pair_j_np, dtype=torch.long, device=CPU),
    )


def assert_valid_transition_table(transitions: np.ndarray, n: int) -> None:
    """Check transition table shape and state-index validity."""

    assert transitions.ndim == 2, "Transition table must be rank two"
    assert transitions.shape[1] == n, (
        f"Transition table second dimension must be n={n}, got {transitions.shape}"
    )
    assert np.issubdtype(transitions.dtype, np.integer), (
        f"Transition table must use integer indices, got {transitions.dtype}"
    )
    assert int(transitions.min()) >= 0, "Transition table contains negative state index"
    assert int(transitions.max()) < n, "Transition table contains out-of-range state index"


def assert_permutation_rows(transitions: np.ndarray, n: int, label: str) -> None:
    """Check that every action row is a permutation."""

    expected = list(range(n))
    for action, row in enumerate(transitions):
        assert sorted(row.tolist()) == expected, (
            f"{label} action {action} is not a permutation of 0..{n - 1}"
        )


def assert_involution_rows(transitions: np.ndarray, n: int, label: str) -> None:
    """Check that every action row is an involution."""

    identity = np.arange(n, dtype=np.int64)
    for action, row in enumerate(transitions):
        assert np.array_equal(row[row], identity), (
            f"{label} action {action} is not an involution"
        )


def theoretical_l1_threshold(system_name: str, n: int) -> int:
    """Analytic gain-one predictive-realization threshold."""

    if system_name == SYSTEM_CYCLE:
        return 2
    if system_name in (SYSTEM_ADJACENT, SYSTEM_ALLSWAP):
        return n - 1
    raise ValueError(f"Unknown system {system_name!r}")


def prepare_system(
    system_name: str,
    n: int,
    build_successor_pairs: bool,
    include_transitions: bool = True,
) -> PreparedSystem:
    """Prepare cached indices.  Avoid huge AllSwap successor tensors in E2."""

    transitions_np = transition_system(system_name, n) if include_transitions else None
    transitions: Optional[torch.Tensor] = None
    succ_i: Optional[torch.Tensor] = None
    succ_j: Optional[torch.Tensor] = None
    if transitions_np is not None:
        assert_valid_transition_table(transitions_np, n)
        transitions = torch.as_tensor(transitions_np, dtype=torch.long, device=CPU)
    pair_i, pair_j = unordered_pair_indices_torch(n)
    if build_successor_pairs:
        if transitions is None:
            raise ValueError("Successor pair cache requires transitions")
        succ_i = transitions[:, pair_i]
        succ_j = transitions[:, pair_j]
    return PreparedSystem(
        name=system_name,
        n=n,
        transitions_np=transitions_np,
        transitions=transitions,
        pair_i=pair_i,
        pair_j=pair_j,
        succ_i=succ_i,
        succ_j=succ_j,
    )


def pairwise_squared_distances(z: torch.Tensor) -> torch.Tensor:
    """Return all squared Euclidean distances between rows of z."""

    diff = z[:, None, :] - z[None, :, :]
    return (diff * diff).sum(dim=-1)


def required_gain_generic(z: torch.Tensor, transitions: torch.Tensor) -> torch.Tensor:
    """Exact hard max over all actions and unordered state pairs."""

    z = z.to(dtype=DTYPE, device=CPU)
    transitions = transitions.to(dtype=torch.long, device=CPU)
    n = int(z.shape[0])
    assert transitions.ndim == 2 and transitions.shape[1] == n
    pair_i, pair_j = unordered_pair_indices_torch(n)
    d2 = pairwise_squared_distances(z)
    base_d2 = d2[pair_i, pair_j]
    if bool(torch.any(base_d2 <= 0.0)) or not bool(torch.all(torch.isfinite(base_d2))):
        return torch.tensor(float("inf"), dtype=DTYPE, device=CPU)
    succ_i = transitions[:, pair_i]
    succ_j = transitions[:, pair_j]
    succ_d2 = d2[succ_i, succ_j]
    ratios = torch.sqrt(succ_d2 / base_d2.view(1, -1))
    if not bool(torch.all(torch.isfinite(ratios))):
        return torch.tensor(float("inf"), dtype=DTYPE, device=CPU)
    return ratios.max()


def required_gain_generic_prepared(z: torch.Tensor, system: PreparedSystem) -> torch.Tensor:
    """Exact hard gain using cached generic successor-pair tensors."""

    if system.succ_i is None or system.succ_j is None:
        if system.transitions is None:
            raise ValueError("Generic gain requires a transition table")
        return required_gain_generic(z, system.transitions)
    d2 = pairwise_squared_distances(z)
    base_d2 = d2[system.pair_i, system.pair_j]
    if bool(torch.any(base_d2 <= 0.0)) or not bool(torch.all(torch.isfinite(base_d2))):
        return torch.tensor(float("inf"), dtype=DTYPE, device=CPU)
    succ_d2 = d2[system.succ_i, system.succ_j]
    ratios = torch.sqrt(succ_d2 / base_d2.view(1, -1))
    if not bool(torch.all(torch.isfinite(ratios))):
        return torch.tensor(float("inf"), dtype=DTYPE, device=CPU)
    return ratios.max()


def required_gain_allswap_fast(z: torch.Tensor) -> torch.Tensor:
    """Exact AllSwap gain via max row aspect ratio."""

    z = z.to(dtype=DTYPE, device=CPU)
    n = int(z.shape[0])
    d = torch.sqrt(torch.clamp(pairwise_squared_distances(z), min=0.0))
    finite = torch.isfinite(d)
    if not bool(torch.all(finite)):
        return torch.tensor(float("inf"), dtype=DTYPE, device=CPU)
    eye = torch.eye(n, dtype=torch.bool, device=CPU)
    row_min = d.masked_fill(eye, float("inf")).min(dim=1).values
    row_max = d.masked_fill(eye, 0.0).max(dim=1).values
    if bool(torch.any(row_min <= 0.0)) or not bool(torch.all(torch.isfinite(row_min))):
        return torch.tensor(float("inf"), dtype=DTYPE, device=CPU)
    ratios = row_max / row_min
    if not bool(torch.all(torch.isfinite(ratios))):
        return torch.tensor(float("inf"), dtype=DTYPE, device=CPU)
    return ratios.max()


def hard_required_gain(z: torch.Tensor, system: PreparedSystem) -> float:
    """Return the reported exact hard gain as a Python float."""

    with torch.no_grad():
        if system.name == SYSTEM_ALLSWAP:
            value = required_gain_allswap_fast(z.detach())
        else:
            value = required_gain_generic_prepared(z.detach(), system)
        return float(value.item())


def smooth_logmax_generic_loss(
    z: torch.Tensor,
    system: PreparedSystem,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Smooth max of generic log distance ratios for optimization only."""

    if system.succ_i is None or system.succ_j is None:
        raise ValueError("Generic objective requires cached successor-pair tensors")
    d2 = pairwise_squared_distances(z)
    base = torch.sqrt(torch.clamp(d2[system.pair_i, system.pair_j], min=0.0))
    succ = torch.sqrt(torch.clamp(d2[system.succ_i, system.succ_j], min=0.0))
    log_ratios = torch.log(succ + eps) - torch.log(base.view(1, -1) + eps)
    return torch.logsumexp(beta * log_ratios.reshape(-1), dim=0) / beta


def smooth_logmax_allswap_fast_loss(z: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    """Smooth AllSwap row-aspect objective for large E2 runs."""

    n = int(z.shape[0])
    d = torch.sqrt(torch.clamp(pairwise_squared_distances(z), min=0.0))
    log_d = torch.log(d + eps)
    eye = torch.eye(n, dtype=torch.bool, device=CPU)
    row_for_max = log_d.masked_fill(eye, float("-inf"))
    row_for_min = log_d.masked_fill(eye, float("inf"))
    smooth_row_max = torch.logsumexp(beta * row_for_max, dim=1) / beta
    smooth_row_min = -torch.logsumexp(-beta * row_for_min, dim=1) / beta
    row_aspects = smooth_row_max - smooth_row_min
    return torch.logsumexp(beta * row_aspects, dim=0) / beta


def beta_at_step(step: int, total_steps: int, config: OptimizeConfig) -> float:
    """Piecewise constant beta schedule."""

    if total_steps <= 0:
        return config.beta_values[-1]
    progress = (step - 1) / float(total_steps)
    if progress < config.beta_boundaries[0]:
        return config.beta_values[0]
    if progress < config.beta_boundaries[1]:
        return config.beta_values[1]
    return config.beta_values[2]


def regular_simplex(n: int) -> torch.Tensor:
    """Construct n centered equidistant points in R^{n-1}."""

    if n < 2:
        raise ValueError("regular_simplex requires n >= 2")
    eye = torch.eye(n, dtype=DTYPE, device=CPU)
    ones = torch.ones((n, n), dtype=DTYPE, device=CPU)
    gram = eye - ones / float(n)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    positive = eigenvalues > 1e-12
    assert int(positive.sum().item()) == n - 1, (
        f"Expected {n - 1} positive simplex eigenvalues, got {int(positive.sum().item())}"
    )
    return eigenvectors[:, positive] * torch.sqrt(eigenvalues[positive]).view(1, -1)


def regular_polygon(n: int) -> torch.Tensor:
    """Construct n equally spaced points on the unit circle in R^2."""

    if n < 3:
        raise ValueError("regular_polygon requires n >= 3")
    angles = 2.0 * math.pi * torch.arange(n, dtype=DTYPE, device=CPU) / float(n)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def grid_configuration(n: int, m: int) -> torch.Tensor:
    """Return the first n points from {0, ..., k-1}^m with k=ceil(n^(1/m))."""

    if n < 1:
        raise ValueError("grid_configuration requires n >= 1")
    if m < 1:
        raise ValueError("grid_configuration requires m >= 1")
    k = int(math.ceil(n ** (1.0 / float(m))))
    coords = np.zeros((n, m), dtype=np.float64)
    for idx in range(n):
        value = idx
        for dim in range(m):
            coords[idx, dim] = value % k
            value //= k
    return torch.as_tensor(coords, dtype=DTYPE, device=CPU)


def pad_embedding(z: torch.Tensor, m: int) -> torch.Tensor:
    """Pad an embedding with zero columns to reach dimension m."""

    if z.shape[1] > m:
        raise ValueError(f"Cannot pad from dimension {z.shape[1]} down to {m}")
    if z.shape[1] == m:
        return z.clone()
    zeros = torch.zeros((z.shape[0], m - z.shape[1]), dtype=DTYPE, device=CPU)
    return torch.cat([z.to(dtype=DTYPE, device=CPU), zeros], dim=1)


def centered_min_distance_normalize(
    z: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], float]:
    """Center rows and scale so the minimum pairwise distance is one."""

    centered = z.to(dtype=DTYPE, device=CPU) - z.to(dtype=DTYPE, device=CPU).mean(
        dim=0,
        keepdim=True,
    )
    d2 = pairwise_squared_distances(centered)
    pair_d = torch.sqrt(torch.clamp(d2[pair_i, pair_j], min=0.0))
    d_min = pair_d.min()
    d_min_value = float(d_min.item())
    if not math.isfinite(d_min_value) or d_min_value < COLLISION_TOL:
        return None, d_min_value
    return centered / d_min, d_min_value


def project_parameter_(raw_z: torch.nn.Parameter, system: PreparedSystem) -> bool:
    """Apply the required no-grad centering/min-distance projection."""

    with torch.no_grad():
        projected, _ = centered_min_distance_normalize(
            raw_z.detach(),
            system.pair_i,
            system.pair_j,
        )
        if projected is None:
            return False
        raw_z.copy_(projected)
    return True


def pairwise_distance_stats(z: torch.Tensor, system: PreparedSystem) -> Tuple[float, float]:
    """Return min and max nonzero pairwise distances."""

    with torch.no_grad():
        d2 = pairwise_squared_distances(z)
        distances = torch.sqrt(torch.clamp(d2[system.pair_i, system.pair_j], min=0.0))
        return float(distances.min().item()), float(distances.max().item())


def make_optimizer(
    raw_z: torch.nn.Parameter,
    config: OptimizeConfig,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """Create the Adam optimizer and cosine learning-rate scheduler."""

    optimizer = torch.optim.Adam([raw_z], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.steps),
    )
    return optimizer, scheduler


def seed_everything(seed: int) -> torch.Generator:
    """Set deterministic NumPy and PyTorch seeds and return a CPU generator."""

    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device=CPU)
    generator.manual_seed(seed)
    return generator


def random_normalized_embedding(
    n: int,
    m: int,
    system: PreparedSystem,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, int]:
    """Sample Gaussian embeddings until the projected configuration is injective."""

    failed_attempts = 0
    while True:
        z0 = torch.randn((n, m), dtype=DTYPE, device=CPU, generator=generator)
        projected, _ = centered_min_distance_normalize(z0, system.pair_i, system.pair_j)
        if projected is not None:
            return projected, failed_attempts
        failed_attempts += 1
        if failed_attempts > 1000:
            raise RuntimeError("Could not initialize a collision-free Gaussian embedding")


def training_loss(
    z: torch.Tensor,
    system: PreparedSystem,
    objective: str,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Dispatch the smooth optimization objective."""

    if objective == "allswap_fast":
        return smooth_logmax_allswap_fast_loss(z, beta=beta, eps=eps)
    if objective == "generic":
        return smooth_logmax_generic_loss(z, system=system, beta=beta, eps=eps)
    raise ValueError(f"Unknown objective {objective!r}")


def save_checkpoint(
    path: Path,
    outcome_row: Dict[str, Any],
    best_z: torch.Tensor,
    final_z: torch.Tensor,
) -> None:
    """Write one embedding checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": outcome_row,
            "Z": best_z.detach().cpu(),
            "final_Z": final_z.detach().cpu(),
            "best_required_gain": float(outcome_row["best_required_gain"]),
        },
        path,
    )


def optimize_embedding(
    system: PreparedSystem,
    m: int,
    config: OptimizeConfig,
    seed: int,
    init_kind: str,
    objective: str,
    save_path: Optional[Path],
    initial_z: Optional[torch.Tensor] = None,
    run_type: str = "random",
) -> OptimizationOutcome:
    """Optimize only a free embedding matrix raw_Z in R^{n x m}."""

    generator = seed_everything(max(0, seed))
    restart_count = 0
    if initial_z is None:
        z0, restart_count = random_normalized_embedding(system.n, m, system, generator)
    else:
        if initial_z.shape != (system.n, m):
            raise ValueError(
                f"Initial embedding shape must be {(system.n, m)}, got {tuple(initial_z.shape)}"
            )
        projected, d_min = centered_min_distance_normalize(
            initial_z,
            system.pair_i,
            system.pair_j,
        )
        if projected is None:
            raise RuntimeError(
                f"Initial embedding {init_kind!r} is not injective; min distance {d_min}"
            )
        z0 = projected

    initial_gain = hard_required_gain(z0, system)
    raw_z = torch.nn.Parameter(z0.clone())
    optimizer, scheduler = make_optimizer(raw_z, config)

    best_z = raw_z.detach().clone()
    best_gain = initial_gain
    final_gain = initial_gain
    final_z = raw_z.detach().clone()
    steps_run = 0
    last_improvement_step = 0
    stopped_reason = "max_steps"
    target_gain = 1.0 + config.exact_tolerance

    for step in range(1, config.steps + 1):
        if best_gain <= target_gain:
            stopped_reason = "gain_one_reached"
            break

        beta = beta_at_step(step, config.steps, config)
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(raw_z, system, objective=objective, beta=beta, eps=config.eps)

        if not bool(torch.isfinite(loss)):
            restart_count += 1
            z0, extra_restarts = random_normalized_embedding(system.n, m, system, generator)
            restart_count += extra_restarts
            raw_z = torch.nn.Parameter(z0.clone())
            optimizer, scheduler = make_optimizer(raw_z, config)
            final_gain = hard_required_gain(raw_z.detach(), system)
            final_z = raw_z.detach().clone()
            if final_gain < best_gain - config.improvement_tol:
                best_gain = final_gain
                best_z = raw_z.detach().clone()
                last_improvement_step = step
            steps_run = step
            stopped_reason = "restarted_after_nonfinite_loss"
            continue

        loss.backward()
        if config.clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([raw_z], max_norm=config.clip_norm)
        optimizer.step()
        scheduler.step()
        steps_run = step

        if not project_parameter_(raw_z, system):
            restart_count += 1
            z0, extra_restarts = random_normalized_embedding(system.n, m, system, generator)
            restart_count += extra_restarts
            raw_z = torch.nn.Parameter(z0.clone())
            optimizer, scheduler = make_optimizer(raw_z, config)
            final_gain = hard_required_gain(raw_z.detach(), system)
            final_z = raw_z.detach().clone()
            if final_gain < best_gain - config.improvement_tol:
                best_gain = final_gain
                best_z = raw_z.detach().clone()
                last_improvement_step = step
            stopped_reason = "restarted_after_collision"
            continue

        final_gain = hard_required_gain(raw_z.detach(), system)
        final_z = raw_z.detach().clone()
        if final_gain < best_gain - config.improvement_tol:
            best_gain = final_gain
            best_z = raw_z.detach().clone()
            last_improvement_step = step

        if best_gain <= target_gain:
            stopped_reason = "gain_one_reached"
            break
        if step - last_improvement_step >= config.patience:
            stopped_reason = "patience"
            break

    min_pairwise, max_pairwise = pairwise_distance_stats(best_z, system)
    row = {
        "system": system.name,
        "n": system.n,
        "m": m,
        "seed": seed,
        "run_type": run_type,
        "init_kind": init_kind,
        "objective": objective,
        "initial_required_gain": initial_gain,
        "best_required_gain": best_gain,
        "final_required_gain": final_gain,
        "min_pairwise_distance": min_pairwise,
        "max_pairwise_distance": max_pairwise,
        "steps_run": steps_run,
        "restart_count": restart_count,
        "converged": bool(best_gain <= target_gain),
        "stopped_reason": stopped_reason,
    }
    if save_path is not None:
        save_checkpoint(save_path, row, best_z, final_z)
    return OptimizationOutcome(row=row, initial_z=z0, final_z=final_z, best_z=best_z)


def allswap_bounds(n: int, m: int) -> Tuple[float, float]:
    """Analytic AllSwap dimension-gain lower and grid upper bounds."""

    lower = max(1.0, (n ** (1.0 / float(m)) - 1.0) / 4.0)
    upper = math.sqrt(float(m)) * (math.ceil(n ** (1.0 / float(m))) - 1.0)
    return float(lower), float(upper)


def grid_upper_bound_value(n: int, m: int) -> float:
    """The analytic gain bound of the explicit grid construction."""

    k = int(math.ceil(n ** (1.0 / float(m))))
    return math.sqrt(float(m)) * float(k - 1)


def e1_explicit_initializations(system_name: str, n: int, m: int) -> List[Tuple[str, torch.Tensor]]:
    """Deterministic candidates kept separate from random seed statistics."""

    inits: List[Tuple[str, torch.Tensor]] = []
    if system_name == SYSTEM_CYCLE and m >= 2:
        inits.append(("regular_polygon", pad_embedding(regular_polygon(n), m)))
    if m >= n - 1:
        inits.append(("regular_simplex", pad_embedding(regular_simplex(n), m)))
    if system_name == SYSTEM_ALLSWAP:
        inits.append(("grid", grid_configuration(n, m)))
    return inits


def e2_explicit_initializations(n: int, m: int) -> List[Tuple[str, torch.Tensor]]:
    """Explicit AllSwap initializations used in E2 optimization."""

    inits = [("grid", grid_configuration(n, m))]
    if m >= n - 1:
        inits.append(("regular_simplex", pad_embedding(regular_simplex(n), m)))
    return inits


def assert_pair_indexing(n: int) -> None:
    """Ensure pair indexing contains every unordered pair exactly once."""

    pair_i, pair_j = unordered_pair_indices_np(n)
    observed = list(zip(pair_i.tolist(), pair_j.tolist()))
    expected = [(i, j) for i in range(n) for j in range(i + 1, n)]
    assert observed == expected, f"Pair ordering mismatch for n={n}"
    assert len(observed) == n * (n - 1) // 2, f"Wrong pair count for n={n}"
    assert len(set(observed)) == len(observed), f"Duplicate pair indices for n={n}"
    assert all(i < j for i, j in observed), f"Non-unordered pair included for n={n}"


def assert_transition_system_tests() -> None:
    """Run transition-table validity, permutation, and involution tests."""

    for n in range(3, 11):
        adj = adjacent_swap_system(n)
        cyc = cycle_system(n)
        allswap = allswap_system(n)
        for transitions in (adj, cyc, allswap):
            assert_valid_transition_table(transitions, n)
        assert_permutation_rows(adj, n, SYSTEM_ADJACENT)
        assert_involution_rows(adj, n, SYSTEM_ADJACENT)
        assert_permutation_rows(allswap, n, SYSTEM_ALLSWAP)
        assert_involution_rows(allswap, n, SYSTEM_ALLSWAP)
        assert_permutation_rows(cyc, n, SYSTEM_CYCLE)


def assert_generic_fast_allswap_agree() -> None:
    """Check generic and fast AllSwap gains on small random configurations."""

    for n in range(3, 9):
        transitions = torch.as_tensor(allswap_system(n), dtype=torch.long, device=CPU)
        for m in range(1, min(5, n)):
            for seed in range(4):
                generator = seed_everything(1000 + 31 * n + 7 * m + seed)
                dummy_system = prepare_system(SYSTEM_ALLSWAP, n, build_successor_pairs=False)
                z, _ = random_normalized_embedding(n, m, dummy_system, generator)
                generic = required_gain_generic(z, transitions)
                fast = required_gain_allswap_fast(z)
                assert torch.allclose(generic, fast, atol=FAST_AGREE_ATOL, rtol=1e-10), (
                    f"AllSwap generic/fast mismatch n={n} m={m}: "
                    f"{float(generic.item())} vs {float(fast.item())}"
                )


def assert_regular_simplex_checks() -> None:
    """Check regular simplex shape, centering, equidistance, and gain one."""

    rng = np.random.default_rng(909)
    for n in range(3, 11):
        z = regular_simplex(n)
        assert z.shape == (n, n - 1), f"Bad simplex shape for n={n}: {tuple(z.shape)}"
        center_norm = float(torch.linalg.norm(z.mean(dim=0)).item())
        assert center_norm <= 1e-12, f"Simplex is not centered for n={n}: {center_norm}"
        pair_i, pair_j = unordered_pair_indices_torch(n)
        d2 = pairwise_squared_distances(z)
        pair_d2 = d2[pair_i, pair_j]
        spread = float((pair_d2.max() - pair_d2.min()).item())
        assert spread <= 1e-10, f"Simplex pairwise distances differ for n={n}: {spread}"

        adj = torch.as_tensor(adjacent_swap_system(n), dtype=torch.long, device=CPU)
        allswap = torch.as_tensor(allswap_system(n), dtype=torch.long, device=CPU)
        assert float(required_gain_generic(z, adj).item()) <= 1.0 + 1e-10
        assert float(required_gain_generic(z, allswap).item()) <= 1.0 + 1e-10

        arbitrary = torch.as_tensor(
            rng.integers(low=0, high=n, size=(3, n), dtype=np.int64),
            dtype=torch.long,
            device=CPU,
        )
        assert float(required_gain_generic(z, arbitrary).item()) <= 1.0 + 1e-10


def assert_regular_polygon_cycle_check() -> None:
    """Check that the regular polygon realizes CycleSystem with gain one."""

    for n in range(3, 21):
        z = regular_polygon(n)
        transitions = torch.as_tensor(cycle_system(n), dtype=torch.long, device=CPU)
        gain = float(required_gain_generic(z, transitions).item())
        assert gain <= 1.0 + 1e-10, f"Cycle polygon gain failed for n={n}: {gain}"


def assert_grid_checks() -> None:
    """Check that grid AllSwap gain is within the analytic grid upper bound."""

    for n in (4, 5, 8, 16, 32):
        for m in (1, 2, 3, 4, 6):
            if m >= n:
                continue
            z = grid_configuration(n, m)
            gain = float(required_gain_allswap_fast(z).item())
            bound = grid_upper_bound_value(n, m)
            assert gain <= bound + 1e-10, (
                f"Grid gain exceeds bound n={n} m={m}: {gain} > {bound}"
            )


def assert_gain_invariances() -> None:
    """Check translation, positive scaling, and orthogonal invariance."""

    rng = np.random.default_rng(12345)
    system = prepare_system(SYSTEM_ADJACENT, 7, build_successor_pairs=True)
    z = torch.as_tensor(rng.normal(size=(7, 4)), dtype=DTYPE, device=CPU)
    projected, _ = centered_min_distance_normalize(z, system.pair_i, system.pair_j)
    assert projected is not None
    z = projected
    base_gain = hard_required_gain(z, system)

    shift = torch.as_tensor([[1.0, -2.0, 0.5, 3.0]], dtype=DTYPE, device=CPU)
    translated = hard_required_gain(z + shift, system)
    scaled = hard_required_gain(6.75 * z, system)

    matrix = torch.as_tensor(rng.normal(size=(4, 4)), dtype=DTYPE, device=CPU)
    q, _ = torch.linalg.qr(matrix)
    rotated = hard_required_gain(z @ q, system)

    assert abs(translated - base_gain) <= 1e-10
    assert abs(scaled - base_gain) <= 1e-10
    assert abs(rotated - base_gain) <= 1e-10


def assert_fixed_seed_determinism() -> None:
    """A fixed quick-mode seed should reproduce initial and final outputs."""

    system = prepare_system(SYSTEM_ADJACENT, 4, build_successor_pairs=True)
    config = OptimizeConfig(
        steps=30,
        seeds=1,
        lr=2e-2,
        beta_values=(10.0, 30.0, 100.0),
        beta_boundaries=(0.3, 0.6),
        eps=1e-12,
        clip_norm=10.0,
        patience=2500,
        improvement_tol=1e-9,
        exact_tolerance=1e-9,
    )
    first = optimize_embedding(
        system=system,
        m=2,
        config=config,
        seed=777,
        init_kind="random_gaussian",
        objective="generic",
        save_path=None,
    )
    second = optimize_embedding(
        system=system,
        m=2,
        config=config,
        seed=777,
        init_kind="random_gaussian",
        objective="generic",
        save_path=None,
    )
    assert torch.allclose(first.initial_z, second.initial_z, atol=0.0, rtol=0.0)
    assert torch.allclose(first.final_z, second.final_z, atol=0.0, rtol=0.0)
    for field in ("initial_required_gain", "best_required_gain", "final_required_gain"):
        assert abs(float(first.row[field]) - float(second.row[field])) <= 1e-12
    assert int(first.row["steps_run"]) == int(second.row["steps_run"])
    assert int(first.row["restart_count"]) == int(second.row["restart_count"])


def run_all_tests() -> None:
    """Run all numerical and analytic sanity checks before experiments."""

    for n in range(3, 12):
        assert_pair_indexing(n)
    assert_transition_system_tests()
    assert_generic_fast_allswap_agree()
    assert_regular_simplex_checks()
    assert_regular_polygon_cycle_check()
    assert_grid_checks()
    assert_gain_invariances()
    assert_fixed_seed_determinism()


def summarize_e1(
    random_df: pd.DataFrame,
    explicit_df: pd.DataFrame,
    feasibility_tol: float,
) -> pd.DataFrame:
    """Aggregate E1 random-seed statistics and separate explicit upper bounds."""

    rows: List[Dict[str, Any]] = []
    group_cols = ["system", "n", "m"]
    explicit_lookup: Dict[Tuple[str, int, int], float] = {}
    if len(explicit_df):
        for key, group in explicit_df.groupby(group_cols, sort=True):
            explicit_lookup[(str(key[0]), int(key[1]), int(key[2]))] = float(
                group["best_required_gain"].min(),
            )

    for key, group in random_df.groupby(group_cols, sort=True):
        system_name, n, m = str(key[0]), int(key[1]), int(key[2])
        gains = group["best_required_gain"].astype(float)
        best_random = float(gains.min())
        best_explicit = explicit_lookup.get((system_name, n, m), math.nan)
        candidates = [best_random]
        if math.isfinite(best_explicit):
            candidates.append(best_explicit)
        best_feasible = min(candidates)
        rows.append(
            {
                "system": system_name,
                "n": n,
                "m": m,
                "theoretical_L1_threshold": theoretical_l1_threshold(system_name, n),
                "best_required_gain": best_random,
                "median_required_gain": float(gains.median()),
                "mean_required_gain": float(gains.mean()),
                "std_required_gain": float(gains.std(ddof=0)),
                "empirical_L1_feasible": bool(best_random <= 1.0 + feasibility_tol),
                "best_explicit_gain": best_explicit,
                "best_feasible_upper_gain": best_feasible,
                "empirical_or_explicit_L1_feasible": bool(best_feasible <= 1.0 + feasibility_tol),
            }
        )
    return pd.DataFrame(rows)


def make_e1_threshold_table(summary: pd.DataFrame, feasibility_tol: float) -> pd.DataFrame:
    """Compare empirical and analytic thresholds for each finite system."""

    rows: List[Dict[str, Any]] = []
    for (system_name, n), group in summary.groupby(["system", "n"], sort=True):
        group = group.sort_values("m")
        random_feasible = group[group["empirical_L1_feasible"]]
        upper_feasible = group[group["empirical_or_explicit_L1_feasible"]]
        random_threshold = (
            int(random_feasible["m"].min()) if len(random_feasible) else math.nan
        )
        upper_threshold = (
            int(upper_feasible["m"].min()) if len(upper_feasible) else math.nan
        )
        rows.append(
            {
                "system": system_name,
                "n": int(n),
                "analytic_L1_threshold": theoretical_l1_threshold(str(system_name), int(n)),
                f"empirical_random_threshold_tol_{feasibility_tol:g}": random_threshold,
                f"empirical_or_explicit_threshold_tol_{feasibility_tol:g}": upper_threshold,
            }
        )
    return pd.DataFrame(rows)


def plot_e1_frontiers(summary: pd.DataFrame, output_path: Path) -> None:
    """Create the E1 dimension-gain frontier figure."""

    available_n = sorted(int(n) for n in summary["n"].unique())
    selected_n = [n for n in PLOT_E1_N_PREFERRED if n in available_n]
    if not selected_n:
        selected_n = available_n[: min(4, len(available_n))]

    systems = [SYSTEM_ADJACENT, SYSTEM_CYCLE, SYSTEM_ALLSWAP]
    colors = {
        SYSTEM_ADJACENT: "tab:blue",
        SYSTEM_CYCLE: "tab:green",
        SYSTEM_ALLSWAP: "tab:orange",
    }
    fig, axes = plt.subplots(
        1,
        len(selected_n),
        figsize=(5.1 * len(selected_n), 4.4),
        sharey=True,
    )
    if len(selected_n) == 1:
        axes = [axes]

    y_values = summary[["best_feasible_upper_gain", "median_required_gain"]].to_numpy(dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    y_min = 0.95
    y_max = max(1.08, float(np.nanmax(y_values)) * 1.08) if y_values.size else 1.2
    if y_max / y_min > 50.0:
        y_max = min(y_max, float(np.nanpercentile(y_values, 98)) * 1.2)

    for ax, n in zip(axes, selected_n):
        for system_name in systems:
            sub = summary[
                (summary["n"] == n) & (summary["system"] == system_name)
            ].sort_values("m")
            if len(sub) == 0:
                continue
            x = sub["m"].to_numpy(dtype=int)
            best = sub["best_feasible_upper_gain"].to_numpy(dtype=float)
            median = sub["median_required_gain"].to_numpy(dtype=float)
            ax.plot(
                x,
                best,
                marker="o",
                linewidth=2.0,
                color=colors[system_name],
                label=f"{system_name} best upper" if n == selected_n[0] else None,
            )
            ax.plot(
                x,
                median,
                linestyle="--",
                linewidth=1.3,
                alpha=0.45,
                color=colors[system_name],
                label=f"{system_name} random median" if n == selected_n[0] else None,
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(2, color="0.35", linestyle=":", linewidth=1.2)
        ax.axvline(n - 1, color="0.1", linestyle="-.", linewidth=1.2)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("latent dimension m")
        ax.set_xticks(range(1, n))
        ax.grid(True, alpha=0.25)
        ax.text(2, y_min + 0.04 * (y_max - y_min), "m=2", ha="center", fontsize=8)
        ax.text(
            n - 1,
            y_min + 0.10 * (y_max - y_min),
            "m=n-1",
            ha="center",
            fontsize=8,
        )

    axes[0].set_ylabel("empirical required gain")
    axes[0].set_ylim(y_min, y_max)
    axes[0].legend(loc="upper right", fontsize=8, frameon=True)
    fig.suptitle("E1: Dimension-gain frontiers for finite transition systems")
    caption = (
        "Optimization estimates feasible upper bounds on L_m^*(T). "
        "The marked gain-one thresholds are certified by analytic theorems, "
        "not by gradient-descent failure below the threshold."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_e1(
    output_dir: Path,
    config: OptimizeConfig,
    n_values: Sequence[int],
    feasibility_tol: float,
) -> List[Path]:
    """Run E1 dimension-gain frontier experiments."""

    generated: List[Path] = []
    embedding_dir = output_dir / "embeddings"
    systems = (SYSTEM_ADJACENT, SYSTEM_CYCLE, SYSTEM_ALLSWAP)
    random_rows: List[Dict[str, Any]] = []
    explicit_rows: List[Dict[str, Any]] = []

    print()
    print(
        "E1 tests direct geometry optimization against analytic gain-one "
        "thresholds for AdjacentSwap, Cycle, and AllSwap."
    )
    for n in n_values:
        for system_name in systems:
            system = prepare_system(system_name, n, build_successor_pairs=True)
            for m in range(1, n):
                objective = "generic"
                for seed in range(config.seeds):
                    save_path = (
                        embedding_dir / f"e1_{system_name}_n{n}_m{m}_seed{seed}.pt"
                    )
                    outcome = optimize_embedding(
                        system=system,
                        m=m,
                        config=config,
                        seed=seed,
                        init_kind="random_gaussian",
                        objective=objective,
                        save_path=save_path,
                        run_type="random",
                    )
                    random_rows.append(outcome.row)
                    generated.append(save_path)
                    print(
                        f"E1 {system_name:12s} n={n:3d} m={m:3d} seed={seed:3d} "
                        f"best={float(outcome.row['best_required_gain']):.9g} "
                        f"final={float(outcome.row['final_required_gain']):.9g} "
                        f"steps={int(outcome.row['steps_run']):5d}"
                    )

                for init_kind, init_z in e1_explicit_initializations(system_name, n, m):
                    save_path = (
                        embedding_dir / f"e1_{system_name}_n{n}_m{m}_{init_kind}.pt"
                    )
                    outcome = optimize_embedding(
                        system=system,
                        m=m,
                        config=config,
                        seed=-1,
                        init_kind=init_kind,
                        objective=objective,
                        save_path=save_path,
                        initial_z=init_z,
                        run_type="explicit_initialization",
                    )
                    explicit_rows.append(outcome.row)
                    generated.append(save_path)
                    print(
                        f"E1 {system_name:12s} n={n:3d} m={m:3d} {init_kind:16s} "
                        f"best={float(outcome.row['best_required_gain']):.9g} "
                        f"steps={int(outcome.row['steps_run']):5d}"
                    )

    runs_df = pd.DataFrame(random_rows)
    runs_path = output_dir / "e1_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    generated.append(runs_path)

    explicit_df = pd.DataFrame(explicit_rows)
    explicit_path = output_dir / "e1_constructions.csv"
    explicit_df.to_csv(explicit_path, index=False)
    generated.append(explicit_path)

    summary_df = summarize_e1(runs_df, explicit_df, feasibility_tol)
    summary_path = output_dir / "e1_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    generated.append(summary_path)

    thresholds_df = make_e1_threshold_table(summary_df, feasibility_tol)
    thresholds_path = output_dir / "e1_thresholds.csv"
    thresholds_df.to_csv(thresholds_path, index=False)
    generated.append(thresholds_path)

    figure_path = output_dir / "e1_frontiers.png"
    plot_e1_frontiers(summary_df, figure_path)
    generated.append(figure_path)

    print_e1_summary(summary_df, thresholds_df, feasibility_tol)
    return generated


def summarize_e2(runs_df: pd.DataFrame, n_values: Sequence[int], m_values: Sequence[int]) -> pd.DataFrame:
    """Build the required E2 AllSwap bounds table."""

    rows: List[Dict[str, Any]] = []
    keys: List[Tuple[int, int]] = []
    for n in n_values:
        for m in m_values:
            if m < n:
                keys.append((int(n), int(m)))
        simplex_m = int(n) - 1
        if (int(n), simplex_m) not in keys:
            keys.append((int(n), simplex_m))

    for n, m in sorted(set(keys)):
        group = runs_df[(runs_df["n"] == n) & (runs_df["m"] == m)]
        if len(group) == 0:
            continue
        analytic_lower, analytic_upper = allswap_bounds(n, m)
        grid_gain = float(required_gain_allswap_fast(grid_configuration(n, m)).item())
        best_empirical = float(group["best_required_gain"].astype(float).min())
        random_group = group[group["run_type"] == "random"]
        median_random = (
            float(random_group["best_required_gain"].astype(float).median())
            if len(random_group)
            else math.nan
        )
        rows.append(
            {
                "n": n,
                "m": m,
                "analytic_lower": analytic_lower,
                "analytic_upper": analytic_upper,
                "grid_actual_gain": grid_gain,
                "best_empirical_gain": best_empirical,
                "median_random_gain": median_random,
                "lower_to_empirical_gap": best_empirical - analytic_lower,
                "empirical_to_grid_improvement": grid_gain - best_empirical,
                "lower_to_empirical_ratio": best_empirical / analytic_lower,
                "analytic_upper_to_lower_ratio": analytic_upper / analytic_lower,
            }
        )
    return pd.DataFrame(rows)


def plot_e2_bounds_vs_dimension(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot E2 bounds against dimension for selected n values."""

    preferred_n = [n for n in (16, 32, 64) if n in set(summary["n"].astype(int))]
    if not preferred_n:
        preferred_n = sorted(summary["n"].astype(int).unique())[:3]
    fig, axes = plt.subplots(
        1,
        len(preferred_n),
        figsize=(5.1 * len(preferred_n), 4.2),
        sharey=True,
    )
    if len(preferred_n) == 1:
        axes = [axes]

    all_y = summary[
        ["analytic_lower", "analytic_upper", "grid_actual_gain", "best_empirical_gain"]
    ].to_numpy(dtype=float)
    finite_y = all_y[np.isfinite(all_y)]
    use_log = bool(finite_y.size and finite_y.max() / max(finite_y.min(), 1e-12) > 30.0)

    for ax, n in zip(axes, preferred_n):
        sub = summary[summary["n"] == n].sort_values("m")
        x = sub["m"].to_numpy(dtype=float)
        lower = sub["analytic_lower"].to_numpy(dtype=float)
        empirical = sub["best_empirical_gain"].to_numpy(dtype=float)
        grid = sub["grid_actual_gain"].to_numpy(dtype=float)
        upper = sub["analytic_upper"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            lower,
            empirical,
            color="tab:blue",
            alpha=0.12,
            label="certified interval" if n == preferred_n[0] else None,
        )
        ax.plot(x, lower, marker="v", color="tab:purple", label="analytic lower" if n == preferred_n[0] else None)
        ax.plot(x, empirical, marker="o", color="tab:blue", linewidth=2.0, label="best empirical upper" if n == preferred_n[0] else None)
        ax.plot(x, grid, marker="s", color="tab:orange", label="grid actual gain" if n == preferred_n[0] else None)
        ax.plot(x, upper, marker="^", color="tab:red", linestyle="--", label="analytic grid upper" if n == preferred_n[0] else None)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(f"AllSwap n = {n}")
        ax.set_xlabel("latent dimension m")
        ax.set_xticks(sorted(set(sub["m"].astype(int))))
        ax.grid(True, alpha=0.25, which="both")
        if use_log:
            ax.set_yscale("log")

    axes[0].set_ylabel("required gain")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("E2: AllSwap bounds versus dimension")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_e2_scaling_vs_n(summary: pd.DataFrame, output_path: Path) -> None:
    """Make the E2 log-log scaling plot against n for fixed dimensions."""

    preferred_m = [m for m in (1, 2, 3, 4) if m in set(summary["m"].astype(int))]
    if not preferred_m:
        preferred_m = sorted(summary["m"].astype(int).unique())[:4]
    fig, axes = plt.subplots(
        1,
        len(preferred_m),
        figsize=(5.0 * len(preferred_m), 4.2),
        sharey=True,
    )
    if len(preferred_m) == 1:
        axes = [axes]

    for ax, m in zip(axes, preferred_m):
        sub = summary[(summary["m"] == m) & (summary["n"] > m)].sort_values("n")
        if len(sub) == 0:
            continue
        n_vals = sub["n"].to_numpy(dtype=float)
        lower = sub["analytic_lower"].to_numpy(dtype=float)
        empirical = sub["best_empirical_gain"].to_numpy(dtype=float)
        grid = sub["grid_actual_gain"].to_numpy(dtype=float)
        upper = sub["analytic_upper"].to_numpy(dtype=float)
        ax.loglog(n_vals, lower, marker="v", color="tab:purple", label="analytic lower" if m == preferred_m[0] else None)
        ax.loglog(n_vals, empirical, marker="o", color="tab:blue", linewidth=2.0, label="best empirical upper" if m == preferred_m[0] else None)
        ax.loglog(n_vals, grid, marker="s", color="tab:orange", label="grid actual gain" if m == preferred_m[0] else None)
        ax.loglog(n_vals, upper, marker="^", color="tab:red", linestyle="--", label="analytic grid upper" if m == preferred_m[0] else None)
        if len(n_vals) >= 2:
            ref = n_vals ** (1.0 / float(m))
            ref *= empirical[0] / ref[0]
            ax.loglog(
                n_vals,
                ref,
                color="0.25",
                linestyle=":",
                linewidth=1.2,
                label=f"n^(1/{m}) reference" if m == preferred_m[0] else None,
            )
        ax.set_title(f"m = {m}")
        ax.set_xlabel("number of states n")
        ax.grid(True, alpha=0.25, which="both")

    axes[0].set_ylabel("required gain")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("E2: AllSwap scaling with state count")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_e2(
    output_dir: Path,
    config: OptimizeConfig,
    n_values: Sequence[int],
    m_values: Sequence[int],
) -> List[Path]:
    """Run E2 AllSwap scaling experiments with the fast exact gain."""

    generated: List[Path] = []
    embedding_dir = output_dir / "embeddings"
    rows: List[Dict[str, Any]] = []

    print()
    print(
        "E2 tests the AllSwap prediction L_m^* = Theta_m(n^(1/m)) "
        "and measures the lower-to-upper gap."
    )
    for n in n_values:
        system = prepare_system(
            SYSTEM_ALLSWAP,
            n,
            build_successor_pairs=False,
            include_transitions=False,
        )
        valid_m = [int(m) for m in m_values if int(m) < int(n)]
        for m in valid_m:
            for seed in range(config.seeds):
                save_path = embedding_dir / f"e2_AllSwap_n{n}_m{m}_seed{seed}.pt"
                outcome = optimize_embedding(
                    system=system,
                    m=m,
                    config=config,
                    seed=seed,
                    init_kind="random_gaussian",
                    objective="allswap_fast",
                    save_path=save_path,
                    run_type="random",
                )
                rows.append(outcome.row)
                generated.append(save_path)
                print(
                    f"E2 AllSwap      n={n:3d} m={m:3d} seed={seed:3d} "
                    f"best={float(outcome.row['best_required_gain']):.9g} "
                    f"steps={int(outcome.row['steps_run']):5d}"
                )

            grid_gain = float(required_gain_allswap_fast(grid_configuration(n, m)).item())
            for init_kind, init_z in e2_explicit_initializations(n, m):
                save_path = embedding_dir / f"e2_AllSwap_n{n}_m{m}_{init_kind}.pt"
                outcome = optimize_embedding(
                    system=system,
                    m=m,
                    config=config,
                    seed=-1,
                    init_kind=init_kind,
                    objective="allswap_fast",
                    save_path=save_path,
                    initial_z=init_z,
                    run_type="explicit_initialization",
                )
                rows.append(outcome.row)
                generated.append(save_path)
                print(
                    f"E2 AllSwap      n={n:3d} m={m:3d} {init_kind:16s} "
                    f"best={float(outcome.row['best_required_gain']):.9g} "
                    f"grid={grid_gain:.9g}"
                )

        simplex_m = int(n) - 1
        if simplex_m not in valid_m:
            simplex_z = pad_embedding(regular_simplex(n), simplex_m)
            save_path = embedding_dir / f"e2_AllSwap_n{n}_m{simplex_m}_regular_simplex.pt"
            outcome = optimize_embedding(
                system=system,
                m=simplex_m,
                config=OptimizeConfig(
                    steps=0,
                    seeds=config.seeds,
                    lr=config.lr,
                    beta_values=config.beta_values,
                    beta_boundaries=config.beta_boundaries,
                    eps=config.eps,
                    clip_norm=config.clip_norm,
                    patience=config.patience,
                    improvement_tol=config.improvement_tol,
                    exact_tolerance=config.exact_tolerance,
                ),
                seed=-1,
                init_kind="regular_simplex",
                objective="allswap_fast",
                save_path=save_path,
                initial_z=simplex_z,
                run_type="analytic_construction",
            )
            rows.append(outcome.row)
            generated.append(save_path)
            print(
                f"E2 AllSwap      n={n:3d} m={simplex_m:3d} regular_simplex  "
                f"best={float(outcome.row['best_required_gain']):.9g}"
            )

    runs_df = pd.DataFrame(rows)
    runs_path = output_dir / "e2_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    generated.append(runs_path)

    summary_df = summarize_e2(runs_df, n_values, m_values)
    for _, row in summary_df.iterrows():
        if int(row["m"]) == int(row["n"]) - 1:
            continue
        assert float(row["best_empirical_gain"]) <= float(row["grid_actual_gain"]) + 1e-8, (
            "E2 empirical upper is worse than grid initialization: "
            f"n={int(row['n'])} m={int(row['m'])} "
            f"best={float(row['best_empirical_gain'])} grid={float(row['grid_actual_gain'])}"
        )
    summary_path = output_dir / "e2_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    generated.append(summary_path)

    figure_a = output_dir / "e2_bounds_vs_dimension.png"
    plot_e2_bounds_vs_dimension(summary_df, figure_a)
    generated.append(figure_a)

    figure_b = output_dir / "e2_scaling_vs_n.png"
    plot_e2_scaling_vs_n(summary_df, figure_b)
    generated.append(figure_b)

    print_e2_summary(summary_df)
    return generated


def print_e1_summary(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    feasibility_tol: float,
) -> None:
    """Print a compact E1 scientific report."""

    print()
    print("E1 compact threshold comparison")
    print(thresholds.to_string(index=False))
    print()
    print(
        f"Empirical optimization feasibility uses best_random_gain <= 1 + {feasibility_tol:g}; "
        "explicit polygon/simplex/grid candidates are tracked separately."
    )
    systems = [SYSTEM_ADJACENT, SYSTEM_CYCLE, SYSTEM_ALLSWAP]
    for system_name in systems:
        sub = summary[summary["system"] == system_name]
        if len(sub) == 0:
            continue
        best_gap = float((sub["best_feasible_upper_gain"] - 1.0).clip(lower=0.0).min())
        print(f"{system_name}: smallest observed feasible-upper excess over 1 is {best_gap:.3g}.")


def print_e2_summary(summary: pd.DataFrame) -> None:
    """Print the E2 gap report requested by the protocol."""

    non_simplex = summary[summary["m"] < summary["n"] - 1].copy()
    if len(non_simplex) == 0:
        return
    ratios = non_simplex["lower_to_empirical_ratio"].astype(float)
    additive = non_simplex["lower_to_empirical_gap"].astype(float)
    print()
    print("E2 lower-to-feasible-upper gap")
    print(
        "Across optimized AllSwap rows, best_empirical_gain / analytic_lower "
        f"ranges from {float(ratios.min()):.4g} to {float(ratios.max()):.4g} "
        f"(median {float(ratios.median()):.4g})."
    )
    print(
        "The largest additive gap best_empirical_gain - analytic_lower is "
        f"{float(additive.max()):.4g}.  Closing this certified lower-to-upper "
        "gap remains an open theoretical problem."
    )


def print_required_scientific_summary(ran_e1: bool, ran_e2: bool) -> None:
    """Print the required interpretation block."""

    print()
    print("Scientific interpretation")
    if ran_e1:
        print(
            "E1 tests whether direct geometry optimization recovers analytically "
            "known gain-one thresholds and whether equal state counts can have "
            "different predictive realization dimensions."
        )
    if ran_e2:
        print(
            "E2 tests the quantitative AllSwap prediction L_m^* = Theta_m(n^(1/m)) "
            "and measures the gap between certified lower and feasible upper bounds."
        )
    print(
        "Numerical optimization supplies feasible embeddings and therefore upper "
        "bounds; it never certifies a new lower bound by failure alone."
    )
    print(
        "These experiments validate the finite-system realization theory. They do "
        "not by themselves establish the same dimension thresholds for continuous "
        "learned world models."
    )


def parse_beta_values(value: str) -> Tuple[float, float, float]:
    """Parse three comma-separated beta values."""

    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three comma-separated beta values")
    if any(part <= 0.0 for part in parts):
        raise argparse.ArgumentTypeError("Beta values must be positive")
    return parts[0], parts[1], parts[2]


def parse_beta_boundaries(value: str) -> Tuple[float, float]:
    """Parse two comma-separated beta schedule boundaries."""

    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated beta boundaries")
    if not (0.0 < parts[0] < parts[1] < 1.0):
        raise argparse.ArgumentTypeError("Beta boundaries must satisfy 0 < first < second < 1")
    return parts[0], parts[1]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description="CPU-only direct-embedding experiments for dimension-gain theory.",
    )
    parser.add_argument(
        "--experiment",
        choices=("e1", "e2", "all"),
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the small smoke-test protocol",
    )
    parser.add_argument("--steps", type=int, default=None, help="Optimizer steps per run")
    parser.add_argument("--seeds", type=int, default=None, help="Random seeds per condition")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dimension_gain"),
        help="Output directory for CSVs, figures, metadata, and checkpoints",
    )
    parser.add_argument("--lr", type=float, default=2e-2, help="Adam learning rate")
    parser.add_argument("--grad-clip", type=float, default=10.0, help="Gradient clipping norm")
    parser.add_argument("--patience", type=int, default=2500, help="Early-stopping patience")
    parser.add_argument("--eps", type=float, default=1e-12, help="Log-distance epsilon")
    parser.add_argument(
        "--beta-values",
        type=parse_beta_values,
        default=(10.0, 30.0, 100.0),
        help="Three comma-separated beta values, default 10,30,100",
    )
    parser.add_argument(
        "--beta-boundaries",
        type=parse_beta_boundaries,
        default=(0.3, 0.6),
        help="Two comma-separated schedule boundaries, default 0.3,0.6",
    )
    parser.add_argument(
        "--improvement-tol",
        type=float,
        default=1e-9,
        help="Minimum hard-gain improvement that resets patience",
    )
    parser.add_argument(
        "--exact-tolerance",
        type=float,
        default=1e-9,
        help="Stop if hard gain reaches 1 plus this tolerance",
    )
    parser.add_argument(
        "--feasibility-tolerance",
        type=float,
        default=FEASIBILITY_TOL_DEFAULT,
        help="Tolerance for empirical L=1 feasibility summaries",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""

    if args.steps is not None and args.steps < 0:
        raise ValueError("--steps must be nonnegative")
    if args.seeds is not None and args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.grad_clip < 0.0:
        raise ValueError("--grad-clip must be nonnegative")
    if args.patience < 1:
        raise ValueError("--patience must be positive")
    if args.eps <= 0.0:
        raise ValueError("--eps must be positive")
    if args.feasibility_tolerance < 0.0:
        raise ValueError("--feasibility-tolerance must be nonnegative")


def make_config(args: argparse.Namespace, experiment: str) -> OptimizeConfig:
    """Resolve default steps and seeds for E1 or E2."""

    if experiment == "e1":
        default_steps = 1000 if args.quick else 12000
        default_seeds = 3 if args.quick else 10
    elif experiment == "e2":
        default_steps = 1000 if args.quick else 8000
        default_seeds = 2 if args.quick else 6
    else:
        raise ValueError(experiment)
    return OptimizeConfig(
        steps=int(args.steps if args.steps is not None else default_steps),
        seeds=int(args.seeds if args.seeds is not None else default_seeds),
        lr=float(args.lr),
        beta_values=tuple(float(v) for v in args.beta_values),
        beta_boundaries=tuple(float(v) for v in args.beta_boundaries),
        eps=float(args.eps),
        clip_norm=float(args.grad_clip),
        patience=int(args.patience),
        improvement_tol=float(args.improvement_tol),
        exact_tolerance=float(args.exact_tolerance),
    )


def resolved_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    """Return the exact experiment protocol selected by the CLI."""

    return {
        "experiment": args.experiment,
        "quick": bool(args.quick),
        "e1_n_values": list(E1_N_QUICK if args.quick else E1_N_DEFAULT),
        "e1_m_values": "1..n-1",
        "e2_n_values": list(E2_N_QUICK if args.quick else E2_N_DEFAULT),
        "e2_m_values": list(E2_M_QUICK if args.quick else E2_M_DEFAULT),
        "e1_config": make_config(args, "e1").__dict__,
        "e2_config": make_config(args, "e2").__dict__,
        "output_dir": str(args.output_dir),
        "feasibility_tolerance": float(args.feasibility_tolerance),
    }


def json_ready_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse Namespace to JSON-serializable values."""

    result: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def write_metadata(args: argparse.Namespace, output_dir: Path) -> Path:
    """Record reproducibility metadata in a sidecar JSON file."""

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "command_line": sys.argv,
        "arguments": json_ready_args(args),
        "protocol": resolved_protocol(args),
        "random_seeds": {
            "e1": list(range(make_config(args, "e1").seeds)),
            "e2": list(range(make_config(args, "e2").seeds)),
        },
        "cpu_thread_count": torch.get_num_threads(),
        "torch_interop_thread_count": torch.get_num_interop_threads(),
        "cpu_only": True,
        "dtype": "torch.float64",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def print_startup_configuration(args: argparse.Namespace) -> None:
    """Print the exact selected configuration at startup."""

    print("Dimension-gain experiments are CPU-only and optimize embeddings directly.")
    print("Selected configuration:")
    print(json.dumps(resolved_protocol(args), indent=2, sort_keys=True))


def main() -> None:
    """CLI entry point."""

    torch.set_default_dtype(DTYPE)
    torch.use_deterministic_algorithms(True)
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_startup_configuration(args)
    metadata_path = write_metadata(args, output_dir)
    generated_paths: List[Path] = [metadata_path]

    print()
    print("Running analytic, numerical, and invariance tests before experiments...")
    run_all_tests()
    print("All pre-experiment tests passed.")

    start = time.time()
    ran_e1 = args.experiment in ("e1", "all")
    ran_e2 = args.experiment in ("e2", "all")
    if ran_e1:
        e1_config = make_config(args, "e1")
        n_values = E1_N_QUICK if args.quick else E1_N_DEFAULT
        generated_paths.extend(
            run_e1(
                output_dir=output_dir,
                config=e1_config,
                n_values=n_values,
                feasibility_tol=float(args.feasibility_tolerance),
            ),
        )
    if ran_e2:
        e2_config = make_config(args, "e2")
        n_values = E2_N_QUICK if args.quick else E2_N_DEFAULT
        m_values = E2_M_QUICK if args.quick else E2_M_DEFAULT
        generated_paths.extend(
            run_e2(
                output_dir=output_dir,
                config=e2_config,
                n_values=n_values,
                m_values=m_values,
            ),
        )

    elapsed = time.time() - start
    print_required_scientific_summary(ran_e1=ran_e1, ran_e2=ran_e2)
    print()
    print(f"Completed in {elapsed:.1f} seconds.")
    print("Generated files:")
    for path in generated_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
