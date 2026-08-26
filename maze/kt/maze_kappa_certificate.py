#!/usr/bin/env python3
"""
Exact kappa_T / gain-one dimension certificates for the final 8x8 Maze A/B/C.

This script uses NO neural-network training.

For each maze it:
  1. builds the exact deterministic transition table for U,D,L,R;
  2. optionally composes all length-h action blocks (default h=5) and
     deduplicates identical transition maps;
  3. builds the directed transition graph on unordered distinct state pairs;
  4. computes strongly connected components (SCCs), i.e. dynamical pair types;
  5. computes kappa_T(X);
  6. computes the full-state few-distance lower bound;
  7. finds EXACTLY the largest subset Y with kappa_T(Y)=1 via maximum clique;
  8. certifies d_PR(T;1) >= |Y|-1 for that subset;
  9. saves CSV/JSON summaries and optional maze visualizations.

Important:
  - These are rigorous LOWER BOUNDS on d_PR(T;1).
  - They are not exact dimensions unless a matching upper bound is known.
  - The kappa=1 subset search is exact for the returned transition family.

Dependencies:
    numpy
    scipy
Optional:
    matplotlib   (only for saving highlighted-maze figures)

Examples:
    python maze_kappa_certificate_final.py
    python maze_kappa_certificate_final.py --family primitive
    python maze_kappa_certificate_final.py --family h5
    python maze_kappa_certificate_final.py --horizon 5 --outdir outputs/kt
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from math import comb
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# =============================================================================
# 1. FINAL MAZE CONFIGURATION
# =============================================================================

GRID_N = 8
NUM_STATES = GRID_N * GRID_N
ACTIONS = ("U", "D", "L", "R")

START = (0, 0)
GOAL = (7, 7)

A_WALLS = {
    ("V", 1, 3),
    ("V", 2, 3),
    ("V", 3, 3),
    ("H", 4, 4),
    ("H", 4, 5),
    ("H", 4, 6),
}

B_WALLS = {
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

C_WALLS = {
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

MAZES = {
    "A": A_WALLS,
    "B": B_WALLS,
    "C": C_WALLS,
}


# =============================================================================
# 2. EXACT MAZE TRANSITIONS
# =============================================================================

def state_id(r: int, c: int) -> int:
    return r * GRID_N + c


def state_rc(s: int) -> Tuple[int, int]:
    return divmod(s, GRID_N)


def step(
    walls: Set[Tuple[str, int, int]],
    s: int,
    action: str,
) -> int:
    """
    Maze semantics:
      V(r,c) blocks (r,c) <-> (r,c+1)
      H(r,c) blocks (r,c) <-> (r+1,c)
      boundary/wall collision -> self-loop
    """
    r, c = state_rc(s)

    if action == "U":
        if r == 0 or ("H", r - 1, c) in walls:
            return s
        return state_id(r - 1, c)

    if action == "D":
        if r == GRID_N - 1 or ("H", r, c) in walls:
            return s
        return state_id(r + 1, c)

    if action == "L":
        if c == 0 or ("V", r, c - 1) in walls:
            return s
        return state_id(r, c - 1)

    if action == "R":
        if c == GRID_N - 1 or ("V", r, c) in walls:
            return s
        return state_id(r, c + 1)

    raise ValueError(action)


def primitive_transition_table(
    walls: Set[Tuple[str, int, int]],
) -> np.ndarray:
    """
    Shape [4, 64].
    T[a, s] is the exact next state after primitive action a.
    """
    return np.asarray(
        [
            [step(walls, s, a) for s in range(NUM_STATES)]
            for a in ACTIONS
        ],
        dtype=np.int32,
    )


def unique_h_step_maps(
    T1: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Enumerate all |A|^h action blocks, compose them exactly, and deduplicate
    blocks that induce the same state-transition map.

    Returns
    -------
    maps : ndarray [num_unique_maps, num_states]
    witness_sequence : dict
        map tuple -> one action-index sequence producing it
    """
    num_actions, num_states = T1.shape
    unique: Dict[Tuple[int, ...], Tuple[int, ...]] = {}

    for seq in product(range(num_actions), repeat=horizon):
        nxt = np.arange(num_states, dtype=np.int32)
        for a in seq:
            nxt = T1[a, nxt]
        key = tuple(int(x) for x in nxt)
        unique.setdefault(key, tuple(int(a) for a in seq))

    maps = np.asarray(list(unique.keys()), dtype=np.int32)
    return maps, unique


# =============================================================================
# 3. PAIR-TRANSITION GRAPH AND kappa_T
# =============================================================================

@dataclass
class PairSCC:
    pairs: List[Tuple[int, int]]
    pair_to_id: Dict[Tuple[int, int], int]
    labels: np.ndarray
    num_scc: int
    scc_sizes: np.ndarray


def build_pair_scc(transition_maps: np.ndarray) -> PairSCC:
    """
    Nodes are unordered distinct state pairs {i,j}.
    For each transition map T, add
        {i,j} -> {T(i),T(j)}
    when T(i) != T(j).

    SCCs are dynamical pair types.
    """
    transition_maps = np.asarray(transition_maps, dtype=np.int32)
    _, n = transition_maps.shape

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_to_id = {p: k for k, p in enumerate(pairs)}

    rows: List[int] = []
    cols: List[int] = []

    for T in transition_maps:
        for src, (i, j) in enumerate(pairs):
            u = int(T[i])
            v = int(T[j])

            if u == v:
                continue

            if u > v:
                u, v = v, u

            rows.append(src)
            cols.append(pair_to_id[(u, v)])

    graph = coo_matrix(
        (
            np.ones(len(rows), dtype=np.uint8),
            (rows, cols),
        ),
        shape=(len(pairs), len(pairs)),
    ).tocsr()

    num_scc, labels = connected_components(
        graph,
        directed=True,
        connection="strong",
    )

    sizes = np.bincount(labels, minlength=num_scc)

    return PairSCC(
        pairs=pairs,
        pair_to_id=pair_to_id,
        labels=labels.astype(np.int32),
        num_scc=int(num_scc),
        scc_sizes=sizes.astype(np.int32),
    )


def kappa_of_subset(
    Y: Sequence[int],
    pair_scc: PairSCC,
) -> int:
    """
    kappa_T(Y) = number of dynamical pair types touched by P_2(Y).
    """
    Y = sorted(set(int(x) for x in Y))

    if len(Y) < 2:
        return 0

    touched = set()

    for a in range(len(Y)):
        for b in range(a + 1, len(Y)):
            i, j = Y[a], Y[b]
            pid = pair_scc.pair_to_id[(i, j)]
            touched.add(int(pair_scc.labels[pid]))

    return len(touched)


def few_distance_lb(num_points: int, kappa: int) -> int:
    """
    Theorem certificate:
        |Y| <= C(m+kappa, kappa)
    so return the smallest integer m satisfying it.
    """
    if num_points < 2:
        return 0
    if kappa < 1:
        raise ValueError("For |Y|>=2, kappa must be >=1.")

    m = 1
    while comb(m + kappa, kappa) < num_points:
        m += 1
    return m


# =============================================================================
# 4. EXACT LARGEST SUBSET WITH kappa_T(Y)=1
# =============================================================================

def pair_label_matrix(n: int, pair_scc: PairSCC) -> np.ndarray:
    M = np.full((n, n), -1, dtype=np.int32)
    for idx, (i, j) in enumerate(pair_scc.pairs):
        lab = int(pair_scc.labels[idx])
        M[i, j] = lab
        M[j, i] = lab
    return M


def maximum_clique_bitset(adjacency: List[int], n: int) -> List[int]:
    """
    Exact maximum clique via Bron-Kerbosch with pivoting and bitsets.
    Suitable for these n=64 state graphs.
    """
    best_mask = 0
    best_size = 0

    def choose_pivot(P: int, X: int) -> int:
        U = P | X
        if U == 0:
            return -1

        best_u = -1
        best_score = -1

        while U:
            bit = U & -U
            u = bit.bit_length() - 1
            score = (P & adjacency[u]).bit_count()

            if score > best_score:
                best_score = score
                best_u = u

            U ^= bit

        return best_u

    def bronk(R: int, P: int, X: int) -> None:
        nonlocal best_mask, best_size

        r_size = R.bit_count()

        # Simple exact branch-and-bound.
        if r_size + P.bit_count() <= best_size:
            return

        if P == 0 and X == 0:
            if r_size > best_size:
                best_size = r_size
                best_mask = R
            return

        u = choose_pivot(P, X)
        candidates = P if u < 0 else P & ~adjacency[u]

        while candidates:
            bit = candidates & -candidates
            v = bit.bit_length() - 1

            bronk(
                R | bit,
                P & adjacency[v],
                X & adjacency[v],
            )

            P &= ~bit
            X |= bit
            candidates &= ~bit

            if r_size + P.bit_count() <= best_size:
                return

    bronk(0, (1 << n) - 1, 0)

    return [
        v
        for v in range(n)
        if (best_mask >> v) & 1
    ]


def largest_kappa1_subset(
    n: int,
    pair_scc: PairSCC,
) -> Tuple[int, List[int]]:
    """
    For a fixed SCC label ell, make a graph on STATES where i--j iff pair
    {i,j} has SCC label ell.

    A clique Y in this graph has every internal pair in the same pair SCC,
    hence kappa_T(Y)=1.

    Search all SCC labels and return the exact largest such Y.
    """
    M = pair_label_matrix(n, pair_scc)

    best_label = -1
    best_Y: List[int] = []

    # Only labels with enough pair edges can possibly beat current best.
    label_edge_counts = np.bincount(
        pair_scc.labels,
        minlength=pair_scc.num_scc,
    )

    label_order = np.argsort(-label_edge_counts)

    for lab in label_order:
        lab = int(lab)

        # A k-clique needs k(k-1)/2 edges.  Skip labels that cannot improve.
        b = len(best_Y) + 1
        needed_edges = b * (b - 1) // 2
        if int(label_edge_counts[lab]) < needed_edges:
            continue

        adjacency = [0] * n

        for i in range(n):
            for j in range(i + 1, n):
                if int(M[i, j]) == lab:
                    adjacency[i] |= 1 << j
                    adjacency[j] |= 1 << i

        Y = maximum_clique_bitset(adjacency, n)

        if len(Y) > len(best_Y):
            best_label = lab
            best_Y = Y

    return best_label, sorted(best_Y)


# =============================================================================
# 5. SANITY CHECKS
# =============================================================================

def cycle_family(n: int) -> np.ndarray:
    return np.asarray(
        [[(i + 1) % n for i in range(n)]],
        dtype=np.int32,
    )


def pair_transitive_family(n: int) -> np.ndarray:
    """
    R = n-cycle
    S = transposition (0 1)
    These generate S_n and are transitive on unordered pairs.
    """
    R = np.asarray([(i + 1) % n for i in range(n)], dtype=np.int32)
    S = np.arange(n, dtype=np.int32)
    S[0], S[1] = 1, 0
    return np.stack([R, S], axis=0)


def sanity_checks() -> None:
    n = 8

    cyc = build_pair_scc(cycle_family(n))
    assert cyc.num_scc == n // 2
    assert few_distance_lb(n, cyc.num_scc) == 2

    pt = build_pair_scc(pair_transitive_family(n))
    assert pt.num_scc == 1
    assert few_distance_lb(n, pt.num_scc) == n - 1

    ident = build_pair_scc(np.arange(n, dtype=np.int32)[None, :])
    assert ident.num_scc == comb(n, 2)
    assert few_distance_lb(n, ident.num_scc) == 1

    print("Sanity checks: PASS")


# =============================================================================
# 6. ANALYSIS
# =============================================================================

def analyze(
    maze_name: str,
    family_name: str,
    transition_maps: np.ndarray,
) -> dict:
    n = transition_maps.shape[1]

    pair_scc = build_pair_scc(transition_maps)

    # Full-state certificate.
    kappa_X = pair_scc.num_scc
    full_lb = few_distance_lb(n, kappa_X)

    # Strong, easy-to-interpret exact kappa=1 certificate.
    best_label, best_Y = largest_kappa1_subset(n, pair_scc)
    assert len(best_Y) >= 2
    assert kappa_of_subset(best_Y, pair_scc) == 1

    kappa1_lb = len(best_Y) - 1

    largest_scc = int(pair_scc.scc_sizes.max())
    largest_scc_fraction = largest_scc / len(pair_scc.pairs)

    row = {
        "maze": maze_name,
        "family": family_name,
        "num_states": n,
        "num_transition_maps": int(transition_maps.shape[0]),
        "num_pair_vertices": len(pair_scc.pairs),
        "kappa_X": kappa_X,
        "full_set_lb": full_lb,
        "largest_pair_scc_size": largest_scc,
        "largest_pair_scc_fraction": largest_scc_fraction,
        "largest_kappa1_subset_size": len(best_Y),
        "kappa1_lb": kappa1_lb,
        "kappa1_scc_label": best_label,
        "kappa1_subset_state_ids": best_Y,
        "kappa1_subset_cells": [state_rc(s) for s in best_Y],
    }

    print()
    print("=" * 92)
    print(f"Maze {maze_name} | {family_name}")
    print("-" * 92)
    print(f"|X|                              : {n}")
    print(f"transition maps                  : {transition_maps.shape[0]}")
    print(f"pair vertices                    : {len(pair_scc.pairs)}")
    print(f"kappa_T(X)                       : {kappa_X}")
    print(f"full-X certificate               : d_PR(T;1) >= {full_lb}")
    print(f"largest pair SCC                 : {largest_scc} "
          f"({100.0*largest_scc_fraction:.2f}% of pairs)")
    print(f"largest exact kappa=1 subset     : |Y| = {len(best_Y)}")
    print(f"kappa=1 certificate              : d_PR(T;1) >= {kappa1_lb}")
    print(f"witness cells                    : {[state_rc(s) for s in best_Y]}")

    return row


# =============================================================================
# 7. OPTIONAL VISUALIZATION
# =============================================================================

def save_maze_figure(
    maze_name: str,
    walls: Set[Tuple[str, int, int]],
    highlighted_states: Sequence[int],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not installed; skipping figures.")
        return

    highlighted = set(int(x) for x in highlighted_states)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Highlight certificate subset using default matplotlib styling.
    for s in highlighted:
        r, c = state_rc(s)
        rect = Rectangle(
            (c, GRID_N - 1 - r),
            1,
            1,
            alpha=0.25,
        )
        ax.add_patch(rect)

    # Draw base grid.
    for x in range(GRID_N + 1):
        ax.plot([x, x], [0, GRID_N], linewidth=0.6)
    for y in range(GRID_N + 1):
        ax.plot([0, GRID_N], [y, y], linewidth=0.6)

    # Draw walls thicker.
    for typ, r, c in walls:
        if typ == "V":
            x = c + 1
            y0 = GRID_N - 1 - r
            ax.plot([x, x], [y0, y0 + 1], linewidth=3.0)
        elif typ == "H":
            y = GRID_N - 1 - r
            x0 = c
            ax.plot([x0, x0 + 1], [y, y], linewidth=3.0)

    ax.set_xlim(0, GRID_N)
    ax.set_ylim(0, GRID_N)
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID_N + 1))
    ax.set_yticks(range(GRID_N + 1))
    ax.set_title(
        f"Maze {maze_name}: highlighted max kappa=1 subset "
        f"(|Y|={len(highlighted)})"
    )
    ax.set_xlabel("column")
    ax.set_ylabel("row (display inverted)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# 8. SAVE RESULTS
# =============================================================================

def save_results(rows: List[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "kappa_summary.csv"
    fieldnames = [
        "maze",
        "family",
        "num_states",
        "num_transition_maps",
        "num_pair_vertices",
        "kappa_X",
        "full_set_lb",
        "largest_pair_scc_size",
        "largest_pair_scc_fraction",
        "largest_kappa1_subset_size",
        "kappa1_lb",
        "kappa1_scc_label",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    json_path = outdir / "kappa_certificates.json"
    json_path.write_text(json.dumps(rows, indent=2))

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


# =============================================================================
# 9. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--family",
        choices=("primitive", "h5", "both"),
        default="both",
        help="Which transition family to analyze.",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Action-block length for the composed family.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/kt"),
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sanity_checks()

    rows: List[dict] = []

    for maze_name, walls in MAZES.items():
        T1 = primitive_transition_table(walls)

        if args.family in ("primitive", "both"):
            row = analyze(
                maze_name=maze_name,
                family_name="primitive",
                transition_maps=T1,
            )
            rows.append(row)

            if not args.no_figures:
                args.outdir.mkdir(parents=True, exist_ok=True)
                save_maze_figure(
                    maze_name,
                    walls,
                    row["kappa1_subset_state_ids"],
                    args.outdir / f"maze_{maze_name}_primitive_kappa1.png",
                )

        if args.family in ("h5", "both"):
            Th, _ = unique_h_step_maps(T1, horizon=args.horizon)

            row = analyze(
                maze_name=maze_name,
                family_name=f"h{args.horizon}-unique",
                transition_maps=Th,
            )
            rows.append(row)

            if not args.no_figures:
                args.outdir.mkdir(parents=True, exist_ok=True)
                save_maze_figure(
                    maze_name,
                    walls,
                    row["kappa1_subset_state_ids"],
                    args.outdir / f"maze_{maze_name}_h{args.horizon}_kappa1.png",
                )

    save_results(rows, args.outdir)

    print()
    print("=" * 92)
    print("COMPACT SUMMARY")
    print("=" * 92)
    print(
        f"{'maze':<6} {'family':<14} {'maps':>6} {'kappa(X)':>10} "
        f"{'full LB':>8} {'max |Y| k=1':>13} {'k=1 LB':>8}"
    )
    print("-" * 92)

    for r in rows:
        print(
            f"{r['maze']:<6} "
            f"{r['family']:<14} "
            f"{r['num_transition_maps']:>6} "
            f"{r['kappa_X']:>10} "
            f"{r['full_set_lb']:>8} "
            f"{r['largest_kappa1_subset_size']:>13} "
            f"{r['kappa1_lb']:>8}"
        )

    print()
    print("Reminder:")
    print("  k=1 LB is a rigorous lower bound on d_PR(T;1).")
    print("  It is NOT automatically the exact dimension.")


if __name__ == "__main__":
    main()
