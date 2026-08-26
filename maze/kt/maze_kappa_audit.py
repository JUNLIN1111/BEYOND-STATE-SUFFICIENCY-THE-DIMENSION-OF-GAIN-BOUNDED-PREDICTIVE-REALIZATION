#!/usr/bin/env python3
"""
Independent audit for maze_kappa_certificate.py.

This is NOT a new experiment and uses NO training.

It performs three checks for the primitive U/D/L/R transition family:

1. Independent SCC audit:
   Rebuild the pair-transition graph and run a pure-Python Tarjan SCC
   implementation, independent of scipy.sparse.csgraph.connected_components.
   Compare the resulting SCC partition against the main script.

2. Witness-subset audit:
   Recompute the main script's largest kappa_T(Y)=1 subset and verify that
   every one of its C(|Y|,2) pairs receives exactly the same SCC label under
   the independent Tarjan partition.

3. Concrete reachability witness:
   Pick two distinct state-pairs from the certified subset and use BFS to
   print an explicit primitive-action sequence taking p -> q and another
   sequence taking q -> p.

Usage:
    Put this file in the same directory as maze_kappa_certificate.py, then run

        python maze_kappa_audit.py

Optional:
        python maze_kappa_audit.py --maze C
        python maze_kappa_audit.py --maze A B C --num-witnesses 3
"""

from __future__ import annotations

import argparse
import sys
sys.setrecursionlimit(10000)
from collections import deque
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

import importlib

def _load_core():
    # Prefer the final script name used in this chat; fall back to the user's
    # local name if they saved it as maze_kappa_certificate.py.
    for name in ("maze_kappa_certificate_final", "maze_kappa_certificate"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            pass
    raise ModuleNotFoundError(
        "Could not import maze_kappa_certificate_final.py or "
        "maze_kappa_certificate.py from the current directory."
    )

core = _load_core()


Pair = Tuple[int, int]


def canonical_pair(i: int, j: int) -> Pair:
    return (i, j) if i < j else (j, i)


def build_labeled_pair_graph(T1: np.ndarray):
    """
    Build the primitive pair graph independently.

    Returns
    -------
    pairs : list of unordered state pairs
    pair_to_id : dict pair -> vertex id
    adjacency : list[list[int]] for Tarjan
    labeled_adjacency : list[list[(dst, action_name)]]
    """
    _, n = T1.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_to_id = {p: idx for idx, p in enumerate(pairs)}

    adjacency: List[List[int]] = [[] for _ in pairs]
    labeled_adjacency: List[List[Tuple[int, str]]] = [[] for _ in pairs]

    for src, (i, j) in enumerate(pairs):
        for a_idx, a_name in enumerate(core.ACTIONS):
            u = int(T1[a_idx, i])
            v = int(T1[a_idx, j])

            if u == v:
                continue

            q = canonical_pair(u, v)
            dst = pair_to_id[q]

            adjacency[src].append(dst)
            labeled_adjacency[src].append((dst, str(a_name)))

    return pairs, pair_to_id, adjacency, labeled_adjacency


def tarjan_scc(adjacency: Sequence[Sequence[int]]):
    """Pure-Python Tarjan SCC; independent of scipy."""
    n = len(adjacency)
    index = 0
    stack: List[int] = []
    on_stack = [False] * n
    indices = [-1] * n
    lowlink = [0] * n
    components: List[List[int]] = []

    def strongconnect(v: int):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1

        stack.append(v)
        on_stack[v] = True

        for w in adjacency[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            components.append(comp)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)

    labels = np.empty(n, dtype=np.int32)
    for lab, comp in enumerate(components):
        for v in comp:
            labels[v] = lab

    return labels, components


def same_partition(labels_a: np.ndarray, labels_b: np.ndarray) -> bool:
    """
    SCC numeric labels can be permuted, so compare equivalence relations.
    O(N^2), with N=2016 this is fine.
    """
    n = len(labels_a)
    for i in range(n):
        ai = labels_a[i]
        bi = labels_b[i]
        for j in range(i + 1, n):
            if (labels_a[j] == ai) != (labels_b[j] == bi):
                return False
    return True


def verify_subset_one_label(
    Y: Sequence[int],
    pair_to_id: Dict[Pair, int],
    independent_labels: np.ndarray,
):
    pair_labels = []
    for i, j in combinations(sorted(Y), 2):
        pair_labels.append(
            int(independent_labels[pair_to_id[(i, j)]])
        )

    distinct = sorted(set(pair_labels))
    return len(pair_labels), distinct


def bfs_path(
    src: int,
    dst: int,
    labeled_adjacency: Sequence[Sequence[Tuple[int, str]]],
):
    """Return one shortest action-labeled path src -> dst."""
    if src == dst:
        return []

    parent: Dict[int, Tuple[int, str]] = {}
    q = deque([src])
    seen = {src}

    while q:
        v = q.popleft()

        for w, action in labeled_adjacency[v]:
            if w in seen:
                continue

            seen.add(w)
            parent[w] = (v, action)

            if w == dst:
                actions = []
                cur = dst
                while cur != src:
                    prev, a = parent[cur]
                    actions.append(a)
                    cur = prev
                actions.reverse()
                return actions

            q.append(w)

    return None


def apply_action_sequence_to_pair(
    pair: Pair,
    actions: Sequence[str],
    T1: np.ndarray,
):
    action_to_idx = {a: i for i, a in enumerate(core.ACTIONS)}

    i, j = pair
    trajectory = [canonical_pair(i, j)]

    for a in actions:
        ai = action_to_idx[a]
        i = int(T1[ai, i])
        j = int(T1[ai, j])

        if i == j:
            raise RuntimeError(
                f"Pair collapsed while replaying witness after action {a}."
            )

        trajectory.append(canonical_pair(i, j))

    return trajectory


def choose_witness_pairs(Y: Sequence[int]):
    """
    Deterministically choose two well-separated pair entries from P_2(Y).
    """
    all_pairs = list(combinations(sorted(Y), 2))
    if len(all_pairs) < 2:
        raise ValueError("Need at least two distinct pairs.")

    p = all_pairs[0]
    q = all_pairs[len(all_pairs) // 2]

    if p == q:
        q = all_pairs[-1]

    return p, q


def audit_maze(maze_name: str, num_witnesses: int = 1):
    walls = core.MAZES[maze_name]
    T1 = core.primitive_transition_table(walls)

    # Main implementation (scipy SCC + exact kappa=1 maximum clique).
    main_scc = core.build_pair_scc(T1)
    _, Y = core.largest_kappa1_subset(
        T1.shape[1],
        main_scc,
    )

    # Independent graph + Tarjan SCC.
    pairs, pair_to_id, adjacency, labeled_adjacency = (
        build_labeled_pair_graph(T1)
    )
    independent_labels, independent_components = tarjan_scc(adjacency)

    print()
    print("=" * 100)
    print(f"MAZE {maze_name}: INDEPENDENT AUDIT")
    print("=" * 100)

    # Check 1: SCC counts and full partition.
    print("[1] SCC audit")
    print(f"    scipy SCC count   : {main_scc.num_scc}")
    print(f"    Tarjan SCC count  : {len(independent_components)}")

    partition_ok = same_partition(
        main_scc.labels,
        independent_labels,
    )
    print(f"    identical partition: {partition_ok}")

    if not partition_ok:
        raise AssertionError(
            f"Maze {maze_name}: scipy and independent Tarjan SCC "
            "partitions do not match."
        )

    scipy_sizes = sorted(
        np.bincount(main_scc.labels).tolist(),
        reverse=True,
    )
    tarjan_sizes = sorted(
        [len(c) for c in independent_components],
        reverse=True,
    )
    sizes_ok = scipy_sizes == tarjan_sizes
    print(f"    SCC size multiset identical: {sizes_ok}")
    print(f"    largest SCC size: {scipy_sizes[0]} / {len(pairs)} pairs")

    if not sizes_ok:
        raise AssertionError("SCC size multisets do not match.")

    # Check 2: all pairs in certificate subset have one independent SCC label.
    num_internal_pairs, distinct_labels = verify_subset_one_label(
        Y,
        pair_to_id,
        independent_labels,
    )

    print()
    print("[2] kappa=1 subset audit")
    print(f"    |Y|                    : {len(Y)}")
    print(f"    internal state pairs   : {num_internal_pairs}")
    print(f"    distinct Tarjan labels : {len(distinct_labels)}")
    print(f"    labels                 : {distinct_labels}")
    print(f"    certified lower bound  : d_PR(T;1) >= {len(Y) - 1}")

    # Optional second implementation for the maximum-clique size.
    try:
        import networkx as nx

        M = np.full((T1.shape[1], T1.shape[1]), -1, dtype=np.int32)
        for idx, (i, j) in enumerate(pairs):
            lab = int(independent_labels[idx])
            M[i, j] = lab
            M[j, i] = lab

        nx_best_size = 0
        nx_best_label = None
        for lab in sorted(set(int(x) for x in independent_labels.tolist())):
            G = nx.Graph()
            G.add_nodes_from(range(T1.shape[1]))
            for i in range(T1.shape[1]):
                for j in range(i + 1, T1.shape[1]):
                    if int(M[i, j]) == lab:
                        G.add_edge(i, j)
            local_best = max(nx.find_cliques(G), key=len, default=[])
            if len(local_best) > nx_best_size:
                nx_best_size = len(local_best)
                nx_best_label = lab

        print(f"    NetworkX exact max clique size: {nx_best_size}")
        print(f"    main exact max clique size    : {len(Y)}")
        print(f"    max-clique size agrees        : {nx_best_size == len(Y)}")
        assert nx_best_size == len(Y)
    except ImportError:
        print("    NetworkX not installed; skipping independent max-clique audit.")

    expected_pairs = len(Y) * (len(Y) - 1) // 2
    assert num_internal_pairs == expected_pairs
    assert len(distinct_labels) == 1, (
        f"Maze {maze_name}: claimed kappa=1 subset touches "
        f"{len(distinct_labels)} independent SCC labels."
    )

    # Check 3: explicit round-trip reachability witnesses.
    print()
    print("[3] Concrete bidirectional reachability witness(es)")

    all_internal_pairs = list(combinations(sorted(Y), 2))

    for w_idx in range(num_witnesses):
        # deterministic but spread through the pair list
        p = all_internal_pairs[
            (w_idx * max(1, len(all_internal_pairs) // max(1, num_witnesses)))
            % len(all_internal_pairs)
        ]
        q = all_internal_pairs[
            ((w_idx + 1) * max(1, len(all_internal_pairs) // (num_witnesses + 1)))
            % len(all_internal_pairs)
        ]
        if p == q:
            q = all_internal_pairs[-1]

        p_id = pair_to_id[p]
        q_id = pair_to_id[q]

        assert independent_labels[p_id] == independent_labels[q_id]

        forward = bfs_path(
            p_id,
            q_id,
            labeled_adjacency,
        )
        backward = bfs_path(
            q_id,
            p_id,
            labeled_adjacency,
        )

        assert forward is not None
        assert backward is not None

        f_traj_ids = []
        # replay directly on actual state-pairs to verify action sequence
        f_traj_pairs = apply_action_sequence_to_pair(
            p,
            forward,
            T1,
        )
        b_traj_pairs = apply_action_sequence_to_pair(
            q,
            backward,
            T1,
        )

        assert f_traj_pairs[-1] == q
        assert b_traj_pairs[-1] == p

        p_cells = tuple(core.state_rc(s) for s in p)
        q_cells = tuple(core.state_rc(s) for s in q)

        print(f"    witness {w_idx + 1}:")
        print(f"      p = {p} = {p_cells}")
        print(f"      q = {q} = {q_cells}")
        print(
            f"      p -> q ({len(forward)} primitive steps): "
            + " ".join(forward)
        )
        print(
            f"      q -> p ({len(backward)} primitive steps): "
            + " ".join(backward)
        )

    print()
    print(f"Maze {maze_name}: ALL AUDITS PASS.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maze",
        nargs="+",
        choices=("A", "B", "C"),
        default=("A", "B", "C"),
    )
    parser.add_argument(
        "--num-witnesses",
        type=int,
        default=1,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for maze_name in args.maze:
        audit_maze(
            maze_name,
            num_witnesses=args.num_witnesses,
        )

    print()
    print("=" * 100)
    print("ALL REQUESTED MAZE AUDITS PASS.")
    print("=" * 100)


if __name__ == "__main__":
    main()
