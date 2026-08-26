from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py
import numpy as np


TASK_DATASETS = {
    "pusht": ["/tmp/pusht_expert_train.h5"],
    "cube": ["/tmp/lewm-cube/ogbench/cube_single_expert.h5", "/tmp/lewm-cube/cube_single_expert.h5"],
    "tworoom": ["/tmp/tworoom_expert_train.h5", "/tmp/two_room_expert_train.h5"],
    "reacher": ["/tmp/reacher_expert_train.h5"],
}


def _find_key(h5: h5py.File, candidates: Iterable[str]) -> str | None:
    for key in candidates:
        if key and key in h5:
            return key
    return None


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _inspect_dataset(path: Path) -> Dict[str, object]:
    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        state_key = _find_key(h5, ["state", "states", "proprio", "observation", "observations", "qpos", "qvel"])
        episode_key = _find_key(h5, ["episode_idx", "episode", "episode_id", "traj_idx", "ep_idx"])
        step_key = _find_key(h5, ["step_idx", "timestep", "step"])
        state_dim = int(h5[state_key].shape[-1]) if state_key is not None and len(h5[state_key].shape) > 1 else 1
    return {"path": str(path), "keys": keys, "state_key": state_key, "episode_key": episode_key, "step_key": step_key, "state_dim": state_dim}


def _first_existing(paths: List[str]) -> Path | None:
    for text in paths:
        path = Path(text)
        if path.exists():
            return path
    return None


def _read_summary(output_dir: Path, task: str, graph_mode: str, state_dim: int) -> Dict[str, object]:
    summary_path = output_dir / task / "plannable_dimension_summary.json"
    if not summary_path.exists():
        return {"task": task, "graph_mode": graph_mode, "physical_state_dim": state_dim, "status": "missing_summary"}
    with summary_path.open() as file:
        payload = json.load(file)["summary"]
    return {
        "task": task,
        "graph_mode": graph_mode,
        "physical_state_dim": state_dim,
        "PCA_rank90": "",
        "local_id_TwoNN": "",
        "d_plan_80": payload.get("d_plan_80"),
        "d_plan_90": payload.get("d_plan_90"),
        "d_plan_95": payload.get("d_plan_95"),
        "d_plan_99": payload.get("d_plan_99"),
        "num_nodes": payload.get("num_nodes"),
        "connected_component_size": "",
        "finite_pair_fraction": "",
        "status": "ok",
    }


def _plot_task_table(path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not path.exists():
        return
    with path.open() as file:
        rows = [row for row in csv.DictReader(file) if row.get("status") == "ok"]
    if not rows:
        return
    labels = [row["task"] for row in rows]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.8))
    for offset, key in [(-1.5, "physical_state_dim"), (-0.5, "d_plan_90"), (0.5, "d_plan_95"), (1.5, "d_plan_99")]:
        ax.bar(x + offset * width, [float(row.get(key) or 0.0) for row in rows], width=width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("dimension")
    ax.set_title("Physical dimension vs plannable spectral dimension")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path.parent / "task_dim_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plannable-dimension spectral diagnostics for all discoverable offline tasks.")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--num_nodes", type=int, default=5000)
    parser.add_argument("--landmarks", type=int, default=1000)
    parser.add_argument("--knn_values", default="10")
    parser.add_argument("--graph_modes", default="temporal_only,temporal_plus_knn")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_dir)
    rows: List[Dict[str, object]] = []
    for task, paths in TASK_DATASETS.items():
        dataset = _first_existing(paths)
        if dataset is None:
            print(f"[all_tasks] warning: no dataset found for {task}")
            continue
        info = _inspect_dataset(dataset)
        if info["state_key"] is None or info["episode_key"] is None:
            print(f"[all_tasks] warning: {task} missing state/episode keys; keys={info['keys']}")
            continue
        task_output = root / task
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("estimate_plannable_dimension.py")),
            "--dataset",
            str(dataset),
            "--feature_mode",
            "state",
            "--state_key",
            str(info["state_key"]),
            "--episode_key",
            str(info["episode_key"]),
            "--num_nodes",
            str(args.num_nodes),
            "--landmarks",
            str(args.landmarks),
            "--knn_values",
            args.knn_values,
            "--graph_modes",
            args.graph_modes,
            "--task_name",
            task,
            "--output_dir",
            str(task_output),
        ]
        print("[all_tasks] " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        rows.append(_read_summary(task_output, task, args.graph_modes, int(info["state_dim"])))
    table_path = root / "summaries" / "task_spectral_table.csv"
    _write_csv(table_path, rows)
    _plot_task_table(table_path)
    print(f"[all_tasks] wrote {table_path}")


if __name__ == "__main__":
    main()
