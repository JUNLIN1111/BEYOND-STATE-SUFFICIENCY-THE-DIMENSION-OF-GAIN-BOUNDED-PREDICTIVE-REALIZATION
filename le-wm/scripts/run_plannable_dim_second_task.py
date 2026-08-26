from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _find_key(keys, candidates: List[str]) -> str:
    lower = {key.lower(): key for key in keys}
    for candidate in candidates:
        if candidate in keys:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return ""


def _detect_keys(dataset: Path) -> Dict[str, str]:
    with h5py.File(dataset, "r") as h5:
        keys = list(h5.keys())
        print("[second_task] dataset keys:")
        for key in keys:
            print(f"  - {key}: {getattr(h5[key], 'shape', None)}")
        state_key = _find_key(keys, ["state", "states", "proprio", "observation", "obs"])
        action_key = _find_key(keys, ["action", "actions"])
        episode_key = _find_key(keys, ["episode_idx", "episode", "traj_idx", "trajectory_idx", "ep_idx"])
        step_key = _find_key(keys, ["step_idx", "timestep", "step", "time"])
    missing = [name for name, key in [("state", state_key), ("episode", episode_key), ("step", step_key)] if not key]
    if missing:
        raise KeyError(
            "Could not robustly detect required Cube keys: "
            + ", ".join(missing)
            + ". Please rerun with explicit compatible key names after inspecting the printed HDF5 keys."
        )
    return {"state_key": state_key, "action_key": action_key, "episode_key": episode_key, "step_key": step_key}


def _candidate_dataset(paths: str) -> Path:
    for item in str(paths).replace(",", " ").split():
        path = Path(item)
        if path.exists():
            return path
    raise FileNotFoundError(f"No candidate second-task dataset exists among: {paths}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline plannable-dimension spectrum on a second task if keys are detectable.")
    parser.add_argument(
        "--dataset_candidates",
        default="/tmp/lewm-cube/ogbench/cube_single_expert.h5,/tmp/lewm-cube/cube_single_expert.h5",
    )
    parser.add_argument("--dataset", default="")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--num_nodes", type=int, default=5000)
    parser.add_argument("--landmarks", type=int, default=1000)
    parser.add_argument("--max_embed_dim", type=int, default=512)
    parser.add_argument("--graph_modes", default="temporal_only,temporal_plus_knn_k10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_estimate", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset) if args.dataset else _candidate_dataset(args.dataset_candidates)
    output_dir = Path(args.output_dir)
    work_dir = output_dir / "second_task_work"
    keys = _detect_keys(dataset)
    if not args.skip_estimate:
        estimator = Path(__file__).with_name("estimate_plannable_dimension.py")
        command = [
            sys.executable,
            str(estimator),
            "--dataset",
            str(dataset),
            "--feature_mode",
            "state",
            "--state_key",
            keys["state_key"],
            "--episode_key",
            keys["episode_key"],
            "--step_key",
            keys["step_key"],
            "--num_nodes",
            str(args.num_nodes),
            "--landmarks",
            str(args.landmarks),
            "--max_embed_dim",
            str(args.max_embed_dim),
            "--graph_modes",
            args.graph_modes,
            "--disconnect_handling",
            "largest_component",
            "--task_name",
            "second_task",
            "--output_dir",
            str(work_dir),
            "--seed",
            str(args.seed),
        ]
        print("[second_task] running estimator:")
        print("  " + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    with (work_dir / "mds_spectrum.json").open() as file:
        spectra = json.load(file)
    rows = []
    for graph_mode, payload in spectra.items():
        rows.append(
            {
                "task": "second_task",
                "dataset": str(dataset),
                "graph_mode": graph_mode,
                "d80": payload.get("rank80"),
                "d90": payload.get("rank90"),
                "d95": payload.get("rank95"),
                "d99": payload.get("rank99"),
                "d100": payload.get("rank100"),
                "positive_eigen_count": payload.get("coverage_summary", {}).get("positive_eigenvalues"),
                "negative_energy_ratio": payload.get("coverage_summary", {}).get("negative_energy_ratio"),
                "largest_component_size": payload.get("component_info", {}).get("largest_component_size"),
                "num_connected_components": payload.get("component_info", {}).get("num_connected_components"),
            }
        )
    _write_csv(output_dir / "spectra" / "second_task_spectral_table.csv", rows)
    summary_lines = [
        "# Second Task Spectral Diagnostic",
        "",
        f"- Dataset: `{dataset}`",
        f"- Detected state key: `{keys['state_key']}`",
        f"- Detected episode key: `{keys['episode_key']}`",
        f"- Detected step key: `{keys['step_key']}`",
        "",
        "This is an offline spectrum-only check. It does not include closed-loop Cube planning evaluation.",
        "",
        "| graph mode | d90 | d95 | d99 | LCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary_lines.append(f"| {row['graph_mode']} | {row['d90']} | {row['d95']} | {row['d99']} | {row['largest_component_size']} |")
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "second_task_spectral_summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"[second_task] wrote {output_dir / 'spectra' / 'second_task_spectral_table.csv'}")


if __name__ == "__main__":
    main()
