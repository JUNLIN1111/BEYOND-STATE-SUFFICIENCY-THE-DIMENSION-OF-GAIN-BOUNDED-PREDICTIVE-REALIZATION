from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import hdf5plugin  # noqa: F401

    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aliasing_experiment import _get_env_state, _make_env, _reset_env_to_state, _step_env  # noqa: E402
from planner_success_reference_diagnostic import (  # noqa: E402
    _call_policy,
    _encode_pixels,
    _extract_pixels_from_obs,
    _find_key,
    _initial_previous_action,
    _load_model,
    _load_stable_worldmodel_policy,
    _policy_info,
    _read_h5_rows,
    _set_policy_env,
)
from task_cost import task_cost  # noqa: E402


EPS = 1e-8
MODEL_DIMS = {
    "state8": 8,
    "state16": 16,
    "state32": 32,
    "state64": 64,
    "baseline192": 192,
    "state302": 302,
    "state502": 502,
}


def _parse_name_paths(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path)
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def _episode_rows(episode_idx: np.ndarray, step_idx: np.ndarray):
    for episode in np.unique(episode_idx):
        rows = np.where(episode_idx == episode)[0]
        rows = rows[np.argsort(step_idx[rows])]
        if rows.size and np.all(np.diff(step_idx[rows]) == 1):
            yield int(episode), rows


def _candidate_starts(episode_idx: np.ndarray, step_idx: np.ndarray, goal_offset: int, eval_budget: int):
    starts = []
    for episode, rows in _episode_rows(episode_idx, step_idx):
        if rows.size <= goal_offset + eval_budget:
            continue
        for local_start in range(0, rows.size - goal_offset - eval_budget):
            starts.append((episode, int(rows[local_start])))
    return starts


def _success_from_info(done: bool, info: object, threshold: float) -> bool:
    if isinstance(info, dict):
        value = float(info.get("success", info.get("is_success", 0.0)))
    else:
        value = 0.0
    if done and value <= 0.0:
        value = 1.0
    return bool(value >= threshold)


def _pad(paths: List[np.ndarray], pad_value: float = np.nan) -> np.ndarray:
    max_len = max(path.shape[0] for path in paths)
    shape = (len(paths), max_len) + paths[0].shape[1:]
    out = np.full(shape, pad_value, dtype=paths[0].dtype)
    for idx, path in enumerate(paths):
        out[idx, : path.shape[0]] = path
    return out


def collect(args: argparse.Namespace) -> None:
    print(
        "[fixed_goal_sa] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[fixed_goal_sa] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    policy = _load_stable_worldmodel_policy(args.policy, Path(args.eval_config))
    env = _make_env(args.env_name)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pixels_paths: List[np.ndarray] = []
    state_paths: List[np.ndarray] = []
    action_paths: List[np.ndarray] = []
    reward_paths: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    num_success = 0
    num_failure = 0

    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        state_key = _find_key(h5, [args.state_key, "state"])
        try:
            action_key: Optional[str] = _find_key(h5, [args.action_key, "action"])
        except KeyError:
            action_key = None
        episode_key = _find_key(h5, [args.episode_key, "episode_idx", "ep_idx"])
        step_key = _find_key(h5, [args.step_key, "step_idx"])
        starts = _candidate_starts(np.asarray(h5[episode_key]).reshape(-1), np.asarray(h5[step_key]).reshape(-1), args.goal_offset_steps, args.eval_budget)
        rng.shuffle(starts)
        print(f"[fixed_goal_sa] candidate starts: {len(starts)}", flush=True)

        for attempt_idx, (episode, start_row) in enumerate(starts[: args.max_attempts]):
            if num_success >= args.num_success and num_failure >= args.num_failure:
                break
            goal_row = start_row + args.goal_offset_steps
            start_state = np.asarray(h5[state_key][start_row], dtype=np.float64)
            goal_state = np.asarray(h5[state_key][goal_row], dtype=np.float64)
            goal_pixels = _read_h5_rows(h5[pixels_key], np.asarray([goal_row]))[0]
            obs, _info = _reset_env_to_state(env, start_state, goal_state, args.reset_state_tol)
            action_dim = _set_policy_env(policy, env)
            previous_action = _initial_previous_action(h5, action_key, start_row, action_dim)
            pixels = [_extract_pixels_from_obs(env, obs)]
            states = [_get_env_state(env, obs=obs)]
            actions = []
            rewards = []
            done = False
            final_info = {}
            for _step in range(args.eval_budget):
                policy_obs = _policy_info(
                    pixels=pixels[-1],
                    goal_pixels=goal_pixels,
                    state=states[-1],
                    goal_state=goal_state,
                    previous_action=previous_action,
                )
                action = _call_policy(policy, policy_obs)
                obs, reward, done, info = _step_env(env, action)
                final_info = info if isinstance(info, dict) else {}
                actions.append(action.astype(np.float32))
                previous_action = action.astype(np.float32)
                rewards.append(float(reward))
                pixels.append(_extract_pixels_from_obs(env, obs))
                states.append(_get_env_state(env, obs=obs, info=info))
                if done:
                    break
            terminal_state = np.asarray(states[-1], dtype=np.float64)[: goal_state.size]
            final_cost = float(task_cost(terminal_state, goal_state))
            success = _success_from_info(done, final_info, args.success_threshold)
            keep = (success and num_success < args.num_success) or ((not success) and final_cost >= args.failure_cost_threshold and num_failure < args.num_failure)
            if keep:
                traj_idx = len(rows)
                pixels_paths.append(np.asarray(pixels))
                state_paths.append(np.asarray(states, dtype=np.float32))
                action_paths.append(np.asarray(actions, dtype=np.float32))
                reward_paths.append(np.asarray(rewards, dtype=np.float32))
                rows.append(
                    {
                        "traj_idx": traj_idx,
                        "attempt_idx": int(attempt_idx),
                        "dataset_episode": int(episode),
                        "start_row": int(start_row),
                        "goal_row": int(goal_row),
                        "goal_offset_steps": int(args.goal_offset_steps),
                        "success": int(success),
                        "done": int(done),
                        "length": int(len(pixels)),
                        "executed_steps": int(len(actions)),
                        "start_task_cost": float(task_cost(start_state, goal_state)),
                        "final_task_cost": final_cost,
                    }
                )
                num_success += int(success)
                num_failure += int(not success)
                print(
                    f"[fixed_goal_sa] kept traj={traj_idx} success={success} final_cost={final_cost:.4f} "
                    f"successes={num_success}/{args.num_success} failures={num_failure}/{args.num_failure}",
                    flush=True,
                )
            elif attempt_idx % args.progress_every == 0:
                print(f"[fixed_goal_sa] attempt={attempt_idx} successes={num_success} failures={num_failure}", flush=True)

    if not rows:
        raise RuntimeError("No trajectories collected. Increase max_attempts or relax thresholds.")
    np.savez_compressed(
        output,
        pixels=_pad(pixels_paths, 0),
        states=_pad(state_paths),
        actions=_pad(action_paths, 0.0),
        rewards=_pad([item[:, None] for item in reward_paths]).squeeze(-1),
        lengths=np.asarray([path.shape[0] for path in pixels_paths], dtype=np.int64),
        start_rows=np.asarray([row["start_row"] for row in rows], dtype=np.int64),
        goal_rows=np.asarray([row["goal_row"] for row in rows], dtype=np.int64),
        success=np.asarray([row["success"] for row in rows], dtype=np.int64),
        goal_offset_steps=np.asarray(args.goal_offset_steps, dtype=np.int64),
        eval_budget=np.asarray(args.eval_budget, dtype=np.int64),
        summary_rows_json=np.asarray(json.dumps(rows)),
        policy=np.asarray(args.policy),
    )
    _write_csv(output.with_suffix(".summary.csv"), rows)
    _write_json(output.with_suffix(".meta.json"), {"args": vars(args), "num_rows": len(rows)})
    print(f"[fixed_goal_sa] wrote {output}", flush=True)


def _distance_curve_rows(z_paths: np.ndarray, z_goals: np.ndarray, lengths: np.ndarray, meta_rows: List[Dict[str, object]], model: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    curve_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for traj_idx, (traj, goal, length, meta) in enumerate(zip(z_paths, z_goals, lengths, meta_rows)):
        length = int(length)
        distances = np.sum((traj[:length] - goal[None]) ** 2, axis=-1)
        d0 = float(distances[0])
        deltas = np.diff(distances)
        positive = np.maximum(deltas, 0.0)
        threshold = float(meta.get("adverse_threshold", 0.01))
        max_idx = int(np.argmax(distances))
        has_drop_after_overshoot = False
        for t1 in range(length):
            if distances[t1] <= d0 + threshold:
                continue
            if np.any(distances[t1 + 1 :] < distances[t1] - threshold):
                has_drop_after_overshoot = True
                break
        for t_idx, value in enumerate(distances):
            curve_rows.append(
                {
                    "model": model,
                    "latent_dim": int(traj.shape[-1]),
                    "traj_idx": int(traj_idx),
                    "success": int(meta["success"]),
                    "goal_offset_steps": int(meta["goal_offset_steps"]),
                    "t": int(t_idx),
                    "D_t": float(value),
                    "D_t_norm": float(value / (d0 + EPS)),
                    "D0": d0,
                    "final_task_cost": float(meta["final_task_cost"]),
                    "start_row": int(meta["start_row"]),
                    "goal_row": int(meta["goal_row"]),
                }
            )
        summary_rows.append(
            {
                "model": model,
                "latent_dim": int(traj.shape[-1]),
                "traj_idx": int(traj_idx),
                "success": int(meta["success"]),
                "goal_offset_steps": int(meta["goal_offset_steps"]),
                "length": length,
                "monotonic_violation_rate": float(np.mean(deltas > 0.0)) if deltas.size else float("nan"),
                "positive_violation_magnitude": float(np.sum(positive) / (d0 + EPS)),
                "max_overshoot": float((np.max(distances) - d0) / (d0 + EPS)),
                "eventual_drop": float(d0 - distances[-1]),
                "eventual_drop_norm": float((d0 - distances[-1]) / (d0 + EPS)),
                "adverse_prefix_flag": int(np.max(distances) > d0 + threshold),
                "先大后小_flag": int(has_drop_after_overshoot),
                "argmax_distance_t": max_idx,
                "D0": d0,
                "DT": float(distances[-1]),
                "Dmax": float(np.max(distances)),
                "final_task_cost": float(meta["final_task_cost"]),
            }
        )
    return curve_rows, summary_rows


def _plot_distance_outputs(curve_rows: List[Dict[str, object]], summary_rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fixed_goal_sa] matplotlib unavailable; skipping plots.", flush=True)
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for success_value, name in ((1, "success"), (0, "failure")):
        subset = [row for row in curve_rows if int(row["success"]) == success_value]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 3.8), facecolor="white")
        plotted = 0
        for (model, traj_idx), rows in _group_curve_rows(subset).items():
            if plotted >= 20:
                break
            rows = sorted(rows, key=lambda row: int(row["t"]))
            ax.plot([row["t"] for row in rows], [row["D_t_norm"] for row in rows], alpha=0.65, label=model if plotted < 5 else None)
            plotted += 1
        ax.set_title(f"Fixed-goal latent distance curves: {name}")
        ax.set_xlabel("env step")
        ax.set_ylabel("D_t / D_0")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"distance_curves_{name}.png", dpi=260)
        fig.savefig(fig_dir / f"distance_curves_{name}.pdf")
        plt.close(fig)

    for key, filename, ylabel in (
        ("max_overshoot", "boxplot_max_overshoot", "max overshoot / D0"),
        ("positive_violation_magnitude", "boxplot_violation_magnitude", "sum positive violations / D0"),
    ):
        groups = _summary_groups(summary_rows)
        if not groups:
            continue
        labels = list(groups)
        data = [[float(row[key]) for row in groups[label]] for label in labels]
        fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 0.9), 3.8), facecolor="white")
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{filename}.png", dpi=260)
        fig.savefig(fig_dir / f"{filename}.pdf")
        plt.close(fig)


def _group_curve_rows(rows: List[Dict[str, object]]) -> Dict[Tuple[str, int], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), int(row["traj_idx"])), []).append(row)
    return grouped


def _summary_groups(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        label = f"{row['model']}\n{'succ' if int(row['success']) else 'fail'}"
        grouped.setdefault(label, []).append(row)
    return grouped


def analyze(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model_paths = _parse_name_paths(args.models)
    trajectory_paths = _parse_name_paths(args.trajectory_npzs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_curve_rows: List[Dict[str, object]] = []
    all_summary_rows: List[Dict[str, object]] = []
    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        for model_name, trajectory_path in trajectory_paths.items():
            if model_name not in model_paths:
                print(f"[fixed_goal_sa] skipping {model_name}: no matching checkpoint in --models", flush=True)
                continue
            data = np.load(trajectory_path, allow_pickle=False)
            pixels = np.asarray(data["pixels"])
            lengths = np.asarray(data["lengths"], dtype=np.int64)
            goal_rows = np.asarray(data["goal_rows"], dtype=np.int64)
            meta_rows = json.loads(str(np.asarray(data["summary_rows_json"]).item()))
            for row in meta_rows:
                row["adverse_threshold"] = args.adverse_threshold
            goal_pixels = _read_h5_rows(h5[pixels_key], goal_rows)
            print(f"[fixed_goal_sa] encoding {trajectory_path} with {model_name}", flush=True)
            model = _load_model(model_paths[model_name], device)
            z_paths = _encode_pixels(model, pixels, args.img_size, args.batch_size, device)
            z_goals = _encode_pixels(model, goal_pixels[:, None], args.img_size, args.batch_size, device)[:, 0]
            curve_rows, summary_rows = _distance_curve_rows(z_paths, z_goals, lengths, meta_rows, model_name)
            all_curve_rows.extend(curve_rows)
            all_summary_rows.extend(summary_rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_csv(output_dir / "traj_distance_curves.csv", all_curve_rows)
    _write_csv(output_dir / "trajectory_sa_summary.csv", all_summary_rows)
    _write_json(output_dir / "fixed_goal_sa_metadata.json", {"args": vars(args)})
    _plot_distance_outputs(all_curve_rows, all_summary_rows, output_dir)
    print(f"[fixed_goal_sa] wrote outputs under {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-goal self-approaching diagnostics for PushT LeWorldModel trajectories.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect closed-loop trajectories with a fixed per-episode goal.")
    collect_parser.add_argument("--policy", required=True)
    collect_parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--eval_config", default="config/eval/pusht.yaml")
    collect_parser.add_argument("--env_name", default="swm/PushT-v1")
    collect_parser.add_argument("--num_success", type=int, default=5)
    collect_parser.add_argument("--num_failure", type=int, default=5)
    collect_parser.add_argument("--max_attempts", type=int, default=100)
    collect_parser.add_argument("--goal_offset_steps", type=int, default=25)
    collect_parser.add_argument("--eval_budget", type=int, default=50)
    collect_parser.add_argument("--success_threshold", type=float, default=0.5)
    collect_parser.add_argument("--failure_cost_threshold", type=float, default=0.10)
    collect_parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    collect_parser.add_argument("--pixels_key", default="pixels")
    collect_parser.add_argument("--state_key", default="state")
    collect_parser.add_argument("--action_key", default="action")
    collect_parser.add_argument("--episode_key", default="episode_idx")
    collect_parser.add_argument("--step_key", default="step_idx")
    collect_parser.add_argument("--seed", type=int, default=0)
    collect_parser.add_argument("--progress_every", type=int, default=25)

    analyze_parser = subparsers.add_parser("analyze", help="Encode collected trajectories and compute fixed-goal SA metrics.")
    analyze_parser.add_argument("--trajectory_npzs", nargs="+", required=True, help="NAME=trajectory.npz")
    analyze_parser.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint_object.ckpt")
    analyze_parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    analyze_parser.add_argument("--output_dir", default="results/self_approaching_fixed_goal")
    analyze_parser.add_argument("--adverse_threshold", type=float, default=0.01)
    analyze_parser.add_argument("--pixels_key", default="pixels")
    analyze_parser.add_argument("--img_size", type=int, default=224)
    analyze_parser.add_argument("--batch_size", type=int, default=16)
    analyze_parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.mode == "collect":
        collect(args)
    elif args.mode == "analyze":
        analyze(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
