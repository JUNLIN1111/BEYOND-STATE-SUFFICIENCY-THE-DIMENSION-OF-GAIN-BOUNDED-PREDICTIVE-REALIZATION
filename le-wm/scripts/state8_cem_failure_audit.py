from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aliasing_experiment import _get_env_state, _make_env, _reset_env_to_state, _step_env  # noqa: E402
from cem_trace import CEMTraceConfig, CEMTracer  # noqa: E402
from planner_success_reference_diagnostic import (  # noqa: E402
    _call_policy,
    _extract_pixels_from_obs,
    _find_key,
    _initial_previous_action,
    _load_stable_worldmodel_policy,
    _policy_info,
    _read_h5_rows,
    _set_policy_env,
)
from task_cost import task_cost  # noqa: E402


def _ensure_stable_worldmodel_dataset_link(dataset: str, dataset_name: str = "pusht_expert_train.h5") -> None:
    source = Path(dataset).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset does not exist: {source}")
    target = Path.home() / ".stable_worldmodel" / dataset_name
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source)
        print(f"[state8_cem_audit] linked stable_worldmodel dataset: {target} -> {source}", flush=True)
    except OSError:
        # Last-resort fallback for filesystems that do not support symlinks.
        import shutil

        shutil.copy2(source, target)
        print(f"[state8_cem_audit] copied stable_worldmodel dataset: {source} -> {target}", flush=True)


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


def _run_episode(
    policy,
    tracer: CEMTracer,
    env,
    h5: h5py.File,
    pixels_key: str,
    action_key: Optional[str],
    state_key: str,
    start_row: int,
    goal_row: int,
    eval_budget: int,
    reset_state_tol: float,
    success_threshold: float,
) -> dict:
    start_state = np.asarray(h5[state_key][start_row], dtype=np.float64)
    goal_state = np.asarray(h5[state_key][goal_row], dtype=np.float64)
    goal_pixels = _read_h5_rows(h5[pixels_key], np.asarray([goal_row]))[0]
    obs, _info = _reset_env_to_state(env, start_state, goal_state, reset_state_tol)
    action_dim = _set_policy_env(policy, env)
    previous_action = _initial_previous_action(h5, action_key, start_row, action_dim)
    states = [_get_env_state(env, obs=obs)]
    pixels = [_extract_pixels_from_obs(env, obs)]
    done = False
    final_info = {}
    for _step in range(eval_budget):
        policy_obs = _policy_info(
            pixels=pixels[-1],
            goal_pixels=goal_pixels,
            state=states[-1],
            goal_state=goal_state,
            previous_action=previous_action,
        )
        action = _call_policy(policy, policy_obs)
        obs, _reward, done, info = _step_env(env, action)
        final_info = info if isinstance(info, dict) else {}
        previous_action = action.astype(np.float32)
        states.append(_get_env_state(env, obs=obs, info=info))
        pixels.append(_extract_pixels_from_obs(env, obs))
        if done:
            break
    terminal_state = np.asarray(states[-1], dtype=np.float64)[: goal_state.size]
    final_cost = float(task_cost(terminal_state, goal_state))
    success = _success_from_info(done, final_info, success_threshold)
    return {
        "success": success,
        "done": bool(done),
        "final_cost": final_cost,
        "executed_steps": len(states) - 1,
        "start_cost": float(task_cost(start_state, goal_state)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PushT eval and keep full CEM audit for selected failed/successful episodes.")
    parser.add_argument("--policy", default="/home/jw3425/.stable_worldmodel/pusht/state8_baseline")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--output_dir", default="results/cem_trace_state8_failure_full")
    parser.add_argument("--eval_config", default="config/eval/pusht.yaml")
    parser.add_argument("--env_name", default="swm/PushT-v1")
    parser.add_argument("--num_failed_episodes", type=int, default=5)
    parser.add_argument("--num_success_episodes", type=int, default=0)
    parser.add_argument("--max_attempts", type=int, default=100)
    parser.add_argument("--goal_offset_steps", type=int, default=25)
    parser.add_argument("--eval_budget", type=int, default=50)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument("--failure_cost_threshold", type=float, default=0.10)
    parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    parser.add_argument("--trace_max_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--episode_key", default="episode_idx")
    parser.add_argument("--step_key", default="step_idx")
    args = parser.parse_args()

    _ensure_stable_worldmodel_dataset_link(args.dataset)
    policy = _load_stable_worldmodel_policy(args.policy, Path(args.eval_config))
    solver = getattr(policy, "solver", None)
    model = getattr(solver, "model", None) or getattr(solver, "_model", None)
    if model is None:
        raise RuntimeError("Could not find policy.solver.model for CEM tracing.")
    trace_config = CEMTraceConfig(
        enabled=True,
        trace_dir=Path(args.output_dir),
        trace_episodes=1,
        trace_max_steps=args.trace_max_steps,
        true_replay_candidates=True,
        model_name=str(args.policy),
        horizon_blocks=5,
        action_block=5,
        receding_horizon_raw=5,
        env_name=args.env_name,
        reset_state_tol=args.reset_state_tol,
    )
    tracer = CEMTracer(trace_config)
    tracer.attach(model=model, policy=policy, cfg=None)
    env = _make_env(args.env_name)
    rng = np.random.default_rng(args.seed)
    failures = []
    successes = []
    attempts = []
    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        state_key = _find_key(h5, [args.state_key, "state"])
        action_key = _find_key(h5, [args.action_key, "action"])
        episode_key = _find_key(h5, [args.episode_key, "episode_idx", "ep_idx"])
        step_key = _find_key(h5, [args.step_key, "step_idx"])
        episode_idx = np.asarray(h5[episode_key]).reshape(-1)
        step_idx = np.asarray(h5[step_key]).reshape(-1)
        starts = _candidate_starts(episode_idx, step_idx, args.goal_offset_steps, args.eval_budget)
        rng.shuffle(starts)
        for attempt_idx, (episode, start_row) in enumerate(starts[: args.max_attempts]):
            if len(failures) >= args.num_failed_episodes and len(successes) >= args.num_success_episodes:
                break
            trace_episode_id = len(attempts)
            tracer.start_episode(trace_episode_id)
            goal_row = start_row + args.goal_offset_steps
            result = _run_episode(
                policy,
                tracer,
                env,
                h5,
                pixels_key,
                action_key,
                state_key,
                start_row,
                goal_row,
                args.eval_budget,
                args.reset_state_tol,
                args.success_threshold,
            )
            is_failure = (not result["success"]) and result["final_cost"] >= args.failure_cost_threshold
            is_success = bool(result["success"])
            keep_failure = is_failure and len(failures) < args.num_failed_episodes
            keep_success = is_success and len(successes) < args.num_success_episodes
            tracer.mark_episode_result(trace_episode_id, result["success"], result["final_cost"])
            attempts.append(
                {
                    "trace_episode_id": trace_episode_id,
                    "attempt_idx": int(attempt_idx),
                    "dataset_episode": int(episode),
                    "start_row": int(start_row),
                    "goal_row": int(goal_row),
                    **result,
                    "kept_failure": bool(keep_failure),
                    "kept_success": bool(keep_success),
                }
            )
            print(
                f"[state8_cem_audit] attempt={attempt_idx} trace_ep={trace_episode_id} "
                f"success={result['success']} final_cost={result['final_cost']:.4f} "
                f"failures={len(failures)}/{args.num_failed_episodes} successes={len(successes)}/{args.num_success_episodes}",
                flush=True,
            )
            if keep_failure:
                failures.append(trace_episode_id)
            if keep_success:
                successes.append(trace_episode_id)
    keep = set(int(item) for item in failures + successes)
    for attr in ("planning_rows", "iter_rows", "pool_rows", "exec_rows", "timeline_rows", "angle_rows", "true_replay_rows"):
        rows = getattr(tracer, attr)
        setattr(tracer, attr, [row for row in rows if int(row.get("episode_id", -1)) in keep])
    tracer.close({"attempts": attempts, "kept_failures": failures, "kept_successes": successes})
    with (Path(args.output_dir) / "failure_attempts.json").open("w") as file:
        json.dump({"attempts": attempts, "kept_failures": failures, "kept_successes": successes, "args": vars(args)}, file, indent=2)
    print(f"[state8_cem_audit] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
