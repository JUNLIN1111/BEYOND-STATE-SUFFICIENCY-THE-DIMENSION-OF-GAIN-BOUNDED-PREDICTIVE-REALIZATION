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

from aliasing_experiment import _action_normalization, _get_env_state, _make_env, _reset_env_to_state, _step_env  # noqa: E402
from planner_success_reference_diagnostic import (  # noqa: E402
    _call_policy,
    _encode_pixels,
    _extract_pixels_from_obs,
    _find_key,
    _initial_previous_action,
    _load_model,
    _load_stable_worldmodel_policy,
    _policy_info,
    _preprocess_pixels,
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
    "global_k32": 32,
    "local_k32": 32,
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


def _episode_rows(episode_idx: np.ndarray, step_idx: np.ndarray) -> Dict[int, np.ndarray]:
    result = {}
    for episode in np.unique(episode_idx):
        rows = np.where(episode_idx == episode)[0]
        rows = rows[np.argsort(step_idx[rows])]
        if rows.size and np.all(np.diff(step_idx[rows]) == 1):
            result[int(episode)] = rows
    return result


def _candidate_start_rows(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    goal_offset_steps: int,
    eval_budget: int,
) -> List[Tuple[int, int]]:
    candidates = []
    for episode, rows in _episode_rows(episode_idx, step_idx).items():
        if rows.size <= goal_offset_steps + eval_budget:
            continue
        for local_start in range(0, rows.size - goal_offset_steps - eval_budget):
            candidates.append((int(episode), int(rows[local_start])))
    return candidates


def _success_from_info(done: bool, info: object, threshold: float) -> float:
    if isinstance(info, dict):
        value = float(info.get("success", info.get("is_success", 0.0)))
    else:
        value = 0.0
    if done and value <= 0.0:
        value = 1.0
    return float(value >= threshold)


def _pad_paths(paths: List[np.ndarray], pad_value: float = np.nan) -> np.ndarray:
    max_len = max(path.shape[0] for path in paths)
    shape = (len(paths), max_len) + paths[0].shape[1:]
    out = np.full(shape, pad_value, dtype=paths[0].dtype)
    for idx, path in enumerate(paths):
        out[idx, : path.shape[0]] = path
    return out


def _rollout_policy(
    policy,
    env,
    h5: h5py.File,
    pixels_key: str,
    action_key: Optional[str],
    start_state: np.ndarray,
    goal_state: np.ndarray,
    goal_pixels: np.ndarray,
    start_row: int,
    eval_budget: int,
    reset_state_tol: float,
    success_threshold: float,
) -> Dict[str, object]:
    obs, _info = _reset_env_to_state(env, start_state, goal_state, reset_state_tol)
    action_dim = _set_policy_env(policy, env)
    previous_action = _initial_previous_action(h5, action_key, start_row, action_dim)

    pixels = [_extract_pixels_from_obs(env, obs)]
    states = [_get_env_state(env, obs=obs)]
    actions = []
    rewards = []
    final_info = {}
    done = False
    for _step in range(eval_budget):
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
    terminal_cost = float(task_cost(terminal_state, goal_state))
    start_cost = float(task_cost(start_state, goal_state))
    return {
        "pixels": np.asarray(pixels),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "length": int(len(pixels)),
        "executed_steps": int(len(actions)),
        "done": bool(done),
        "success": _success_from_info(done, final_info, success_threshold),
        "start_cost": start_cost,
        "terminal_cost": terminal_cost,
        "progress": float(start_cost - terminal_cost),
        "terminal_state": terminal_state.astype(np.float32),
    }


def collect(args: argparse.Namespace) -> None:
    print(
        "[paired_goal] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[paired_goal] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    wrong_policy = _load_stable_worldmodel_policy(args.wrong_policy, Path(args.eval_config))
    good_policy = _load_stable_worldmodel_policy(args.good_policy, Path(args.eval_config))
    env = _make_env(args.env_name)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    wrong_pixels, good_pixels = [], []
    wrong_states, good_states = [], []
    wrong_actions, good_actions = [], []
    start_rows, goal_rows, episodes = [], [], []
    rows = []

    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        state_key = _find_key(h5, [args.state_key, "state"])
        try:
            action_key: Optional[str] = _find_key(h5, [args.action_key, "action"])
        except KeyError:
            action_key = None
            print("[paired_goal] no action key found; using zero previous-action history.", flush=True)
        episode_key = _find_key(h5, [args.episode_key, "episode_idx", "ep_idx"])
        step_key = _find_key(h5, [args.step_key, "step_idx"])
        episode_idx = np.asarray(h5[episode_key]).reshape(-1)
        step_idx = np.asarray(h5[step_key]).reshape(-1)
        starts = _candidate_start_rows(episode_idx, step_idx, args.goal_offset_steps, args.eval_budget)
        rng.shuffle(starts)
        print(f"[paired_goal] candidate starts: {len(starts)}", flush=True)

        for attempt_idx, (episode, start_row) in enumerate(starts):
            if len(rows) >= args.num_pairs:
                break
            goal_row = start_row + args.goal_offset_steps
            start_state = np.asarray(h5[state_key][start_row], dtype=np.float64)
            goal_state = np.asarray(h5[state_key][goal_row], dtype=np.float64)
            goal_pixels = _read_h5_rows(h5[pixels_key], np.asarray([goal_row]))[0]

            wrong = _rollout_policy(
                wrong_policy,
                env,
                h5,
                pixels_key,
                action_key,
                start_state,
                goal_state,
                goal_pixels,
                start_row,
                args.eval_budget,
                args.reset_state_tol,
                args.success_threshold,
            )
            good = _rollout_policy(
                good_policy,
                env,
                h5,
                pixels_key,
                action_key,
                start_state,
                goal_state,
                goal_pixels,
                start_row,
                args.eval_budget,
                args.reset_state_tol,
                args.success_threshold,
            )

            wrong_is_bad = wrong["terminal_cost"] >= args.wrong_min_terminal_cost and wrong["success"] < 1.0
            good_is_good = good["terminal_cost"] <= args.good_max_terminal_cost or good["success"] >= 1.0
            if wrong_is_bad and good_is_good:
                pair_idx = len(rows)
                wrong_pixels.append(wrong["pixels"])
                good_pixels.append(good["pixels"])
                wrong_states.append(wrong["states"])
                good_states.append(good["states"])
                wrong_actions.append(wrong["actions"])
                good_actions.append(good["actions"])
                start_rows.append(start_row)
                goal_rows.append(goal_row)
                episodes.append(episode)
                rows.append(
                    {
                        "pair_idx": pair_idx,
                        "attempt_idx": int(attempt_idx),
                        "episode": int(episode),
                        "start_row": int(start_row),
                        "goal_row": int(goal_row),
                        "start_cost": float(wrong["start_cost"]),
                        "wrong_terminal_cost": float(wrong["terminal_cost"]),
                        "good_terminal_cost": float(good["terminal_cost"]),
                        "terminal_cost_gap_wrong_minus_good": float(wrong["terminal_cost"] - good["terminal_cost"]),
                        "wrong_progress": float(wrong["progress"]),
                        "good_progress": float(good["progress"]),
                        "progress_gap_good_minus_wrong": float(good["progress"] - wrong["progress"]),
                        "wrong_success": float(wrong["success"]),
                        "good_success": float(good["success"]),
                        "wrong_executed_steps": int(wrong["executed_steps"]),
                        "good_executed_steps": int(good["executed_steps"]),
                    }
                )
                print(
                    f"[paired_goal] collected {len(rows)}/{args.num_pairs} at attempt {attempt_idx}: "
                    f"wrong_cost={wrong['terminal_cost']:.4f}, good_cost={good['terminal_cost']:.4f}",
                    flush=True,
                )
            elif attempt_idx % args.progress_every == 0:
                print(
                    f"[paired_goal] attempt {attempt_idx}, pairs={len(rows)}, "
                    f"wrong_cost={wrong['terminal_cost']:.4f}, good_cost={good['terminal_cost']:.4f}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No paired wrong/good planner cases collected. Relax cost thresholds or increase eval_budget.")

    np.savez_compressed(
        output,
        wrong_pixels=_pad_paths(wrong_pixels, 0),
        good_pixels=_pad_paths(good_pixels, 0),
        wrong_states=_pad_paths(wrong_states),
        good_states=_pad_paths(good_states),
        wrong_actions=_pad_paths(wrong_actions, 0.0),
        good_actions=_pad_paths(good_actions, 0.0),
        wrong_lengths=np.asarray([path.shape[0] for path in wrong_pixels], dtype=np.int64),
        good_lengths=np.asarray([path.shape[0] for path in good_pixels], dtype=np.int64),
        start_rows=np.asarray(start_rows, dtype=np.int64),
        goal_rows=np.asarray(goal_rows, dtype=np.int64),
        episodes=np.asarray(episodes, dtype=np.int64),
        summary_rows_json=np.asarray(json.dumps(rows)),
        wrong_policy=np.asarray(args.wrong_policy),
        good_policy=np.asarray(args.good_policy),
        goal_offset_steps=np.asarray(args.goal_offset_steps),
        eval_budget=np.asarray(args.eval_budget),
    )
    _write_csv(output.with_suffix(".summary.csv"), rows)
    _write_json(output.with_suffix(".meta.json"), {"args": vars(args), "num_pairs": len(rows)})
    print(f"[paired_goal] wrote {output}", flush=True)


def _squared_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum((a - b) ** 2, axis=-1)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + EPS
    return numerator / denominator


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def _latent_dim(model_name: str, z: np.ndarray) -> int:
    return MODEL_DIMS.get(model_name, int(z.shape[-1]))


def _last_valid(pixels: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.stack([pixels[idx, int(length) - 1] for idx, length in enumerate(lengths)], axis=0)


def _at_raw_step(paths: np.ndarray, lengths: np.ndarray, raw_step: int) -> np.ndarray:
    values = []
    for idx, length in enumerate(lengths):
        step = min(int(raw_step), int(length) - 1)
        values.append(paths[idx, step])
    return np.stack(values, axis=0)


def _state_cost_at_raw_step(states: np.ndarray, lengths: np.ndarray, goal_states: np.ndarray, raw_step: int) -> np.ndarray:
    costs = []
    for idx, length in enumerate(lengths):
        step = min(int(raw_step), int(length) - 1)
        costs.append(task_cost(states[idx, step], goal_states[idx]))
    return np.asarray(costs, dtype=np.float64)


def _rollout_predicted_terminals(
    model,
    start_pixels: np.ndarray,
    actions_raw: np.ndarray,
    action_lengths: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    prediction_raw_steps: int,
    frameskip: int,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if prediction_raw_steps % frameskip != 0:
        raise ValueError(f"--prediction_raw_steps={prediction_raw_steps} must be divisible by --frameskip={frameskip}.")
    num_chunks = prediction_raw_steps // frameskip
    action_dim = int(action_mean.reshape(-1).shape[0])
    flat_dim = frameskip * action_dim
    valid = np.asarray(action_lengths, dtype=np.int64) >= prediction_raw_steps
    with torch.no_grad():
        output_dim = int(model.encode({"pixels": _preprocess_pixels(start_pixels[:1, None], img_size, device)})["emb"].shape[-1])
    pred = np.full((start_pixels.shape[0], output_dim), np.nan, dtype=np.float32)
    if not np.any(valid):
        return pred, valid

    valid_indices = np.where(valid)[0]
    chunks = []
    for idx in valid_indices:
        raw = actions_raw[idx, :prediction_raw_steps].astype(np.float32)
        norm = (raw - action_mean.reshape(1, -1)) / action_std.reshape(1, -1)
        chunks.append(norm.reshape(num_chunks, flat_dim))
    action_sequence = np.stack(chunks, axis=0).astype(np.float32)
    with torch.no_grad():
        for start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[start:start + batch_size]
            pixels = _preprocess_pixels(start_pixels[batch_indices, None], img_size, device)
            actions = torch.from_numpy(action_sequence[start:start + batch_size]).float().to(device)
            rollout = model.rollout({"pixels": pixels.unsqueeze(0)}, actions.unsqueeze(0), history_size=3)
            terminal = rollout["predicted_emb"][0, :, -1, :]
            pred[batch_indices] = terminal.detach().float().cpu().numpy()
    return pred, valid


def _plot(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[paired_goal] matplotlib unavailable; skipping plots.", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in rows}, key=lambda m: min(int(r["latent_dim"]) for r in rows if r["model"] == m))
    labels = [f"{model}\nD={min(int(r['latent_dim']) for r in rows if r['model'] == model)}" for model in models]
    x = np.arange(len(models), dtype=np.float64)
    width = 0.34

    def values(model: str, key: str, traj: str) -> List[float]:
        return [float(row[key]) for row in rows if row["model"] == model and row["trajectory"] == traj]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), facecolor="white")
    wrong_means = [np.nanmean(values(model, "latent_cost_to_goal", "wrong")) for model in models]
    good_means = [np.nanmean(values(model, "latent_cost_to_goal", "good")) for model in models]
    axes[0].bar(x - width / 2, good_means, width=width, label="baseline192 good")
    axes[0].bar(x + width / 2, wrong_means, width=width, label="state8 wrong")
    axes[0].set_title("A. Encoded terminal cost to goal")
    axes[0].set_ylabel("||z_terminal - z_goal||^2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].legend(frameon=False)

    ratio_values = [np.nanmean(values(model, "log_ratio_wrong_over_good", "wrong")) for model in models]
    axes[1].bar(x, ratio_values, width=0.55)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("B. Wrong/good log ratio")
    axes[1].set_ylabel("log(cost_wrong / cost_good)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")

    cos_gap_values = [np.nanmean(values(model, "cos_gap_wrong_minus_good", "wrong")) for model in models]
    axes[2].bar(x, cos_gap_values, width=0.55)
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("C. Goal-direction angle gap")
    axes[2].set_ylabel("cos_wrong - cos_good")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_paired_false_goal_neighborhood.png", dpi=260)
    fig.savefig(output_dir / "fig_paired_false_goal_neighborhood.pdf")
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models = _parse_name_paths(args.models)
    pairs = np.load(args.pairs_npz, allow_pickle=False)
    wrong_pixels = np.asarray(pairs["wrong_pixels"])
    good_pixels = np.asarray(pairs["good_pixels"])
    wrong_states = np.asarray(pairs["wrong_states"])
    good_states = np.asarray(pairs["good_states"])
    wrong_actions = np.asarray(pairs["wrong_actions"])
    good_actions = np.asarray(pairs["good_actions"])
    wrong_lengths = np.asarray(pairs["wrong_lengths"], dtype=np.int64)
    good_lengths = np.asarray(pairs["good_lengths"], dtype=np.int64)
    wrong_action_lengths = np.maximum(wrong_lengths - 1, 0)
    good_action_lengths = np.maximum(good_lengths - 1, 0)
    goal_rows = np.asarray(pairs["goal_rows"], dtype=np.int64)
    start_rows = np.asarray(pairs["start_rows"], dtype=np.int64)
    summary_rows = json.loads(str(np.asarray(pairs["summary_rows_json"]).item()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrong_terminal_pixels = _last_valid(wrong_pixels, wrong_lengths)
    good_terminal_pixels = _last_valid(good_pixels, good_lengths)
    wrong_step_pixels = _at_raw_step(wrong_pixels, wrong_lengths, args.prediction_raw_steps)
    good_step_pixels = _at_raw_step(good_pixels, good_lengths, args.prediction_raw_steps)

    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        state_key = _find_key(h5, [args.state_key, "state"])
        action_key = _find_key(h5, [args.action_key, "action"])
        start_pixels = _read_h5_rows(h5[pixels_key], start_rows)
        goal_pixels = _read_h5_rows(h5[pixels_key], goal_rows)
        goal_states = _read_h5_rows(h5[state_key], goal_rows).astype(np.float64)
        actions_dataset = np.asarray(h5[action_key]).astype(np.float32)
        action_mean, action_std = _action_normalization(actions_dataset)
    true_wrong_cost_at_pred_step = _state_cost_at_raw_step(wrong_states, wrong_lengths, goal_states, args.prediction_raw_steps)
    true_good_cost_at_pred_step = _state_cost_at_raw_step(good_states, good_lengths, goal_states, args.prediction_raw_steps)

    rows: List[Dict[str, object]] = []
    summary_by_model: List[Dict[str, object]] = []
    for model_name, checkpoint in models.items():
        print(f"[paired_goal] encoding terminals in {model_name} space", flush=True)
        model = _load_model(checkpoint, device)
        z_start = _encode_pixels(model, start_pixels[:, None], args.img_size, args.batch_size, device)[:, 0]
        z_goal = _encode_pixels(model, goal_pixels[:, None], args.img_size, args.batch_size, device)[:, 0]
        z_wrong = _encode_pixels(model, wrong_terminal_pixels[:, None], args.img_size, args.batch_size, device)[:, 0]
        z_good = _encode_pixels(model, good_terminal_pixels[:, None], args.img_size, args.batch_size, device)[:, 0]
        z_wrong_step = _encode_pixels(model, wrong_step_pixels[:, None], args.img_size, args.batch_size, device)[:, 0]
        z_good_step = _encode_pixels(model, good_step_pixels[:, None], args.img_size, args.batch_size, device)[:, 0]
        z_pred_wrong, pred_wrong_valid = _rollout_predicted_terminals(
            model,
            start_pixels,
            wrong_actions,
            wrong_action_lengths,
            action_mean,
            action_std,
            args.prediction_raw_steps,
            args.frameskip,
            args.img_size,
            args.batch_size,
            device,
        )
        z_pred_good, pred_good_valid = _rollout_predicted_terminals(
            model,
            start_pixels,
            good_actions,
            good_action_lengths,
            action_mean,
            action_std,
            args.prediction_raw_steps,
            args.frameskip,
            args.img_size,
            args.batch_size,
            device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        q = z_goal - z_start
        d_wrong = z_wrong - z_start
        d_good = z_good - z_start
        cost_wrong = _squared_distance(z_wrong, z_goal)
        cost_good = _squared_distance(z_good, z_goal)
        log_ratio = np.log((cost_wrong + EPS) / (cost_good + EPS))
        cos_wrong = _cosine(d_wrong, q)
        cos_good = _cosine(d_good, q)
        cos_gap = cos_wrong - cos_good
        real_step_cost_wrong = _squared_distance(z_wrong_step, z_goal)
        real_step_cost_good = _squared_distance(z_good_step, z_goal)
        pred_cost_wrong = _squared_distance(z_pred_wrong, z_goal)
        pred_cost_good = _squared_distance(z_pred_good, z_goal)
        pred_log_ratio = np.log((pred_cost_wrong + EPS) / (pred_cost_good + EPS))
        pred_real_error_wrong = _squared_distance(z_pred_wrong, z_wrong_step)
        pred_real_error_good = _squared_distance(z_pred_good, z_good_step)
        pred_cos_wrong = _cosine(z_pred_wrong - z_start, q)
        pred_cos_good = _cosine(z_pred_good - z_start, q)
        pred_cos_gap = pred_cos_wrong - pred_cos_good
        latent_dim = _latent_dim(model_name, z_goal)

        for idx, base in enumerate(summary_rows):
            common = {
                "pair_idx": int(idx),
                "model": model_name,
                "latent_dim": int(latent_dim),
                "true_wrong_terminal_cost": float(base["wrong_terminal_cost"]),
                "true_good_terminal_cost": float(base["good_terminal_cost"]),
                "true_terminal_cost_gap_wrong_minus_good": float(base["terminal_cost_gap_wrong_minus_good"]),
                "wrong_executed_steps": int(base["wrong_executed_steps"]),
                "good_executed_steps": int(base["good_executed_steps"]),
                "prediction_raw_steps": int(args.prediction_raw_steps),
                "pred_wrong_valid": bool(pred_wrong_valid[idx]),
                "pred_good_valid": bool(pred_good_valid[idx]),
                "true_wrong_cost_at_prediction_step": float(true_wrong_cost_at_pred_step[idx]),
                "true_good_cost_at_prediction_step": float(true_good_cost_at_pred_step[idx]),
                "latent_cost_wrong": float(cost_wrong[idx]),
                "latent_cost_good": float(cost_good[idx]),
                "latent_gap_wrong_minus_good": float(cost_wrong[idx] - cost_good[idx]),
                "log_ratio_wrong_over_good": float(log_ratio[idx]),
                "cos_wrong": float(cos_wrong[idx]),
                "cos_good": float(cos_good[idx]),
                "cos_gap_wrong_minus_good": float(cos_gap[idx]),
                "real_step_latent_cost_wrong": float(real_step_cost_wrong[idx]),
                "real_step_latent_cost_good": float(real_step_cost_good[idx]),
                "pred_latent_cost_wrong": float(pred_cost_wrong[idx]),
                "pred_latent_cost_good": float(pred_cost_good[idx]),
                "pred_latent_gap_wrong_minus_good": float(pred_cost_wrong[idx] - pred_cost_good[idx]),
                "pred_log_ratio_wrong_over_good": float(pred_log_ratio[idx]),
                "pred_vs_real_error_wrong": float(pred_real_error_wrong[idx]),
                "pred_vs_real_error_good": float(pred_real_error_good[idx]),
                "pred_vs_real_error_ratio_wrong": float(pred_real_error_wrong[idx] / (real_step_cost_wrong[idx] + EPS)),
                "pred_vs_real_error_ratio_good": float(pred_real_error_good[idx] / (real_step_cost_good[idx] + EPS)),
                "pred_goal_dist_ratio_wrong": float(pred_cost_wrong[idx] / (real_step_cost_wrong[idx] + EPS)),
                "pred_goal_dist_ratio_good": float(pred_cost_good[idx] / (real_step_cost_good[idx] + EPS)),
                "pred_goal_dist_bias_wrong": float(pred_cost_wrong[idx] - real_step_cost_wrong[idx]),
                "pred_goal_dist_bias_good": float(pred_cost_good[idx] - real_step_cost_good[idx]),
                "pred_cos_wrong": float(pred_cos_wrong[idx]),
                "pred_cos_good": float(pred_cos_good[idx]),
                "pred_cos_gap_wrong_minus_good": float(pred_cos_gap[idx]),
                "wrong_in_goal_neighborhood": int(cost_wrong[idx] <= args.goal_neighborhood_multiplier * (cost_good[idx] + EPS)),
                "pred_wrong_in_goal_neighborhood": int(pred_cost_wrong[idx] <= args.goal_neighborhood_multiplier * (pred_cost_good[idx] + EPS)),
            }
            rows.append({**common, "trajectory": "wrong", "latent_cost_to_goal": float(cost_wrong[idx]), "cos_to_goal_direction": float(cos_wrong[idx])})
            rows.append({**common, "trajectory": "good", "latent_cost_to_goal": float(cost_good[idx]), "cos_to_goal_direction": float(cos_good[idx])})

        summary_by_model.append(
            {
                "model": model_name,
                "latent_dim": int(latent_dim),
                "num_pairs": int(cost_wrong.shape[0]),
                "mean_true_wrong_terminal_cost": float(np.mean([row["wrong_terminal_cost"] for row in summary_rows])),
                "mean_true_good_terminal_cost": float(np.mean([row["good_terminal_cost"] for row in summary_rows])),
                "mean_latent_cost_wrong": float(np.mean(cost_wrong)),
                "mean_latent_cost_good": float(np.mean(cost_good)),
                "mean_latent_gap_wrong_minus_good": float(np.mean(cost_wrong - cost_good)),
                "median_log_ratio_wrong_over_good": float(np.median(log_ratio)),
                "mean_log_ratio_wrong_over_good": float(np.mean(log_ratio)),
                "wrong_closer_than_good_rate": float(np.mean(cost_wrong < cost_good)),
                "wrong_in_goal_neighborhood_rate": float(np.mean(cost_wrong <= args.goal_neighborhood_multiplier * (cost_good + EPS))),
                "mean_cos_wrong": float(np.mean(cos_wrong)),
                "mean_cos_good": float(np.mean(cos_good)),
                "mean_cos_gap_wrong_minus_good": float(np.mean(cos_gap)),
                "prediction_raw_steps": int(args.prediction_raw_steps),
                "pred_valid_wrong_rate": float(np.mean(pred_wrong_valid)),
                "pred_valid_good_rate": float(np.mean(pred_good_valid)),
                "mean_true_wrong_cost_at_prediction_step": float(np.mean(true_wrong_cost_at_pred_step)),
                "mean_true_good_cost_at_prediction_step": float(np.mean(true_good_cost_at_pred_step)),
                "mean_real_step_latent_cost_wrong": float(np.nanmean(real_step_cost_wrong)),
                "mean_real_step_latent_cost_good": float(np.nanmean(real_step_cost_good)),
                "mean_pred_latent_cost_wrong": float(np.nanmean(pred_cost_wrong)),
                "mean_pred_latent_cost_good": float(np.nanmean(pred_cost_good)),
                "mean_pred_latent_gap_wrong_minus_good": float(np.nanmean(pred_cost_wrong - pred_cost_good)),
                "mean_pred_log_ratio_wrong_over_good": float(np.nanmean(pred_log_ratio)),
                "pred_wrong_closer_than_good_rate": float(np.nanmean(pred_cost_wrong < pred_cost_good)),
                "pred_wrong_in_goal_neighborhood_rate": float(np.nanmean(pred_cost_wrong <= args.goal_neighborhood_multiplier * (pred_cost_good + EPS))),
                "mean_pred_vs_real_error_wrong": float(np.nanmean(pred_real_error_wrong)),
                "mean_pred_vs_real_error_good": float(np.nanmean(pred_real_error_good)),
                "mean_real_step_latent_cost_wrong_pred_valid": _masked_mean(real_step_cost_wrong, pred_wrong_valid),
                "mean_real_step_latent_cost_good_pred_valid": _masked_mean(real_step_cost_good, pred_good_valid),
                "mean_pred_vs_real_error_ratio_wrong_pred_valid": _masked_mean(pred_real_error_wrong / (real_step_cost_wrong + EPS), pred_wrong_valid),
                "mean_pred_vs_real_error_ratio_good_pred_valid": _masked_mean(pred_real_error_good / (real_step_cost_good + EPS), pred_good_valid),
                "mean_pred_goal_dist_ratio_wrong_pred_valid": _masked_mean(pred_cost_wrong / (real_step_cost_wrong + EPS), pred_wrong_valid),
                "mean_pred_goal_dist_ratio_good_pred_valid": _masked_mean(pred_cost_good / (real_step_cost_good + EPS), pred_good_valid),
                "mean_pred_goal_dist_bias_wrong_pred_valid": _masked_mean(pred_cost_wrong - real_step_cost_wrong, pred_wrong_valid),
                "mean_pred_goal_dist_bias_good_pred_valid": _masked_mean(pred_cost_good - real_step_cost_good, pred_good_valid),
                "mean_pred_cos_gap_wrong_minus_good": float(np.nanmean(pred_cos_gap)),
            }
        )

    _write_csv(output_dir / "paired_goal_neighborhood_by_pair.csv", rows)
    _write_csv(output_dir / "paired_goal_neighborhood_summary.csv", summary_by_model)
    _write_json(output_dir / "paired_goal_neighborhood_metadata.json", {"args": vars(args), "num_pairs": int(len(summary_rows))})
    _plot(rows, output_dir / "paper_figures")
    print(f"[paired_goal] wrote outputs under {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired state8-fail vs baseline-success goal-neighborhood diagnostic.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect same-start state8-fail and baseline-success planner pairs.")
    collect_parser.add_argument("--wrong_policy", required=True, help="Policy expected to fail, e.g. /path/to/state8.")
    collect_parser.add_argument("--good_policy", required=True, help="Policy expected to succeed, e.g. /path/to/baseline.")
    collect_parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--eval_config", default="config/eval/pusht.yaml")
    collect_parser.add_argument("--env_name", default="swm/PushT-v1")
    collect_parser.add_argument("--num_pairs", type=int, default=30)
    collect_parser.add_argument("--goal_offset_steps", type=int, default=25)
    collect_parser.add_argument("--eval_budget", type=int, default=50)
    collect_parser.add_argument("--success_threshold", type=float, default=0.5)
    collect_parser.add_argument("--wrong_min_terminal_cost", type=float, default=0.10)
    collect_parser.add_argument("--good_max_terminal_cost", type=float, default=0.05)
    collect_parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    collect_parser.add_argument("--pixels_key", default="pixels")
    collect_parser.add_argument("--state_key", default="state")
    collect_parser.add_argument("--action_key", default="action")
    collect_parser.add_argument("--episode_key", default="episode_idx")
    collect_parser.add_argument("--step_key", default="step_idx")
    collect_parser.add_argument("--seed", type=int, default=0)
    collect_parser.add_argument("--progress_every", type=int, default=10)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze encoded terminal goal neighborhoods for collected pairs.")
    analyze_parser.add_argument("--pairs_npz", required=True)
    analyze_parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    analyze_parser.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint_object.ckpt")
    analyze_parser.add_argument("--output_dir", default="results/paired_goal_neighborhood")
    analyze_parser.add_argument("--goal_neighborhood_multiplier", type=float, default=2.0)
    analyze_parser.add_argument("--prediction_raw_steps", type=int, default=25)
    analyze_parser.add_argument("--frameskip", type=int, default=5)
    analyze_parser.add_argument("--pixels_key", default="pixels")
    analyze_parser.add_argument("--state_key", default="state")
    analyze_parser.add_argument("--action_key", default="action")
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
