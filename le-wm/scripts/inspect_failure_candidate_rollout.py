from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aliasing_experiment import (  # noqa: E402
    _action_chunks_from_rows,
    _action_normalization,
    _build_model_action_sequence,
    _expected_action_dim,
    _find_key,
    _get_env_state,
    _load_model,
    _make_env,
    _preprocess_pixels,
    _rank_1_based,
    _read_rows,
    _reset_env_to_state,
    _step_env,
)
from task_cost import task_cost  # noqa: E402


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


def _plot_trace(path_prefix: Path, rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[inspect_failure] matplotlib unavailable; skipping plot.", flush=True)
        return
    future_rows = [row for row in rows if int(row["raw_offset_from_start"]) >= 0]
    if not future_rows:
        return
    models = sorted({str(row["model"]) for row in future_rows})
    labels = ["task_best", "focus_wrong"]
    colors = {"task_best": "#2563eb", "focus_wrong": "#dc2626"}
    linestyles = {"task_best": "-", "focus_wrong": "--"}

    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 3.6), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_rows = [row for row in future_rows if row["model"] == model]
        for label in labels:
            series = sorted(
                [row for row in model_rows if row["candidate_label"] == label],
                key=lambda row: int(row["raw_offset_from_start"]),
            )
            if not series:
                continue
            ax.plot(
                [int(row["raw_offset_from_start"]) for row in series],
                [float(row["latent_cost_to_goal"]) for row in series],
                marker="o",
                linewidth=2.0,
                color=colors[label],
                linestyle=linestyles[label],
                label=label.replace("_", " "),
            )
        ax.set_title(model)
        ax.set_xlabel("raw steps from start")
        ax.set_ylabel("latent cost to goal")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Task-best vs state8-wrong candidate rollout cost", y=1.03)
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _candidate_label(candidate_idx: int, task_best_idx: int, focus_wrong_idx: int) -> str:
    if candidate_idx == task_best_idx:
        return "task_best"
    if candidate_idx == focus_wrong_idx:
        return "focus_wrong"
    return f"candidate_{candidate_idx}"


def _load_raws(raw_paths: Dict[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    raws = {}
    for name, path in raw_paths.items():
        raw = np.load(path, allow_pickle=False)
        raws[name] = {key: np.asarray(raw[key]) for key in raw.files}
    return raws


def _selected_candidates(focus_raw: Dict[str, np.ndarray], window_idx: int) -> Dict[str, int]:
    progress = np.asarray(focus_raw["progress"], dtype=np.float64)[window_idx]
    focus_scores = np.asarray(focus_raw["latent_scores"], dtype=np.float64)[window_idx]
    task_best_idx = int(np.argmax(progress))
    focus_wrong_idx = int(np.argmin(focus_scores))
    return {"task_best": task_best_idx, "focus_wrong": focus_wrong_idx}


def _summary_rows(
    raws: Dict[str, Dict[str, np.ndarray]],
    window_idx: int,
    task_best_idx: int,
    focus_wrong_idx: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    progress = np.asarray(next(iter(raws.values()))["progress"], dtype=np.float64)[window_idx]
    terminal_cost = np.asarray(next(iter(raws.values()))["terminal_cost"], dtype=np.float64)[window_idx]
    for model_name, raw in raws.items():
        scores = np.asarray(raw["latent_scores"], dtype=np.float64)[window_idx]
        model_best_idx = int(np.argmin(scores))
        for candidate_idx in (task_best_idx, focus_wrong_idx, model_best_idx):
            rows.append(
                {
                    "model": model_name,
                    "window_idx": window_idx,
                    "candidate_idx": int(candidate_idx),
                    "candidate_label": _candidate_label(candidate_idx, task_best_idx, focus_wrong_idx),
                    "model_best_idx": model_best_idx,
                    "task_best_idx": task_best_idx,
                    "focus_wrong_idx": focus_wrong_idx,
                    "latent_score": float(scores[candidate_idx]),
                    "rank_by_latent_score": _rank_1_based(scores, candidate_idx, lower_is_better=True),
                    "progress": float(progress[candidate_idx]),
                    "terminal_cost": float(terminal_cost[candidate_idx]),
                    "score_wrong_minus_task": float(scores[focus_wrong_idx] - scores[task_best_idx]),
                    "progress_task_minus_wrong": float(progress[task_best_idx] - progress[focus_wrong_idx]),
                }
            )
    return rows


def _rollout_trace_rows(
    model_name: str,
    checkpoint: Path,
    raw: Dict[str, np.ndarray],
    h5: h5py.File,
    window_idx: int,
    candidate_indices: List[int],
    pixels_key: str,
    action_key: str,
    frameskip: int,
    img_size: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    model = _load_model(checkpoint, device)
    history_size = int(getattr(model.predictor, "pos_embedding").shape[1])
    expected_action_dim = _expected_action_dim(model)
    raw_action_dim = int(np.asarray(h5[action_key]).shape[-1])
    if expected_action_dim is not None and expected_action_dim != frameskip * raw_action_dim:
        raise ValueError(
            f"{model_name}: model expects action chunk dim {expected_action_dim}, "
            f"but frameskip*raw_action_dim={frameskip * raw_action_dim}"
        )

    start_row = int(np.asarray(raw["start_rows"])[window_idx])
    goal_row = int(np.asarray(raw["goal_rows"])[window_idx])
    context_rows = start_row - (history_size - 1) * frameskip + np.arange(history_size) * frameskip
    if context_rows[0] < 0 or context_rows[-1] != start_row:
        raise ValueError(f"{model_name}: invalid reconstructed context_rows={context_rows.tolist()}")

    dataset_actions = np.asarray(h5[action_key]).astype(np.float32)
    action_mean, action_std = _action_normalization(dataset_actions)
    dataset_actions_norm = ((dataset_actions - action_mean) / action_std).astype(np.float32)
    history_chunks_norm = _action_chunks_from_rows(dataset_actions_norm, context_rows[:-1], frameskip)
    future_chunks = np.asarray(raw["candidate_future_actions"], dtype=np.float32)[window_idx, candidate_indices]
    actions_model = _build_model_action_sequence(history_chunks_norm, future_chunks)
    context_pixels = _read_rows(h5[pixels_key], context_rows)
    goal_pixels = _read_rows(h5[pixels_key], np.asarray([goal_row], dtype=np.int64))[0]

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        goal_tensor = _preprocess_pixels(goal_pixels[None, None], img_size, device)
        goal_latent = model.encode({"pixels": goal_tensor})["emb"][:, 0]
        context_batch = np.repeat(context_pixels[None], len(candidate_indices), axis=0)
        pixels = _preprocess_pixels(context_batch, img_size, device)
        actions_all = torch.from_numpy(actions_model).float().to(device)
        rollout = model.rollout({"pixels": pixels.unsqueeze(0)}, actions_all.unsqueeze(0), history_size=history_size)
        predicted = rollout["predicted_emb"][0].detach().float()
        if predicted.dim() != 3:
            raise ValueError(f"{model_name}: expected predicted_emb[0] shape (candidates,steps,D), got {tuple(predicted.shape)}")
        start_latent = model.encode({"pixels": pixels})["emb"][:, -1].detach().float()
        for local_idx, candidate_idx in enumerate(candidate_indices):
            initial_cost = torch.sum((start_latent[local_idx] - goal_latent[0]) ** 2).item()
            rows.append(
                {
                    "model": model_name,
                    "window_idx": window_idx,
                    "candidate_idx": int(candidate_idx),
                    "rollout_step": 0,
                    "model_rollout_index": -1,
                    "raw_offset_from_start": 0,
                    "phase": "start_encoded",
                    "latent_cost_to_goal": float(initial_cost),
                    "delta_cost_from_start": 0.0,
                }
            )
            for step_idx in range(predicted.shape[1]):
                cost = torch.sum((predicted[local_idx, step_idx] - goal_latent[0]) ** 2).item()
                raw_offset = int((step_idx + 1 - history_size) * frameskip)
                rows.append(
                    {
                        "model": model_name,
                        "window_idx": window_idx,
                        "candidate_idx": int(candidate_idx),
                        "rollout_step": int(step_idx + 1),
                        "model_rollout_index": int(step_idx),
                        "raw_offset_from_start": raw_offset,
                        "phase": "future" if raw_offset > 0 else ("current" if raw_offset == 0 else "history"),
                        "latent_cost_to_goal": float(cost),
                        "delta_cost_from_start": float(cost - initial_cost),
                    }
                )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _env_raw_step_trace_rows(
    raw: Dict[str, np.ndarray],
    h5: h5py.File,
    window_idx: int,
    candidate_indices: List[int],
    state_key: str,
    action_key: str,
    env_name: str,
    frameskip: int,
    reset_tol: float,
) -> List[Dict[str, object]]:
    env = _make_env(env_name)
    start_row = int(np.asarray(raw["start_rows"])[window_idx])
    goal_row = int(np.asarray(raw["goal_rows"])[window_idx])
    start_state = np.asarray(h5[state_key][start_row], dtype=np.float64)
    goal_state = np.asarray(h5[state_key][goal_row], dtype=np.float64)
    dataset_actions = np.asarray(h5[action_key]).astype(np.float32)
    action_mean, action_std = _action_normalization(dataset_actions)
    future_chunks_norm = np.asarray(raw["candidate_future_actions"], dtype=np.float32)[window_idx, candidate_indices]
    future_chunks_raw = future_chunks_norm * action_std.reshape(1, 1, 1, -1) + action_mean.reshape(1, 1, 1, -1)

    rows: List[Dict[str, object]] = []
    start_cost = task_cost(start_state, goal_state)
    for local_idx, candidate_idx in enumerate(candidate_indices):
        obs, _info = _reset_env_to_state(env, start_state, goal_state, reset_tol)
        current_state = _get_env_state(env, obs=obs)
        if current_state is None:
            current_state = start_state
        current_state = np.asarray(current_state, dtype=np.float64)[: goal_state.shape[0]]
        rows.append(
            {
                "window_idx": int(window_idx),
                "candidate_idx": int(candidate_idx),
                "raw_step": 0,
                "task_cost": float(task_cost(current_state, goal_state)),
                "progress_from_start": float(start_cost - task_cost(current_state, goal_state)),
                "block_x": float(current_state[2]) if current_state.size > 2 else float("nan"),
                "block_y": float(current_state[3]) if current_state.size > 3 else float("nan"),
                "block_theta": float(current_state[4]) if current_state.size > 4 else float("nan"),
                "goal_block_x": float(goal_state[2]) if goal_state.size > 2 else float("nan"),
                "goal_block_y": float(goal_state[3]) if goal_state.size > 3 else float("nan"),
                "goal_block_theta": float(goal_state[4]) if goal_state.size > 4 else float("nan"),
                "block_pose_l2": float(np.linalg.norm(current_state[2:4] - goal_state[2:4])) if current_state.size > 3 and goal_state.size > 3 else float("nan"),
                "block_angle_error": _wrap_to_pi(float(current_state[4] - goal_state[4])) if current_state.size > 4 and goal_state.size > 4 else float("nan"),
            }
        )
        flat_actions = future_chunks_raw[local_idx].reshape(-1, future_chunks_raw.shape[-1])
        for raw_step, action in enumerate(flat_actions, start=1):
            obs, _reward, done, info = _step_env(env, action)
            current_state = _get_env_state(env, obs=obs, info=info)
            if current_state is None:
                raise RuntimeError("Could not extract env state during raw-step trace.")
            current_state = np.asarray(current_state, dtype=np.float64)[: goal_state.shape[0]]
            cost = task_cost(current_state, goal_state)
            rows.append(
                {
                    "window_idx": int(window_idx),
                    "candidate_idx": int(candidate_idx),
                    "raw_step": int(raw_step),
                    "task_cost": float(cost),
                    "progress_from_start": float(start_cost - cost),
                    "block_x": float(current_state[2]) if current_state.size > 2 else float("nan"),
                    "block_y": float(current_state[3]) if current_state.size > 3 else float("nan"),
                    "block_theta": float(current_state[4]) if current_state.size > 4 else float("nan"),
                    "goal_block_x": float(goal_state[2]) if goal_state.size > 2 else float("nan"),
                    "goal_block_y": float(goal_state[3]) if goal_state.size > 3 else float("nan"),
                    "goal_block_theta": float(goal_state[4]) if goal_state.size > 4 else float("nan"),
                    "block_pose_l2": float(np.linalg.norm(current_state[2:4] - goal_state[2:4])) if current_state.size > 3 and goal_state.size > 3 else float("nan"),
                    "block_angle_error": _wrap_to_pi(float(current_state[4] - goal_state[4])) if current_state.size > 4 and goal_state.size > 4 else float("nan"),
                    "done": int(done),
                }
            )
            if done:
                break
    try:
        env.close()
    except Exception:
        pass
    return rows


def _env_candidate_summary_rows(env_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[int, str], List[Dict[str, object]]] = {}
    for row in env_rows:
        grouped.setdefault((int(row["candidate_idx"]), str(row["candidate_label"])), []).append(row)

    summary: List[Dict[str, object]] = []
    for (candidate_idx, candidate_label), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        rows = sorted(rows, key=lambda row: int(row["raw_step"]))
        start = rows[0]
        terminal = rows[-1]
        executed_steps = int(terminal["raw_step"])
        done = int(any(int(row.get("done", 0)) for row in rows))
        summary.append(
            {
                "window_idx": int(terminal["window_idx"]),
                "candidate_idx": candidate_idx,
                "candidate_label": candidate_label,
                "executed_raw_steps": executed_steps,
                "done": done,
                "start_task_cost": float(start["task_cost"]),
                "terminal_task_cost": float(terminal["task_cost"]),
                "terminal_progress_from_start": float(terminal["progress_from_start"]),
                "start_block_x": float(start["block_x"]),
                "start_block_y": float(start["block_y"]),
                "start_block_theta": float(start["block_theta"]),
                "terminal_block_x": float(terminal["block_x"]),
                "terminal_block_y": float(terminal["block_y"]),
                "terminal_block_theta": float(terminal["block_theta"]),
                "goal_block_x": float(terminal["goal_block_x"]),
                "goal_block_y": float(terminal["goal_block_y"]),
                "goal_block_theta": float(terminal["goal_block_theta"]),
                "terminal_block_pose_l2": float(terminal["block_pose_l2"]),
                "terminal_block_angle_error": float(terminal["block_angle_error"]),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect task-best vs low-dimensional wrong candidate rollout traces.")
    parser.add_argument("--raw_pools", nargs="+", required=True, help="NAME=raw_pool.npz")
    parser.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint_object.ckpt")
    parser.add_argument("--focus_model", default="state8")
    parser.add_argument("--window_idx", type=int, default=23)
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence/failure_candidate_rescoring_state8/window_inspect")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--env_name", default="swm/PushT-v1")
    parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    raw_paths = _parse_name_paths(args.raw_pools)
    model_paths = _parse_name_paths(args.models)
    if args.focus_model not in raw_paths:
        raise KeyError(f"focus model {args.focus_model!r} missing from raw pools")
    raws = _load_raws(raw_paths)
    selected = _selected_candidates(raws[args.focus_model], args.window_idx)
    candidate_indices = [selected["task_best"], selected["focus_wrong"]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_rows(raws, args.window_idx, selected["task_best"], selected["focus_wrong"])
    _write_csv(output_dir / f"window_{args.window_idx}_candidate_score_summary.csv", summary)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    trace_rows: List[Dict[str, object]] = []
    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "observation/pixels"])
        action_key = _find_key(h5, [args.action_key, "actions"])
        state_key = _find_key(h5, [args.state_key, "state"])
        for model_name, checkpoint in model_paths.items():
            if model_name not in raws:
                raise KeyError(f"model {model_name!r} missing from raw pools")
            trace_rows.extend(
                _rollout_trace_rows(
                    model_name,
                    checkpoint,
                    raws[model_name],
                    h5,
                    args.window_idx,
                    candidate_indices,
                    pixels_key,
                    action_key,
                    args.frameskip,
                    args.img_size,
                    device,
                )
            )
        env_rows = _env_raw_step_trace_rows(
            raws[args.focus_model],
            h5,
            args.window_idx,
            candidate_indices,
            state_key,
            action_key,
            args.env_name,
            args.frameskip,
            args.reset_state_tol,
        )
    for row in trace_rows:
        row["candidate_label"] = _candidate_label(int(row["candidate_idx"]), selected["task_best"], selected["focus_wrong"])
    for row in env_rows:
        row["candidate_label"] = _candidate_label(int(row["candidate_idx"]), selected["task_best"], selected["focus_wrong"])
    env_summary = _env_candidate_summary_rows(env_rows)
    _write_csv(output_dir / f"window_{args.window_idx}_rollout_trace.csv", trace_rows)
    _write_csv(output_dir / f"window_{args.window_idx}_env_raw_step_trace.csv", env_rows)
    _write_csv(output_dir / f"window_{args.window_idx}_env_candidate_summary.csv", env_summary)
    _plot_trace(output_dir / f"window_{args.window_idx}_rollout_trace", trace_rows)
    print(f"[inspect_failure] window={args.window_idx} task_best={selected['task_best']} focus_wrong={selected['focus_wrong']}", flush=True)
    print(f"[inspect_failure] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
