from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aliasing_metrics import compute_aliasing_and_ranking_metrics  # noqa: E402
from task_cost import task_cost  # noqa: E402
from aliasing_experiment import (  # noqa: E402
    HDF5PLUGIN_AVAILABLE,
    _action_chunks_from_rows,
    _action_normalization,
    _build_model_action_sequence,
    _expected_action_dim,
    _find_key,
    _generate_expert_injected_candidates,
    _generate_future_action_chunks,
    _load_model,
    _make_env,
    _read_rows,
    _rollout_model_candidates,
    _rollout_real_env,
    _sample_contexts,
    _str_to_bool,
)


def _parse_list(value: str, caster=float):
    if isinstance(value, (list, tuple)):
        return [caster(item) for item in value]
    return [caster(item) for item in str(value).split(",") if str(item).strip()]


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _metadata_path(npz_path: Path) -> Path:
    return npz_path.with_suffix(npz_path.suffix + ".meta.json")


def _aggregate_rows(rows: List[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    aggregate_rows = []
    for key_values, group in grouped.items():
        out = {key: value for key, value in zip(group_keys, key_values)}
        metric_keys = sorted(set().union(*(row.keys() for row in group)) - set(group_keys))
        for metric_key in metric_keys:
            values = []
            for row in group:
                value = row.get(metric_key)
                if isinstance(value, (int, float, np.integer, np.floating, bool)):
                    values.append(float(value))
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            out[f"{metric_key}_mean"] = float(np.nanmean(array))
            out[f"{metric_key}_std"] = float(np.nanstd(array))
            out[f"{metric_key}_stderr"] = float(np.nanstd(array) / np.sqrt(np.sum(~np.isnan(array))))
            out[f"{metric_key}_count"] = int(np.sum(~np.isnan(array)))
        aggregate_rows.append(out)
    return aggregate_rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w") as file:
            file.write("")
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _subset_indices(
    Nmax: int,
    N: int,
    mode: str,
    repeat_idx: int,
    window_idx: int,
    subset_seed: int,
    nested_perm: np.ndarray,
) -> np.ndarray:
    if N > Nmax:
        raise ValueError(f"Requested subset N={N} exceeds Nmax={Nmax}.")
    if mode == "nested":
        return nested_perm[:N].copy()
    if mode == "random":
        seed = subset_seed + 100000 * window_idx + 1000 * N + repeat_idx
        rng = np.random.default_rng(seed)
        return rng.choice(Nmax, size=N, replace=False)
    raise ValueError(f"Unknown subset_mode: {mode}")


def _candidate_pool_labels(labels: List[List[str]], window_idx: int, indices: np.ndarray) -> List[str]:
    if not labels:
        return []
    return [labels[window_idx][int(index)] for index in indices]


def _expert_subset_metrics(
    progress: np.ndarray,
    terminal_cost: np.ndarray,
    latent_scores: np.ndarray,
    original_indices: np.ndarray,
) -> Dict[str, object]:
    expert_locations = np.where(original_indices == 0)[0]
    out = {"expert_in_subset": bool(expert_locations.size > 0)}
    if expert_locations.size == 0:
        out.update(
            {
                "expert_rank_by_progress": float("nan"),
                "expert_rank_by_latent_score": float("nan"),
                "expert_progress_percentile": float("nan"),
                "expert_score_percentile": float("nan"),
            }
        )
        return out
    loc = int(expert_locations[0])
    out["expert_rank_by_progress"] = int(np.where(np.argsort(-progress) == loc)[0][0] + 1)
    out["expert_rank_by_latent_score"] = int(np.where(np.argsort(latent_scores) == loc)[0][0] + 1)
    out["expert_progress_percentile"] = float(np.mean(progress <= progress[loc]))
    out["expert_score_percentile"] = float(np.mean(latent_scores >= latent_scores[loc]))
    return out


def _compute_subset_rows(
    pool: Dict[str, np.ndarray],
    labels: List[List[str]],
    args,
    metadata: Dict[str, object],
) -> List[Dict[str, object]]:
    rows = []
    W, Nmax = pool["progress"].shape[:2]
    D = pool["terminal_latents"].shape[-1]
    pool_sizes = _parse_list(args.candidate_pool_sizes, int)
    gamma_values = _parse_list(args.gamma_values, float)
    rho_values = _parse_list(args.rho_values, float)
    score_tau_values = _parse_list(args.score_tau_values, float)
    score_rho_values = _parse_list(args.score_rho_values, float)
    topk_values = _parse_list(args.topk_values, int)
    subset_modes = _parse_list(args.subset_mode, str)

    for window_idx in range(W):
        nested_rng = np.random.default_rng(args.subset_seed + 100000 * window_idx)
        nested_perm = nested_rng.permutation(Nmax) if args.nested_shuffle else np.arange(Nmax)
        for subset_mode in subset_modes:
            repeats = 1 if subset_mode == "nested" else args.num_subset_repeats
            for N in pool_sizes:
                for repeat_idx in range(repeats):
                    indices = _subset_indices(Nmax, N, subset_mode, repeat_idx, window_idx, args.subset_seed, nested_perm)
                    for gamma in gamma_values:
                        metrics = compute_aliasing_and_ranking_metrics(
                            Z=pool["terminal_latents"][window_idx, indices],
                            z_goal=pool["goal_latents"][window_idx],
                            progress=pool["progress"][window_idx, indices],
                            terminal_cost=pool["terminal_cost"][window_idx, indices],
                            latent_scores=pool["latent_scores"][window_idx, indices],
                            true_metric=args.true_metric,
                            gamma_values=[gamma],
                            rho_values=rho_values,
                            score_tau_values=score_tau_values,
                            score_rho_values=score_rho_values,
                            topk_values=topk_values,
                            effective_dim=D,
                        )
                        row = {
                            "model": args.model_name,
                            "source": "trained",
                            "effective_dim": D,
                            "window_idx": window_idx,
                            "N": int(N),
                            "subset_repeat": int(repeat_idx),
                            "subset_mode": subset_mode,
                            "gamma": float(gamma),
                            "candidate_labels": ";".join(_candidate_pool_labels(labels, window_idx, indices)),
                        }
                        row.update(_expert_subset_metrics(
                            pool["progress"][window_idx, indices],
                            pool["terminal_cost"][window_idx, indices],
                            pool["latent_scores"][window_idx, indices],
                            pool["original_candidate_indices"][window_idx, indices],
                        ))
                        row.update(metrics)
                        rows.append(row)
    return rows


def _load_raw_pool(npz_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    pool = {key: data[key] for key in data.files}
    labels = []
    metadata = {}
    meta_path = _metadata_path(npz_path)
    if meta_path.exists():
        with meta_path.open() as file:
            metadata = json.load(file)
        labels = metadata.get("candidate_labels", [])
    return pool, labels, metadata


def _generate_raw_pool(args):
    checkpoint = Path(args.checkpoint or args.checkpoint_object)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _load_model(checkpoint, device)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    history_size = int(getattr(model.predictor, "pos_embedding").shape[1])
    expected_action_dim = _expected_action_dim(model)
    effective_action_dim = args.frameskip * args.action_dim
    if expected_action_dim is not None and expected_action_dim != effective_action_dim:
        raise ValueError(
            f"Model action encoder expects dim {expected_action_dim}, but frameskip*action_dim={effective_action_dim}."
        )

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "observation/pixels"])
        state_key = _find_key(h5, [args.state_key])
        action_key = _find_key(h5, [args.action_key, "actions"])
        episode_key = _find_key(h5, [args.episode_idx_key, "ep_idx"])
        step_key = _find_key(h5, [args.step_idx_key])
        print("[varyK] dataset keys and shapes:", flush=True)
        for key in h5.keys():
            print(f"  - {key}: {getattr(h5[key], 'shape', None)}", flush=True)
        states = np.asarray(h5[state_key]).astype(np.float64)
        dataset_actions = np.asarray(h5[action_key]).astype(np.float32)
        episode_idx = np.asarray(h5[episode_key]).reshape(-1)
        step_idx = np.asarray(h5[step_key]).reshape(-1)
        action_mean, action_std = _action_normalization(dataset_actions)
        dataset_actions_norm = ((dataset_actions - action_mean) / action_std).astype(np.float32)
        contexts = _sample_contexts(
            episode_idx,
            step_idx,
            history_size,
            args.frameskip,
            args.horizon,
            args.num_windows,
            args.goal_mode,
            args.goal_offset_steps,
            rng,
        )

        env = _make_env(args.env_name)
        arrays = defaultdict(list)
        all_labels = []
        with torch.no_grad():
            pixels_ds = h5[pixels_key]
            for window_idx, (start_row, goal_row, context_rows, goal_source) in enumerate(contexts):
                print(f"[varyK] rolling window {window_idx + 1}/{len(contexts)}", flush=True)
                history_chunk_rows = context_rows[:-1]
                future_chunk_rows = start_row + np.arange(args.horizon) * args.frameskip
                history_chunks_norm = _action_chunks_from_rows(dataset_actions_norm, history_chunk_rows, args.frameskip)
                expert_future_chunks_norm = _action_chunks_from_rows(dataset_actions_norm, future_chunk_rows, args.frameskip)
                if args.candidate_mode == "expert_injected":
                    actions_future_norm, labels = _generate_expert_injected_candidates(
                        rng,
                        args.num_candidates_max,
                        args.horizon,
                        args.frameskip,
                        args.action_dim,
                        args.action_clip,
                        expert_future_chunks_norm,
                        args.expert_small_noise,
                        args.expert_medium_noise,
                        args.expert_large_noise,
                        args.num_expert_small,
                        args.num_expert_medium,
                        args.num_expert_large,
                        args.include_zero_action,
                        args.include_sign_flip,
                        args.include_shuffle,
                        args.action_noise_sigma,
                    )
                else:
                    actions_future_norm, labels = _generate_future_action_chunks(
                        rng,
                        args.candidate_mode,
                        args.num_candidates_max,
                        args.horizon,
                        args.frameskip,
                        args.action_dim,
                        args.action_noise_sigma,
                        args.action_clip,
                        expert_future_chunks_norm,
                        args.cem_var_scale,
                    )
                actions_future_raw = actions_future_norm * action_std.reshape(1, 1, 1, -1) + action_mean.reshape(1, 1, 1, -1)
                actions_model = _build_model_action_sequence(history_chunks_norm, actions_future_norm)
                context_pixels = _read_rows(pixels_ds, context_rows)
                goal_pixels = _read_rows(pixels_ds, np.asarray([goal_row], dtype=np.int64))[0]
                start_state = states[start_row]
                goal_state = states[goal_row]
                terminal_cost, terminal_states, _ = _rollout_real_env(
                    env,
                    start_state,
                    goal_state,
                    actions_future_raw,
                    args.horizon,
                    args.frameskip,
                    args.reset_state_tol,
                )
                terminal_latents, latent_scores, goal_latent = _rollout_model_candidates(
                    model,
                    context_pixels,
                    goal_pixels,
                    actions_model,
                    args.horizon,
                    history_size,
                    args.img_size,
                    device,
                    args.batch_size,
                    debug_compare_get_cost=(window_idx == 0 and args.debug_alignment),
                )
                start_cost = task_cost(start_state, goal_state)
                progress = start_cost - terminal_cost
                arrays["start_rows"].append(int(start_row))
                arrays["goal_rows"].append(int(goal_row))
                arrays["terminal_latents"].append(terminal_latents.astype(np.float32))
                arrays["goal_latents"].append(goal_latent.astype(np.float32))
                arrays["latent_scores"].append(latent_scores.astype(np.float32))
                arrays["progress"].append(progress.astype(np.float32))
                arrays["terminal_cost"].append(terminal_cost.astype(np.float32))
                arrays["terminal_states"].append(terminal_states.astype(np.float32))
                arrays["candidate_future_actions"].append(actions_future_norm.astype(np.float32))
                arrays["original_candidate_indices"].append(np.arange(args.num_candidates_max, dtype=np.int64))
                arrays["start_cost"].append(float(start_cost))
                all_labels.append(labels)
        if hasattr(env, "close"):
            env.close()

    pool = {
        "start_rows": np.asarray(arrays["start_rows"], dtype=np.int64),
        "goal_rows": np.asarray(arrays["goal_rows"], dtype=np.int64),
        "terminal_latents": np.stack(arrays["terminal_latents"], axis=0),
        "goal_latents": np.stack(arrays["goal_latents"], axis=0),
        "latent_scores": np.stack(arrays["latent_scores"], axis=0),
        "progress": np.stack(arrays["progress"], axis=0),
        "terminal_cost": np.stack(arrays["terminal_cost"], axis=0),
        "terminal_states": np.stack(arrays["terminal_states"], axis=0),
        "candidate_future_actions": np.stack(arrays["candidate_future_actions"], axis=0),
        "original_candidate_indices": np.stack(arrays["original_candidate_indices"], axis=0),
        "start_cost": np.asarray(arrays["start_cost"], dtype=np.float32),
    }
    metadata = {
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "latent_dim": int(pool["terminal_latents"].shape[-1]),
        "num_windows": int(pool["progress"].shape[0]),
        "num_candidates_max": int(args.num_candidates_max),
        "horizon": int(args.horizon),
        "goal_mode": args.goal_mode,
        "goal_offset_steps": int(args.goal_offset_steps),
        "candidate_mode": args.candidate_mode,
        "true_metric": args.true_metric,
        "expert_noise": {
            "small": args.expert_small_noise,
            "medium": args.expert_medium_noise,
            "large": args.expert_large_noise,
            "num_small": args.num_expert_small,
            "num_medium": args.num_expert_medium,
            "num_large": args.num_expert_large,
        },
        "reset_state_tol": args.reset_state_tol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "candidate_labels": all_labels,
    }
    return pool, all_labels, metadata


def main():
    parser = argparse.ArgumentParser(description="Vary-K future aliasing experiment from a cached candidate pool.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_object", default=None)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--num_windows", type=int, default=20)
    parser.add_argument("--num_candidates_max", type=int, default=1024)
    parser.add_argument("--candidate_pool_sizes", default="32,64,128,256,512,1024")
    parser.add_argument("--subset_mode", default="random")
    parser.add_argument("--num_subset_repeats", type=int, default=10)
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--nested_shuffle", type=_str_to_bool, default=True)
    parser.add_argument("--save_raw_pool_npz", default=None)
    parser.add_argument("--load_raw_pool_npz", default=None)
    parser.add_argument("--save_subset_metrics_csv", required=True)
    parser.add_argument("--save_subset_metrics_json", default=None)
    parser.add_argument("--save_aggregate_metrics_csv", default=None)
    parser.add_argument("--gamma_values", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--rho_values", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--score_tau_values", default="0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--score_rho_values", default="0.001,0.002,0.005,0.01,0.02")
    parser.add_argument("--topk_values", default="10,30")
    parser.add_argument("--candidate_mode", choices=["gaussian", "cem_initial", "expert_perturb", "expert_injected"], default="expert_injected")
    parser.add_argument("--true_metric", choices=["progress", "terminal_cost"], default="progress")
    parser.add_argument("--goal_mode", choices=["eval_offset", "episode_final"], default="eval_offset")
    parser.add_argument("--goal_offset_steps", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--elite_k", type=int, default=30)
    parser.add_argument("--action_noise_sigma", type=float, default=0.5)
    parser.add_argument("--action_clip", type=float, default=3.0)
    parser.add_argument("--cem_var_scale", type=float, default=1.0)
    parser.add_argument("--expert_small_noise", type=float, default=0.05)
    parser.add_argument("--expert_medium_noise", type=float, default=0.2)
    parser.add_argument("--expert_large_noise", type=float, default=0.5)
    parser.add_argument("--num_expert_small", type=int, default=80)
    parser.add_argument("--num_expert_medium", type=int, default=80)
    parser.add_argument("--num_expert_large", type=int, default=80)
    parser.add_argument("--include_zero_action", type=_str_to_bool, default=True)
    parser.add_argument("--include_sign_flip", type=_str_to_bool, default=True)
    parser.add_argument("--include_shuffle", type=_str_to_bool, default=True)
    parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--env_name", default="swm/PushT-v1")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--episode_idx_key", default="episode_idx")
    parser.add_argument("--step_idx_key", default="step_idx")
    parser.add_argument("--debug_alignment", action="store_true")
    args = parser.parse_args()

    if not args.load_raw_pool_npz and not (args.checkpoint or args.checkpoint_object):
        raise ValueError("Provide --checkpoint/--checkpoint_object unless --load_raw_pool_npz is used.")
    if args.load_raw_pool_npz:
        pool, labels, metadata = _load_raw_pool(Path(args.load_raw_pool_npz))
    else:
        pool, labels, metadata = _generate_raw_pool(args)
        if args.save_raw_pool_npz:
            output_npz = Path(args.save_raw_pool_npz)
            output_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_npz, **pool)
            with _metadata_path(output_npz).open("w") as file:
                json.dump(_jsonify(metadata), file, indent=2)

    rows = _compute_subset_rows(pool, labels, args, metadata)
    _write_csv(Path(args.save_subset_metrics_csv), rows)
    if args.save_subset_metrics_json:
        Path(args.save_subset_metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.save_subset_metrics_json).open("w") as file:
            json.dump(_jsonify({"metadata": metadata, "rows": rows}), file, indent=2)
    aggregate_path = Path(args.save_aggregate_metrics_csv) if args.save_aggregate_metrics_csv else Path(args.save_subset_metrics_csv).with_name(
        Path(args.save_subset_metrics_csv).stem + "_aggregate.csv"
    )
    aggregate_rows = _aggregate_rows(rows, ["model", "effective_dim", "N", "gamma", "subset_mode"])
    _write_csv(aggregate_path, aggregate_rows)
    print(f"[varyK] wrote {len(rows)} subset rows to {args.save_subset_metrics_csv}")
    print(f"[varyK] wrote aggregate rows to {aggregate_path}")


if __name__ == "__main__":
    print(
        "[varyK] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[varyK] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    main()
