from __future__ import annotations

import argparse
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _find_key(h5: h5py.File, candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in h5:
            return key
    raise KeyError(f"None of these keys were found in dataset: {list(candidates)}")


def _read_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    values = np.asarray(dataset[unique_rows])
    return values[inverse]


def _normalize_action_stats(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = actions.reshape(-1, actions.shape[-1]).astype(np.float32)
    valid = ~np.isnan(flat).any(axis=1)
    mean = flat[valid].mean(axis=0, keepdims=True)
    std = flat[valid].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize_state(state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (state - mean) / std


def _state_l2_cost(states: np.ndarray, goal_state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    states_std = _standardize_state(states, mean, std)
    goal_std = _standardize_state(goal_state[None], mean, std)[0]
    diff = states_std - goal_std[None]
    return np.sum(diff * diff, axis=-1)


def _to_chw_float(images: torch.Tensor) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError(f"Expected image batch (N,H,W,C) or (N,C,H,W), got {tuple(images.shape)}")
    if images.shape[1] not in (1, 3, 4) and images.shape[-1] in (1, 3, 4):
        images = images.permute(0, 3, 1, 2)
    if images.shape[1] == 4:
        images = images[:, :3]
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)
    images = images.float()
    if images.max() > 2.0:
        images = images / 255.0
    return images


def _preprocess_pixels(pixels: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(pixels).to(device)
    batch, steps = tensor.shape[:2]
    flat = tensor.reshape(batch * steps, *tensor.shape[2:])
    flat = _to_chw_float(flat)
    if flat.shape[-2:] != (img_size, img_size):
        flat = F.interpolate(flat, size=(img_size, img_size), mode="bilinear", align_corners=False)
    flat = (flat - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return flat.reshape(batch, steps, *flat.shape[1:])


def _load_model(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    model = obj.model if hasattr(obj, "model") else obj
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def _apply_transition_bottleneck(model, pred_raw: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    bottleneck = getattr(model, "transition_bottleneck", None)
    if bottleneck is None:
        return pred_raw
    delta_raw = pred_raw - anchor
    if bottleneck.__class__.__name__ == "StateConditionedTangentBottleneck":
        delta_rec, _, _ = bottleneck(delta_raw, anchor)
    else:
        delta_rec, _ = bottleneck(delta_raw)
    return anchor + delta_rec


def _model_step(model, emb_seq: torch.Tensor, action_seq: torch.Tensor, history_size: int) -> torch.Tensor:
    act_emb = model.action_encoder(action_seq)
    ctx_emb = emb_seq[:, -history_size:]
    ctx_act = act_emb[:, -history_size:]
    pred_raw = model.predict(ctx_emb, ctx_act)[:, -1:]
    return _apply_transition_bottleneck(model, pred_raw, ctx_emb[:, -1:])


def _expected_action_dim(model) -> Optional[int]:
    patch_embed = getattr(getattr(model, "action_encoder", None), "patch_embed", None)
    return int(patch_embed.in_channels) if patch_embed is not None and hasattr(patch_embed, "in_channels") else None


def _rollout_model_candidates(
    model,
    context_pixels: np.ndarray,
    goal_pixels: np.ndarray,
    candidate_model_actions: np.ndarray,
    history_size: int,
    horizon: int,
    img_size: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    goal_tensor = _preprocess_pixels(goal_pixels[None, None], img_size, device)
    goal_latent = model.encode({"pixels": goal_tensor})["emb"][:, 0]
    terminal_latents = []
    latent_costs = []
    for start in range(0, candidate_model_actions.shape[0], batch_size):
        action_batch_np = candidate_model_actions[start:start + batch_size]
        batch = action_batch_np.shape[0]
        context_batch = np.repeat(context_pixels[None], batch, axis=0)
        pixels = _preprocess_pixels(context_batch, img_size, device)
        emb_seq = model.encode({"pixels": pixels})["emb"]
        action_roll = torch.from_numpy(action_batch_np[:, :history_size]).float().to(device)
        actions_all = torch.from_numpy(action_batch_np).float().to(device)
        terminal = None
        for step in range(1, horizon + 1):
            terminal = _model_step(model, emb_seq, action_roll, history_size)
            emb_seq = torch.cat([emb_seq, terminal], dim=1)
            next_action_idx = history_size + step - 1
            if next_action_idx < actions_all.shape[1]:
                action_roll = torch.cat([action_roll, actions_all[:, next_action_idx:next_action_idx + 1]], dim=1)
        if terminal is None:
            raise ValueError("horizon must be positive for model rollout.")
        terminal = terminal[:, 0]
        cost = torch.sum((terminal - goal_latent.expand_as(terminal)) ** 2, dim=-1)
        terminal_latents.append(terminal.detach().float().cpu().numpy())
        latent_costs.append(cost.detach().float().cpu().numpy())
    return np.concatenate(terminal_latents, axis=0), np.concatenate(latent_costs, axis=0), goal_latent[0].detach().float().cpu().numpy()


def _make_env(env_name: str):
    try:
        import stable_worldmodel  # noqa: F401
    except ImportError:
        pass
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError as exc:
            raise ImportError("Neither gymnasium nor gym is installed; cannot run real PushT env rollout.") from exc
    return gym.make(env_name)


def _maybe_unwrapped(env):
    return getattr(env, "unwrapped", env)


def _call_state_setter(env, names: Sequence[str], state: np.ndarray) -> bool:
    targets = [env, _maybe_unwrapped(env)]
    for target in targets:
        for name in names:
            method = getattr(target, name, None)
            if method is None:
                continue
            try:
                method(state)
                return True
            except TypeError:
                try:
                    method(state=state)
                    return True
                except TypeError:
                    continue
    return False


def _extract_state(obj) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in ("state", "proprio"):
            if key in obj:
                value = np.asarray(obj[key], dtype=np.float64).reshape(-1)
                if value.size >= 1:
                    return value
    for attr in ("state", "_state"):
        if hasattr(obj, attr):
            value = np.asarray(getattr(obj, attr), dtype=np.float64).reshape(-1)
            if value.size >= 1:
                return value
    return None


def _get_env_state(env, obs=None, info=None) -> Optional[np.ndarray]:
    for source in (info, obs, env, _maybe_unwrapped(env)):
        state = _extract_state(source)
        if state is not None:
            return state
    for target in (env, _maybe_unwrapped(env)):
        for name in ("get_state", "_get_state"):
            method = getattr(target, name, None)
            if method is not None:
                try:
                    return np.asarray(method(), dtype=np.float64).reshape(-1)
                except TypeError:
                    continue
    return None


def _reset_env_to_state(env, start_state: np.ndarray, goal_state: np.ndarray):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    if not _call_state_setter(env, ("_set_state", "set_state"), start_state):
        raise RuntimeError(
            "Could not reset PushT env to dataset start state. "
            "Tried _set_state/set_state on env and env.unwrapped."
        )
    _call_state_setter(env, ("_set_goal_state", "set_goal_state"), goal_state)
    return obs, info


def _step_env(env, action: np.ndarray):
    out = env.step(action.astype(np.float32))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out
    return obs, reward, done, info


def _rollout_true_env_costs(
    env,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    candidate_raw_actions: np.ndarray,
    horizon: int,
    frameskip: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    terminal_states = []
    for candidate in candidate_raw_actions:
        obs, info = _reset_env_to_state(env, start_state, goal_state)
        terminal_state = _get_env_state(env, obs=obs, info=info)
        for step in range(horizon * frameskip):
            action = candidate.reshape(-1, candidate.shape[-1])[step]
            obs, _, done, info = _step_env(env, action)
            terminal_state = _get_env_state(env, obs=obs, info=info)
            if done:
                break
        if terminal_state is None:
            raise RuntimeError("Could not extract terminal state from PushT env observation/info/env.")
        if terminal_state.size < goal_state.size:
            raise RuntimeError(
                f"Extracted terminal state has dim {terminal_state.size}, but dataset goal state has dim {goal_state.size}."
            )
        terminal_states.append(terminal_state[: goal_state.shape[0]])
    terminal_states = np.stack(terminal_states, axis=0)
    return _state_l2_cost(terminal_states, goal_state, state_mean, state_std), terminal_states


def cost_packing_number(costs: np.ndarray, gamma: float) -> int:
    values = sorted(float(c) for c in costs)
    if len(values) == 0:
        return 0
    count = 1
    last = values[0]
    for value in values[1:]:
        if value - last >= gamma:
            count += 1
            last = value
    return count


def _pairwise_distances(latents: np.ndarray) -> np.ndarray:
    diff = latents[:, None, :] - latents[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    upper = np.triu_indices(latents.shape[0], k=1)
    return dist[upper]


def _distance_stats(values: np.ndarray, prefix: str = "") -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}mean": float("nan"),
            f"{prefix}median": float("nan"),
            f"{prefix}p10": float("nan"),
            f"{prefix}p90": float("nan"),
            f"{prefix}min": float("nan"),
            f"{prefix}max": float("nan"),
        }
    return {
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}p10": float(np.percentile(values, 10)),
        f"{prefix}p90": float(np.percentile(values, 90)),
        f"{prefix}min": float(np.min(values)),
        f"{prefix}max": float(np.max(values)),
    }


def latent_aliasing_metrics(costs: np.ndarray, latents: np.ndarray, gamma: float, eta: float) -> Dict[str, float]:
    cost_diff = np.abs(costs[:, None] - costs[None, :])
    latent_diff = np.linalg.norm(latents[:, None, :] - latents[None, :, :], axis=-1)
    upper = np.triu_indices(len(costs), k=1)
    separated = cost_diff[upper] >= gamma
    separated_distances = latent_diff[upper][separated]
    aliased = separated_distances <= eta
    stats = {
        "num_task_separated_pairs": int(separated.sum()),
        "num_aliased_pairs": int(aliased.sum()),
        "aliasing_rate": float(np.mean(aliased)) if separated_distances.size > 0 else float("nan"),
    }
    stats.update(
        {
            "task_separated_distance_mean": float(np.mean(separated_distances)) if separated_distances.size else float("nan"),
            "task_separated_distance_median": float(np.median(separated_distances)) if separated_distances.size else float("nan"),
            "task_separated_distance_p10": float(np.percentile(separated_distances, 10)) if separated_distances.size else float("nan"),
            "task_separated_distance_p90": float(np.percentile(separated_distances, 90)) if separated_distances.size else float("nan"),
        }
    )
    return stats


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan")
    corr = spearmanr(x, y, nan_policy="omit").correlation
    return float(corr) if corr is not None else float("nan")


def _parse_start_indices(value: Optional[str]) -> Optional[np.ndarray]:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        text = path.read_text()
    else:
        text = value
    parts = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    return np.asarray([int(part) for part in parts], dtype=np.int64)


def _build_episode_lookup(episode_idx: np.ndarray, step_idx: np.ndarray):
    lookup = {}
    for ep in np.unique(episode_idx):
        rows = np.where(episode_idx == ep)[0]
        rows = rows[np.argsort(step_idx[rows])]
        lookup[int(ep)] = rows
    return lookup


def _valid_contexts(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    history_size: int,
    frameskip: int,
    goal_offset_steps: int,
    requested_start_rows: Optional[np.ndarray],
) -> List[Tuple[int, int, np.ndarray]]:
    contexts = []
    episode_rows = _build_episode_lookup(episode_idx, step_idx)
    requested = set(int(row) for row in requested_start_rows) if requested_start_rows is not None else None
    for _, rows in episode_rows.items():
        step_values = step_idx[rows]
        if not np.all(np.diff(step_values) == 1):
            continue
        for pos, current_row in enumerate(rows):
            if requested is not None and int(current_row) not in requested:
                continue
            context_start = pos - (history_size - 1) * frameskip
            goal_pos = pos + goal_offset_steps
            if context_start < 0 or goal_pos >= len(rows):
                continue
            context_positions = context_start + np.arange(history_size) * frameskip
            if context_positions[-1] != pos:
                continue
            contexts.append((int(current_row), int(rows[goal_pos]), rows[context_positions]))
    return contexts


def _generate_candidate_actions(
    rng: np.random.Generator,
    num_candidates: int,
    total_model_steps: int,
    frameskip: int,
    action_dim: int,
    action_std: float,
    action_clip: float,
) -> np.ndarray:
    actions = rng.normal(0.0, action_std, size=(num_candidates, total_model_steps, frameskip, action_dim))
    return np.clip(actions, -action_clip, action_clip).astype(np.float32)


def _context_metrics(
    context_id: int,
    start_row: int,
    goal_row: int,
    true_costs: np.ndarray,
    latent_costs: np.ndarray,
    terminal_latents: np.ndarray,
    goal_latent: np.ndarray,
    gamma_values: Sequence[float],
    eta_values: Sequence[float],
) -> Dict[str, object]:
    true_argmin = int(np.argmin(true_costs))
    latent_argmin = int(np.argmin(latent_costs))
    regret = float(true_costs[latent_argmin] - true_costs[true_argmin])
    top5 = np.argsort(latent_costs)[: min(5, len(latent_costs))]
    top10 = np.argsort(latent_costs)[: min(10, len(latent_costs))]
    norms = np.linalg.norm(terminal_latents, axis=-1)
    goal_norm = float(np.linalg.norm(goal_latent))
    r_max = float(max(norms.max(), goal_norm))
    pairwise = _pairwise_distances(terminal_latents)
    d = terminal_latents.shape[-1]
    metrics: Dict[str, object] = {
        "context_index": int(context_id),
        "start_row": int(start_row),
        "goal_row": int(goal_row),
        "true_cost_min": float(true_costs.min()),
        "true_cost_mean": float(true_costs.mean()),
        "true_cost_std": float(true_costs.std()),
        "latent_cost_min": float(latent_costs.min()),
        "latent_cost_mean": float(latent_costs.mean()),
        "latent_cost_std": float(latent_costs.std()),
        "spearman": _spearman(true_costs, latent_costs),
        "pearson": _pearson(true_costs, latent_costs),
        "regret": regret,
        "selected_true_cost": float(true_costs[latent_argmin]),
        "oracle_true_cost": float(true_costs[true_argmin]),
        "top1_agreement": float(latent_argmin == true_argmin),
        "top5_recall": float(true_argmin in top5),
        "top10_recall": float(true_argmin in top10),
        "K_gamma": {},
        "aliasing": {},
        "packing": {},
        "pairwise_latent_distance": _distance_stats(pairwise),
        "R_max": r_max,
        "R_p95": float(np.percentile(norms, 95)),
        "R_mean": float(norms.mean()),
        "terminal_norm_mean": float(norms.mean()),
        "terminal_norm_p95": float(np.percentile(norms, 95)),
        "goal_norm": goal_norm,
    }
    for gamma in gamma_values:
        gamma_key = str(gamma)
        k_gamma = cost_packing_number(true_costs, gamma)
        metrics["K_gamma"][gamma_key] = int(k_gamma)
        metrics["packing"][gamma_key] = {}
        for eta in eta_values:
            eta_key = str(eta)
            alias_key = f"gamma={gamma},eta={eta}"
            metrics["aliasing"][alias_key] = latent_aliasing_metrics(true_costs, terminal_latents, gamma, eta)
            if eta > 0 and r_max > 0:
                base = 1.0 + 2.0 * r_max / eta
                log_capacity = float(d * np.log(base))
                metrics["packing"][gamma_key][eta_key] = {
                    "packing_capacity_bound": float(np.exp(log_capacity)) if log_capacity < 700 else float("inf"),
                    "packing_capacity_bound_log": log_capacity,
                    "D_lower_bound": float(np.log(max(k_gamma, 1)) / np.log(base)),
                }
            else:
                metrics["packing"][gamma_key][eta_key] = {
                    "packing_capacity_bound": float("nan"),
                    "packing_capacity_bound_log": float("nan"),
                    "D_lower_bound": float("nan"),
                }
    return metrics


def _aggregate(contexts: List[Dict[str, object]], gamma_values: Sequence[float], eta_values: Sequence[float]) -> Dict[str, object]:
    def collect(key: str) -> np.ndarray:
        return np.asarray([ctx[key] for ctx in contexts], dtype=np.float64)

    aggregate: Dict[str, object] = {
        "mean_regret": float(np.nanmean(collect("regret"))),
        "median_regret": float(np.nanmedian(collect("regret"))),
        "p90_regret": float(np.nanpercentile(collect("regret"), 90)),
        "mean_spearman": float(np.nanmean(collect("spearman"))),
        "mean_pearson": float(np.nanmean(collect("pearson"))),
        "top1_agreement": float(np.nanmean(collect("top1_agreement"))),
        "top5_recall": float(np.nanmean(collect("top5_recall"))),
        "top10_recall": float(np.nanmean(collect("top10_recall"))),
        "R_max": {
            "mean": float(np.nanmean(collect("R_max"))),
            "p95": float(np.nanpercentile(collect("R_max"), 95)),
        },
        "K_gamma": {},
        "aliasing": {},
    }
    for gamma in gamma_values:
        values = np.asarray([ctx["K_gamma"][str(gamma)] for ctx in contexts], dtype=np.float64)
        aggregate["K_gamma"][str(gamma)] = {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "p90": float(np.nanpercentile(values, 90)),
        }
        for eta in eta_values:
            key = f"gamma={gamma},eta={eta}"
            values = np.asarray([ctx["aliasing"][key]["aliasing_rate"] for ctx in contexts], dtype=np.float64)
            aggregate["aliasing"][key] = {
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "p90": float(np.nanpercentile(values, 90)),
            }
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Candidate future aliasing diagnostic for PushT world models.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint_object", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_eval_contexts", type=int, default=50)
    parser.add_argument("--num_candidates", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate_mode", choices=["gaussian"], default="gaussian")
    parser.add_argument("--action_std", type=float, default=0.5)
    parser.add_argument("--action_clip", type=float, default=1.0)
    parser.add_argument("--gamma_values", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--eta_values", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--cost_mode", choices=["state_l2"], default="state_l2")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--episode_idx_key", default="episode_idx")
    parser.add_argument("--step_idx_key", default="step_idx")
    parser.add_argument("--start_indices", default=None)
    parser.add_argument("--goal_offset_steps", type=int, default=25)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--env_name", default="swm/PushT-v1")
    args = parser.parse_args()

    if args.num_eval_contexts <= 0 or args.num_candidates <= 0 or args.horizon <= 0:
        raise ValueError("num_eval_contexts, num_candidates, and horizon must be positive.")
    rng = np.random.default_rng(args.seed)
    dataset_path = Path(args.dataset)
    print(
        "[aliasing] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[aliasing] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )

    with h5py.File(dataset_path, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "observation/pixels"])
        state_key = _find_key(h5, [args.state_key])
        action_key = _find_key(h5, [args.action_key, "actions"])
        episode_key = _find_key(h5, [args.episode_idx_key, "ep_idx"])
        step_key = _find_key(h5, [args.step_idx_key])
        print("[aliasing] dataset keys and shapes:", flush=True)
        for key in h5.keys():
            print(f"  - {key}: {getattr(h5[key], 'shape', None)}", flush=True)
        states = np.asarray(h5[state_key]).astype(np.float64)
        actions_dataset = np.asarray(h5[action_key]).astype(np.float32)
        episode_idx = np.asarray(h5[episode_key]).reshape(-1)
        step_idx = np.asarray(h5[step_key]).reshape(-1)
        state_mean = np.nanmean(states, axis=0)
        state_std = np.nanstd(states, axis=0)
        state_std = np.where(state_std < 1e-8, 1.0, state_std)
        action_mean, action_std = _normalize_action_stats(actions_dataset)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _load_model(Path(args.checkpoint_object), device)
    model_class = model.__class__.__name__
    has_bottleneck = getattr(model, "transition_bottleneck", None) is not None
    history_size = int(getattr(model.predictor, "pos_embedding").shape[1])
    expected_action_dim = _expected_action_dim(model)
    effective_action_dim = args.frameskip * args.action_dim
    if expected_action_dim is not None and expected_action_dim != effective_action_dim:
        raise ValueError(
            f"Model action encoder expects dim {expected_action_dim}, but frameskip*action_dim="
            f"{args.frameskip}*{args.action_dim}={effective_action_dim}."
        )
    total_model_action_steps = history_size + args.horizon - 1
    requested_starts = _parse_start_indices(args.start_indices)
    contexts = _valid_contexts(
        episode_idx,
        step_idx,
        history_size,
        args.frameskip,
        args.goal_offset_steps,
        requested_starts,
    )
    if len(contexts) == 0:
        raise ValueError("No valid start contexts found. Check start_indices/history/frameskip/goal_offset_steps.")
    selected_ids = rng.choice(len(contexts), size=min(args.num_eval_contexts, len(contexts)), replace=False)
    selected_contexts = [contexts[int(i)] for i in selected_ids]
    print(
        f"[aliasing] model_class={model_class}, latent_dim=?, has_transition_bottleneck={has_bottleneck}",
        flush=True,
    )
    print(
        f"[aliasing] selected contexts={len(selected_contexts)}, history_size={history_size}, "
        f"total_model_action_steps={total_model_action_steps}",
        flush=True,
    )

    env = _make_env(args.env_name)
    context_metrics = []
    terminal_latent_dim = None
    with h5py.File(dataset_path, "r") as h5, torch.no_grad():
        pixels_ds = h5[pixels_key]
        for context_id, (start_row, goal_row, context_rows) in enumerate(selected_contexts):
            normalized_actions = _generate_candidate_actions(
                rng,
                args.num_candidates,
                total_model_action_steps,
                args.frameskip,
                args.action_dim,
                args.action_std,
                args.action_clip,
            )
            raw_actions = normalized_actions * action_std.reshape(1, 1, 1, -1) + action_mean.reshape(1, 1, 1, -1)
            model_actions = normalized_actions.reshape(args.num_candidates, total_model_action_steps, effective_action_dim)
            context_pixels = _read_rows(pixels_ds, context_rows)
            goal_pixels = _read_rows(pixels_ds, np.asarray([goal_row], dtype=np.int64))[0]
            terminal_latents, latent_costs, goal_latent = _rollout_model_candidates(
                model,
                context_pixels,
                goal_pixels,
                model_actions,
                history_size,
                args.horizon,
                args.img_size,
                device,
                args.batch_size,
            )
            true_costs, _ = _rollout_true_env_costs(
                env,
                states[start_row],
                states[goal_row],
                raw_actions[:, :args.horizon],
                args.horizon,
                args.frameskip,
                state_mean,
                state_std,
            )
            if context_id == 0:
                terminal_latent_dim = int(terminal_latents.shape[-1])
                print(
                    f"[aliasing] terminal_latents.shape={terminal_latents.shape}, "
                    f"goal_latent.shape={goal_latent.shape}, latent_dim={terminal_latents.shape[-1]}",
                    flush=True,
                )
            metrics = _context_metrics(
                context_id,
                start_row,
                goal_row,
                true_costs,
                latent_costs,
                terminal_latents,
                goal_latent,
                args.gamma_values,
                args.eta_values,
            )
            context_metrics.append(metrics)
            print(
                f"[aliasing] context {context_id + 1}/{len(selected_contexts)} "
                f"regret={metrics['regret']:.4f}, spearman={metrics['spearman']:.4f}, "
                f"top1={metrics['top1_agreement']:.0f}",
                flush=True,
            )
    if hasattr(env, "close"):
        env.close()

    aggregate = _aggregate(context_metrics, args.gamma_values, args.eta_values)
    config = vars(args).copy()
    output = {
        "config": config,
        "model": {
            "checkpoint": str(Path(args.checkpoint_object)),
            "latent_dim": terminal_latent_dim,
            "terminal_latent_dim": terminal_latent_dim,
            "has_transition_bottleneck": bool(has_bottleneck),
            "model_class": model_class,
        },
        "aggregate": aggregate,
        "contexts": context_metrics,
        "notes": {
            "candidate_actions": "Gaussian actions are sampled in normalized action space and de-normalized for env stepping.",
            "true_cost": "state_l2 uses terminal env state vs dataset goal state, standardized over dataset state dimensions.",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(_jsonify(output), f, indent=2)

    print("\nAggregate candidate aliasing diagnostics")
    print(
        f"mean_regret={aggregate['mean_regret']:.4f} "
        f"median_regret={aggregate['median_regret']:.4f} "
        f"p90_regret={aggregate['p90_regret']:.4f} "
        f"mean_spearman={aggregate['mean_spearman']:.4f} "
        f"mean_pearson={aggregate['mean_pearson']:.4f} "
        f"top1={aggregate['top1_agreement']:.4f} "
        f"top5={aggregate['top5_recall']:.4f} "
        f"top10={aggregate['top10_recall']:.4f}"
    )
    print("\nK_gamma")
    for gamma, stats in aggregate["K_gamma"].items():
        print(f"gamma={gamma:>5}: mean={stats['mean']:.2f} median={stats['median']:.2f} p90={stats['p90']:.2f}")
    print("\nAliasing rates")
    for key, stats in aggregate["aliasing"].items():
        print(f"{key}: mean={stats['mean']:.4f} median={stats['median']:.4f} p90={stats['p90']:.4f}")
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
