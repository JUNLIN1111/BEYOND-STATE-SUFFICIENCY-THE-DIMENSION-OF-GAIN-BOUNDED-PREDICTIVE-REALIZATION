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
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task_cost import task_cost  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EPS = 1e-8


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


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _rank_1_based(values: np.ndarray, index: int, lower_is_better: bool = True) -> int:
    order = np.argsort(values if lower_is_better else -values)
    return int(np.where(order == index)[0][0] + 1)


def _good_percentile(values: np.ndarray, index: int, lower_is_better: bool) -> float:
    value = values[index]
    if lower_is_better:
        return float(np.mean(values >= value))
    return float(np.mean(values <= value))


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


def _action_normalization(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = actions.reshape(-1, actions.shape[-1]).astype(np.float32)
    valid = ~np.isnan(flat).any(axis=1)
    mean = flat[valid].mean(axis=0, keepdims=True).astype(np.float32)
    std = flat[valid].std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


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


def _expected_action_dim(model) -> Optional[int]:
    patch_embed = getattr(getattr(model, "action_encoder", None), "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "in_channels"):
        return int(patch_embed.in_channels)
    return None


def _rollout_model_candidates(
    model,
    context_pixels: np.ndarray,
    goal_pixels: np.ndarray,
    actions_model: np.ndarray,
    horizon: int,
    history_size: int,
    img_size: int,
    device: torch.device,
    batch_size: int,
    debug_compare_get_cost: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    goal_tensor = _preprocess_pixels(goal_pixels[None, None], img_size, device)
    goal_latent = model.encode({"pixels": goal_tensor})["emb"][:, 0]
    terminal_latents = []
    latent_scores = []
    printed_get_cost_diff = False
    for start in range(0, actions_model.shape[0], batch_size):
        action_batch = actions_model[start:start + batch_size]
        batch = action_batch.shape[0]
        context_batch = np.repeat(context_pixels[None], batch, axis=0)
        pixels = _preprocess_pixels(context_batch, img_size, device)
        actions_all = torch.from_numpy(action_batch).float().to(device)
        rollout_info = {"pixels": pixels.unsqueeze(0)}
        rollout = model.rollout(rollout_info, actions_all.unsqueeze(0), history_size=history_size)
        terminal = rollout["predicted_emb"][0, :, -1, :]
        score = torch.sum((terminal - goal_latent.expand_as(terminal)) ** 2, dim=-1)
        if debug_compare_get_cost and not printed_get_cost_diff:
            goal_batch = goal_tensor.unsqueeze(1).expand(1, batch, -1, -1, -1, -1)
            cost_info = {
                "pixels": pixels.unsqueeze(0),
                "goal": goal_batch,
                "action": actions_all[:, :history_size].unsqueeze(0),
            }
            get_cost_score = model.get_cost(cost_info, actions_all.unsqueeze(0))[0]
            max_diff = torch.max(torch.abs(score.detach() - get_cost_score.detach())).item()
            print(f"[aliasing:debug] max |manual latent score - model.get_cost| = {max_diff:.8f}", flush=True)
            printed_get_cost_diff = True
        terminal_latents.append(terminal.detach().float().cpu().numpy())
        latent_scores.append(score.detach().float().cpu().numpy())
    return (
        np.concatenate(terminal_latents, axis=0),
        np.concatenate(latent_scores, axis=0),
        goal_latent[0].detach().float().cpu().numpy(),
    )


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
    try:
        return gym.make(env_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not instantiate env '{env_name}'. Make sure stable_worldmodel registers this gym env."
        ) from exc


def _unwrapped(env):
    return getattr(env, "unwrapped", env)


def _call_state_setter(
    env,
    names: Sequence[str],
    state: np.ndarray,
    keyword_names: Sequence[str] = ("state",),
) -> bool:
    for target in (_unwrapped(env), env):
        for name in names:
            method = getattr(target, name, None)
            if method is None:
                continue
            for keyword_name in keyword_names:
                try:
                    method(**{keyword_name: state})
                    return True
                except TypeError:
                    continue
            try:
                method(state)
                return True
            except TypeError:
                continue
    return False


def _direct_pusht_state(obj) -> Optional[np.ndarray]:
    if obj is None:
        return None
    agent = getattr(obj, "agent", None)
    block = getattr(obj, "block", None)
    if agent is None or block is None:
        return None
    try:
        agent_pos = np.asarray(agent.position, dtype=np.float64).reshape(-1)
        block_pos = np.asarray(block.position, dtype=np.float64).reshape(-1)
        theta = float(getattr(block, "angle"))
        agent_vel = np.asarray(getattr(agent, "velocity", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, AttributeError):
        return None
    if agent_pos.size < 2 or block_pos.size < 2:
        return None
    if agent_vel.size < 2:
        agent_vel = np.zeros(2, dtype=np.float64)
    return np.asarray(
        [agent_pos[0], agent_pos[1], block_pos[0], block_pos[1], theta, agent_vel[0], agent_vel[1]],
        dtype=np.float64,
    )


def _direct_set_pusht_state(env, state: np.ndarray) -> bool:
    target = _unwrapped(env)
    agent = getattr(target, "agent", None)
    block = getattr(target, "block", None)
    if agent is None or block is None:
        return False
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.size < 5:
        return False
    try:
        agent.position = (float(state[0]), float(state[1]))
        block.position = (float(state[2]), float(state[3]))
        block.angle = float(state[4])
        if hasattr(agent, "velocity"):
            agent.velocity = (float(state[5]), float(state[6])) if state.size >= 7 else (0.0, 0.0)
        if hasattr(block, "velocity"):
            block.velocity = (0.0, 0.0)
        if hasattr(block, "angular_velocity"):
            block.angular_velocity = 0.0
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def _direct_correct_pusht_state(env, expected_state: np.ndarray, tol: float, max_iters: int = 8) -> bool:
    target_state = np.asarray(expected_state, dtype=np.float64).reshape(-1).copy()
    if target_state.size < 5:
        return False
    for _ in range(max_iters):
        if not _direct_set_pusht_state(env, target_state):
            return False
        env_state = _direct_pusht_state(_unwrapped(env))
        if env_state is None:
            env_state = _direct_pusht_state(env)
        if env_state is None:
            return False
        _, pose_diff = _pose_comparable_diff(env_state, expected_state)
        if pose_diff <= tol:
            return True
        correction = expected_state[:5] - env_state[:5]
        correction[4] = (correction[4] + np.pi) % (2.0 * np.pi) - np.pi
        target_state[:5] += correction
    return False


def _extract_state(obj) -> Optional[np.ndarray]:
    if obj is None:
        return None
    direct_state = _direct_pusht_state(obj)
    if direct_state is not None:
        return direct_state
    if isinstance(obj, dict):
        for key in ("state", "proprio"):
            if key in obj:
                value = np.asarray(obj[key], dtype=np.float64).reshape(-1)
                if value.size >= 5:
                    return value
    for attr in ("state", "_state"):
        if hasattr(obj, attr):
            value = np.asarray(getattr(obj, attr), dtype=np.float64).reshape(-1)
            if value.size >= 5:
                return value
    return None


def _get_env_state(env, obs=None, info=None) -> Optional[np.ndarray]:
    for target in (_unwrapped(env), env):
        state = _direct_pusht_state(target)
        if state is not None:
            return state
    for target in (_unwrapped(env), env):
        for name in ("get_state", "_get_state"):
            method = getattr(target, name, None)
            if method is None:
                continue
            try:
                value = np.asarray(method(), dtype=np.float64).reshape(-1)
                if value.size >= 5:
                    return value
            except TypeError:
                continue
    for source in (_unwrapped(env), env, info, obs):
        state = _extract_state(source)
        if state is not None:
            return state
    return None


def _pose_comparable_diff(env_state: np.ndarray, expected_state: np.ndarray) -> Tuple[float, float]:
    compare_dim = min(env_state.size, expected_state.size)
    full_diff = float(np.max(np.abs(env_state[:compare_dim] - expected_state[:compare_dim])))
    pose_dim = min(compare_dim, 5)
    pose_delta = env_state[:pose_dim] - expected_state[:pose_dim]
    if pose_dim >= 5:
        pose_delta[4] = (pose_delta[4] + np.pi) % (2.0 * np.pi) - np.pi
    pose_diff = float(np.max(np.abs(pose_delta))) if pose_delta.size else float("inf")
    return full_diff, pose_diff


def _validate_env_state_after_reset(
    env,
    expected_state: np.ndarray,
    tol: float,
    context: str,
) -> np.ndarray:
    env_state = _get_env_state(env)
    if env_state is None:
        raise RuntimeError(
            f"Could not extract env state after {context}. Refusing to compute true costs without verified reset."
        )
    compare_dim = min(env_state.size, expected_state.size)
    full_diff, pose_diff = _pose_comparable_diff(env_state, expected_state)
    if pose_diff > tol:
        raise RuntimeError(
            f"Env pose mismatch after {context}: pose_max_abs_diff={pose_diff:.8f} > {tol}. "
            f"full_max_abs_diff={full_diff:.8f}. "
            f"dataset_state={np.array2string(expected_state[:compare_dim], precision=4)}, "
            f"env_state={np.array2string(env_state[:compare_dim], precision=4)}. "
            "True-cost rollout would be invalid."
        )
    return env_state[:compare_dim]


def _reset_env_to_state(env, start_state: np.ndarray, goal_state: np.ndarray, reset_tol: float):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    if not _call_state_setter(env, ("_set_state", "set_state"), start_state, ("state",)):
        raise RuntimeError(
            "Could not reset PushT env to dataset start state. Tried _set_state/set_state on env and env.unwrapped."
        )
    if not _call_state_setter(env, ("_set_goal_state", "set_goal_state"), goal_state, ("goal_state", "state")):
        if not getattr(_reset_env_to_state, "_warned_goal_setter", False):
            print("[aliasing] warning: env goal setter not found; continuing with dataset goal state for cost.", flush=True)
            _reset_env_to_state._warned_goal_setter = True
    try:
        _validate_env_state_after_reset(env, start_state, reset_tol, "_set_state")
    except RuntimeError as exc:
        if not _direct_correct_pusht_state(env, start_state, reset_tol):
            raise
        try:
            _validate_env_state_after_reset(env, start_state, reset_tol, "direct PushT body state fallback")
            if not getattr(_reset_env_to_state, "_warned_direct_state", False):
                print(
                    "[aliasing] warning: _set_state did not produce a verified state; "
                    "used direct PushT agent/block body assignment fallback.",
                    flush=True,
                )
                _reset_env_to_state._warned_direct_state = True
        except RuntimeError:
            raise exc
    return obs, info


def _step_env(env, action: np.ndarray):
    out = env.step(action.astype(np.float32))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def _extract_scalar_info(info: Optional[dict]) -> Dict[str, float]:
    if not isinstance(info, dict):
        return {}
    preferred = (
        "success",
        "is_success",
        "coverage",
        "overlap",
        "iou",
        "IoU",
        "reward",
        "distance",
    )
    out = {}
    for key in preferred:
        if key not in info:
            continue
        value = info[key]
        if np.isscalar(value):
            out[str(key)] = float(value)
        else:
            array = np.asarray(value)
            if array.size == 1:
                out[str(key)] = float(array.reshape(-1)[0])
    return out


def _rollout_real_env(
    env,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    actions_raw: np.ndarray,
    horizon: int,
    frameskip: int,
    reset_tol: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    terminal_states = []
    costs = []
    feedback = []
    for candidate in actions_raw[:, :horizon]:
        _reset_env_to_state(env, start_state, goal_state, reset_tol)
        terminal_state = _get_env_state(env)
        flat_actions = candidate.reshape(horizon * frameskip, candidate.shape[-1])
        total_reward = 0.0
        final_reward = float("nan")
        final_info = {}
        for action in flat_actions:
            obs, reward, done, info = _step_env(env, action)
            final_reward = float(reward)
            total_reward += float(reward)
            final_info = _extract_scalar_info(info)
            terminal_state = _get_env_state(env, obs=obs, info=info)
            if done:
                break
        if terminal_state is None:
            raise RuntimeError(
                "Could not extract terminal state after env rollout. "
                "This diagnostic requires real env reset/step and state extraction."
            )
        if terminal_state.size < goal_state.size:
            raise RuntimeError(
                f"Env terminal state dim {terminal_state.size} is smaller than goal state dim {goal_state.size}."
            )
        terminal_state = terminal_state[: goal_state.size]
        terminal_states.append(terminal_state)
        costs.append(task_cost(terminal_state, goal_state))
        feedback.append(
            {
                "total_reward": total_reward,
                "final_reward": final_reward,
                "final_info": final_info,
            }
        )
    return np.asarray(costs, dtype=np.float64), np.stack(terminal_states, axis=0), feedback


def _episode_rows(episode_idx: np.ndarray, step_idx: np.ndarray) -> Dict[int, np.ndarray]:
    result = {}
    for episode in np.unique(episode_idx):
        rows = np.where(episode_idx == episode)[0]
        rows = rows[np.argsort(step_idx[rows])]
        if len(rows) > 0 and np.all(np.diff(step_idx[rows]) == 1):
            result[int(episode)] = rows
    return result


def _sample_contexts(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    history_size: int,
    frameskip: int,
    horizon: int,
    num_windows: int,
    goal_mode: str,
    goal_offset_steps: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int, np.ndarray, str]]:
    contexts = []
    for episode, rows in _episode_rows(episode_idx, step_idx).items():
        final_row = int(rows[-1])
        for pos, start_row in enumerate(rows):
            context_start = pos - (history_size - 1) * frameskip
            if context_start < 0:
                continue
            context_positions = context_start + np.arange(history_size) * frameskip
            if context_positions[-1] != pos:
                continue
            if pos + horizon * frameskip >= len(rows):
                continue
            if goal_mode == "eval_offset":
                goal_pos = pos + goal_offset_steps
                if goal_pos >= len(rows):
                    continue
                goal_row = int(rows[goal_pos])
                goal_source = f"eval_offset_{goal_offset_steps}_steps"
            elif goal_mode == "episode_final":
                goal_row = final_row
                goal_source = "episode_final_state"
            else:
                raise ValueError(f"Unknown goal_mode: {goal_mode}")
            contexts.append((int(start_row), goal_row, rows[context_positions], goal_source))
    if len(contexts) == 0:
        raise ValueError("No valid contexts found with the requested history_size/frameskip.")
    selected = rng.choice(len(contexts), size=min(num_windows, len(contexts)), replace=False)
    return [contexts[int(idx)] for idx in selected]


def _action_chunks_from_rows(
    actions_norm: np.ndarray,
    start_rows: np.ndarray,
    frameskip: int,
) -> np.ndarray:
    chunks = []
    for row in np.asarray(start_rows, dtype=np.int64).reshape(-1):
        end = int(row) + frameskip
        if end > len(actions_norm):
            raise ValueError(f"Action chunk starting at row {row} would exceed action dataset length.")
        chunks.append(actions_norm[int(row):end])
    return np.stack(chunks, axis=0).astype(np.float32)


def _generate_future_action_chunks(
    rng: np.random.Generator,
    candidate_mode: str,
    num_candidates: int,
    horizon: int,
    frameskip: int,
    action_dim: int,
    noise_sigma: float,
    clip: float,
    expert_future_chunks: Optional[np.ndarray],
    cem_var_scale: float,
) -> Tuple[np.ndarray, List[str]]:
    if candidate_mode == "gaussian":
        sigma = noise_sigma
        base = 0.0
        labels = ["gaussian"] * num_candidates
    elif candidate_mode == "cem_initial":
        sigma = cem_var_scale
        base = 0.0
        labels = ["cem_initial"] * num_candidates
    elif candidate_mode == "expert_perturb":
        if expert_future_chunks is None:
            raise ValueError("expert_future_chunks is required for candidate_mode=expert_perturb.")
        sigma = noise_sigma
        base = expert_future_chunks[None]
        labels = ["expert_perturb"] * num_candidates
    else:
        raise ValueError(f"Unknown candidate_mode: {candidate_mode}")
    actions = base + rng.normal(0.0, sigma, size=(num_candidates, horizon, frameskip, action_dim))
    return np.clip(actions, -clip, clip).astype(np.float32), labels


def _generate_expert_injected_candidates(
    rng: np.random.Generator,
    num_candidates: int,
    horizon: int,
    frameskip: int,
    action_dim: int,
    clip: float,
    expert_future_chunks: np.ndarray,
    small_noise: float,
    medium_noise: float,
    large_noise: float,
    num_small: int,
    num_medium: int,
    num_large: int,
    include_zero: bool,
    include_sign_flip: bool,
    include_shuffle: bool,
    random_sigma: float,
) -> Tuple[np.ndarray, List[str]]:
    candidates = [expert_future_chunks.astype(np.float32)]
    labels = ["exact_expert"]

    def add_noisy(count: int, sigma: float, label: str):
        for _ in range(max(0, count)):
            noise = rng.normal(0.0, sigma, size=expert_future_chunks.shape)
            candidates.append((expert_future_chunks + noise).astype(np.float32))
            labels.append(label)

    add_noisy(num_small, small_noise, "small_noise")
    add_noisy(num_medium, medium_noise, "medium_noise")
    add_noisy(num_large, large_noise, "large_noise")

    if include_zero:
        candidates.append(np.zeros_like(expert_future_chunks, dtype=np.float32))
        labels.append("zero")
    if include_sign_flip:
        candidates.append((-expert_future_chunks).astype(np.float32))
        labels.append("sign_flip")
    if include_shuffle:
        shuffled = expert_future_chunks[rng.permutation(expert_future_chunks.shape[0])].copy()
        candidates.append(shuffled.astype(np.float32))
        labels.append("shuffle")

    while len(candidates) < num_candidates:
        random_actions = rng.normal(0.0, random_sigma, size=(horizon, frameskip, action_dim)).astype(np.float32)
        candidates.append(random_actions)
        labels.append("random")

    actions = np.stack(candidates[:num_candidates], axis=0)
    labels = labels[:num_candidates]
    actions = np.clip(actions, -clip, clip).astype(np.float32)
    actions[0] = expert_future_chunks.astype(np.float32)
    return actions, labels


def _build_model_action_sequence(
    history_chunks_norm: np.ndarray,
    future_chunks_norm: np.ndarray,
) -> np.ndarray:
    prefix = np.repeat(history_chunks_norm[None], future_chunks_norm.shape[0], axis=0)
    all_chunks = np.concatenate([prefix, future_chunks_norm], axis=1)
    return all_chunks.reshape(all_chunks.shape[0], all_chunks.shape[1], -1)


def _debug_print_window(
    context_idx: int,
    start_row: int,
    goal_row: int,
    goal_source: str,
    context_rows: np.ndarray,
    states: np.ndarray,
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    goal_pixels: np.ndarray,
    future_chunks_norm: np.ndarray,
    future_chunks_raw: np.ndarray,
    actions_model: np.ndarray,
):
    print("\n[aliasing:debug] window alignment", flush=True)
    print(f"  context_idx={context_idx}", flush=True)
    print(f"  goal_source={goal_source}", flush=True)
    print(f"  start_row={start_row}, goal_row={goal_row}", flush=True)
    print(
        f"  episode_idx[start]={episode_idx[start_row]}, step_idx[start]={step_idx[start_row]}, "
        f"episode_idx[goal]={episode_idx[goal_row]}, step_idx[goal]={step_idx[goal_row]}",
        flush=True,
    )
    print(f"  context_rows={context_rows.tolist()}", flush=True)
    print("  all candidates in this window share the same start_state, context pixels, goal_state, and goal image.", flush=True)
    print(f"  goal image row == goal state row: {goal_row}", flush=True)
    print(f"  goal observation shape={tuple(goal_pixels.shape)}", flush=True)
    print(f"  start_state={np.array2string(states[start_row], precision=4)}", flush=True)
    print(f"  goal_block_pose={np.array2string(states[goal_row][2:5], precision=4)}", flush=True)
    print(f"  candidate0 future normalized chunks shape={future_chunks_norm[0].shape}", flush=True)
    print(np.array2string(future_chunks_norm[0], precision=4, suppress_small=False), flush=True)
    print(f"  candidate0 future raw env chunks shape={future_chunks_raw[0].shape}", flush=True)
    print(np.array2string(future_chunks_raw[0], precision=4, suppress_small=False), flush=True)
    print(f"  candidate0 model action tensor shape={actions_model[0].shape}", flush=True)
    print(np.array2string(actions_model[0], precision=4, suppress_small=False), flush=True)


def _debug_env_reset_check(env, start_state: np.ndarray, goal_state: np.ndarray, reset_tol: float):
    _reset_env_to_state(env, start_state, goal_state, reset_tol)
    env_state = _get_env_state(env)
    compare_dim = min(env_state.size, start_state.size)
    full_diff, pose_diff = _pose_comparable_diff(env_state, start_state)
    print("[aliasing:debug] env reset check", flush=True)
    print(f"  dataset_start_state={np.array2string(start_state, precision=4)}", flush=True)
    print(f"  env_state_after_reset={np.array2string(env_state[:compare_dim], precision=4)}", flush=True)
    print(f"  max_abs_diff={full_diff:.8f}", flush=True)
    print(f"  pose_max_abs_diff_first5_angle_wrapped={pose_diff:.8f}", flush=True)


def _debug_expert_action_rollout_check(
    env,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    dataset_future_state: np.ndarray,
    expert_future_raw_chunks: np.ndarray,
    reset_tol: float,
):
    costs, terminal_states, feedback = _rollout_real_env(
        env,
        start_state,
        goal_state,
        expert_future_raw_chunks[None],
        expert_future_raw_chunks.shape[0],
        expert_future_raw_chunks.shape[1],
        reset_tol,
    )
    terminal = terminal_states[0]
    block_pose_diff = terminal[2:5] - dataset_future_state[2:5]
    block_pose_diff[2] = (block_pose_diff[2] + np.pi) % (2.0 * np.pi) - np.pi
    print("[aliasing:debug] expert action env rollout sanity", flush=True)
    print(f"  env_terminal_block_pose={np.array2string(terminal[2:5], precision=4)}", flush=True)
    print(f"  dataset_future_block_pose={np.array2string(dataset_future_state[2:5], precision=4)}", flush=True)
    print(f"  block_pose_diff=[dx, dy, dtheta_wrapped]={np.array2string(block_pose_diff, precision=6)}", flush=True)
    print(f"  task_cost(env_terminal, goal)={costs[0]:.6f}", flush=True)
    print(f"  env_feedback={feedback[0]}", flush=True)


def _debug_ranking_summary(
    terminal_costs,
    progress,
    true_scores,
    true_metric,
    latent_scores,
    terminal_states,
    goal_state,
    elite_k,
    candidate_labels=None,
    expert_index=None,
    start_cost=None,
    env_feedback=None,
):
    true_order = np.argsort(true_scores)
    progress_order = np.argsort(-progress)
    latent_order = np.argsort(latent_scores)
    argmin_true = int(true_order[0])
    argmin_latent = int(latent_order[0])
    overlap = len(set(true_order[:elite_k].tolist()) & set(latent_order[:elite_k].tolist()))
    print("[aliasing:debug] single-window failure diagnosis", flush=True)
    if start_cost is not None:
        print(f"  start_cost={start_cost:.6f}", flush=True)
    print(f"  true_metric={true_metric}", flush=True)
    print(f"  argmin_true={argmin_true}", flush=True)
    print(f"  argmin_latent={argmin_latent}", flush=True)
    print(f"  terminal_cost[argmin_true]={terminal_costs[argmin_true]:.6f}", flush=True)
    print(f"  terminal_cost[argmin_latent]={terminal_costs[argmin_latent]:.6f}", flush=True)
    print(f"  progress[argmin_true]={progress[argmin_true]:.6f}", flush=True)
    print(f"  progress[argmin_latent]={progress[argmin_latent]:.6f}", flush=True)
    print(f"  ell[argmin_true]={latent_scores[argmin_true]:.6f}", flush=True)
    print(f"  ell[argmin_latent]={latent_scores[argmin_latent]:.6f}", flush=True)
    print(f"  goal_block_pose={np.array2string(goal_state[2:5], precision=4)}", flush=True)
    print(f"  true_best_final_block_pose={np.array2string(terminal_states[argmin_true][2:5], precision=4)}", flush=True)
    print(f"  latent_best_final_block_pose={np.array2string(terminal_states[argmin_latent][2:5], precision=4)}", flush=True)
    print(f"  top10_true_indices={true_order[:10].tolist()}", flush=True)
    print(f"  top10_progress_indices={progress_order[:10].tolist()}", flush=True)
    print(f"  top10_latent_indices={latent_order[:10].tolist()}", flush=True)
    print(f"  top{elite_k}_overlap={overlap}/{elite_k} = {overlap / elite_k:.4f}", flush=True)
    if candidate_labels is not None:
        print(f"  candidate_labels={list(candidate_labels)}", flush=True)
    if expert_index is not None:
        print("  exact expert candidate:", flush=True)
        print(f"    index={expert_index}", flush=True)
        print(f"    final_block_pose={np.array2string(terminal_states[expert_index][2:5], precision=4)}", flush=True)
        print(f"    terminal_cost={terminal_costs[expert_index]:.6f}", flush=True)
        print(f"    progress={progress[expert_index]:.6f}", flush=True)
        print(f"    latent_score={latent_scores[expert_index]:.6f}", flush=True)
        print(f"    rank_by_progress={_rank_1_based(progress, expert_index, lower_is_better=False)}", flush=True)
        print(f"    rank_by_latent_score={_rank_1_based(latent_scores, expert_index, lower_is_better=True)}", flush=True)
    if env_feedback is not None:
        print(f"  true_best_env_feedback={env_feedback[argmin_true]}", flush=True)
        print(f"  latent_best_env_feedback={env_feedback[argmin_latent]}", flush=True)
    print("  top5 true candidates: idx | block_pose | terminal_cost | progress | latent_score", flush=True)
    for idx in true_order[:5]:
        print(
            f"    {int(idx):4d} | {np.array2string(terminal_states[idx][2:5], precision=4)} | "
            f"{terminal_costs[idx]:.6f} | {progress[idx]:.6f} | {latent_scores[idx]:.6f}",
            flush=True,
        )
    print("  top5 latent candidates: idx | block_pose | terminal_cost | progress | latent_score", flush=True)
    for idx in latent_order[:5]:
        print(
            f"    {int(idx):4d} | {np.array2string(terminal_states[idx][2:5], precision=4)} | "
            f"{terminal_costs[idx]:.6f} | {progress[idx]:.6f} | {latent_scores[idx]:.6f}",
            flush=True,
        )


def cost_packing_number(costs: np.ndarray, gamma: float) -> int:
    values = sorted(float(value) for value in costs)
    if len(values) == 0:
        return 0
    count = 1
    last = values[0]
    for value in values[1:]:
        if value - last >= gamma:
            count += 1
            last = value
    return count


def _pairwise_arrays(costs: np.ndarray, latent_scores: np.ndarray, terminal_latents: np.ndarray):
    upper = np.triu_indices(len(costs), k=1)
    real_gap = np.abs(costs[:, None] - costs[None, :])[upper]
    score_gap = np.abs(latent_scores[:, None] - latent_scores[None, :])[upper]
    latent_dist = np.linalg.norm(terminal_latents[:, None, :] - terminal_latents[None, :, :], axis=-1)[upper]
    return real_gap, latent_dist, score_gap


def _quantile_stats(values: np.ndarray, prefix: str = "") -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}mean": float("nan"),
            f"{prefix}median": float("nan"),
            f"{prefix}p10": float("nan"),
            f"{prefix}p90": float("nan"),
        }
    return {
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}p10": float(np.percentile(values, 10)),
        f"{prefix}p90": float(np.percentile(values, 90)),
    }


def _alias_stats(real_gap: np.ndarray, values: np.ndarray, gamma: float, threshold: float, value_name: str):
    mask = real_gap >= gamma
    selected = values[mask]
    aliased = selected <= threshold
    out = {
        "num_task_distinct_pairs": int(mask.sum()),
        "num_aliased_pairs": int(aliased.sum()),
        "alias_rate": float(np.mean(aliased)) if selected.size else float("nan"),
    }
    out.update(_quantile_stats(selected, prefix=f"{value_name}_distinct_"))
    return out


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


def _elite_metrics(
    true_scores: np.ndarray,
    terminal_costs: np.ndarray,
    progress: np.ndarray,
    latent_scores: np.ndarray,
    elite_k: int,
    true_metric: str,
) -> Dict[str, object]:
    if elite_k <= 0 or elite_k >= len(true_scores):
        raise ValueError(f"elite_k must be in [1, num_candidates - 1], got {elite_k} for {len(true_scores)} candidates.")

    latent_order = np.argsort(latent_scores)
    true_order = np.argsort(true_scores)
    latent_elite = latent_order[:elite_k]
    true_elite = true_order[:elite_k]

    latent_elite_mask = np.zeros(len(true_scores), dtype=bool)
    true_elite_mask = np.zeros(len(true_scores), dtype=bool)
    latent_elite_mask[latent_elite] = True
    true_elite_mask[true_elite] = True

    rest = np.where(~latent_elite_mask)[0]
    non_true_elite = np.where(~true_elite_mask)[0]
    elite_costs = terminal_costs[latent_elite]
    rest_costs = terminal_costs[rest]
    elite_progress = progress[latent_elite]
    rest_progress = progress[rest]
    true_elite_scores = latent_scores[true_elite]
    non_true_elite_scores = latent_scores[non_true_elite]

    elite_true_cost_mean = float(np.mean(elite_costs))
    rest_true_cost_mean = float(np.mean(rest_costs))
    elite_improvement = rest_true_cost_mean - elite_true_cost_mean
    elite_pairwise_win_rate = float(np.mean(elite_costs[:, None] < rest_costs[None, :]))
    elite_progress_mean = float(np.mean(elite_progress))
    rest_progress_mean = float(np.mean(rest_progress))
    elite_progress_improvement = elite_progress_mean - rest_progress_mean
    elite_progress_win_rate = float(np.mean(elite_progress[:, None] > rest_progress[None, :]))
    true_elite_latent_score_mean = float(np.mean(true_elite_scores))
    non_true_elite_latent_score_mean = float(np.mean(non_true_elite_scores))

    return {
        "elite_k": int(elite_k),
        "true_metric": true_metric,
        "elite_true_cost_mean": elite_true_cost_mean,
        "rest_true_cost_mean": rest_true_cost_mean,
        "elite_improvement": float(elite_improvement),
        "elite_improvement_norm": float(elite_improvement / (np.std(terminal_costs) + EPS)),
        "elite_true_cost_median": float(np.median(elite_costs)),
        "rest_true_cost_median": float(np.median(rest_costs)),
        "elite_pairwise_win_rate": elite_pairwise_win_rate,
        "elite_progress_mean": elite_progress_mean,
        "rest_progress_mean": rest_progress_mean,
        "elite_progress_improvement": float(elite_progress_improvement),
        "elite_progress_win_rate": elite_progress_win_rate,
        "elite_overlap": float(len(set(latent_elite.tolist()) & set(true_elite.tolist())) / elite_k),
        "true_elite_latent_score_mean": true_elite_latent_score_mean,
        "non_true_elite_latent_score_mean": non_true_elite_latent_score_mean,
        "true_elite_score_improvement": float(non_true_elite_latent_score_mean - true_elite_latent_score_mean),
        "latent_elite_indices": latent_elite.astype(int).tolist(),
        "true_elite_indices": true_elite.astype(int).tolist(),
    }


def _expert_metrics(
    terminal_costs: np.ndarray,
    progress: np.ndarray,
    true_scores: np.ndarray,
    latent_scores: np.ndarray,
    elite_k: int,
    expert_index: Optional[int],
) -> Dict[str, object]:
    if expert_index is None:
        return {}
    latent_order = np.argsort(latent_scores)
    true_order = np.argsort(true_scores)
    return {
        "index": int(expert_index),
        "terminal_cost": float(terminal_costs[expert_index]),
        "progress": float(progress[expert_index]),
        "latent_score": float(latent_scores[expert_index]),
        "rank_by_terminal_cost": _rank_1_based(terminal_costs, expert_index, lower_is_better=True),
        "rank_by_progress": _rank_1_based(progress, expert_index, lower_is_better=False),
        "rank_by_latent_score": _rank_1_based(latent_scores, expert_index, lower_is_better=True),
        "rank_by_primary_true_metric": _rank_1_based(true_scores, expert_index, lower_is_better=True),
        "in_latent_topk": bool(expert_index in set(latent_order[:elite_k].tolist())),
        "in_true_topk": bool(expert_index in set(true_order[:elite_k].tolist())),
        "expert_score_percentile": _good_percentile(latent_scores, expert_index, lower_is_better=True),
        "expert_progress_percentile": _good_percentile(progress, expert_index, lower_is_better=False),
    }


def _good_progress_metrics(progress: np.ndarray, latent_scores: np.ndarray, elite_k: int, threshold: float):
    latent_order = np.argsort(latent_scores)
    elite = latent_order[:elite_k]
    elite_mask = np.zeros(len(progress), dtype=bool)
    elite_mask[elite] = True
    rest = np.where(~elite_mask)[0]
    good = progress >= threshold
    latent_topk_good_frac = float(np.mean(good[elite])) if elite.size else float("nan")
    rest_good_frac = float(np.mean(good[rest])) if rest.size else float("nan")
    return {
        "threshold": float(threshold),
        "num_good": int(np.sum(good)),
        "frac_good": float(np.mean(good)),
        "latent_topk_good_frac": latent_topk_good_frac,
        "rest_good_frac": rest_good_frac,
        "good_enrichment": float(latent_topk_good_frac - rest_good_frac),
    }


def _context_metrics(
    context_idx: int,
    start_row: int,
    goal_row: int,
    goal_source: str,
    terminal_costs: np.ndarray,
    progress: np.ndarray,
    true_scores: np.ndarray,
    true_gap_values: np.ndarray,
    true_metric: str,
    start_cost: float,
    latent_scores: np.ndarray,
    terminal_latents: np.ndarray,
    goal_latent: np.ndarray,
    candidate_labels: Sequence[str],
    expert_index: Optional[int],
    good_progress_threshold: float,
    gamma_values: Sequence[float],
    eta_values: Sequence[float],
    rho_values: Sequence[float],
    tau_values: Sequence[float],
    elite_k: int,
) -> Dict[str, object]:
    real_gap, latent_dist, score_gap = _pairwise_arrays(true_gap_values, latent_scores, terminal_latents)
    terminal_norms = np.linalg.norm(terminal_latents, axis=-1)
    goal_norm = float(np.linalg.norm(goal_latent))
    r_max = float(max(np.max(terminal_norms), goal_norm))
    latent_dist_norm = latent_dist / max(r_max, EPS)
    true_order = np.argsort(true_scores)
    latent_order = np.argsort(latent_scores)
    top10 = min(10, len(terminal_costs))
    regret = float(true_scores[latent_order[0]] - true_scores[true_order[0]])
    context = {
        "context_index": int(context_idx),
        "start_row": int(start_row),
        "goal_row": int(goal_row),
        "goal_source": goal_source,
        "true_metric": true_metric,
        "start_cost": float(start_cost),
        "terminal_cost_min": float(np.min(terminal_costs)),
        "terminal_cost_mean": float(np.mean(terminal_costs)),
        "progress_max": float(np.max(progress)),
        "progress_mean": float(np.mean(progress)),
        "true_cost_min": float(np.min(true_scores)),
        "true_cost_mean": float(np.mean(true_scores)),
        "terminal_costs": terminal_costs.astype(float).tolist(),
        "progress": progress.astype(float).tolist(),
        "true_scores": true_scores.astype(float).tolist(),
        "latent_scores": latent_scores.astype(float).tolist(),
        "candidate_labels": list(candidate_labels),
        "latent_score_min": float(np.min(latent_scores)),
        "latent_score_mean": float(np.mean(latent_scores)),
        "R_max": r_max,
        "R_p95": float(np.percentile(terminal_norms, 95)),
        "R_mean": float(np.mean(terminal_norms)),
        "goal_norm": goal_norm,
        "pairwise_latent_distance": _quantile_stats(latent_dist),
        "pairwise_latent_distance_norm": _quantile_stats(latent_dist_norm),
        "K_gamma": {},
        "alias_geom": {},
        "alias_norm": {},
        "alias_score": {},
        "packing": {},
        "elite": _elite_metrics(true_scores, terminal_costs, progress, latent_scores, elite_k, true_metric),
        "expert": _expert_metrics(terminal_costs, progress, true_scores, latent_scores, elite_k, expert_index),
        "good_progress": _good_progress_metrics(progress, latent_scores, elite_k, good_progress_threshold),
        "auxiliary": {
            "spearman": _spearman(latent_scores, true_scores),
            "pearson": _pearson(latent_scores, true_scores),
            "top10_precision": float(
                len(set(latent_order[:top10].tolist()) & set(true_order[:top10].tolist())) / top10
            ),
            "candidate_regret": regret,
        },
    }
    for gamma in gamma_values:
        gamma_key = str(gamma)
        k_gamma = cost_packing_number(true_gap_values, gamma)
        context["K_gamma"][gamma_key] = int(k_gamma)
        context["packing"][gamma_key] = {}
        for eta in eta_values:
            key = f"gamma={gamma},eta={eta}"
            context["alias_geom"][key] = _alias_stats(real_gap, latent_dist, gamma, eta, "latent_dist")
        for rho in rho_values:
            key = f"gamma={gamma},rho={rho}"
            context["alias_norm"][key] = _alias_stats(real_gap, latent_dist_norm, gamma, rho, "latent_dist_norm")
        for tau in tau_values:
            key = f"gamma={gamma},tau={tau}"
            context["alias_score"][key] = _alias_stats(real_gap, score_gap, gamma, 2.0 * tau, "score_gap")
            if k_gamma > 1 and tau > 0 and r_max > 0:
                context["packing"][gamma_key][str(tau)] = float(
                    np.log(k_gamma) / np.log(1.0 + 4.0 * r_max * r_max / tau)
                )
            else:
                context["packing"][gamma_key][str(tau)] = float("nan")
    return context


def _aggregate(contexts: List[Dict[str, object]], gamma_values, eta_values, rho_values, tau_values, elite_k: int):
    def values(path: Sequence[str]) -> np.ndarray:
        out = []
        for context in contexts:
            item = context
            try:
                for key in path:
                    item = item[key]
                out.append(item)
            except KeyError:
                out.append(float("nan"))
        return np.asarray(out, dtype=np.float64)

    aggregate = {
        "K_gamma": {},
        "alias_geom": {},
        "alias_norm": {},
        "alias_score": {},
        "latent_scale": {
            "R_max_mean": float(np.nanmean(values(["R_max"]))),
            "R_max_p95": float(np.nanpercentile(values(["R_max"]), 95)),
            "pairwise_latent_distance_median": float(np.nanmean(values(["pairwise_latent_distance", "median"]))),
            "pairwise_latent_distance_p10": float(np.nanmean(values(["pairwise_latent_distance", "p10"]))),
            "pairwise_latent_distance_p90": float(np.nanmean(values(["pairwise_latent_distance", "p90"]))),
            "pairwise_latent_distance_norm_median": float(np.nanmean(values(["pairwise_latent_distance_norm", "median"]))),
            "pairwise_latent_distance_norm_p10": float(np.nanmean(values(["pairwise_latent_distance_norm", "p10"]))),
            "pairwise_latent_distance_norm_p90": float(np.nanmean(values(["pairwise_latent_distance_norm", "p90"]))),
        },
        "auxiliary": {
            "mean_spearman": float(np.nanmean(values(["auxiliary", "spearman"]))),
            "mean_pearson": float(np.nanmean(values(["auxiliary", "pearson"]))),
            "top10_precision": float(np.nanmean(values(["auxiliary", "top10_precision"]))),
            "candidate_regret_mean": float(np.nanmean(values(["auxiliary", "candidate_regret"]))),
            "candidate_regret_median": float(np.nanmedian(values(["auxiliary", "candidate_regret"]))),
        },
        "elite": {
            "elite_k": int(elite_k),
            "mean_elite_true_cost_mean": float(np.nanmean(values(["elite", "elite_true_cost_mean"]))),
            "mean_rest_true_cost_mean": float(np.nanmean(values(["elite", "rest_true_cost_mean"]))),
            "mean_elite_improvement": float(np.nanmean(values(["elite", "elite_improvement"]))),
            "median_elite_improvement": float(np.nanmedian(values(["elite", "elite_improvement"]))),
            "p10_elite_improvement": float(np.nanpercentile(values(["elite", "elite_improvement"]), 10)),
            "p90_elite_improvement": float(np.nanpercentile(values(["elite", "elite_improvement"]), 90)),
            "mean_elite_improvement_norm": float(np.nanmean(values(["elite", "elite_improvement_norm"]))),
            "mean_elite_pairwise_win_rate": float(np.nanmean(values(["elite", "elite_pairwise_win_rate"]))),
            "mean_elite_progress_mean": float(np.nanmean(values(["elite", "elite_progress_mean"]))),
            "mean_rest_progress_mean": float(np.nanmean(values(["elite", "rest_progress_mean"]))),
            "mean_elite_progress_improvement": float(np.nanmean(values(["elite", "elite_progress_improvement"]))),
            "mean_elite_progress_win_rate": float(np.nanmean(values(["elite", "elite_progress_win_rate"]))),
            "mean_elite_overlap": float(np.nanmean(values(["elite", "elite_overlap"]))),
            "mean_true_elite_score_improvement": float(
                np.nanmean(values(["elite", "true_elite_score_improvement"]))
            ),
        },
        "expert": {
            "mean_rank_by_progress": float(np.nanmean(values(["expert", "rank_by_progress"]))),
            "mean_rank_by_latent_score": float(np.nanmean(values(["expert", "rank_by_latent_score"]))),
            "expert_in_latent_topk_rate": float(np.nanmean(values(["expert", "in_latent_topk"]))),
            "mean_expert_progress": float(np.nanmean(values(["expert", "progress"]))),
            "mean_expert_latent_score": float(np.nanmean(values(["expert", "latent_score"]))),
            "mean_expert_terminal_cost": float(np.nanmean(values(["expert", "terminal_cost"]))),
            "mean_expert_score_percentile": float(np.nanmean(values(["expert", "expert_score_percentile"]))),
            "mean_expert_progress_percentile": float(np.nanmean(values(["expert", "expert_progress_percentile"]))),
        },
        "good_progress": {
            "mean_num_good": float(np.nanmean(values(["good_progress", "num_good"]))),
            "mean_frac_good": float(np.nanmean(values(["good_progress", "frac_good"]))),
            "mean_latent_topk_good_frac": float(np.nanmean(values(["good_progress", "latent_topk_good_frac"]))),
            "mean_rest_good_frac": float(np.nanmean(values(["good_progress", "rest_good_frac"]))),
            "mean_good_enrichment": float(np.nanmean(values(["good_progress", "good_enrichment"]))),
        },
    }
    for gamma in gamma_values:
        gamma_key = str(gamma)
        k_values = values(["K_gamma", gamma_key])
        aggregate["K_gamma"][gamma_key] = {
            "mean": float(np.nanmean(k_values)),
            "median": float(np.nanmedian(k_values)),
            "p90": float(np.nanpercentile(k_values, 90)),
        }
        for eta in eta_values:
            key = f"gamma={gamma},eta={eta}"
            rates = values(["alias_geom", key, "alias_rate"])
            aggregate["alias_geom"][key] = {
                "mean": float(np.nanmean(rates)),
                "median": float(np.nanmedian(rates)),
                "p90": float(np.nanpercentile(rates, 90)),
            }
        for rho in rho_values:
            key = f"gamma={gamma},rho={rho}"
            rates = values(["alias_norm", key, "alias_rate"])
            aggregate["alias_norm"][key] = {
                "mean": float(np.nanmean(rates)),
                "median": float(np.nanmedian(rates)),
                "p90": float(np.nanpercentile(rates, 90)),
            }
        for tau in tau_values:
            key = f"gamma={gamma},tau={tau}"
            rates = values(["alias_score", key, "alias_rate"])
            aggregate["alias_score"][key] = {
                "mean": float(np.nanmean(rates)),
                "median": float(np.nanmedian(rates)),
                "p90": float(np.nanpercentile(rates, 90)),
            }
    return aggregate


def _print_elite_summary(aggregate: Dict[str, object], num_candidates: int):
    elite = aggregate["elite"]
    expert = aggregate.get("expert", {})
    good = aggregate.get("good_progress", {})
    print("\nMPC-style elite validation")
    print(f"N candidates: {num_candidates}")
    print(f"Elite K: {elite['elite_k']}")
    print(f"elite true cost mean:      {elite['mean_elite_true_cost_mean']:.4f}")
    print(f"rest true cost mean:       {elite['mean_rest_true_cost_mean']:.4f}")
    print(f"elite improvement:         {elite['mean_elite_improvement']:.4f}")
    print(f"elite improvement norm:    {elite['mean_elite_improvement_norm']:.4f}")
    print(f"elite pairwise win rate:   {elite['mean_elite_pairwise_win_rate']:.4f}")
    print(f"elite progress mean:       {elite['mean_elite_progress_mean']:.4f}")
    print(f"rest progress mean:        {elite['mean_rest_progress_mean']:.4f}")
    print(f"elite progress improve:    {elite['mean_elite_progress_improvement']:.4f}")
    print(f"elite progress win rate:   {elite['mean_elite_progress_win_rate']:.4f}")
    print(f"top-{elite['elite_k']} overlap:            {elite['mean_elite_overlap']:.4f}")
    print(f"true elite score improve:  {elite['mean_true_elite_score_improvement']:.4f}")
    if expert:
        print("\nExpert candidate diagnostics")
        print(f"mean rank by progress:     {expert['mean_rank_by_progress']:.2f}")
        print(f"mean rank by latent score: {expert['mean_rank_by_latent_score']:.2f}")
        print(f"in latent top-k rate:      {expert['expert_in_latent_topk_rate']:.4f}")
        print(f"mean expert progress:      {expert['mean_expert_progress']:.4f}")
        print(f"mean expert latent score:  {expert['mean_expert_latent_score']:.4f}")
    if good:
        print("\nGood-progress enrichment")
        print(f"mean num good:             {good['mean_num_good']:.2f}")
        print(f"mean frac good:            {good['mean_frac_good']:.4f}")
        print(f"latent top-k good frac:    {good['mean_latent_topk_good_frac']:.4f}")
        print(f"rest good frac:            {good['mean_rest_good_frac']:.4f}")
        print(f"good enrichment:           {good['mean_good_enrichment']:.4f}")


def _print_matrix(title: str, aggregate: Dict[str, object], row_values, col_values, key_template: str):
    print(f"\n{title}")
    print("gamma \\ threshold | " + " ".join(f"{value:>8}" for value in col_values))
    print("-" * (18 + 9 * len(col_values)))
    for gamma in row_values:
        cells = []
        for value in col_values:
            key = key_template.format(gamma=gamma, value=value)
            cells.append(f"{aggregate[key]['mean']:>8.3f}")
        print(f"{gamma:>16} | " + " ".join(cells))


def main():
    parser = argparse.ArgumentParser(description="Core candidate-future aliasing experiment for PushT.")
    parser.add_argument("--checkpoint_object", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_windows", type=int, default=100)
    parser.add_argument("--num_candidates", type=int, default=200)
    parser.add_argument("--elite_k", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--candidate_mode",
        choices=["gaussian", "cem_initial", "expert_perturb", "expert_injected"],
        default="gaussian",
    )
    parser.add_argument("--true_metric", choices=["terminal_cost", "progress"], default="terminal_cost")
    parser.add_argument("--action_noise_sigma", type=float, default=0.5)
    parser.add_argument("--action_clip", type=float, default=1.0)
    parser.add_argument("--cem_var_scale", type=float, default=1.0)
    parser.add_argument("--expert_small_noise", type=float, default=0.05)
    parser.add_argument("--expert_medium_noise", type=float, default=0.2)
    parser.add_argument("--expert_large_noise", type=float, default=0.5)
    parser.add_argument("--num_expert_small", type=int, default=50)
    parser.add_argument("--num_expert_medium", type=int, default=50)
    parser.add_argument("--num_expert_large", type=int, default=50)
    parser.add_argument("--include_zero_action", type=_str_to_bool, default=True)
    parser.add_argument("--include_sign_flip", type=_str_to_bool, default=True)
    parser.add_argument("--include_shuffle", type=_str_to_bool, default=True)
    parser.add_argument("--good_progress_threshold", type=float, default=0.02)
    parser.add_argument("--goal_mode", choices=["eval_offset", "episode_final"], default="eval_offset")
    parser.add_argument("--goal_offset_steps", type=int, default=25)
    parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    parser.add_argument("--debug_alignment", action="store_true")
    parser.add_argument("--gamma_values", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--eta_values", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--rho_values", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--tau_values", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
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
    args = parser.parse_args()
    if args.num_windows <= 0:
        raise ValueError("--num_windows must be positive.")
    if args.num_candidates < 2:
        raise ValueError("--num_candidates must be at least 2 for pairwise aliasing metrics.")
    if args.elite_k <= 0 or args.elite_k >= args.num_candidates:
        raise ValueError("--elite_k must be positive and smaller than --num_candidates.")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.action_noise_sigma < 0:
        raise ValueError("--action_noise_sigma must be non-negative.")
    if args.action_clip <= 0:
        raise ValueError("--action_clip must be positive.")
    if args.cem_var_scale < 0:
        raise ValueError("--cem_var_scale must be non-negative.")
    if args.expert_small_noise < 0 or args.expert_medium_noise < 0 or args.expert_large_noise < 0:
        raise ValueError("Expert injection noise scales must be non-negative.")
    if args.num_expert_small < 0 or args.num_expert_medium < 0 or args.num_expert_large < 0:
        raise ValueError("Expert injection counts must be non-negative.")
    if args.goal_offset_steps <= 0:
        raise ValueError("--goal_offset_steps must be positive.")
    if args.reset_state_tol < 0:
        raise ValueError("--reset_state_tol must be non-negative.")

    rng = np.random.default_rng(args.seed)
    dataset_path = Path(args.dataset)
    print(
        "[aliasing] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[aliasing] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _load_model(Path(args.checkpoint_object), device)
    model_class = model.__class__.__name__
    has_bottleneck = getattr(model, "transition_bottleneck", None) is not None
    history_size = int(getattr(model.predictor, "pos_embedding").shape[1])
    expected_action_dim = _expected_action_dim(model)
    effective_action_dim = args.frameskip * args.action_dim
    if expected_action_dim is not None and expected_action_dim != effective_action_dim:
        raise ValueError(
            f"Model action encoder expects dim {expected_action_dim}, but frameskip*action_dim "
            f"is {effective_action_dim}. Adjust --frameskip/--action_dim."
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
    context_outputs = []
    terminal_latent_dim = None
    print(
        f"[aliasing] model={model_class}, has_transition_bottleneck={has_bottleneck}, "
        f"history_size={history_size}, model_action_dim={effective_action_dim}",
        flush=True,
    )
    with h5py.File(dataset_path, "r") as h5, torch.no_grad():
        pixels_ds = h5[pixels_key]
        for context_idx, (start_row, goal_row, context_rows, goal_source) in enumerate(contexts):
            history_chunk_rows = context_rows[:-1]
            future_chunk_rows = start_row + np.arange(args.horizon) * args.frameskip
            history_chunks_norm = _action_chunks_from_rows(dataset_actions_norm, history_chunk_rows, args.frameskip)
            expert_future_chunks_norm = _action_chunks_from_rows(dataset_actions_norm, future_chunk_rows, args.frameskip)
            expert_index = 0 if args.candidate_mode == "expert_injected" else None
            if args.candidate_mode == "expert_injected":
                actions_future_norm_blocks, candidate_labels = _generate_expert_injected_candidates(
                    rng,
                    args.num_candidates,
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
                actions_future_norm_blocks, candidate_labels = _generate_future_action_chunks(
                    rng,
                    args.candidate_mode,
                    args.num_candidates,
                    args.horizon,
                    args.frameskip,
                    args.action_dim,
                    args.action_noise_sigma,
                    args.action_clip,
                    expert_future_chunks_norm,
                    args.cem_var_scale,
                )
            actions_future_raw_blocks = (
                actions_future_norm_blocks * action_std.reshape(1, 1, 1, -1)
                + action_mean.reshape(1, 1, 1, -1)
            )
            actions_model = _build_model_action_sequence(history_chunks_norm, actions_future_norm_blocks)
            context_pixels = _read_rows(pixels_ds, context_rows)
            goal_pixels = _read_rows(pixels_ds, np.asarray([goal_row], dtype=np.int64))[0]
            start_state = states[start_row]
            goal_state = states[goal_row]
            if context_idx == 0 and args.debug_alignment:
                _debug_print_window(
                    context_idx,
                    start_row,
                    goal_row,
                    goal_source,
                    context_rows,
                    states,
                    episode_idx,
                    step_idx,
                    goal_pixels,
                    actions_future_norm_blocks,
                    actions_future_raw_blocks,
                    actions_model,
                )
                _debug_env_reset_check(env, start_state, goal_state, args.reset_state_tol)
                expert_future_raw_chunks = (
                    expert_future_chunks_norm * action_std.reshape(1, 1, -1)
                    + action_mean.reshape(1, 1, -1)
                )
                dataset_future_row = start_row + args.horizon * args.frameskip
                _debug_expert_action_rollout_check(
                    env,
                    start_state,
                    goal_state,
                    states[dataset_future_row],
                    expert_future_raw_chunks,
                    args.reset_state_tol,
                )
            terminal_costs, terminal_states, env_feedback = _rollout_real_env(
                env,
                start_state,
                goal_state,
                actions_future_raw_blocks,
                args.horizon,
                args.frameskip,
                args.reset_state_tol,
            )
            start_cost = task_cost(start_state, goal_state)
            progress = start_cost - terminal_costs
            if args.true_metric == "terminal_cost":
                true_scores = terminal_costs
                true_gap_values = terminal_costs
            elif args.true_metric == "progress":
                true_scores = -progress
                true_gap_values = progress
            else:
                raise ValueError(f"Unknown true_metric: {args.true_metric}")
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
                debug_compare_get_cost=(context_idx == 0 and args.debug_alignment),
            )
            if context_idx == 0:
                terminal_latent_dim = int(terminal_latents.shape[-1])
                print(
                    f"[aliasing] env action shape={actions_future_raw_blocks.shape}, "
                    f"model action shape={actions_model.shape}",
                    flush=True,
                )
                print(
                    f"[aliasing] terminal latent shape={terminal_latents.shape}, "
                    f"goal latent shape={goal_latent.shape}",
                    flush=True,
                )
                if args.debug_alignment:
                    _debug_ranking_summary(
                        terminal_costs,
                        progress,
                        true_scores,
                        args.true_metric,
                        latent_scores,
                        terminal_states,
                        goal_state,
                        args.elite_k,
                        candidate_labels,
                        expert_index,
                        start_cost,
                        env_feedback,
                    )
            context = _context_metrics(
                context_idx,
                start_row,
                goal_row,
                goal_source,
                terminal_costs,
                progress,
                true_scores,
                true_gap_values,
                args.true_metric,
                start_cost,
                latent_scores,
                terminal_latents,
                goal_latent,
                candidate_labels,
                expert_index,
                args.good_progress_threshold,
                args.gamma_values,
                args.eta_values,
                args.rho_values,
                args.tau_values,
                args.elite_k,
            )
            context_outputs.append(context)
            print(
                f"[aliasing] window {context_idx + 1}/{len(contexts)} "
                f"spearman={context['auxiliary']['spearman']:.4f}, "
                f"regret={context['auxiliary']['candidate_regret']:.4f}, "
                f"R_max={context['R_max']:.4f}",
                flush=True,
            )
    if hasattr(env, "close"):
        env.close()

    aggregate = _aggregate(
        context_outputs,
        args.gamma_values,
        args.eta_values,
        args.rho_values,
        args.tau_values,
        args.elite_k,
    )
    output = {
        "config": vars(args),
        "model": {
            "checkpoint": str(Path(args.checkpoint_object)),
            "latent_dim": terminal_latent_dim,
            "has_transition_bottleneck": bool(has_bottleneck),
            "model_class": model_class,
        },
        "aggregate": aggregate,
        "contexts": context_outputs,
        "notes": {
            "task_cost": "scripts/task_cost.py block-pose cost: block xy / 512 + wrapped angle / pi.",
            "goal_source": "Default goal_mode=eval_offset uses the same raw-step offset idea as eval.py goal_offset_steps; episode_final is available for older final-state diagnostics.",
            "action_normalization": "Candidates are sampled in normalized dataset-action coordinates; env receives de-normalized raw future actions; model receives dataset history action chunks plus candidate future chunks flattened by frameskip.",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(_jsonify(output), f, indent=2)

    print("\nModel:", Path(args.checkpoint_object).name)
    print(f"Num windows: {len(context_outputs)}")
    print(f"Num candidates per window: {args.num_candidates}")
    print("Cost: block_pose task_cost")
    print(f"Primary true metric: {args.true_metric}")
    if args.candidate_mode == "expert_injected":
        print("Expert candidate index: 0")
    _print_elite_summary(aggregate, args.num_candidates)
    print("\nK_gamma:")
    print("gamma | mean | median | p90")
    for gamma, stats in aggregate["K_gamma"].items():
        print(f"{gamma:>5} | {stats['mean']:>6.2f} | {stats['median']:>6.2f} | {stats['p90']:>6.2f}")
    _print_matrix("Geometric aliasing", aggregate["alias_geom"], args.gamma_values, args.eta_values, "gamma={gamma},eta={value}")
    _print_matrix("Normalized geometric aliasing", aggregate["alias_norm"], args.gamma_values, args.rho_values, "gamma={gamma},rho={value}")
    _print_matrix("Score aliasing", aggregate["alias_score"], args.gamma_values, args.tau_values, "gamma={gamma},tau={value}")
    latent_scale = aggregate["latent_scale"]
    aux = aggregate["auxiliary"]
    print("\nLatent scale:")
    print(
        f"R_max mean={latent_scale['R_max_mean']:.4f}, p95={latent_scale['R_max_p95']:.4f}; "
        f"pairwise median={latent_scale['pairwise_latent_distance_median']:.4f}, "
        f"p10={latent_scale['pairwise_latent_distance_p10']:.4f}, "
        f"p90={latent_scale['pairwise_latent_distance_p90']:.4f}"
    )
    print("\nAuxiliary:")
    print(
        f"spearman={aux['mean_spearman']:.4f}, pearson={aux['mean_pearson']:.4f}, "
        f"top10_precision={aux['top10_precision']:.4f}, "
        f"candidate_regret_mean={aux['candidate_regret_mean']:.4f}, "
        f"candidate_regret_median={aux['candidate_regret_median']:.4f}"
    )
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
