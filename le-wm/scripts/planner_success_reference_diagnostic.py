from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
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

from aliasing_experiment import _get_env_state, _make_env, _reset_env_to_state, _step_env  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EPS = 1e-8


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


def _find_key(h5: h5py.File, candidates: Sequence[str]) -> str:
    for key in candidates:
        if key and key in h5:
            return key
    raise KeyError(f"None of these keys exist in dataset: {list(candidates)}")


def _read_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    values = np.asarray(dataset[unique_rows])
    return values[inverse]


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


def _encode_pixels(model, pixels: np.ndarray, img_size: int, batch_size: int, device: torch.device) -> np.ndarray:
    if pixels.ndim == 4:
        pixels = pixels[:, None]
    chunks = []
    with torch.no_grad():
        for start in range(0, pixels.shape[0], batch_size):
            batch = pixels[start:start + batch_size]
            tensor = _preprocess_pixels(batch, img_size, device)
            emb = model.encode({"pixels": tensor})["emb"]
            chunks.append(emb.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + EPS
    return numerator / denominator


def _extract_pixels_from_obs(env, obs) -> np.ndarray:
    if isinstance(obs, dict):
        for key in ("pixels", "image", "rgb", "observation"):
            value = obs.get(key)
            if value is not None:
                arr = np.asarray(value)
                if arr.ndim >= 3:
                    return arr
    if hasattr(env, "render"):
        frame = env.render()
        if frame is not None:
            return np.asarray(frame)
    raise RuntimeError("Could not extract pixels from env observation and env.render() returned nothing.")


class _ActionSpaceAdapter:
    def __init__(self, action_dim: int):
        self.shape = (1, int(action_dim))


class _PolicyEnvAdapter:
    def __init__(self, action_dim: int):
        self.num_envs = 1
        self.action_space = _ActionSpaceAdapter(action_dim)


def _set_policy_env(policy, env) -> int:
    raw_action_dim = int(np.prod(env.action_space.shape))
    policy.env = _PolicyEnvAdapter(raw_action_dim)
    if getattr(policy, "_action_buffer", None) is None:
        policy._action_buffer = deque()
    elif hasattr(policy, "_action_buffer"):
        policy._action_buffer.clear()
    if hasattr(policy, "_next_init"):
        policy._next_init = None
    action_block = int(getattr(policy.cfg, "action_block", 1))
    horizon = int(getattr(policy.cfg, "horizon", 1))
    solver = getattr(policy, "solver", None)
    if solver is not None:
        solver._config = SimpleNamespace(horizon=horizon, action_block=action_block)
        solver._n_envs = 1
        solver._horizon = horizon
        solver._action_dim = raw_action_dim
    return raw_action_dim


def _proprio_from_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.size >= 7:
        return state[[0, 1, 5, 6]]
    if state.size >= 4:
        return state[:4]
    return state


def _history(value: np.ndarray) -> np.ndarray:
    return np.asarray(value)[None, None]


def _policy_info(
    pixels: np.ndarray,
    goal_pixels: np.ndarray,
    state: np.ndarray,
    goal_state: np.ndarray,
    previous_action: np.ndarray,
) -> Dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    goal_state = np.asarray(goal_state, dtype=np.float32)
    return {
        "pixels": _history(np.asarray(pixels)),
        "goal": _history(np.asarray(goal_pixels)),
        "action": _history(np.asarray(previous_action, dtype=np.float32)),
        "state": _history(state),
        "goal_state": _history(goal_state),
        "proprio": _history(_proprio_from_state(state)),
        "goal_proprio": _history(_proprio_from_state(goal_state)),
    }


def _call_policy(policy, info_dict: dict) -> np.ndarray:
    try:
        action = policy.get_action(info_dict)
    except Exception as exc:  # noqa: BLE001
        shapes = {key: tuple(np.asarray(value).shape) for key, value in info_dict.items()}
        raise RuntimeError(f"WorldModelPolicy.get_action failed with info shapes {shapes}") from exc
    action = np.asarray(action, dtype=np.float32)
    if action.size == 0:
        raise RuntimeError("WorldModelPolicy.get_action returned an empty action.")
    return action.reshape(-1)[-2:]


def _initial_previous_action(h5: h5py.File, action_key: Optional[str], start_row: int, action_dim: int) -> np.ndarray:
    if action_key is None:
        return np.zeros(action_dim, dtype=np.float32)
    previous_action = np.asarray(h5[action_key][start_row], dtype=np.float32).reshape(-1)
    if previous_action.size != action_dim:
        print(
            f"[planner_ref] dataset action dim {previous_action.size} != policy action dim {action_dim}; "
            "using zeros for initial previous action.",
            flush=True,
        )
        return np.zeros(action_dim, dtype=np.float32)
    return previous_action


def _load_stable_worldmodel_policy(policy_name: str, eval_config: Path):
    try:
        import hydra
        import stable_pretraining as spt
        import stable_worldmodel as swm
        from hydra.core.global_hydra import GlobalHydra
        from sklearn import preprocessing
        from torchvision.transforms import v2 as transforms
    except ImportError as exc:
        raise ImportError("Collect mode requires stable_worldmodel, hydra, stable_pretraining, sklearn, and torchvision.") from exc

    config_path = eval_config.resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        cfg = hydra.compose(config_name=config_path.stem)
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    transforms_by_key = {"pixels": transform, "goal": transform}
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    stats_dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    model = swm.policy.AutoCostModel(policy_name).to("cuda").eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True
    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    return swm.policy.WorldModelPolicy(solver=solver, config=plan_config, process=process, transform=transforms_by_key)


def _candidate_start_rows(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    goal_offset_steps: int,
    eval_budget: int,
) -> List[Tuple[int, int]]:
    candidates = []
    for episode in np.unique(episode_idx):
        rows = np.where(episode_idx == episode)[0]
        rows = rows[np.argsort(step_idx[rows])]
        if rows.size <= goal_offset_steps + eval_budget:
            continue
        if not np.all(np.diff(step_idx[rows]) == 1):
            continue
        for local_start in range(0, rows.size - goal_offset_steps - eval_budget):
            candidates.append((int(episode), int(rows[local_start])))
    return candidates


def collect(args: argparse.Namespace) -> None:
    print(
        "[planner_ref] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[planner_ref] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    policy = _load_stable_worldmodel_policy(args.policy, Path(args.eval_config))
    env = _make_env(args.env_name)
    action_dim = _set_policy_env(policy, env)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pixels_paths = []
    state_paths = []
    action_paths = []
    reward_paths = []
    start_rows = []
    goal_rows = []
    successes = []

    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        state_key = _find_key(h5, [args.state_key, "state"])
        try:
            action_key: Optional[str] = _find_key(h5, [args.action_key, "action"])
        except KeyError:
            action_key = None
            print("[planner_ref] no action key found; using zero previous-action history.", flush=True)
        episode_key = _find_key(h5, [args.episode_key, "episode_idx", "ep_idx"])
        step_key = _find_key(h5, [args.step_key, "step_idx"])
        episode_idx = np.asarray(h5[episode_key]).reshape(-1)
        step_idx = np.asarray(h5[step_key]).reshape(-1)
        starts = _candidate_start_rows(episode_idx, step_idx, args.goal_offset_steps, args.eval_budget)
        rng.shuffle(starts)
        print(f"[planner_ref] candidate starts: {len(starts)}", flush=True)

        for attempt_idx, (_episode, start_row) in enumerate(starts):
            if len(pixels_paths) >= args.num_success:
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
            final_info = {}
            done = False
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

            success_value = float(final_info.get("success", final_info.get("is_success", 0.0))) if isinstance(final_info, dict) else 0.0
            if done and success_value <= 0.0:
                success_value = 1.0
            if success_value >= args.success_threshold:
                pixels_paths.append(np.asarray(pixels))
                state_paths.append(np.asarray(states, dtype=np.float32))
                action_paths.append(np.asarray(actions, dtype=np.float32))
                reward_paths.append(np.asarray(rewards, dtype=np.float32))
                start_rows.append(start_row)
                goal_rows.append(goal_row)
                successes.append(success_value)
                print(f"[planner_ref] collected success {len(pixels_paths)}/{args.num_success} at attempt {attempt_idx}", flush=True)
            elif attempt_idx % 25 == 0:
                print(f"[planner_ref] attempt {attempt_idx}, successes={len(pixels_paths)}", flush=True)

    if not pixels_paths:
        raise RuntimeError("No successful planner trajectories collected.")
    max_len = max(path.shape[0] for path in pixels_paths)

    def pad_array(paths: List[np.ndarray], pad_value: float = np.nan) -> np.ndarray:
        shape = (len(paths), max_len) + paths[0].shape[1:]
        out = np.full(shape, pad_value, dtype=paths[0].dtype)
        for idx, path in enumerate(paths):
            out[idx, : path.shape[0]] = path
        return out

    np.savez_compressed(
        output,
        pixels=pad_array(pixels_paths, 0),
        states=pad_array(state_paths),
        actions=pad_array(action_paths, 0.0),
        rewards=pad_array([r[:, None] for r in reward_paths]).squeeze(-1),
        lengths=np.asarray([path.shape[0] for path in pixels_paths], dtype=np.int64),
        start_rows=np.asarray(start_rows, dtype=np.int64),
        goal_rows=np.asarray(goal_rows, dtype=np.int64),
        success=np.asarray(successes, dtype=np.float32),
        policy=np.asarray(args.policy),
        goal_offset_steps=np.asarray(args.goal_offset_steps),
        eval_budget=np.asarray(args.eval_budget),
    )
    print(f"[planner_ref] wrote {output}", flush=True)


def _reference_geometry_rows(z_paths: np.ndarray, z_goals: np.ndarray, lengths: np.ndarray, model_name: str, latent_dim: int, horizons: List[int]) -> List[Dict[str, object]]:
    rows = []
    for horizon in horizons:
        cos_values = []
        rho_values = []
        vplus_values = []
        vplus_cond_values = []
        signed_values = []
        for traj, goal, length in zip(z_paths, z_goals, lengths):
            length = int(length)
            if length <= horizon:
                continue
            for t_idx in range(0, length - horizon):
                z_t = traj[t_idx]
                z_h = traj[t_idx + horizon]
                q = goal - z_t
                d = z_h - z_t
                cos_values.append(float(_cosine(d, q)))
                before = float(np.sum((z_t - goal) ** 2))
                after = float(np.sum((z_h - goal) ** 2))
                delta = after - before
                norm_delta = delta / (before + EPS)
                signed_values.append(norm_delta)
                rho_values.append(float(delta > 0.0))
                vplus_values.append(max(norm_delta, 0.0))
                if delta > 0.0:
                    vplus_cond_values.append(norm_delta)
        cos_array = np.asarray(cos_values, dtype=np.float64)
        rho_array = np.asarray(rho_values, dtype=np.float64)
        rows.append(
            {
                "model": model_name,
                "latent_dim": int(latent_dim),
                "h_raw": int(horizon),
                "num_valid_pairs": int(cos_array.size),
                "p_sa": float(1.0 - np.mean(rho_array)) if rho_array.size else float("nan"),
                "rho_sa": float(np.mean(rho_array)) if rho_array.size else float("nan"),
                "vplus_sa_norm": float(np.mean(vplus_values)) if vplus_values else float("nan"),
                "vplus_sa_norm_cond": float(np.mean(vplus_cond_values)) if vplus_cond_values else 0.0,
                "signed_progress_norm": float(np.mean(signed_values)) if signed_values else float("nan"),
                "mean_cos_progress": float(np.mean(cos_array)) if cos_array.size else float("nan"),
                "progress_obtuse_rate": float(np.mean(cos_array < 0.0)) if cos_array.size else float("nan"),
                "mean_neg_cos_progress": float(np.mean(np.maximum(-cos_array, 0.0))) if cos_array.size else float("nan"),
                "cos_progress_q10": float(np.quantile(cos_array, 0.10)) if cos_array.size else float("nan"),
                "cos_progress_q25": float(np.quantile(cos_array, 0.25)) if cos_array.size else float("nan"),
                "cos_progress_q50": float(np.quantile(cos_array, 0.50)) if cos_array.size else float("nan"),
            }
        )
    return rows


def _plot_analyze(rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[planner_ref] matplotlib unavailable; skipping plots.", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted({int(row["h_raw"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows}, key=lambda model: min(int(row["latent_dim"]) for row in rows if row["model"] == model))
    dims = {model: min(int(row["latent_dim"]) for row in rows if row["model"] == model) for model in models}
    labels = [f"{model}\nD={dims[model]}" for model in models]
    x = np.arange(len(models), dtype=np.float64)
    width = min(0.18, 0.75 / max(len(horizons), 1))

    def value(model: str, horizon: int, key: str) -> float:
        for row in rows:
            if row["model"] == model and int(row["h_raw"]) == horizon:
                return float(row[key])
        return float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), facecolor="white")
    offsets = (np.arange(len(horizons)) - (len(horizons) - 1) / 2.0) * width
    for offset, horizon in zip(offsets, horizons):
        axes[0].bar(x + offset, [value(model, horizon, "p_sa") for model in models], width=width, label=f"h={horizon}")
        axes[1].bar(x + offset, [value(model, horizon, "mean_neg_cos_progress") for model in models], width=width, label=f"h={horizon}")
        axes[2].bar(x + offset, [value(model, horizon, "vplus_sa_norm") for model in models], width=width, label=f"h={horizon}")
    axes[0].set_title("A. Planner-success reference p_SA")
    axes[0].set_ylabel("p_sa")
    axes[1].set_title("B. Negative progress cosine")
    axes[1].set_ylabel("mean max(-cos,0)")
    axes[2].set_title("C. Positive distance violation")
    axes[2].set_ylabel("vplus_sa_norm")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_baseline_success_reference_geometry.png", dpi=260)
    fig.savefig(output_dir / "fig_baseline_success_reference_geometry.pdf")
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models = _parse_name_paths(args.models)
    reference = np.load(args.reference_npz)
    pixels = np.asarray(reference["pixels"])
    lengths = np.asarray(reference["lengths"], dtype=np.int64)
    goal_rows = np.asarray(reference["goal_rows"], dtype=np.int64)
    horizons = [int(item) for item in str(args.horizons).replace(",", " ").split() if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        goal_pixels = _read_h5_rows(h5[pixels_key], goal_rows)

    rows: List[Dict[str, object]] = []
    for model_name, checkpoint in models.items():
        print(f"[planner_ref] encoding baseline-success reference paths for {model_name}", flush=True)
        model = _load_model(checkpoint, device)
        z_paths = _encode_pixels(model, pixels, args.img_size, args.batch_size, device)
        z_goals = _encode_pixels(model, goal_pixels[:, None], args.img_size, args.batch_size, device)[:, 0]
        rows.extend(_reference_geometry_rows(z_paths, z_goals, lengths, model_name, z_paths.shape[-1], horizons))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(output_dir / "baseline_success_reference_geometry.csv", rows)
    with (output_dir / "baseline_success_reference_geometry.json").open("w") as file:
        json.dump({"rows": rows, "args": vars(args)}, file, indent=2)
    _plot_analyze(rows, output_dir / "paper_figures")
    print(f"[planner_ref] wrote outputs under {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline-success common-reference geometry diagnostic for LeWM PushT.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect successful baseline planner trajectories.")
    collect_parser.add_argument("--policy", required=True, help="stable_worldmodel policy name, e.g. pusht/baseline")
    collect_parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--eval_config", default="config/eval/pusht.yaml")
    collect_parser.add_argument("--env_name", default="swm/PushT-v1")
    collect_parser.add_argument("--num_success", type=int, default=50)
    collect_parser.add_argument("--goal_offset_steps", type=int, default=25)
    collect_parser.add_argument("--eval_budget", type=int, default=50)
    collect_parser.add_argument("--success_threshold", type=float, default=0.5)
    collect_parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    collect_parser.add_argument("--pixels_key", default="pixels")
    collect_parser.add_argument("--state_key", default="state")
    collect_parser.add_argument("--action_key", default="action")
    collect_parser.add_argument("--episode_key", default="episode_idx")
    collect_parser.add_argument("--step_key", default="step_idx")
    collect_parser.add_argument("--seed", type=int, default=0)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze common successful reference trajectories with multiple encoders.")
    analyze_parser.add_argument("--reference_npz", required=True)
    analyze_parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    analyze_parser.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint_object.ckpt")
    analyze_parser.add_argument("--output_dir", default="results_baseline_success_reference")
    analyze_parser.add_argument("--horizons", default="5,25,50")
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
