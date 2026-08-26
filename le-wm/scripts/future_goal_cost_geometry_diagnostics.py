from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _find_key(h5: h5py.File, candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in h5:
            return key
    raise KeyError(f"None of these keys were found in dataset: {list(candidates)}")


def _parse_indices(values: Optional[List[int]], default_dim: int) -> np.ndarray:
    if values is None or len(values) == 0:
        return np.arange(default_dim)
    return np.asarray(values, dtype=np.int64)


def _read_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    unique_values = np.asarray(dataset[unique_rows])
    return unique_values[inverse]


def _sample_rows(valid_rows: np.ndarray, count: int, rng: np.random.Generator, label: str) -> np.ndarray:
    if valid_rows.size == 0:
        raise ValueError(f"No valid rows available for {label}.")
    replace = count > valid_rows.size
    if replace:
        print(
            f"[future_goal] requested {count} {label} rows but only {valid_rows.size} valid rows exist; sampling with replacement.",
            flush=True,
        )
    return rng.choice(valid_rows, size=count, replace=replace)


def _standardized_state_cost(
    state: np.ndarray,
    future_rows: np.ndarray,
    goal_rows: np.ndarray,
    future_indices: np.ndarray,
    goal_indices: np.ndarray,
) -> np.ndarray:
    if future_indices.size != goal_indices.size:
        raise ValueError(
            "future_state_indices and goal_state_indices must select the same number of dimensions. "
            f"Got {future_indices.size} and {goal_indices.size}."
        )
    mean = np.nanmean(state, axis=0)
    std = np.nanstd(state, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    futures = (state[future_rows][:, future_indices] - mean[future_indices]) / std[future_indices]
    goals = (state[goal_rows][:, goal_indices] - mean[goal_indices]) / std[goal_indices]
    diff = futures[:, None, :] - goals[None, :, :]
    return np.sum(diff * diff, axis=-1, dtype=np.float64)


def _double_center(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + matrix.mean()


def _energy_dimension(singular_values: np.ndarray, fraction: float) -> int:
    energy = singular_values * singular_values
    total = energy.sum()
    if total <= 0:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, fraction, side="left") + 1)


def _spectral_metrics(matrix: np.ndarray, include_spectrum: bool = True) -> Dict[str, object]:
    centered = _double_center(matrix)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.size == 0:
        return {
            "exact_rank_diagnostic": 0,
            "d90": 0,
            "d95": 0,
            "d99": 0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "singular_values": [] if include_spectrum else None,
        }

    tol = max(centered.shape) * np.finfo(singular_values.dtype).eps * singular_values[0]
    exact_rank = int(np.sum(singular_values > tol))
    d90 = _energy_dimension(singular_values, 0.90)
    d95 = _energy_dimension(singular_values, 0.95)
    d99 = _energy_dimension(singular_values, 0.99)
    singular_sum = singular_values.sum()
    if singular_sum > 0:
        probabilities = singular_values / singular_sum
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12))))
    else:
        effective_rank = 0.0
    stable_rank = (
        float(np.sum(singular_values * singular_values) / (singular_values[0] ** 2 + 1e-12))
        if singular_values[0] > 0
        else 0.0
    )
    metrics: Dict[str, object] = {
        "exact_rank_diagnostic": exact_rank,
        "d90": d90,
        "d95": d95,
        "d99": d99,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
    }
    if include_spectrum:
        metrics["singular_values"] = singular_values.tolist()
    return metrics


def _sample_size_stability(matrix: np.ndarray, subset_sizes: List[int]) -> Dict[str, Dict[str, object]]:
    stability = {}
    max_rows, max_cols = matrix.shape
    for size in subset_sizes:
        size = int(size)
        if size <= 0:
            continue
        subset_n = min(size, max_rows, max_cols)
        if subset_n < size:
            print(
                f"[future_goal] subset size {size} exceeds matrix dimensions {matrix.shape}; using {subset_n}.",
                flush=True,
            )
        stability[str(size)] = _spectral_metrics(matrix[:subset_n, :subset_n], include_spectrum=False)
    return stability


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


def _preprocess_image_batch(images: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(images).to(device)
    tensor = _to_chw_float(tensor)
    if tensor.shape[-2:] != (img_size, img_size):
        tensor = F.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = (tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return tensor


def _load_model(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    model = obj.model if hasattr(obj, "model") else obj
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def _encode_rows(
    h5_path: Path,
    rows: np.ndarray,
    pixels_key: str,
    model,
    device: torch.device,
    batch_size: int,
    img_size: int,
) -> np.ndarray:
    latents = []
    with h5py.File(h5_path, "r") as h5:
        pixels_ds = h5[pixels_key]
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            images = _read_rows(pixels_ds, batch_rows)
            pixels = _preprocess_image_batch(images, img_size, device).unsqueeze(1)
            encoded = model.encode({"pixels": pixels})
            latents.append(encoded["emb"][:, 0].detach().float().cpu().numpy())
            print(
                f"[future_goal] encoded {min(start + batch_size, len(rows))}/{len(rows)} rows from {pixels_key}",
                flush=True,
            )
    return np.concatenate(latents, axis=0)


def _squared_l2_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    distances = left_norm + right_norm - 2.0 * (left @ right.T)
    return np.maximum(distances, 0.0)


def _pearson(flat_a: np.ndarray, flat_b: np.ndarray) -> float:
    std_a = flat_a.std()
    std_b = flat_b.std()
    if std_a < 1e-12 or std_b < 1e-12:
        return float("nan")
    return float(np.corrcoef(flat_a, flat_b)[0, 1])


def _spearman(flat_a: np.ndarray, flat_b: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("[future_goal] scipy not available; skipping Spearman correlation.", flush=True)
        return float("nan")
    corr = spearmanr(flat_a, flat_b, nan_policy="omit").correlation
    return float(corr) if corr is not None else float("nan")


def _alignment_metrics(true_cost: np.ndarray, latent_cost: np.ndarray) -> Dict[str, float]:
    flat_a = true_cost.reshape(-1).astype(np.float64)
    flat_b = latent_cost.reshape(-1).astype(np.float64)
    b_var = np.var(flat_b)
    if b_var < 1e-12:
        affine_scale = 0.0
    else:
        affine_scale = float(np.mean((flat_b - flat_b.mean()) * (flat_a - flat_a.mean())) / b_var)
    affine_bias = float(flat_a.mean() - affine_scale * flat_b.mean())
    calibrated = affine_scale * flat_b + affine_bias

    true_best = np.argmin(true_cost, axis=0)
    latent_best = np.argmin(latent_cost, axis=0)
    goal_ids = np.arange(true_cost.shape[1])
    regret = true_cost[latent_best, goal_ids] - true_cost[true_best, goal_ids]
    return {
        "pearson": _pearson(flat_a, flat_b),
        "spearman": _spearman(flat_a, flat_b),
        "affine_scale": affine_scale,
        "affine_bias": affine_bias,
        "calibrated_mse": float(np.mean((calibrated - flat_a) ** 2)),
        "top1_agreement": float(np.mean(true_best == latent_best)),
        "mean_regret": float(np.mean(regret)),
        "median_regret": float(np.median(regret)),
        "p90_regret": float(np.percentile(regret, 90)),
    }


def _print_spectral_summary(name: str, metrics: Dict[str, object]):
    print(
        f"{name:>12} | "
        f"d90={metrics['d90']:>4} "
        f"d95={metrics['d95']:>4} "
        f"d99={metrics['d99']:>4} "
        f"eff_rank={metrics['effective_rank']:>8.2f} "
        f"stable_rank={metrics['stable_rank']:>8.2f} "
        f"rank_diag={metrics['exact_rank_diagnostic']:>4}"
    )


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items() if val is not None}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Empirical finite-sample future-goal cost geometry diagnostic for PushT-style datasets."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_futures", type=int, default=1000)
    parser.add_argument("--num_goals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost_mode", choices=["state_l2"], default="state_l2")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--goal_state_indices", type=int, nargs="*", default=None)
    parser.add_argument("--future_state_indices", type=int, nargs="*", default=None)
    parser.add_argument("--subset_sizes", type=int, nargs="*", default=[100, 200, 500, 1000])
    parser.add_argument("--checkpoint_object", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rng = np.random.default_rng(args.seed)
    print(
        "[future_goal] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[future_goal] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )

    with h5py.File(dataset_path, "r") as h5:
        print("[future_goal] dataset keys and shapes:", flush=True)
        for key in h5.keys():
            print(f"  - {key}: {getattr(h5[key], 'shape', None)}", flush=True)
        state_key = _find_key(h5, [args.state_key])
        pixels_key = _find_key(h5, [args.pixels_key, "observation/pixels"])
        state = np.asarray(h5[state_key]).astype(np.float64)

    future_indices = _parse_indices(args.future_state_indices, state.shape[-1])
    goal_indices = _parse_indices(args.goal_state_indices, state.shape[-1])
    if args.future_state_indices is not None and args.goal_state_indices is None:
        goal_indices = future_indices.copy()
    if args.goal_state_indices is not None and args.future_state_indices is None:
        future_indices = goal_indices.copy()
    if (
        np.any(future_indices < 0)
        or np.any(goal_indices < 0)
        or np.any(future_indices >= state.shape[-1])
        or np.any(goal_indices >= state.shape[-1])
    ):
        raise ValueError(
            f"State indices out of bounds for state dimension {state.shape[-1]}: "
            f"future={future_indices.tolist()}, goal={goal_indices.tolist()}"
        )
    selected_for_valid = np.unique(np.concatenate([future_indices, goal_indices]))
    valid_rows = np.nonzero(~np.isnan(state[:, selected_for_valid]).any(axis=1))[0]
    future_rows = _sample_rows(valid_rows, args.num_futures, rng, "future")
    goal_rows = _sample_rows(valid_rows, args.num_goals, rng, "goal")
    print(
        f"[future_goal] sampled futures={len(future_rows)}, goals={len(goal_rows)}, valid_rows={len(valid_rows)}",
        flush=True,
    )

    true_cost = _standardized_state_cost(state, future_rows, goal_rows, future_indices, goal_indices)
    true_metrics = _spectral_metrics(true_cost, include_spectrum=True)
    stability = _sample_size_stability(true_cost, args.subset_sizes)
    output = {
        "dataset_path": str(dataset_path),
        "num_futures": int(args.num_futures),
        "num_goals": int(args.num_goals),
        "seed": int(args.seed),
        "cost_mode": args.cost_mode,
        "state_key": state_key,
        "pixels_key": pixels_key,
        "future_state_indices": future_indices.tolist(),
        "goal_state_indices": goal_indices.tolist(),
        "note": "Finite-sample empirical spectral geometry under the chosen sampling distribution; not task-intrinsic rank.",
        "true_cost": true_metrics,
        "sample_size_stability": stability,
    }

    print("\nSpectral geometry")
    print("------------ | d90  d95  d99  effective  stable   rank")
    _print_spectral_summary("true_cost", true_metrics)

    if args.checkpoint_object is not None:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
        model = _load_model(Path(args.checkpoint_object), device)
        with torch.no_grad():
            future_latents = _encode_rows(
                dataset_path,
                future_rows,
                pixels_key,
                model,
                device,
                args.batch_size,
                args.img_size,
            )
            goal_latents = _encode_rows(
                dataset_path,
                goal_rows,
                pixels_key,
                model,
                device,
                args.batch_size,
                args.img_size,
            )
        latent_cost = _squared_l2_matrix(future_latents, goal_latents)
        latent_metrics = _spectral_metrics(latent_cost, include_spectrum=True)
        alignment = _alignment_metrics(true_cost, latent_cost)
        latent_dim = int(future_latents.shape[-1])
        latent_metrics["latent_dim"] = latent_dim
        latent_metrics["centered_rank_le_latent_dim"] = (
            int(latent_metrics["exact_rank_diagnostic"]) <= latent_dim
        )
        latent_metrics["latent_dim_over_d90_A"] = (
            float(latent_dim / true_metrics["d90"]) if true_metrics["d90"] else float("nan")
        )
        latent_metrics["latent_dim_over_effective_rank_A"] = (
            float(latent_dim / true_metrics["effective_rank"])
            if true_metrics["effective_rank"]
            else float("nan")
        )
        output["checkpoint_object"] = str(Path(args.checkpoint_object))
        output["latent_cost"] = latent_metrics
        output["alignment"] = alignment

        _print_spectral_summary("latent", latent_metrics)
        print("\nAlignment true cost A vs latent cost B")
        print(
            f"pearson={alignment['pearson']:.4f} "
            f"spearman={alignment['spearman']:.4f} "
            f"top1={alignment['top1_agreement']:.4f} "
            f"mean_regret={alignment['mean_regret']:.4f} "
            f"median_regret={alignment['median_regret']:.4f} "
            f"p90_regret={alignment['p90_regret']:.4f} "
            f"calibrated_mse={alignment['calibrated_mse']:.4f}"
        )

    print("\nSample-size stability")
    print("subset | d90  d95  d99  effective  stable")
    for size, metrics in stability.items():
        print(
            f"{size:>6} | "
            f"{metrics['d90']:>3} "
            f"{metrics['d95']:>4} "
            f"{metrics['d99']:>4} "
            f"{metrics['effective_rank']:>9.2f} "
            f"{metrics['stable_rank']:>7.2f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(_jsonify(output), f, indent=2)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
