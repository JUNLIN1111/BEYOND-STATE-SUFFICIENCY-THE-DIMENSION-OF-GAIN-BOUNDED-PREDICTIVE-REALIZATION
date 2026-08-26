from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    _find_key,
    _load_model,
    _preprocess_pixels,
    _read_rows,
    _sample_contexts,
)
from multistep_rollout_diagnostics import _expected_action_dim, _model_step, _raw_action_dim  # noqa: E402


def _parse_models(specs: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected model spec as name=path, got: {spec}")
        name, path = spec.split("=", 1)
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


def _aggregate(rows: List[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out_rows = []
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
            arr = np.asarray(values, dtype=np.float64)
            count = max(int(np.sum(np.isfinite(arr))), 1)
            out[f"{metric_key}_mean"] = float(np.nanmean(arr))
            out[f"{metric_key}_stderr"] = float(np.nanstd(arr) / math.sqrt(count))
            out[f"{metric_key}_count"] = int(np.sum(np.isfinite(arr)))
        out_rows.append(out)
    return out_rows


def _infer_frameskip(model, dataset_path: Path, frameskip_arg: int | None) -> int:
    if frameskip_arg is not None:
        return int(frameskip_arg)
    expected = _expected_action_dim(model)
    raw = _raw_action_dim(dataset_path)
    if expected % raw != 0:
        raise ValueError(f"Cannot infer frameskip: model action dim {expected}, raw action dim {raw}")
    frameskip = expected // raw
    print(f"[pred_loss] inferred frameskip={frameskip} from expected_action_dim={expected} and raw_action_dim={raw}")
    return frameskip


def _rollout_prediction_loss_for_window(
    model,
    pixels_ds,
    actions_norm: np.ndarray,
    start_row: int,
    goal_row: int,
    context_rows: np.ndarray,
    frameskip: int,
    horizon: int,
    img_size: int,
    device: torch.device,
    model_name: str,
    checkpoint_path: Path,
    window_idx: int,
) -> Dict[str, object]:
    history_size = int(getattr(getattr(model, "predictor", None), "pos_embedding").shape[1])
    future_rows = start_row + np.arange(1, horizon + 1) * frameskip
    model_rows = np.concatenate([context_rows, future_rows]).astype(np.int64)
    pixels_np = _read_rows(pixels_ds, model_rows)[None]
    action_chunks = _action_chunks_from_rows(actions_norm, model_rows, frameskip).reshape(len(model_rows), -1)[None]

    pixels = _preprocess_pixels(pixels_np, img_size, device)
    actions = torch.from_numpy(action_chunks).float().to(device)
    encoded = model.encode({"pixels": pixels, "action": actions})
    true_emb = encoded["emb"]

    emb_roll = true_emb[:, :history_size].clone()
    action_roll = actions[:, :history_size].clone()
    pred_by_step = {}
    for step in range(1, horizon + 1):
        pred = _model_step(model, emb_roll, action_roll, history_size)
        emb_roll = torch.cat([emb_roll, pred], dim=1)
        pred_by_step[step] = pred[:, 0]
        next_action_idx = history_size + step - 1
        if next_action_idx < actions.shape[1]:
            action_roll = torch.cat([action_roll, actions[:, next_action_idx:next_action_idx + 1]], dim=1)

    row: Dict[str, object] = {
        "model": model_name,
        "checkpoint_path": str(checkpoint_path),
        "latent_dim": int(true_emb.shape[-1]),
        "window_idx": int(window_idx),
        "start_row": int(start_row),
        "goal_row": int(goal_row),
        "horizon": int(horizon),
    }
    mse_values = []
    l2_values = []
    norm_l2_values = []
    for step in range(1, horizon + 1):
        pred = pred_by_step[step]
        target_idx = history_size - 1 + step
        true = true_emb[:, target_idx]
        err = pred - true
        mse = err.pow(2).mean(dim=-1)
        l2 = torch.norm(err, dim=-1)
        true_norm = torch.norm(true, dim=-1)
        normalized_l2 = l2 / (true_norm + 1e-8)
        row[f"latent_mse_step_{step}"] = float(mse.mean().detach().cpu())
        row[f"latent_l2_step_{step}"] = float(l2.mean().detach().cpu())
        row[f"normalized_l2_step_{step}"] = float(normalized_l2.mean().detach().cpu())
        mse_values.append(mse)
        l2_values.append(l2)
        norm_l2_values.append(normalized_l2)

    terminal_mse = mse_values[-1].mean()
    terminal_l2 = l2_values[-1].mean()
    row["terminal_latent_mse"] = float(terminal_mse.detach().cpu())
    row["terminal_latent_l2"] = float(terminal_l2.detach().cpu())
    row["normalized_terminal_l2"] = float(norm_l2_values[-1].mean().detach().cpu())
    row["mean_multistep_latent_mse"] = float(torch.stack(mse_values).mean().detach().cpu())
    row["mean_multistep_latent_l2"] = float(torch.stack(l2_values).mean().detach().cpu())
    row["normalized_mean_multistep_l2"] = float(torch.stack(norm_l2_values).mean().detach().cpu())
    row["per_dim_terminal_mse"] = row["terminal_latent_mse"]
    row["per_dim_mean_multistep_mse"] = row["mean_multistep_latent_mse"]
    future_true = true_emb[:, history_size:history_size + horizon].reshape(-1, true_emb.shape[-1])
    variance = torch.var(future_true.float(), dim=0, unbiased=False).mean()
    row["variance_normalized_mean_mse"] = float((torch.stack(mse_values).mean() / (variance + 1e-8)).detach().cpu())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out multi-step latent prediction loss diagnostic.")
    parser.add_argument("--models", nargs="+", required=True, help="List of model=object_checkpoint.ckpt")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--num_windows", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--goal_mode", choices=["eval_offset", "episode_final"], default="eval_offset")
    parser.add_argument("--goal_offset_steps", type=int, default=25)
    parser.add_argument("--frameskip", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="rollout_results/prediction_loss")
    args = parser.parse_args()

    try:
        import hdf5plugin  # noqa: F401
        print("[pred_loss] hdf5plugin available; compressed HDF5 filters enabled.")
    except ImportError:
        print("[pred_loss] hdf5plugin not available; continuing with default HDF5 filters.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    rows: List[Dict[str, object]] = []
    model_specs = _parse_models(args.models)

    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, ["pixels", "observation/pixels"])
        action_key = _find_key(h5, ["action", "actions"])
        episode_key = _find_key(h5, ["episode_idx", "ep_idx"])
        step_key = _find_key(h5, ["step_idx"])
        actions_raw = np.asarray(h5[action_key]).astype(np.float32)
        action_mean, action_std = _action_normalization(actions_raw)
        actions_norm = ((actions_raw - action_mean) / action_std).astype(np.float32)
        episode_idx = np.asarray(h5[episode_key]).reshape(-1)
        step_idx = np.asarray(h5[step_key]).reshape(-1)
        pixels_ds = h5[pixels_key]

        for model_name, checkpoint_path in model_specs.items():
            print(f"[pred_loss] loading {model_name}: {checkpoint_path}", flush=True)
            model = _load_model(checkpoint_path, device)
            frameskip = _infer_frameskip(model, Path(args.dataset), args.frameskip)
            history_size = int(getattr(getattr(model, "predictor", None), "pos_embedding").shape[1])
            contexts = _sample_contexts(
                episode_idx,
                step_idx,
                history_size,
                frameskip,
                args.horizon,
                args.num_windows,
                args.goal_mode,
                args.goal_offset_steps,
                np.random.default_rng(args.seed),
            )
            with torch.no_grad():
                for window_idx, (start_row, goal_row, context_rows, _goal_source) in enumerate(contexts):
                    row = _rollout_prediction_loss_for_window(
                        model,
                        pixels_ds,
                        actions_norm,
                        start_row,
                        goal_row,
                        context_rows,
                        frameskip,
                        args.horizon,
                        args.img_size,
                        device,
                        model_name,
                        checkpoint_path,
                        window_idx,
                    )
                    if window_idx == 0:
                        print(
                            f"[pred_loss] {model_name} shapes: context_rows={context_rows.shape}, "
                            f"horizon={args.horizon}, latent_dim={row['latent_dim']}, frameskip={frameskip}",
                            flush=True,
                        )
                    rows.append(row)

    per_window_path = output_dir / "prediction_loss_per_window.csv"
    summary_path = output_dir / "prediction_loss_summary.csv"
    _write_csv(per_window_path, rows)
    _write_csv(summary_path, _aggregate(rows, ["model", "checkpoint_path", "latent_dim", "horizon"]))
    print(f"[pred_loss] wrote {per_window_path}")
    print(f"[pred_loss] wrote {summary_path}")


if __name__ == "__main__":
    main()
