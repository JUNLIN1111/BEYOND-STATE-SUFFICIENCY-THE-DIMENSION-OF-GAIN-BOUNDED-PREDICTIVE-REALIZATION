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
    _load_model,
    _preprocess_pixels,
    _read_rows,
)


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


def _load_margin_windows(path: Path, max_windows: int) -> List[Dict[str, int]]:
    with path.open() as file:
        rows = list(csv.DictReader(file))
    windows: Dict[int, Dict[str, int]] = {}
    for row in rows:
        window_idx = int(row["window_idx"])
        windows.setdefault(
            window_idx,
            {
                "window_idx": window_idx,
                "task_best_idx": int(row["task_best_idx"]),
                "focus_wrong_idx": int(row["focus_wrong_idx"]),
            },
        )
    ordered = list(windows.values())
    if max_windows > 0:
        ordered = ordered[:max_windows]
    return ordered


def _load_raw(path: Path) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    return {key: np.asarray(raw[key]) for key in raw.files}


def _rollout_cost_series_for_model(
    model_name: str,
    checkpoint: Path,
    raw: Dict[str, np.ndarray],
    h5: h5py.File,
    windows: List[Dict[str, int]],
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
    dataset_actions = np.asarray(h5[action_key]).astype(np.float32)
    action_mean, action_std = _action_normalization(dataset_actions)
    dataset_actions_norm = ((dataset_actions - action_mean) / action_std).astype(np.float32)

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for item in windows:
            window_idx = int(item["window_idx"])
            task_best_idx = int(item["task_best_idx"])
            focus_wrong_idx = int(item["focus_wrong_idx"])
            candidate_indices = [task_best_idx, focus_wrong_idx]
            start_row = int(np.asarray(raw["start_rows"])[window_idx])
            goal_row = int(np.asarray(raw["goal_rows"])[window_idx])
            context_rows = start_row - (history_size - 1) * frameskip + np.arange(history_size) * frameskip
            if context_rows[0] < 0 or context_rows[-1] != start_row:
                raise ValueError(f"{model_name}: invalid context_rows={context_rows.tolist()} at window {window_idx}")
            history_chunks_norm = _action_chunks_from_rows(dataset_actions_norm, context_rows[:-1], frameskip)
            future_chunks = np.asarray(raw["candidate_future_actions"], dtype=np.float32)[window_idx, candidate_indices]
            actions_model = _build_model_action_sequence(history_chunks_norm, future_chunks)
            context_pixels = _read_rows(h5[pixels_key], context_rows)
            goal_pixels = _read_rows(h5[pixels_key], np.asarray([goal_row], dtype=np.int64))[0]
            goal_tensor = _preprocess_pixels(goal_pixels[None, None], img_size, device)
            goal_latent = model.encode({"pixels": goal_tensor})["emb"][:, 0]
            context_batch = np.repeat(context_pixels[None], len(candidate_indices), axis=0)
            pixels = _preprocess_pixels(context_batch, img_size, device)
            actions_all = torch.from_numpy(actions_model).float().to(device)
            rollout = model.rollout({"pixels": pixels.unsqueeze(0)}, actions_all.unsqueeze(0), history_size=history_size)
            predicted = rollout["predicted_emb"][0].detach().float()
            start_latent = model.encode({"pixels": pixels})["emb"][:, -1].detach().float()

            for local_idx, candidate_idx in enumerate(candidate_indices):
                label = "task_best" if candidate_idx == task_best_idx else "focus_wrong"
                costs = {0: float(torch.sum((start_latent[local_idx] - goal_latent[0]) ** 2).item())}
                for step_idx in range(predicted.shape[1]):
                    raw_offset = int((step_idx + 1 - history_size) * frameskip)
                    if raw_offset < 0:
                        continue
                    cost = float(torch.sum((predicted[local_idx, step_idx] - goal_latent[0]) ** 2).item())
                    costs[raw_offset] = cost
                ordered_offsets = sorted(offset for offset in costs if offset >= 0)
                preterminal = [costs[offset] for offset in ordered_offsets if offset < max(ordered_offsets)]
                c0 = costs.get(0, float("nan"))
                cH = costs[ordered_offsets[-1]]
                max_pre = max(preterminal) if preterminal else c0
                rows.append(
                    {
                        "model": model_name,
                        "window_idx": window_idx,
                        "candidate_label": label,
                        "candidate_idx": candidate_idx,
                        "task_best_idx": task_best_idx,
                        "focus_wrong_idx": focus_wrong_idx,
                        "c0": c0,
                        "c5": costs.get(5, float("nan")),
                        "c10": costs.get(10, float("nan")),
                        "c15": costs.get(15, float("nan")),
                        "c20": costs.get(20, float("nan")),
                        "c25": costs.get(25, float("nan")),
                        "terminal_cost": cH,
                        "terminal_drop": c0 - cH,
                        "max_preterminal_cost": max_pre,
                        "overshoot": max_pre - c0,
                        "has_overshoot": int(max_pre > c0 + 1e-8),
                        "monotone_nonincreasing": int(all(costs[b] <= costs[a] + 1e-8 for a, b in zip(ordered_offsets[:-1], ordered_offsets[1:]))),
                    }
                )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _paired_margin_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, int], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), int(row["window_idx"])), {})[str(row["candidate_label"])] = row
    out = []
    for (model, window_idx), pair in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        if "task_best" not in pair or "focus_wrong" not in pair:
            continue
        task = pair["task_best"]
        wrong = pair["focus_wrong"]
        out.append(
            {
                "model": model,
                "window_idx": window_idx,
                "task_best_idx": int(task["candidate_idx"]),
                "focus_wrong_idx": int(wrong["candidate_idx"]),
                "task_c25": float(task["c25"]),
                "wrong_c25": float(wrong["c25"]),
                "wrong_minus_task_c25": float(wrong["c25"]) - float(task["c25"]),
                "task_terminal_drop": float(task["terminal_drop"]),
                "wrong_terminal_drop": float(wrong["terminal_drop"]),
                "task_has_overshoot": int(task["has_overshoot"]),
                "wrong_has_overshoot": int(wrong["has_overshoot"]),
                "task_overshoot": float(task["overshoot"]),
                "wrong_overshoot": float(wrong["overshoot"]),
                "task_monotone_nonincreasing": int(task["monotone_nonincreasing"]),
                "wrong_monotone_nonincreasing": int(wrong["monotone_nonincreasing"]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch latent cost overshoot traces for focus-model failure windows.")
    parser.add_argument("--margin_csv", required=True, help="focus_error_margin_by_dim.csv")
    parser.add_argument("--raw_pools", nargs="+", required=True, help="NAME=raw_pool.npz")
    parser.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint_object.ckpt")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--max_windows", type=int, default=25)
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence/failure_candidate_rescoring_state8/overshoot_batch")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    raw_paths = _parse_name_paths(args.raw_pools)
    model_paths = _parse_name_paths(args.models)
    windows = _load_margin_windows(Path(args.margin_csv), args.max_windows)
    raws = {name: _load_raw(path) for name, path in raw_paths.items()}
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows: List[Dict[str, object]] = []
    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "observation/pixels"])
        action_key = _find_key(h5, [args.action_key, "actions"])
        for model_name, checkpoint in model_paths.items():
            if model_name not in raws:
                raise KeyError(f"{model_name!r} missing from --raw_pools")
            print(f"[overshoot] rolling {model_name} on {len(windows)} windows", flush=True)
            rows.extend(
                _rollout_cost_series_for_model(
                    model_name,
                    checkpoint,
                    raws[model_name],
                    h5,
                    windows,
                    pixels_key,
                    action_key,
                    args.frameskip,
                    args.img_size,
                    device,
                )
            )
    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "failure_overshoot_traces.csv", rows)
    _write_csv(output_dir / "failure_overshoot_paired_margins.csv", _paired_margin_rows(rows))
    print(f"[overshoot] wrote outputs under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
