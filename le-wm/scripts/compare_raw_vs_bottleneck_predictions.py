from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from multistep_rollout_diagnostics import (
    _apply_transition_bottleneck,
    _infer_frameskip,
    _load_model,
    _make_batches,
    _metric,
    _preprocess_pixels,
)


def _history_size(model) -> int:
    return int(getattr(model.predictor, "pos_embedding").shape[1])


def _one_step_raw_and_bottleneck(
    model,
    context_emb: torch.Tensor,
    context_action: torch.Tensor,
):
    context_act_emb = model.action_encoder(context_action)
    pred_raw = model.predict(context_emb, context_act_emb)[:, -1:]
    anchor = context_emb[:, -1:]
    pred_bottleneck = _apply_transition_bottleneck(model, pred_raw, anchor)
    return pred_raw, pred_bottleneck, anchor


def _metrics_for_prediction(
    pred: torch.Tensor,
    target: torch.Tensor,
    anchor: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_delta = pred - anchor
    target_delta = target - anchor
    target_norm = torch.norm(target_delta, dim=-1)
    return {
        "mse": (pred - target).pow(2).mean(),
        "cosine": F.cosine_similarity(pred_delta, target_delta, dim=-1, eps=1e-8).mean(),
        "normalized_error": (torch.norm(pred - target, dim=-1) / (target_norm + 1e-8)).mean(),
        "norm_ratio": (torch.norm(pred_delta, dim=-1) / (target_norm + 1e-8)).mean(),
    }


def _append_metrics(store: Dict[str, list], prefix: str, metrics: Dict[str, torch.Tensor]):
    for name, value in metrics.items():
        store.setdefault(f"{prefix}/{name}", []).append(value.detach())


def _teacher_forced_metrics(
    store: Dict[str, list],
    model,
    true_emb: torch.Tensor,
    actions: torch.Tensor,
    horizons: List[int],
    history_size: int,
):
    for h in horizons:
        context_start = h - 1
        context_end = context_start + history_size
        target_idx = context_end
        if target_idx >= true_emb.shape[1]:
            continue
        context_emb = true_emb[:, context_start:context_end]
        context_action = actions[:, context_start:context_end]
        target = true_emb[:, target_idx:target_idx + 1]
        pred_raw, pred_bottleneck, anchor = _one_step_raw_and_bottleneck(model, context_emb, context_action)
        raw_metrics = _metrics_for_prediction(pred_raw, target, anchor)
        bottleneck_metrics = _metrics_for_prediction(pred_bottleneck, target, anchor)
        _append_metrics(store, f"raw_h{h}", raw_metrics)
        _append_metrics(store, f"bottleneck_h{h}", bottleneck_metrics)
        if h == 1:
            _append_metrics(store, "raw", raw_metrics)
            _append_metrics(store, "bottleneck", bottleneck_metrics)


def _rollout_step(model, emb_roll: torch.Tensor, action_roll: torch.Tensor, history_size: int, use_bottleneck: bool):
    context_emb = emb_roll[:, -history_size:]
    context_action = action_roll[:, -history_size:]
    pred_raw, pred_bottleneck, _ = _one_step_raw_and_bottleneck(model, context_emb, context_action)
    return pred_bottleneck if use_bottleneck else pred_raw


def _free_rollout_metrics(
    store: Dict[str, list],
    model,
    true_emb: torch.Tensor,
    actions: torch.Tensor,
    horizons: List[int],
    history_size: int,
):
    max_h = max(horizons)
    raw_roll = true_emb[:, :history_size].clone()
    bottleneck_roll = true_emb[:, :history_size].clone()
    action_roll = actions[:, :history_size].clone()
    initial_anchor = true_emb[:, history_size - 1:history_size]
    raw_by_h = {}
    bottleneck_by_h = {}

    for step in range(1, max_h + 1):
        next_raw = _rollout_step(model, raw_roll, action_roll, history_size, use_bottleneck=False)
        next_bottleneck = _rollout_step(model, bottleneck_roll, action_roll, history_size, use_bottleneck=True)
        raw_roll = torch.cat([raw_roll, next_raw], dim=1)
        bottleneck_roll = torch.cat([bottleneck_roll, next_bottleneck], dim=1)
        raw_by_h[step] = next_raw
        bottleneck_by_h[step] = next_bottleneck
        next_action_idx = history_size + step - 1
        if next_action_idx < actions.shape[1]:
            action_roll = torch.cat([action_roll, actions[:, next_action_idx:next_action_idx + 1]], dim=1)

    for h in horizons:
        target_idx = history_size - 1 + h
        if target_idx >= true_emb.shape[1] or h not in raw_by_h:
            continue
        target = true_emb[:, target_idx:target_idx + 1]
        _append_metrics(
            store,
            f"raw_rollout_h{h}",
            _metrics_for_prediction(raw_by_h[h], target, initial_anchor),
        )
        _append_metrics(
            store,
            f"bottleneck_rollout_h{h}",
            _metrics_for_prediction(bottleneck_by_h[h], target, initial_anchor),
        )


def _mean_store(store: Dict[str, list]) -> Dict[str, float]:
    return {key: _metric(torch.stack(values).mean()) for key, values in sorted(store.items()) if values}


def _print_summary(metrics: Dict[str, float], horizons: List[int]):
    print("\nhorizon | raw_mse | bottleneck_mse | raw_cos | bottleneck_cos | raw_roll_mse | bottleneck_roll_mse")
    print("-" * 101)
    for h in horizons:
        print(
            f"{h:>7} | "
            f"{metrics.get(f'raw_h{h}/mse', float('nan')):>7.5f} | "
            f"{metrics.get(f'bottleneck_h{h}/mse', float('nan')):>14.5f} | "
            f"{metrics.get(f'raw_h{h}/cosine', float('nan')):>7.5f} | "
            f"{metrics.get(f'bottleneck_h{h}/cosine', float('nan')):>14.5f} | "
            f"{metrics.get(f'raw_rollout_h{h}/mse', float('nan')):>12.5f} | "
            f"{metrics.get(f'bottleneck_rollout_h{h}/mse', float('nan')):>19.5f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare raw predictor output vs transition-bottleneck output.")
    parser.add_argument("--checkpoint_object", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--frameskip", type=int, default=None)
    parser.add_argument("--max_candidate_starts", type=int, default=200000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    horizons = sorted(set(args.horizons))
    model = _load_model(Path(args.checkpoint_object), device)
    history_size = _history_size(model)
    window_len = history_size + max(horizons)
    frameskip = _infer_frameskip(model, Path(args.dataset), args.frameskip)
    tb = getattr(model, "transition_bottleneck", None)
    print(
        "[compare] "
        f"transition_bottleneck_exists={tb is not None}, "
        f"class={tb.__class__.__name__ if tb is not None else 'None'}, "
        f"history_size={history_size}, window_len={window_len}",
        flush=True,
    )

    store = {}
    num_samples = 0
    with torch.no_grad():
        for batch in _make_batches(
            Path(args.dataset),
            args.batch_size,
            args.num_batches,
            window_len,
            frameskip,
            args.max_candidate_starts,
        ):
            pixels = _preprocess_pixels(batch["pixels"], args.img_size, device)
            actions = batch["action"].float().to(device)
            encoded = model.encode({"pixels": pixels, "action": actions})
            true_emb = encoded["emb"]
            _teacher_forced_metrics(store, model, true_emb, actions, horizons, history_size)
            _free_rollout_metrics(store, model, true_emb, actions, horizons, history_size)
            num_samples += int(batch["pixels"].shape[0])

    metrics = _mean_store(store)
    output = {
        "checkpoint_path": str(Path(args.checkpoint_object)),
        "dataset_path": str(Path(args.dataset)),
        "num_samples": num_samples,
        "horizons": horizons,
        "transition_bottleneck_exists": tb is not None,
        "transition_bottleneck_class": tb.__class__.__name__ if tb is not None else None,
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    _print_summary(metrics, horizons)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
