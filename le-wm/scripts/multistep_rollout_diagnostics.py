from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_SPEARMAN_WARNED = False


def _metric(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def _load_model(checkpoint: Path, device: torch.device):
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    model = obj.model if hasattr(obj, "model") else obj
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def _find_key(h5: h5py.File, candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in h5:
            return key
    raise KeyError(f"None of these keys were found in dataset: {list(candidates)}")


def _normalize_actions(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    flat = action.reshape(-1, action.shape[-1]).astype(np.float32)
    valid = ~np.isnan(flat).any(axis=1)
    mean = flat[valid].mean(axis=0, keepdims=True)
    std = flat[valid].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((action - mean) / std).astype(np.float32), mean, std


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


def _preprocess_pixels(pixels: torch.Tensor, img_size: int, device: torch.device) -> torch.Tensor:
    batch, steps = pixels.shape[:2]
    flat = pixels.reshape(batch * steps, *pixels.shape[2:]).to(device)
    flat = _to_chw_float(flat)
    if flat.shape[-2:] != (img_size, img_size):
        flat = F.interpolate(flat, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    flat = (flat - mean) / std
    return flat.reshape(batch, steps, *flat.shape[1:])


def _candidate_starts(
    episode_idx: np.ndarray,
    step_idx: np.ndarray,
    raw_window_len: int,
    max_candidate_starts: int,
) -> List[Tuple[np.ndarray, int]]:
    candidates = []
    for ep in np.unique(episode_idx):
        rows = np.where(episode_idx == ep)[0]
        rows = rows[np.argsort(step_idx[rows])]
        if len(rows) < raw_window_len:
            continue
        for start in range(0, len(rows) - raw_window_len + 1):
            if step_idx[rows[start + raw_window_len - 1]] - step_idx[rows[start]] == raw_window_len - 1:
                if np.all(np.diff(step_idx[rows[start:start + raw_window_len]]) == 1):
                    candidates.append((rows, start))
    if len(candidates) > max_candidate_starts:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(candidates), size=max_candidate_starts, replace=False)
        candidates = [candidates[i] for i in keep]
    return candidates


def _read_pixels_windows(pixels_ds, idx: np.ndarray) -> np.ndarray:
    flat_idx = idx.reshape(-1)
    unique_idx, inverse = np.unique(flat_idx, return_inverse=True)
    unique_pixels = np.asarray(pixels_ds[unique_idx])
    flat_pixels = unique_pixels[inverse]
    return flat_pixels.reshape(*idx.shape, *flat_pixels.shape[1:])


def _make_batches(
    dataset_path: Path,
    batch_size: int,
    num_batches: int,
    model_window_len: int,
    frameskip: int,
    max_candidate_starts: int,
):
    try:
        import hdf5plugin  # noqa: F401
        print("[rollout] hdf5plugin available; compressed HDF5 filters enabled.", flush=True)
    except ImportError:
        print("[rollout] hdf5plugin not available; continuing with default HDF5 filters.", flush=True)

    print(f"[rollout] opening dataset: {dataset_path}", flush=True)
    with h5py.File(dataset_path, "r") as h5:
        print("[rollout] dataset keys and shapes:", flush=True)
        for key in h5.keys():
            obj = h5[key]
            shape = getattr(obj, "shape", None)
            print(f"  - {key}: {shape}", flush=True)

        pixels_key = _find_key(h5, ["pixels", "observation/pixels"])
        action_key = _find_key(h5, ["action", "actions"])
        episode_key = _find_key(h5, ["episode_idx", "ep_idx"])
        step_key = _find_key(h5, ["step_idx"])

        pixels_ds = h5[pixels_key]
        actions = np.asarray(h5[action_key]).astype(np.float32)
        episode_idx = np.asarray(h5[episode_key]).reshape(-1)
        step_idx = np.asarray(h5[step_key]).reshape(-1)
        actions, _, _ = _normalize_actions(actions)

        raw_window_len = model_window_len * frameskip
        print(
            f"[rollout] model_window_len={model_window_len}, frameskip={frameskip}, "
            f"raw_window_len={raw_window_len}",
            flush=True,
        )
        candidates = _candidate_starts(episode_idx, step_idx, raw_window_len, max_candidate_starts)
        print(f"[rollout] candidate starts found: {len(candidates)}", flush=True)
        if len(candidates) == 0:
            raise ValueError(f"No contiguous raw windows of length {raw_window_len} found in {dataset_path}")

        rng = np.random.default_rng(0)
        needed = min(len(candidates), batch_size * num_batches)
        selected = rng.choice(len(candidates), size=needed, replace=False)

        for batch_idx, offset in enumerate(range(0, needed, batch_size), start=1):
            selected_idx = selected[offset:offset + batch_size]
            if selected_idx.size == 0:
                continue
            rows = [candidates[i][0] for i in selected_idx]
            starts = [candidates[i][1] for i in selected_idx]
            pixel_idx = np.stack(
                [row[start + np.arange(model_window_len) * frameskip] for row, start in zip(rows, starts)],
                axis=0,
            )
            action_idx = np.stack(
                [
                    np.stack(
                        [row[start + t * frameskip:start + (t + 1) * frameskip] for t in range(model_window_len)],
                        axis=0,
                    )
                    for row, start in zip(rows, starts)
                ],
                axis=0,
            )
            print(
                f"[rollout] yielding batch {batch_idx}/{int(np.ceil(needed / batch_size))} "
                f"with {pixel_idx.shape[0]} windows",
                flush=True,
            )
            pixels_batch = _read_pixels_windows(pixels_ds, pixel_idx)
            actions_batch = actions[action_idx].reshape(
                action_idx.shape[0], action_idx.shape[1], frameskip * actions.shape[-1]
            )
            yield {
                "pixels": torch.from_numpy(pixels_batch),
                "action": torch.from_numpy(actions_batch),
            }

            if batch_idx >= num_batches:
                break


def _apply_transition_bottleneck(model, pred_raw: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    tb = getattr(model, "transition_bottleneck", None)
    if tb is None:
        return pred_raw
    delta_raw = pred_raw - anchor
    if tb.__class__.__name__ == "StateConditionedTangentBottleneck":
        delta_rec, _, _ = tb(delta_raw, anchor)
    else:
        delta_rec, _ = tb(delta_raw)
    return anchor + delta_rec


def _apply_oracle_chart_bottleneck(tb, pred_raw: torch.Tensor, oracle_anchor: torch.Tensor, true_anchor: torch.Tensor):
    delta_raw = pred_raw - oracle_anchor
    batch, steps, dim = delta_raw.shape
    delta_flat = delta_raw.reshape(batch * steps, dim)
    code_flat = tb.code_net(tb.delta_norm(delta_flat))
    basis = _basis_from_anchor(tb, true_anchor).reshape(batch * steps, dim, tb.tangent_dim)
    delta_rec_flat = torch.einsum("bdk,bk->bd", basis, code_flat)
    delta_rec = delta_rec_flat.reshape(batch, steps, dim)
    return oracle_anchor + delta_rec


def _basis_from_anchor(tb, anchor: torch.Tensor) -> torch.Tensor:
    if tb is None or tb.__class__.__name__ != "StateConditionedTangentBottleneck":
        raise ValueError("chart drift is only available for StateConditionedTangentBottleneck")
    if anchor.dim() == 2:
        anchor = anchor.unsqueeze(1)
    batch, steps, dim = anchor.shape
    flat = anchor.reshape(batch * steps, dim)
    basis = tb.basis_net(tb.anchor_norm(flat)).reshape(batch * steps, dim, tb.tangent_dim)
    if tb.basis_normalization == "column_norm":
        basis = F.normalize(basis, dim=1, eps=1e-8)
    return basis.reshape(batch, steps, dim, tb.tangent_dim)


def _projector_distance_from_bases(basis_a: torch.Tensor, basis_b: torch.Tensor) -> torch.Tensor:
    qa, _ = torch.linalg.qr(basis_a.float(), mode="reduced")
    qb, _ = torch.linalg.qr(basis_b.float(), mode="reduced")
    k = qa.shape[-1]
    overlap = torch.einsum("bdk,bdl->bkl", qa, qb).pow(2).sum(dim=(1, 2))
    return torch.sqrt(torch.clamp(2 * k - 2 * overlap, min=0.0) / (2 * k + 1e-8))


def _expected_action_dim(model) -> int:
    patch_embed = getattr(getattr(model, "action_encoder", None), "patch_embed", None)
    if patch_embed is None or not hasattr(patch_embed, "in_channels"):
        raise ValueError("Could not infer expected action dimension from model.action_encoder.patch_embed")
    return int(patch_embed.in_channels)


def _raw_action_dim(dataset_path: Path) -> int:
    with h5py.File(dataset_path, "r") as h5:
        action_key = _find_key(h5, ["action", "actions"])
        return int(h5[action_key].shape[-1])


def _infer_frameskip(model, dataset_path: Path, frameskip_arg: int | None) -> int:
    if frameskip_arg is not None:
        return int(frameskip_arg)
    expected_dim = _expected_action_dim(model)
    raw_dim = _raw_action_dim(dataset_path)
    if expected_dim % raw_dim != 0:
        raise ValueError(
            f"Cannot infer frameskip: model expects action dim {expected_dim}, "
            f"raw dataset action dim is {raw_dim}."
        )
    frameskip = expected_dim // raw_dim
    print(
        f"[rollout] inferred frameskip={frameskip} from expected_action_dim={expected_dim} "
        f"and raw_action_dim={raw_dim}",
        flush=True,
    )
    return frameskip


def _model_step(model, emb_seq: torch.Tensor, action_seq: torch.Tensor, history_size: int) -> torch.Tensor:
    act_emb = model.action_encoder(action_seq)
    ctx_emb = emb_seq[:, -history_size:]
    ctx_act = act_emb[:, -history_size:]
    pred_raw = model.predict(ctx_emb, ctx_act)[:, -1:]
    anchor = ctx_emb[:, -1:]
    return _apply_transition_bottleneck(model, pred_raw, anchor)


def _oracle_chart_step(
    model,
    emb_seq: torch.Tensor,
    action_seq: torch.Tensor,
    true_anchor: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    tb = getattr(model, "transition_bottleneck", None)
    if tb is None or tb.__class__.__name__ != "StateConditionedTangentBottleneck":
        raise ValueError("oracle chart rollout requires StateConditionedTangentBottleneck")
    act_emb = model.action_encoder(action_seq)
    ctx_emb = emb_seq[:, -history_size:]
    ctx_act = act_emb[:, -history_size:]
    pred_raw = model.predict(ctx_emb, ctx_act)[:, -1:]
    oracle_anchor = ctx_emb[:, -1:]
    return _apply_oracle_chart_bottleneck(tb, pred_raw, oracle_anchor, true_anchor)


def _rollout_batch(
    model,
    batch: Dict[str, torch.Tensor],
    horizons: List[int],
    img_size: int,
    device: torch.device,
    oracle_chart: bool,
):
    max_h = max(horizons)
    history_size = int(getattr(getattr(model, "predictor", None), "pos_embedding").shape[1])
    pixels = _preprocess_pixels(batch["pixels"], img_size, device)
    actions = batch["action"].float().to(device)
    encoded = model.encode({"pixels": pixels, "action": actions})
    true_emb = encoded["emb"]

    emb_roll = true_emb[:, :history_size].clone()
    action_roll = actions[:, :history_size].clone()
    pred_by_h = {}
    tb = getattr(model, "transition_bottleneck", None)
    run_oracle = oracle_chart and tb is not None and tb.__class__.__name__ == "StateConditionedTangentBottleneck"
    oracle_roll = true_emb[:, :history_size].clone() if run_oracle else None
    oracle_action_roll = actions[:, :history_size].clone() if run_oracle else None
    oracle_by_h = {}
    for step in range(1, max_h + 1):
        next_pred = _model_step(model, emb_roll, action_roll, history_size)
        emb_roll = torch.cat([emb_roll, next_pred], dim=1)
        pred_by_h[step] = next_pred[:, 0]
        if run_oracle:
            true_anchor_idx = history_size + step - 2
            true_anchor = true_emb[:, true_anchor_idx:true_anchor_idx + 1]
            next_oracle = _oracle_chart_step(
                model,
                oracle_roll,
                oracle_action_roll,
                true_anchor,
                history_size,
            )
            oracle_roll = torch.cat([oracle_roll, next_oracle], dim=1)
            oracle_by_h[step] = next_oracle[:, 0]
        next_action_idx = history_size + step - 1
        if next_action_idx < actions.shape[1]:
            action_roll = torch.cat([action_roll, actions[:, next_action_idx:next_action_idx + 1]], dim=1)
            if run_oracle:
                oracle_action_roll = torch.cat(
                    [oracle_action_roll, actions[:, next_action_idx:next_action_idx + 1]], dim=1
                )
    return true_emb, pred_by_h, oracle_by_h, history_size, run_oracle, actions


def _accumulate_metrics(
    store: Dict[str, list],
    model,
    true_emb,
    pred_by_h,
    oracle_by_h,
    history_size,
    horizons,
    oracle_requested: bool,
):
    anchor = true_emb[:, history_size - 1]
    tb = getattr(model, "transition_bottleneck", None)
    is_state_tangent = tb is not None and tb.__class__.__name__ == "StateConditionedTangentBottleneck"
    for h in horizons:
        target_idx = history_size - 1 + h
        if target_idx >= true_emb.shape[1] or h not in pred_by_h:
            continue
        pred = pred_by_h[h]
        true = true_emb[:, target_idx]
        pred_delta = pred - anchor
        true_delta = true - anchor
        true_norm = torch.norm(true_delta, dim=-1)
        store.setdefault(f"rollout/mse_h{h}", []).append((pred - true).pow(2).mean())
        store.setdefault(f"rollout/relative_error_h{h}", []).append(
            (torch.norm(pred - true, dim=-1) / (true_norm + 1e-8)).mean()
        )
        store.setdefault(f"rollout/cosine_transition_h{h}", []).append(
            F.cosine_similarity(pred_delta, true_delta, dim=-1, eps=1e-8).mean()
        )
        store.setdefault(f"rollout/norm_ratio_h{h}", []).append(
            (torch.norm(pred_delta, dim=-1) / (true_norm + 1e-8)).mean()
        )
        normal_mse = (pred - true).pow(2).mean()
        normal_cos = F.cosine_similarity(pred_delta, true_delta, dim=-1, eps=1e-8).mean()
        if oracle_requested:
            store.setdefault(f"oracle_chart/normal_mse_h{h}", []).append(normal_mse)
            store.setdefault(f"oracle_chart/normal_cos_h{h}", []).append(normal_cos)
        if is_state_tangent:
            basis_true = _basis_from_anchor(tb, true.unsqueeze(1))[:, 0]
            basis_pred = _basis_from_anchor(tb, pred.unsqueeze(1))[:, 0]
            chart_drift = _projector_distance_from_bases(basis_true, basis_pred).mean()
            store.setdefault(f"chart_drift/projector_distance_h{h}", []).append(chart_drift)
            if oracle_requested:
                store.setdefault(f"oracle_chart/chart_drift_h{h}", []).append(chart_drift)
        if h in oracle_by_h:
            oracle_pred = oracle_by_h[h]
            oracle_delta = oracle_pred - anchor
            oracle_mse = (oracle_pred - true).pow(2).mean()
            oracle_cos = F.cosine_similarity(oracle_delta, true_delta, dim=-1, eps=1e-8).mean()
            store.setdefault(f"oracle_chart/oracle_mse_h{h}", []).append(oracle_mse)
            store.setdefault(f"oracle_chart/mse_gap_h{h}", []).append(normal_mse - oracle_mse)
            store.setdefault(f"oracle_chart/oracle_cos_h{h}", []).append(oracle_cos)
            store.setdefault(f"oracle_chart/cos_gap_h{h}", []).append(oracle_cos - normal_cos)


def _mean_store(store: Dict[str, list]) -> Dict[str, float]:
    metrics = {}
    for key, values in sorted(store.items()):
        if len(values) == 0:
            continue
        if values[0].dim() == 0:
            reduced = torch.nanmean(torch.stack(values).float())
        else:
            reduced = torch.nanmean(torch.cat([value.reshape(-1) for value in values], dim=0).float())
        metrics[key] = _metric(reduced)
    return metrics


def _fmt_metric(value, width: int = 10):
    if value is None or (isinstance(value, float) and value != value):
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.5f}"


def _spearman_corr(x: torch.Tensor, y: torch.Tensor):
    global _SPEARMAN_WARNED
    try:
        from scipy.stats import spearmanr
    except ImportError:
        if not _SPEARMAN_WARNED:
            print("[rollout] scipy not available; skipping expert_goal Spearman metrics.", flush=True)
            _SPEARMAN_WARNED = True
        return None
    x_np = x.detach().float().cpu().numpy()
    y_np = y.detach().float().cpu().numpy()
    if x_np.size < 2 or np.std(x_np) < 1e-12 or np.std(y_np) < 1e-12:
        return None
    corr = spearmanr(x_np, y_np, nan_policy="omit").correlation
    if corr is None or np.isnan(corr):
        return None
    return torch.tensor(float(corr), device=x.device)


def _accumulate_expert_goal_metrics(
    store: Dict[str, list],
    true_emb: torch.Tensor,
    pred_by_h: Dict[int, torch.Tensor],
    history_size: int,
    horizons: List[int],
    goal_offset: int,
    goal_ratio_epsilon: float,
):
    goal_idx = history_size - 1 + goal_offset
    if goal_idx >= true_emb.shape[1]:
        return
    z_goal = true_emb[:, goal_idx]
    for h in horizons:
        target_idx = history_size - 1 + h
        if target_idx >= true_emb.shape[1] or h not in pred_by_h:
            continue
        pred = pred_by_h[h]
        true = true_emb[:, target_idx]
        pred_goal_dist = torch.norm(pred - z_goal, dim=-1)
        true_goal_dist = torch.norm(true - z_goal, dim=-1)
        store.setdefault(f"expert_goal/pred_goal_dist_h{h}", []).append(pred_goal_dist.detach())
        store.setdefault(f"expert_goal/true_goal_dist_h{h}", []).append(true_goal_dist.detach())
        store.setdefault(f"expert_goal/goal_dist_error_h{h}", []).append(
            torch.abs(pred_goal_dist - true_goal_dist).detach()
        )
        valid_ratio = true_goal_dist > goal_ratio_epsilon
        valid_frac = valid_ratio.float().mean()
        store.setdefault(f"expert_goal/goal_dist_ratio_valid_frac_h{h}", []).append(valid_frac.detach())
        if valid_ratio.any():
            ratio = pred_goal_dist[valid_ratio] / (true_goal_dist[valid_ratio] + 1e-8)
            store.setdefault(f"expert_goal/goal_dist_ratio_h{h}", []).append(ratio.detach())
        else:
            store.setdefault(f"expert_goal/goal_dist_ratio_h{h}", []).append(
                torch.full((), float("nan"), device=true_emb.device)
            )
        spearman = _spearman_corr(pred_goal_dist, true_goal_dist)
        if spearman is not None:
            store.setdefault(f"expert_goal/goal_dist_spearman_h{h}", []).append(spearman)


def _rollout_with_actions(
    model,
    true_emb: torch.Tensor,
    actions: torch.Tensor,
    horizons: List[int],
    history_size: int,
) -> Dict[int, torch.Tensor]:
    max_h = max(horizons)
    emb_roll = true_emb[:, :history_size].clone()
    action_roll = actions[:, :history_size].clone()
    pred_by_h = {}
    for step in range(1, max_h + 1):
        next_pred = _model_step(model, emb_roll, action_roll, history_size)
        emb_roll = torch.cat([emb_roll, next_pred], dim=1)
        pred_by_h[step] = next_pred[:, 0]
        next_action_idx = history_size + step - 1
        if next_action_idx < actions.shape[1]:
            action_roll = torch.cat([action_roll, actions[:, next_action_idx:next_action_idx + 1]], dim=1)
    return pred_by_h


def _accumulate_action_sensitivity_metrics(
    store: Dict[str, list],
    model,
    true_emb: torch.Tensor,
    expert_pred_by_h: Dict[int, torch.Tensor],
    actions: torch.Tensor,
    history_size: int,
    horizons: List[int],
    goal_offset: int,
    num_action_perturbations: int,
    action_noise_sigma: float,
    action_noise_clip: float,
):
    if num_action_perturbations <= 0:
        return
    goal_idx = history_size - 1 + goal_offset
    if goal_idx >= true_emb.shape[1]:
        return
    z_goal = true_emb[:, goal_idx]
    perturb_dists = {h: [] for h in horizons}
    for _ in range(num_action_perturbations):
        perturbed_actions = actions + action_noise_sigma * torch.randn_like(actions)
        perturbed_actions = torch.clamp(perturbed_actions, -action_noise_clip, action_noise_clip)
        perturb_pred_by_h = _rollout_with_actions(model, true_emb, perturbed_actions, horizons, history_size)
        for h in horizons:
            if h in perturb_pred_by_h:
                perturb_dists[h].append(torch.norm(perturb_pred_by_h[h] - z_goal, dim=-1))

    for h in horizons:
        if h not in expert_pred_by_h or len(perturb_dists[h]) == 0:
            continue
        expert_goal_dist = torch.norm(expert_pred_by_h[h] - z_goal, dim=-1)
        perturb_stack = torch.stack(perturb_dists[h], dim=1)
        perturb_mean_per_sample = perturb_stack.mean(dim=1)
        store.setdefault(f"action_sensitivity/expert_goal_dist_h{h}", []).append(expert_goal_dist.detach())
        store.setdefault(f"action_sensitivity/perturb_goal_dist_mean_h{h}", []).append(
            perturb_stack.reshape(-1).detach()
        )
        store.setdefault(f"action_sensitivity/perturb_minus_expert_h{h}", []).append(
            (perturb_mean_per_sample - expert_goal_dist).detach()
        )
        store.setdefault(f"action_sensitivity/action_sensitivity_h{h}", []).append(
            perturb_stack.std(dim=1, unbiased=False).detach()
        )
        store.setdefault(f"action_sensitivity/goal_rank_consistency_h{h}", []).append(
            (perturb_stack > expert_goal_dist.unsqueeze(1)).float().mean(dim=1).detach()
        )


def _print_summary(metrics: Dict[str, float], horizons: List[int]):
    print("\nhorizon | mse | rel_err | cos_transition | norm_ratio | chart_drift")
    print("-" * 76)
    for h in horizons:
        print(
            f"{h:>7} | "
            f"{metrics.get(f'rollout/mse_h{h}', float('nan')):>8.5f} | "
            f"{metrics.get(f'rollout/relative_error_h{h}', float('nan')):>8.5f} | "
            f"{metrics.get(f'rollout/cosine_transition_h{h}', float('nan')):>14.5f} | "
            f"{metrics.get(f'rollout/norm_ratio_h{h}', float('nan')):>10.5f} | "
            f"{metrics.get(f'chart_drift/projector_distance_h{h}', float('nan')):>11.5f}"
        )


def _print_expert_goal_summary(metrics: Dict[str, float], horizons: List[int]):
    print("\nhorizon | pred_goal_dist | true_goal_dist | goal_dist_error | goal_dist_ratio | spearman")
    print("-" * 88)
    for h in horizons:
        print(
            f"{h:>7} | "
            f"{_fmt_metric(metrics.get(f'expert_goal/pred_goal_dist_h{h}'), width=14)} | "
            f"{_fmt_metric(metrics.get(f'expert_goal/true_goal_dist_h{h}'), width=14)} | "
            f"{_fmt_metric(metrics.get(f'expert_goal/goal_dist_error_h{h}'), width=15)} | "
            f"{_fmt_metric(metrics.get(f'expert_goal/goal_dist_ratio_h{h}'), width=15)} | "
            f"{_fmt_metric(metrics.get(f'expert_goal/goal_dist_spearman_h{h}'), width=8)}"
        )


def _print_action_sensitivity_summary(metrics: Dict[str, float], horizons: List[int]):
    print(
        "\nhorizon | expert_goal_dist | perturb_goal_dist_mean | perturb_minus_expert | "
        "action_sensitivity | goal_rank_consistency"
    )
    print("-" * 121)
    for h in horizons:
        print(
            f"{h:>7} | "
            f"{_fmt_metric(metrics.get(f'action_sensitivity/expert_goal_dist_h{h}'), width=16)} | "
            f"{_fmt_metric(metrics.get(f'action_sensitivity/perturb_goal_dist_mean_h{h}'), width=22)} | "
            f"{_fmt_metric(metrics.get(f'action_sensitivity/perturb_minus_expert_h{h}'), width=21)} | "
            f"{_fmt_metric(metrics.get(f'action_sensitivity/action_sensitivity_h{h}'), width=18)} | "
            f"{_fmt_metric(metrics.get(f'action_sensitivity/goal_rank_consistency_h{h}'), width=21)}"
        )


def _fmt(value, width: int = 10):
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.5f}"


def _print_oracle_summary(metrics: Dict[str, float], horizons: List[int], oracle_available: bool):
    print("\nhorizon | normal_mse | oracle_mse | mse_gap | normal_cos | oracle_cos | cos_gap | chart_drift")
    print("-" * 101)
    for h in horizons:
        oracle_mse = metrics.get(f"oracle_chart/oracle_mse_h{h}")
        oracle_cos = metrics.get(f"oracle_chart/oracle_cos_h{h}")
        mse_gap = metrics.get(f"oracle_chart/mse_gap_h{h}")
        cos_gap = metrics.get(f"oracle_chart/cos_gap_h{h}")
        chart = metrics.get(f"oracle_chart/chart_drift_h{h}")
        print(
            f"{h:>7} | "
            f"{_fmt(metrics.get(f'oracle_chart/normal_mse_h{h}'))} | "
            f"{_fmt(oracle_mse if oracle_available else None)} | "
            f"{_fmt(mse_gap if oracle_available else None, width=7)} | "
            f"{_fmt(metrics.get(f'oracle_chart/normal_cos_h{h}'))} | "
            f"{_fmt(oracle_cos if oracle_available else None)} | "
            f"{_fmt(cos_gap if oracle_available else None, width=7)} | "
            f"{_fmt(chart, width=11)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Offline multi-step latent rollout diagnostics.")
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
    parser.add_argument("--oracle_chart", action="store_true")
    parser.add_argument("--expert_goal_eval", action="store_true")
    parser.add_argument("--goal_offset", type=int, default=8)
    parser.add_argument("--goal_ratio_epsilon", type=float, default=0.1)
    parser.add_argument("--action_sensitivity_eval", action="store_true")
    parser.add_argument("--num_action_perturbations", type=int, default=8)
    parser.add_argument("--action_noise_sigma", type=float, default=0.5)
    parser.add_argument("--action_noise_clip", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    horizons = sorted(set(args.horizons))
    uses_goal_diagnostic = args.expert_goal_eval or args.action_sensitivity_eval
    goal_diagnostic_horizons = sorted(set(horizons + [args.goal_offset])) if uses_goal_diagnostic else horizons
    if uses_goal_diagnostic and args.goal_offset <= 0:
        raise ValueError(f"--goal_offset must be positive, got {args.goal_offset}")
    if args.goal_ratio_epsilon < 0:
        raise ValueError(f"--goal_ratio_epsilon must be non-negative, got {args.goal_ratio_epsilon}")
    if args.action_sensitivity_eval and args.num_action_perturbations <= 0:
        raise ValueError(
            f"--num_action_perturbations must be positive, got {args.num_action_perturbations}"
        )
    if args.action_sensitivity_eval and args.action_noise_sigma < 0:
        raise ValueError(f"--action_noise_sigma must be non-negative, got {args.action_noise_sigma}")
    if args.action_sensitivity_eval and args.action_noise_clip <= 0:
        raise ValueError(f"--action_noise_clip must be positive, got {args.action_noise_clip}")
    max_required_horizon = max(goal_diagnostic_horizons) if uses_goal_diagnostic else max(horizons)
    model = _load_model(Path(args.checkpoint_object), device)
    history_size = int(getattr(model.predictor, "pos_embedding").shape[1])
    window_len = history_size + max_required_horizon
    frameskip = _infer_frameskip(model, Path(args.dataset), args.frameskip)
    store = {}
    num_samples = 0
    tb = getattr(model, "transition_bottleneck", None)
    oracle_available = args.oracle_chart and tb is not None and tb.__class__.__name__ == "StateConditionedTangentBottleneck"
    if args.oracle_chart and not oracle_available:
        print("[rollout] --oracle_chart requested, but model is not state_tangent; running normal rollout only.", flush=True)
    if args.expert_goal_eval:
        print("[expert_goal] goal_offset is in model steps; goal is at horizon=goal_offset.", flush=True)

    with torch.no_grad():
        for batch in _make_batches(
            Path(args.dataset),
            args.batch_size,
            args.num_batches,
            window_len,
            frameskip,
            args.max_candidate_starts,
        ):
            true_emb, pred_by_h, oracle_by_h, history_size, _, actions = _rollout_batch(
                model,
                batch,
                goal_diagnostic_horizons,
                args.img_size,
                device,
                args.oracle_chart,
            )
            _accumulate_metrics(
                store,
                model,
                true_emb,
                pred_by_h,
                oracle_by_h,
                history_size,
                horizons,
                args.oracle_chart,
            )
            if args.expert_goal_eval:
                _accumulate_expert_goal_metrics(
                    store,
                    true_emb,
                    pred_by_h,
                    history_size,
                    goal_diagnostic_horizons,
                    args.goal_offset,
                    args.goal_ratio_epsilon,
                )
            if args.action_sensitivity_eval:
                _accumulate_action_sensitivity_metrics(
                    store,
                    model,
                    true_emb,
                    pred_by_h,
                    actions,
                    history_size,
                    goal_diagnostic_horizons,
                    args.goal_offset,
                    args.num_action_perturbations,
                    args.action_noise_sigma,
                    args.action_noise_clip,
                )
            num_samples += int(batch["pixels"].shape[0])

    metrics = _mean_store(store)
    output = {
        "checkpoint_path": str(Path(args.checkpoint_object)),
        "dataset_path": str(Path(args.dataset)),
        "num_samples": num_samples,
        "horizons": horizons,
        "expert_goal_eval": bool(args.expert_goal_eval),
        "goal_offset": int(args.goal_offset),
        "goal_ratio_epsilon": float(args.goal_ratio_epsilon),
        "goal_diagnostic_horizons": goal_diagnostic_horizons,
        "action_sensitivity_eval": bool(args.action_sensitivity_eval),
        "num_action_perturbations": int(args.num_action_perturbations),
        "action_noise_sigma": float(args.action_noise_sigma),
        "action_noise_clip": float(args.action_noise_clip),
        "oracle_chart_requested": bool(args.oracle_chart),
        "oracle_chart_available": bool(oracle_available),
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    _print_summary(metrics, horizons)
    if args.oracle_chart:
        _print_oracle_summary(metrics, horizons, oracle_available)
    if args.expert_goal_eval:
        _print_expert_goal_summary(metrics, goal_diagnostic_horizons)
    if args.action_sensitivity_eval:
        _print_action_sensitivity_summary(metrics, goal_diagnostic_horizons)
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
