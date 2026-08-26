from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from learned_branching_maze_gain import (
    EPS32,
    EPS64,
    LearnedMazeModel,
    all_unordered_pairs,
    normalize_coords,
    torch_load_checkpoint,
)


def _dtype_from_config(config: Dict[str, object]) -> torch.dtype:
    dtype_name = str(config.get("dtype", "float64"))
    return torch.float32 if dtype_name == "float32" else torch.float64


def _quantiles(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p1": float(np.quantile(values, 0.01)),
        f"{prefix}_p5": float(np.quantile(values, 0.05)),
        f"{prefix}_median": float(np.quantile(values, 0.50)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_max": float(np.max(values)),
    }


def _nearest_state_accuracy(decoded_next: torch.Tensor, coords_t: torch.Tensor, true_next: torch.Tensor, chunk: int) -> float:
    flat_decoded = decoded_next.reshape(-1, decoded_next.shape[-1])
    flat_true = true_next.reshape(-1)
    correct = 0
    total = int(flat_decoded.shape[0])
    for start in range(0, total, int(chunk)):
        block = flat_decoded[start : start + int(chunk)]
        d2 = torch.sum(torch.square(block[:, None, :] - coords_t[None, :, :]), dim=-1)
        pred = torch.argmin(d2, dim=-1)
        correct += int(torch.sum(pred == flat_true[start : start + int(chunk)]).detach().cpu().item())
    return float(correct / max(total, 1))


def _nearest_state_predictions(decoded: torch.Tensor, coords_t: torch.Tensor, chunk: int) -> torch.Tensor:
    total = int(decoded.shape[0])
    preds = []
    for start in range(0, total, int(chunk)):
        block = decoded[start : start + int(chunk)]
        d2 = torch.sum(torch.square(block[:, None, :] - coords_t[None, :, :]), dim=-1)
        preds.append(torch.argmin(d2, dim=-1))
    return torch.cat(preds, dim=0)


def _round_normalized_to_lattice(decoded: np.ndarray, coords_lattice: np.ndarray) -> np.ndarray:
    coords_lattice = np.asarray(coords_lattice, dtype=np.float64)
    low = coords_lattice.min(axis=0)
    high = coords_lattice.max(axis=0)
    span = high - low
    out = np.zeros_like(decoded, dtype=np.float64)
    for axis in range(2):
        if span[axis] > 0:
            out[:, axis] = (decoded[:, axis] + 1.0) * span[axis] / 2.0 + low[axis]
        else:
            out[:, axis] = low[axis]
    return np.rint(out).astype(np.int64)


def _rank_diagnostics(z: np.ndarray, err: np.ndarray) -> Dict[str, float]:
    centered = z - np.mean(z, axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        eigvals = np.zeros((centered.shape[1],), dtype=np.float64)
        eigvecs = np.eye(centered.shape[1], dtype=np.float64)
    else:
        cov = centered.T @ centered / float(centered.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

    total = float(np.sum(eigvals))
    if total <= 0.0:
        probs = np.full_like(eigvals, 1.0 / max(len(eigvals), 1), dtype=np.float64)
        cumulative = np.cumsum(probs)
        participation = 0.0
        entropy_rank = 0.0
    else:
        probs = eigvals / total
        cumulative = np.cumsum(probs)
        participation = float(total * total / max(float(np.sum(np.square(eigvals))), 1e-300))
        positive = probs[probs > 0.0]
        entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))

    k95 = int(np.searchsorted(cumulative, 0.95) + 1) if len(cumulative) else 0
    k99 = int(np.searchsorted(cumulative, 0.99) + 1) if len(cumulative) else 0
    flat_err = err.reshape(-1, err.shape[-1])

    def projected_stats(k: int, label: str) -> Dict[str, float]:
        if k <= 0:
            top = np.zeros_like(flat_err)
        else:
            basis = eigvecs[:, :k]
            top = (flat_err @ basis) @ basis.T
        residual = flat_err - top
        top_l2 = np.linalg.norm(top, axis=-1)
        residual_l2 = np.linalg.norm(residual, axis=-1)
        return {
            f"pred_l2_top_{label}_mean": float(np.mean(top_l2)),
            f"pred_l2_top_{label}_q99": float(np.quantile(top_l2, 0.99)),
            f"pred_l2_residual_{label}_mean": float(np.mean(residual_l2)),
            f"pred_l2_residual_{label}_q99": float(np.quantile(residual_l2, 0.99)),
        }

    out = {
        "cov_eig_max": float(eigvals[0]) if len(eigvals) else 0.0,
        "cov_eig_min": float(eigvals[-1]) if len(eigvals) else 0.0,
        "cov_eig_sum": total,
        "participation_effective_rank": participation,
        "entropy_effective_rank": entropy_rank,
        "k95_variance": int(k95),
        "k99_variance": int(k99),
    }
    for idx, value in enumerate(eigvals[: min(16, len(eigvals))], start=1):
        out[f"cov_eig_{idx}"] = float(value)
    out.update(projected_stats(k95, "k95"))
    out.update(projected_stats(k99, "k99"))
    return out


def diagnose_checkpoint(path: Path, device: torch.device, nearest_chunk: int) -> Dict[str, object]:
    payload = torch_load_checkpoint(path)
    model_config = dict(payload["model_config"])
    dtype = _dtype_from_config(model_config)
    eps = EPS32 if dtype == torch.float32 else EPS64
    model = LearnedMazeModel(
        latent_dim=int(model_config["latent_dim"]),
        num_macro_actions=int(model_config["num_macro_actions"]),
        hidden_dim=int(model_config["hidden_dim"]),
        encoder_hidden_layers=int(model_config["encoder_hidden_layers"]),
        predictor_hidden_layers=int(model_config["predictor_hidden_layers"]),
        decoder_hidden_layers=int(model_config["decoder_hidden_layers"]),
        action_embedding_dim=int(model_config["action_embedding_dim"]),
        activation=str(model_config["activation"]),
    ).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    map_payload = dict(payload["map"])
    coords_lattice = np.asarray(map_payload["coords"], dtype=np.int64)
    coords_np = normalize_coords(coords_lattice)
    macro_next_np = np.asarray(payload["macro_next"], dtype=np.int64)
    coords_t = torch.as_tensor(coords_np, dtype=dtype, device=device)
    macro_next_t = torch.as_tensor(macro_next_np, dtype=torch.long, device=device)
    k, n = macro_next_np.shape

    with torch.no_grad():
        z = model.encoder(coords_t)
        recon = model.decoder(z)
        predictions = []
        for alpha in range(k):
            alpha_ids = torch.full((n,), int(alpha), dtype=torch.long, device=device)
            predictions.append(model.predictor(z, alpha_ids))
        z_pred = torch.stack(predictions, dim=0)
        z_next = z[macro_next_t]
        err = z_pred - z_next
        pred_l2 = torch.linalg.norm(err, dim=-1)
        decoded_next = model.decoder(z_pred.reshape(k * n, z.shape[1])).reshape(k, n, 2)
        next_coords = coords_t[macro_next_t]
        decoded_next_rmse = torch.sqrt(torch.mean(torch.sum(torch.square(decoded_next - next_coords), dim=-1)))
        nearest_acc = _nearest_state_accuracy(decoded_next, coords_t, macro_next_t, nearest_chunk)
        current_recon_rmse = torch.sqrt(torch.mean(torch.sum(torch.square(recon - coords_t), dim=-1)))
        current_pred = _nearest_state_predictions(recon, coords_t, nearest_chunk)
        current_true = torch.arange(n, dtype=torch.long, device=device)
        current_exact_acc = float(torch.mean((current_pred == current_true).to(torch.float64)).detach().cpu().item())

    z_np = z.detach().cpu().numpy().astype(np.float64)
    z_pred_np = z_pred.detach().cpu().numpy().astype(np.float64)
    z_next_np = z_next.detach().cpu().numpy().astype(np.float64)
    err_np = z_pred_np - z_next_np
    pred_l2_np = pred_l2.detach().cpu().numpy().astype(np.float64)
    recon_np = recon.detach().cpu().numpy().astype(np.float64)
    latent_dim = int(z_np.shape[1])

    pair_i, pair_j = all_unordered_pairs(n)
    denom = np.linalg.norm(z_np[pair_i] - z_np[pair_j], axis=-1)
    denom = np.maximum(denom, eps)
    denom_p1 = float(np.quantile(denom, 0.01))
    denom_p5 = float(np.quantile(denom, 0.05))
    keep_trim1 = denom > denom_p1
    keep_trimmed = denom > denom_p5
    collision_count = int(np.sum(denom <= eps))
    rounded_lattice = _round_normalized_to_lattice(recon_np, coords_lattice)
    grid_cell_accuracy = float(np.mean(np.all(rounded_lattice == coords_lattice, axis=1)))

    r_true_values: List[np.ndarray] = []
    r_model_values: List[np.ndarray] = []
    ratio_deficit_values: List[np.ndarray] = []
    norm_pair_values: List[np.ndarray] = []
    norm_pair_trim1_values: List[np.ndarray] = []
    norm_pair_trimmed_values: List[np.ndarray] = []
    top_den_candidates: List[np.ndarray] = []
    for alpha in range(k):
        d_true = np.linalg.norm(z_next_np[alpha, pair_i] - z_next_np[alpha, pair_j], axis=-1)
        d_model = np.linalg.norm(z_pred_np[alpha, pair_i] - z_pred_np[alpha, pair_j], axis=-1)
        r_true_values.append(d_true / denom)
        r_model = d_model / denom
        r_model_values.append(r_model)
        ratio_deficit_values.append(np.maximum(d_true / denom - r_model, 0.0))
        state_error = np.linalg.norm(z_pred_np[alpha] - z_next_np[alpha], axis=-1)
        pair_error = np.sqrt((np.square(state_error[pair_i]) + np.square(state_error[pair_j])) / 2.0)
        norm_pair = pair_error / denom
        norm_pair_values.append(norm_pair)
        norm_pair_trim1_values.append(norm_pair[keep_trim1])
        norm_pair_trimmed_values.append(norm_pair[keep_trimmed])
        threshold = float(np.quantile(norm_pair, 0.99))
        top_den_candidates.append(denom[norm_pair >= threshold])

    r_true_all = np.concatenate(r_true_values, axis=0)
    r_model_all = np.concatenate(r_model_values, axis=0)
    ratio_deficit_all = np.concatenate(ratio_deficit_values, axis=0)
    norm_pair_all = np.concatenate(norm_pair_values, axis=0)
    norm_pair_trim1_all = np.concatenate(norm_pair_trim1_values, axis=0)
    norm_pair_trimmed_all = np.concatenate(norm_pair_trimmed_values, axis=0)
    top_den = np.concatenate(top_den_candidates, axis=0)

    row: Dict[str, object] = {
        "checkpoint_path": str(path),
        "checkpoint_kind": str(payload.get("checkpoint_kind", "final")),
        "map_name": str(map_payload.get("name", path.stem)),
        "m": latent_dim,
        "seed": int(path.stem.rsplit("_seed", 1)[-1]) if "_seed" in path.stem else -1,
        "num_states": int(n),
        "num_macro_actions": int(k),
        "current_reconstruction_rmse": float(current_recon_rmse.detach().cpu().item()),
        "exact_current_state_accuracy": current_exact_acc,
        "exact_grid_cell_accuracy": grid_cell_accuracy,
        "current_state_decoding_mistakes": int(n - round(current_exact_acc * n)),
        "latent_collision_count": collision_count,
        "min_pair_distance": float(np.min(denom)),
        "mean_raw_latent_pred_l2": float(np.mean(pred_l2_np)),
        "q95_raw_latent_pred_l2": float(np.quantile(pred_l2_np, 0.95)),
        "q99_raw_latent_pred_l2": float(np.quantile(pred_l2_np, 0.99)),
        "mean_raw_latent_pred_l2_over_sqrt_m": float(np.mean(pred_l2_np) / math.sqrt(latent_dim)),
        "q95_raw_latent_pred_l2_over_sqrt_m": float(np.quantile(pred_l2_np, 0.95) / math.sqrt(latent_dim)),
        "q99_raw_latent_pred_l2_over_sqrt_m": float(np.quantile(pred_l2_np, 0.99) / math.sqrt(latent_dim)),
        "per_coordinate_pred_mse": float(np.mean(np.square(err_np))),
        "decoded_next_rmse": float(decoded_next_rmse.detach().cpu().item()),
        "nearest_state_next_accuracy": nearest_acc,
        "decoded_next_accuracy": nearest_acc,
        "q95_r_true": float(np.quantile(r_true_all, 0.95)),
        "q99_r_true": float(np.quantile(r_true_all, 0.99)),
        "max_r_true": float(np.max(r_true_all)),
        "q99_r_model": float(np.quantile(r_model_all, 0.99)),
        "q99_ratio_deficit": float(np.quantile(ratio_deficit_all, 0.99)),
        "q99_norm_pair_error_now": float(np.quantile(norm_pair_all, 0.99)),
        "q99_norm_pair_error_exclude_bottom1_denom": float(np.quantile(norm_pair_trim1_all, 0.99)),
        "q99_norm_pair_error_exclude_bottom5_denom": float(np.quantile(norm_pair_trimmed_all, 0.99)),
    }
    row.update(_quantiles(np.linalg.norm(z_np, axis=-1), "latent_norm"))
    row.update(_quantiles(denom, "pair_distance"))
    row.update(_quantiles(top_den, "top1_norm_error_pair_distance"))
    row.update(_rank_diagnostics(z_np, err_np))
    return row


def iter_checkpoint_paths(args: argparse.Namespace) -> Iterable[Path]:
    if args.checkpoints:
        for item in args.checkpoints:
            yield Path(item)
        return
    yield from sorted(Path(args.checkpoint_dir).glob("*_m*_seed*.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline diagnostics for learned branching maze checkpoints.")
    parser.add_argument("--checkpoint-dir", default="outputs/learned_branching_maze_gain/hard/checkpoints")
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--output-csv", default="outputs/learned_branching_maze_gain/hard/checkpoint_diagnostics.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nearest-chunk", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    for path in iter_checkpoint_paths(args):
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"diagnosing {path}", flush=True)
        rows.append(diagnose_checkpoint(path, device=device, nearest_chunk=args.nearest_chunk))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
