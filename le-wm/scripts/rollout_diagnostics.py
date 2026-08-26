import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def _as_model(obj):
    if hasattr(obj, "model"):
        return obj.model
    return obj


def _metric_tensor(value: torch.Tensor):
    return float(value.detach().cpu())


def compute_rollout_metrics(pred_emb: torch.Tensor, true_emb: torch.Tensor, horizons: Iterable[int]) -> Dict[str, float]:
    """Compute latent rollout metrics from predicted and true latent sequences.

    pred_emb and true_emb should be shaped (B,T,D), aligned at T.
    Metrics use the first latent as the rollout anchor.
    """
    if pred_emb.shape != true_emb.shape:
        raise ValueError(f"pred_emb and true_emb shapes must match, got {pred_emb.shape} and {true_emb.shape}")
    anchor = true_emb[:, :1]
    max_steps = pred_emb.shape[1] - 1
    out = {}
    for horizon in horizons:
        if horizon > max_steps:
            continue
        pred_delta = pred_emb[:, horizon] - anchor[:, 0]
        true_delta = true_emb[:, horizon] - anchor[:, 0]
        diff = pred_emb[:, horizon] - true_emb[:, horizon]
        true_norm = torch.norm(true_delta, dim=-1)
        pred_norm = torch.norm(pred_delta, dim=-1)
        out[f"rollout/cosine_h{horizon}"] = _metric_tensor(
            F.cosine_similarity(pred_delta, true_delta, dim=-1, eps=1e-8).mean()
        )
        out[f"rollout/mse_h{horizon}"] = _metric_tensor(diff.pow(2).mean())
        out[f"rollout/norm_ratio_h{horizon}"] = _metric_tensor((pred_norm / (true_norm + 1e-8)).mean())
        out[f"rollout/error_ratio_h{horizon}"] = _metric_tensor(
            (torch.norm(diff, dim=-1) / (true_norm + 1e-8)).mean()
        )
    return out


def load_object_checkpoint(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return _as_model(obj)


def main():
    parser = argparse.ArgumentParser(description="Offline latent rollout diagnostics for LeWorldModel objects.")
    parser.add_argument("--checkpoint_object", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_name", default="lewm")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_object)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    model = load_object_checkpoint(checkpoint_path)

    result = {
        "model_name": args.model_name,
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "num_samples": 0,
        "metrics": {},
        "note": (
            "This script provides rollout metric computation and generic object loading. "
            "Dataset window construction is project-specific; pass precomputed pred_emb/true_emb "
            "or adapt this script to the exact HDF5 schema and transforms used for training."
        ),
    }

    if not hasattr(model, "encode") or not hasattr(model, "predict"):
        raise RuntimeError(
            "Loaded checkpoint object does not expose encode() and predict(). "
            "Please pass the serialized JEPA/world model object checkpoint."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    raise RuntimeError(
        "Rollout dataset window construction is not implemented for this HDF5 schema yet. "
        f"Wrote metadata stub to {output_path}. Use compute_rollout_metrics(pred_emb, true_emb, horizons) "
        "after constructing aligned latent rollout tensors."
    )


if __name__ == "__main__":
    main()
