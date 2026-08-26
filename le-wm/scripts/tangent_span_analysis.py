from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import hdf5plugin  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except ImportError:
    HDF5PLUGIN_AVAILABLE = False

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aliasing_experiment import _find_key, _load_model, _preprocess_pixels, _read_rows  # noqa: E402


def _sample_rows(num_rows: int, num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(int(num_points), int(num_rows))
    return np.sort(rng.choice(num_rows, size=count, replace=False).astype(np.int64))


def _encode_latents(
    model,
    pixels_ds,
    rows: np.ndarray,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            pixels_np = _read_rows(pixels_ds, batch_rows)[:, None]
            pixels = _preprocess_pixels(pixels_np, img_size, device)
            emb = model.encode({"pixels": pixels})["emb"][:, 0]
            latents.append(emb.detach().float().cpu())
            print(
                f"[tangent_span] encoded {min(start + batch_size, len(rows))}/{len(rows)} points",
                flush=True,
            )
    return torch.cat(latents, dim=0)


def _rank_at_energy(singular_values: torch.Tensor, threshold: float) -> int:
    energy = singular_values.float().pow(2)
    total = torch.sum(energy)
    if total <= 0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(float(threshold), device=cumulative.device)).item() + 1)


def _effective_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    energy = singular_values.float().pow(2)
    total = torch.sum(energy)
    if total <= 0:
        return float("nan")
    p = energy / total
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    return float(torch.exp(entropy).cpu())


def _local_tangent_bases(
    Z: torch.Tensor,
    num_neighbors: int,
    local_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if Z.ndim != 2:
        raise ValueError(f"Expected Z shape [N,D], got {tuple(Z.shape)}")
    num_points, latent_dim = Z.shape
    if num_neighbors >= num_points:
        raise ValueError(f"num_neighbors={num_neighbors} must be < num_points={num_points}")
    if local_dim <= 0 or local_dim > min(num_neighbors, latent_dim):
        raise ValueError(
            f"local_dim={local_dim} must be in [1, min(num_neighbors, latent_dim)]="
            f"[1, {min(num_neighbors, latent_dim)}]"
        )

    Z_device = Z.to(device=device, dtype=torch.float32)
    distances = torch.cdist(Z_device, Z_device)
    neighbor_idx = torch.topk(distances, k=num_neighbors + 1, largest=False).indices[:, 1:]
    bases = []
    for idx in range(num_points):
        neighbors = Z_device[neighbor_idx[idx]]
        centered = neighbors - neighbors.mean(dim=0, keepdim=True)
        _u, _s, vh = torch.linalg.svd(centered.float(), full_matrices=False)
        bases.append(vh[:local_dim].detach().cpu())
        if (idx + 1) % 100 == 0 or idx + 1 == num_points:
            print(f"[tangent_span] local PCA {idx + 1}/{num_points}", flush=True)
    return torch.cat(bases, dim=0)


def _analyze_tangent_span(V_all: torch.Tensor) -> Dict[str, object]:
    singular_values = torch.linalg.svdvals(V_all.float())
    singular_values, _ = torch.sort(singular_values, descending=True)
    squared_energy = singular_values.float().pow(2)
    squared_energy_cumsum = torch.cumsum(squared_energy, dim=0) / torch.sum(squared_energy)
    return {
        "effective_rank": _effective_rank_from_singular_values(singular_values),
        "rank_50": _rank_at_energy(singular_values, 0.50),
        "rank_90": _rank_at_energy(singular_values, 0.90),
        "rank_99": _rank_at_energy(singular_values, 0.99),
        "singular_values": [float(value) for value in singular_values.cpu().tolist()],
        "squared_energy_cumsum": [float(value) for value in squared_energy_cumsum.cpu().tolist()],
    }


def _print_report(result: Dict[str, object]) -> None:
    print("=========================================")
    print("TANGENT SPAN ANALYSIS")
    print("=========================================")
    print(f"Model: {Path(str(result['checkpoint_object'])).name}")
    print(f"Latent dim D: {result['latent_dim']}")
    print(f"Points sampled: {result['num_points']}")
    print(f"Neighbors per point: {result['num_neighbors']}")
    print(f"Local tangent dim: {result['local_dim']}")
    print("")
    print(f"Tangent span effective rank: {result['effective_rank']:.2f}")
    print(f"Tangent span rank (50% energy): {result['rank_50']}")
    print(f"Tangent span rank (90% energy): {result['rank_90']}")
    print(f"Tangent span rank (99% energy): {result['rank_99']}")
    print("Energy definition: cumulative sum(s_k^2) / sum(s_j^2) [confirmed]")
    print("")
    print("Top 20 singular values:")
    top = result["singular_values"][:20]
    for offset in range(0, len(top), 5):
        chunk = top[offset:offset + 5]
        print("  " + "  ".join(f"s{offset + idx + 1}={value:.4f}" for idx, value in enumerate(chunk)))
    print("=========================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure global span of local tangent planes in latent space.")
    parser.add_argument("--checkpoint_object", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--num_neighbors", type=int, default=50)
    parser.add_argument("--local_dim", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixels_key", default="pixels")
    args = parser.parse_args()

    print(
        "[tangent_span] hdf5plugin available; compressed HDF5 filters enabled."
        if HDF5PLUGIN_AVAILABLE
        else "[tangent_span] hdf5plugin not available; continuing with default HDF5 filters.",
        flush=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = _load_model(Path(args.checkpoint_object), device)
    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "observation/pixels"])
        rows = _sample_rows(h5[pixels_key].shape[0], args.num_points, args.seed)
        print(f"[tangent_span] sampled {len(rows)} rows from {h5[pixels_key].shape[0]} observations", flush=True)
        Z = _encode_latents(model, h5[pixels_key], rows, args.img_size, args.batch_size, device)

    V_all = _local_tangent_bases(Z, args.num_neighbors, args.local_dim, device)
    analysis = _analyze_tangent_span(V_all)
    result = {
        "checkpoint_object": str(args.checkpoint_object),
        "dataset": str(args.dataset),
        "latent_dim": int(Z.shape[-1]),
        "num_points": int(Z.shape[0]),
        "num_neighbors": int(args.num_neighbors),
        "local_dim": int(args.local_dim),
        "effective_rank": analysis["effective_rank"],
        "rank_50": analysis["rank_50"],
        "rank_90": analysis["rank_90"],
        "rank_99": analysis["rank_99"],
        "rank50": analysis["rank_50"],
        "rank90": analysis["rank_90"],
        "rank99": analysis["rank_99"],
        "singular_values": analysis["singular_values"],
        "squared_energy_cumsum": analysis["squared_energy_cumsum"],
        "sampled_rows": [int(row) for row in rows.tolist()],
        "energy_definition": "rank thresholds use cumulative sum(s_k^2) / sum(s_j^2).",
    }
    _print_report(result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(result, file, indent=2)
    print(f"[tangent_span] saved {output_path}")


if __name__ == "__main__":
    main()
