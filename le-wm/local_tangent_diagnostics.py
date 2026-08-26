from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-1])
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected (B,T,D) or (N,D), got shape={tuple(x.shape)}")


def _nan(device: torch.device) -> torch.Tensor:
    return torch.full((), float("nan"), device=device)


def _median(x: torch.Tensor) -> torch.Tensor:
    return x.median() if x.numel() > 0 else _nan(x.device)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() < 2 or y.numel() < 2:
        return _nan(x.device)
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.pow(2).mean() * y.pow(2).mean()) + eps
    return (x * y).mean() / denom


def _pca_dims_and_basis(local_delta: torch.Tensor, r_basis: int, eps: float):
    centered = local_delta.float() - local_delta.float().mean(dim=0, keepdim=True)
    _, singvals, vh = torch.linalg.svd(centered, full_matrices=False)
    eig = singvals.pow(2)
    total = eig.sum()
    if total <= eps:
        dim90 = torch.ones((), device=local_delta.device)
        dim95 = torch.ones((), device=local_delta.device)
        cov_pr = torch.ones((), device=local_delta.device)
        top_frac = torch.ones((), device=local_delta.device)
    else:
        cumsum = torch.cumsum(eig, dim=0) / (total + eps)
        dim90 = (torch.searchsorted(cumsum, torch.tensor(0.90, device=cumsum.device)) + 1).float()
        dim95 = (torch.searchsorted(cumsum, torch.tensor(0.95, device=cumsum.device)) + 1).float()
        cov_pr = total.pow(2) / (eig.pow(2).sum() + eps)
        top_frac = eig[0] / (total + eps)

    r = min(r_basis, vh.shape[0])
    basis = vh[:r].T
    if r < r_basis:
        pad = torch.zeros(basis.shape[0], r_basis - r, device=basis.device, dtype=basis.dtype)
        basis = torch.cat([basis, pad], dim=1)
    return dim90, dim95, cov_pr, top_frac, basis


def _projector_pair_distances(bases: torch.Tensor, eps: float):
    # bases: (M,D,r), assumed orthonormal columns.
    m, _, r = bases.shape
    if m < 2:
        return None
    idx_i, idx_j = torch.triu_indices(m, m, offset=1, device=bases.device)
    ui = bases[idx_i]
    uj = bases[idx_j]
    overlap = torch.einsum("bdr,bds->brs", ui, uj).pow(2).sum(dim=(1, 2))
    dist_sq = torch.clamp(2 * r - 2 * overlap, min=0.0)
    return idx_i, idx_j, torch.sqrt(dist_sq / (2 * r + eps))


def _model_pca_alignment(model_basis: torch.Tensor, pca_basis: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
    q_model, _ = torch.linalg.qr(model_basis.float(), mode="reduced")
    q_pca, _ = torch.linalg.qr(pca_basis.float(), mode="reduced")
    k = q_model.shape[-1]
    r = q_pca.shape[-1]
    min_dim = max(min(k, r), 1)
    cross = torch.einsum("bdk,bdr->bkr", q_model, q_pca)
    overlap_raw = cross.pow(2).sum(dim=(1, 2))
    overlap = overlap_raw / min_dim
    proj_dist = torch.sqrt(torch.clamp(k + r - 2 * overlap_raw, min=0.0) / (k + r + eps))
    return {
        "model_pca_overlap_mean": overlap.mean(),
        "model_pca_overlap_median": _median(overlap),
        "model_pca_projector_distance_mean": proj_dist.mean(),
    }


def compute_local_tangent_diagnostics(
    z: torch.Tensor,
    delta: torch.Tensor,
    num_anchors: int = 128,
    num_neighbors: int = 64,
    r_basis: int = 16,
    model_basis: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Estimate local transition tangent geometry from a batch."""
    with torch.no_grad():
        z_flat = _flatten_bt(z).detach().float()
        delta_flat = _flatten_bt(delta).detach().float()
        if z_flat.shape != delta_flat.shape:
            raise ValueError(f"z and delta must have same flattened shape, got {z_flat.shape} and {delta_flat.shape}")

        n, dim = z_flat.shape
        if n < 3:
            return {"skipped": torch.ones((), device=z_flat.device)}

        k_neighbors = min(num_neighbors, n - 1)
        m = min(num_anchors, n)
        anchor_idx = torch.randperm(n, device=z_flat.device)[:m]
        dist = torch.cdist(z_flat[anchor_idx], z_flat)
        dist[torch.arange(m, device=z_flat.device), anchor_idx] = float("inf")
        knn_idx = torch.topk(dist, k=k_neighbors, dim=1, largest=False).indices

        dim90, dim95, cov_pr, top_frac, bases = [], [], [], [], []
        for row in range(m):
            local_delta = delta_flat[knn_idx[row]]
            d90, d95, pr, top, basis = _pca_dims_and_basis(local_delta, r_basis, eps)
            dim90.append(d90)
            dim95.append(d95)
            cov_pr.append(pr)
            top_frac.append(top)
            bases.append(basis)

        dim90 = torch.stack(dim90)
        dim95 = torch.stack(dim95)
        cov_pr = torch.stack(cov_pr)
        top_frac = torch.stack(top_frac)
        bases = torch.stack(bases)

        out = {
            "dim90_mean": dim90.mean(),
            "dim90_median": _median(dim90),
            "dim95_mean": dim95.mean(),
            "dim95_median": _median(dim95),
            "cov_pr_rank_mean": cov_pr.mean(),
            "cov_pr_rank_median": _median(cov_pr),
            "top_eig_fraction_mean": top_frac.mean(),
            "top_eig_fraction_median": _median(top_frac),
        }

        pair_data = _projector_pair_distances(bases, eps)
        if pair_data is not None:
            idx_i, idx_j, subspace_dist = pair_data
            z_anchor = z_flat[anchor_idx]
            delta_anchor = delta_flat[anchor_idx]
            state_dist = torch.norm(z_anchor[idx_i] - z_anchor[idx_j], dim=-1)
            transition_dist = 1.0 - F.cosine_similarity(delta_anchor[idx_i], delta_anchor[idx_j], dim=-1, eps=eps)
            out.update(
                {
                    "subspace_distance_mean": subspace_dist.mean(),
                    "subspace_distance_median": _median(subspace_dist),
                    "subspace_distance_p90": torch.quantile(subspace_dist, 0.9),
                    "state_distance_mean": state_dist.mean(),
                    "subspace_state_corr": _pearson_corr(subspace_dist, state_dist, eps),
                    "subspace_transition_corr": _pearson_corr(subspace_dist, transition_dist, eps),
                }
            )

        if model_basis is not None:
            model_basis_flat = model_basis.detach().reshape(-1, dim, model_basis.shape[-1]).float()
            out.update(_model_pca_alignment(model_basis_flat[anchor_idx], bases, eps))

        return out


def compute_state_tangent_basis_ablation(
    code: torch.Tensor,
    basis: torch.Tensor,
    anchor: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Keep code fixed and shuffle the state-conditioned basis."""
    with torch.no_grad():
        code_flat = _flatten_bt(code).detach().float()
        basis_flat = basis.detach().reshape(-1, basis.shape[-2], basis.shape[-1]).float()
        anchor_flat = _flatten_bt(anchor).detach().float()
        target_flat = _flatten_bt(target).detach().float()
        n = code_flat.shape[0]
        if n < 2:
            return {"skipped": torch.ones((), device=code_flat.device)}

        perm = torch.randperm(n, device=code_flat.device)
        if torch.equal(perm, torch.arange(n, device=code_flat.device)):
            perm = torch.roll(perm, shifts=1)

        delta_correct = torch.einsum("ndk,nk->nd", basis_flat, code_flat)
        delta_shuffled = torch.einsum("ndk,nk->nd", basis_flat[perm], code_flat)
        delta_tgt = target_flat - anchor_flat

        cos_correct = F.cosine_similarity(delta_correct, delta_tgt, dim=-1, eps=eps)
        cos_shuffled = F.cosine_similarity(delta_shuffled, delta_tgt, dim=-1, eps=eps)
        err_correct = torch.norm(delta_correct - delta_tgt, dim=-1) / (torch.norm(delta_tgt, dim=-1) + eps)
        err_shuffled = torch.norm(delta_shuffled - delta_tgt, dim=-1) / (torch.norm(delta_tgt, dim=-1) + eps)
        norm_ratio_correct = torch.norm(delta_correct, dim=-1) / (torch.norm(delta_tgt, dim=-1) + eps)
        norm_ratio_shuffled = torch.norm(delta_shuffled, dim=-1) / (torch.norm(delta_tgt, dim=-1) + eps)

        return {
            "cosine_alignment_correct_basis": cos_correct.mean(),
            "cosine_alignment_shuffled_basis": cos_shuffled.mean(),
            "normalized_error_correct_basis": err_correct.mean(),
            "normalized_error_shuffled_basis": err_shuffled.mean(),
            "norm_ratio_correct_basis": norm_ratio_correct.mean(),
            "norm_ratio_shuffled_basis": norm_ratio_shuffled.mean(),
            "cosine_gap_correct_minus_shuffled": cos_correct.mean() - cos_shuffled.mean(),
            "error_gap_shuffled_minus_correct": err_shuffled.mean() - err_correct.mean(),
        }
