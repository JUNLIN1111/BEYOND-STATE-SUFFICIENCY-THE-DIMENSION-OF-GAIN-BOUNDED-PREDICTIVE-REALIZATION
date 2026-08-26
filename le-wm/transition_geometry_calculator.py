import warnings
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def flatten_bt(x: torch.Tensor) -> torch.Tensor:
    """Flatten (B,T,D)->(B*T,D) and keep (B,D) unchanged."""
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-1])
    if x.dim() == 2:
        return x
    raise ValueError(f"flatten_bt expects shape (B,T,D) or (B,D), got {tuple(x.shape)}")


class TransitionGeometryCalculator:
    """Diagnostics for action-aligned latent transition geometry."""

    def __init__(
        self,
        max_jacobian_samples: int = 3,
        eps: float = 1e-8,
        min_two_nn_samples: int = 10,
        enable_standardized_two_nn: bool = True,
    ):
        self.max_jacobian_samples = max_jacobian_samples
        self.eps = eps
        self.min_two_nn_samples = min_two_nn_samples
        self.enable_standardized_two_nn = enable_standardized_two_nn

    def _warn(self, msg: str):
        warnings.warn(f"[TransitionGeometryCalculator] {msg}", RuntimeWarning, stacklevel=2)

    def _autocast_off(self, device: torch.device):
        if device.type in ("cuda", "cpu"):
            return torch.autocast(device_type=device.type, enabled=False)
        return nullcontext()

    def _nan(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.full((), float("nan"), device=device, dtype=dtype)

    def flatten_bt(self, x: torch.Tensor) -> torch.Tensor:
        return flatten_bt(x)

    def _participation_rank(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        s1 = values.sum()
        s2 = values.pow(2).sum()
        if (not torch.isfinite(s1)) or (not torch.isfinite(s2)) or (s2 <= self.eps):
            return self._nan(values.device, values.dtype)
        return (s1.pow(2)) / (s2 + self.eps)

    def _covariance_eigs(self, z_flat: torch.Tensor) -> torch.Tensor:
        n = z_flat.shape[0]
        if n < 2:
            return self._nan(z_flat.device, z_flat.dtype).reshape(1)
        with self._autocast_off(z_flat.device):
            z = z_flat.to(dtype=torch.float32)
            z_centered = z - z.mean(dim=0, keepdim=True)
            cov = (z_centered.T @ z_centered) / (n - 1)
            cov = cov.to(dtype=torch.float32)
            eigs = torch.linalg.eigvalsh(cov)
            eigs = torch.clamp(eigs.real, min=0.0)
        return eigs

    def _coord_var_pr_rank(self, z_flat: torch.Tensor) -> torch.Tensor:
        if z_flat.shape[0] < 2:
            return self._nan(z_flat.device, z_flat.dtype)
        with self._autocast_off(z_flat.device):
            var_vec = z_flat.to(dtype=torch.float32).var(dim=0, unbiased=False)
        return self._participation_rank(var_vec)

    def _cov_pr_rank(self, z_flat: torch.Tensor) -> torch.Tensor:
        eigs = self._covariance_eigs(z_flat)
        if eigs.numel() == 1 and torch.isnan(eigs[0]):
            return eigs[0]
        return self._participation_rank(eigs)

    def _entropy_effective_rank(self, z_flat: torch.Tensor) -> torch.Tensor:
        eigs = self._covariance_eigs(z_flat)
        if eigs.numel() == 1 and torch.isnan(eigs[0]):
            return eigs[0]
        total = eigs.sum()
        if (not torch.isfinite(total)) or (total <= self.eps):
            return self._nan(z_flat.device, z_flat.dtype)
        p = eigs / (total + self.eps)
        p = p[p > 0]
        if p.numel() == 0:
            return self._nan(z_flat.device, z_flat.dtype)
        entropy = -(p * torch.log(p + self.eps)).sum()
        return torch.exp(entropy)

    def _two_nn_id(self, z_flat: torch.Tensor, standardize: bool = False) -> torch.Tensor:
        with self._autocast_off(z_flat.device):
            z = z_flat.to(dtype=torch.float32)
            n, d = z.shape
            if n < self.min_two_nn_samples:
                self._warn(f"Two-NN ID skipped: only {n} samples (need >= {self.min_two_nn_samples}).")
                return self._nan(z.device, z.dtype)
            if standardize:
                mean = z.mean(dim=0, keepdim=True)
                std = z.std(dim=0, keepdim=True, unbiased=False)
                z = (z - mean) / (std + self.eps)

            dist = torch.cdist(z, z, p=2)
            diag = torch.arange(n, device=z.device)
            dist[diag, diag] = float("inf")

            knn = torch.topk(dist, k=2, dim=1, largest=False).values
            r1 = torch.clamp(knn[:, 0], min=self.eps)
            r2 = torch.clamp(knn[:, 1], min=self.eps)
            mu = r2 / (r1 + self.eps)
            mu = mu[mu > 1.0]
            if mu.numel() < 5:
                self._warn("Two-NN ID failed: too few valid mu ratios (>1.0).")
                return self._nan(z.device, z.dtype)

            mu_sorted, _ = torch.sort(mu)
            k = mu_sorted.numel()
            idx = torch.arange(1, k + 1, device=z.device, dtype=z.dtype)
            f_mu = idx / (k + 1)
            y = -torch.log(1.0 - f_mu + self.eps)
            x = torch.log(mu_sorted + self.eps)
            denom = torch.dot(x, x)
            if denom <= self.eps:
                self._warn("Two-NN ID failed: degenerate log-ratio distribution.")
                return self._nan(z.device, z.dtype)

            did = torch.dot(x, y) / (denom + self.eps)
            return torch.clamp(did, min=0.0, max=float(d))

    def _geometry_metrics_for_object(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_flat = self.flatten_bt(x.detach())
        metrics = {
            "coord_var_pr_rank": self._coord_var_pr_rank(z_flat),
            "cov_pr_rank": self._cov_pr_rank(z_flat),
            "entropy_effective_rank": self._entropy_effective_rank(z_flat),
            "id_two_nn": self._two_nn_id(z_flat, standardize=False),
        }
        if self.enable_standardized_two_nn:
            metrics["id_two_nn_std"] = self._two_nn_id(z_flat, standardize=True)
        return metrics

    def _prepare_anchor(self, pred_emb: torch.Tensor, ctx_emb: torch.Tensor) -> torch.Tensor:
        if pred_emb.dim() == 3:
            anchor = ctx_emb[:, -1:, :]
            if anchor.shape[1] != pred_emb.shape[1]:
                anchor = anchor.expand(-1, pred_emb.shape[1], -1)
            return anchor
        if pred_emb.dim() == 2:
            if ctx_emb.dim() == 3:
                return ctx_emb[:, -1, :]
            if ctx_emb.dim() == 2:
                return ctx_emb
            raise ValueError(f"ctx_emb shape incompatible with 2D pred_emb: {tuple(ctx_emb.shape)}")
        raise ValueError(f"pred_emb must be 2D or 3D, got {tuple(pred_emb.shape)}")

    def _transition_alignment_metrics(
        self, delta_pred: torch.Tensor, delta_tgt: torch.Tensor, residual: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        dp = self.flatten_bt(delta_pred.detach())
        dt = self.flatten_bt(delta_tgt.detach())
        res = self.flatten_bt(residual.detach())

        cos = F.cosine_similarity(dp, dt, dim=-1, eps=self.eps)
        diff_norm = torch.norm(dp - dt, dim=-1)
        tgt_norm = torch.norm(dt, dim=-1)
        pred_norm = torch.norm(dp, dim=-1)
        normalized_error = diff_norm / (tgt_norm + self.eps)
        norm_ratio = pred_norm / (tgt_norm + self.eps)

        return {
            "cosine_alignment_mean": cos.mean(),
            "cosine_alignment_median": cos.median(),
            "cosine_alignment_std": cos.std(unbiased=False),
            "normalized_transition_error": normalized_error.mean(),
            "normalized_transition_error_median": normalized_error.median(),
            "normalized_transition_error_p90": torch.quantile(normalized_error, 0.9),
            "delta_norm_ratio": norm_ratio.mean(),
            "delta_norm_ratio_median": norm_ratio.median(),
            "delta_norm_ratio_p90": torch.quantile(norm_ratio, 0.9),
            "normalized_transition_error_ratio_of_means": diff_norm.mean() / (tgt_norm.mean() + self.eps),
            "delta_norm_ratio_ratio_of_means": pred_norm.mean() / (tgt_norm.mean() + self.eps),
            "delta_tgt_norm_mean": tgt_norm.mean(),
            "delta_tgt_norm_median": tgt_norm.median(),
            "delta_pred_norm_mean": pred_norm.mean(),
            "near_zero_delta_tgt_frac": (tgt_norm < 1e-6).float().mean(),
            "near_zero_delta_tgt_frac_1e4": (tgt_norm < 1e-4).float().mean(),
            "near_zero_delta_tgt_frac_1e3": (tgt_norm < 1e-3).float().mean(),
            "near_zero_delta_tgt_frac_1e2": (tgt_norm < 1e-2).float().mean(),
            "residual_mse": res.pow(2).mean(),
        }

    def jacobian_action_emb_pr_rank(self, pred_emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Compute participation rank of singular values of J = d(pred_emb) / d(act_emb)."""
        try:
            if pred_emb.dim() != 3 or act_emb.dim() != 3:
                raise ValueError(
                    f"Jacobian expects 3D tensors pred_emb(B,T,D) and act_emb(B,T,A), got "
                    f"{tuple(pred_emb.shape)} and {tuple(act_emb.shape)}"
                )
            batch_size, pred_t, pred_d = pred_emb.shape
            _, act_t, act_d = act_emb.shape
            n = min(self.max_jacobian_samples, batch_size)
            ranks = []
            for i in range(n):
                pred_vec = pred_emb[i].reshape(-1)
                jac = torch.zeros(pred_t * pred_d, act_t * act_d, device=act_emb.device, dtype=pred_emb.dtype)
                for out_idx in range(pred_vec.numel()):
                    grads = torch.autograd.grad(
                        outputs=pred_vec[out_idx],
                        inputs=act_emb,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                    if grads is not None:
                        jac[out_idx] = grads[i].reshape(-1)
                sigma = torch.linalg.svdvals(jac.float())
                ranks.append(self._participation_rank(sigma))
            if len(ranks) == 0:
                self._warn("Jacobian PR rank failed: empty sample set.")
                return self._nan(act_emb.device, act_emb.dtype)
            return torch.stack(ranks).mean()
        except Exception as exc:
            self._warn(f"Jacobian PR rank failed: {exc}")
            return self._nan(act_emb.device, act_emb.dtype)

    def compute(
        self,
        emb: torch.Tensor,
        pred_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        act_emb: torch.Tensor,
        ctx_emb: torch.Tensor,
        anchor_emb: Optional[torch.Tensor] = None,
        raw_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del raw_action  # Reserved for future analysis hooks.
        out: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            pred_emb_det = pred_emb.detach()
            tgt_emb_det = tgt_emb.detach()
            emb_det = emb.detach()
            ctx_emb_det = ctx_emb.detach()
            if anchor_emb is None:
                anchor_emb_det = self._prepare_anchor(pred_emb_det, ctx_emb_det)
            else:
                anchor_emb_det = anchor_emb.detach()

            delta_pred = pred_emb_det - anchor_emb_det
            delta_tgt = tgt_emb_det - anchor_emb_det
            residual = pred_emb_det - tgt_emb_det

            transition_metrics = self._transition_alignment_metrics(delta_pred, delta_tgt, residual)
            out["transition/cosine_alignment_mean"] = transition_metrics["cosine_alignment_mean"]
            out["transition/cosine_alignment_median"] = transition_metrics["cosine_alignment_median"]
            out["transition/cosine_alignment_std"] = transition_metrics["cosine_alignment_std"]
            out["transition/normalized_transition_error"] = transition_metrics["normalized_transition_error"]
            out["transition/normalized_transition_error_median"] = transition_metrics[
                "normalized_transition_error_median"
            ]
            out["transition/normalized_transition_error_p90"] = transition_metrics[
                "normalized_transition_error_p90"
            ]
            out["transition/delta_norm_ratio"] = transition_metrics["delta_norm_ratio"]
            out["transition/delta_norm_ratio_median"] = transition_metrics["delta_norm_ratio_median"]
            out["transition/delta_norm_ratio_p90"] = transition_metrics["delta_norm_ratio_p90"]
            out["transition/normalized_transition_error_ratio_of_means"] = transition_metrics[
                "normalized_transition_error_ratio_of_means"
            ]
            out["transition/delta_norm_ratio_ratio_of_means"] = transition_metrics[
                "delta_norm_ratio_ratio_of_means"
            ]
            out["transition/delta_tgt_norm_mean"] = transition_metrics["delta_tgt_norm_mean"]
            out["transition/delta_tgt_norm_median"] = transition_metrics["delta_tgt_norm_median"]
            out["transition/delta_pred_norm_mean"] = transition_metrics["delta_pred_norm_mean"]
            out["transition/near_zero_delta_tgt_frac"] = transition_metrics["near_zero_delta_tgt_frac"]
            out["transition/near_zero_delta_tgt_frac_1e4"] = transition_metrics["near_zero_delta_tgt_frac_1e4"]
            out["transition/near_zero_delta_tgt_frac_1e3"] = transition_metrics["near_zero_delta_tgt_frac_1e3"]
            out["transition/near_zero_delta_tgt_frac_1e2"] = transition_metrics["near_zero_delta_tgt_frac_1e2"]
            out["residual/mse"] = transition_metrics["residual_mse"]

            objects = {
                "emb": emb_det,
                "pred_emb": pred_emb_det,
                "tgt_emb": tgt_emb_det,
                "delta_pred": delta_pred,
                "delta_tgt": delta_tgt,
                "residual": residual,
            }
            for obj_name, obj in objects.items():
                metrics = self._geometry_metrics_for_object(obj)
                for metric_name, value in metrics.items():
                    out[f"{obj_name}/{metric_name}"] = value

        out["jacobian_action_emb_pr_rank"] = self.jacobian_action_emb_pr_rank(pred_emb, act_emb)
        return out


def transition_geometry_sanity_checks(device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Unit-test-like synthetic checks:
    - low-rank data => low covariance PR rank
    - isotropic data => high covariance PR rank
    - delta_pred == delta_tgt => cosine ~1, normalized error ~0
    - shuffled transitions reduce alignment on correlated synthetic data
    """
    dev = device or torch.device("cpu")
    calc = TransitionGeometryCalculator()

    n, d, k = 256, 32, 3
    base = torch.randn(n, k, device=dev)
    proj = torch.randn(k, d, device=dev)
    low_rank = base @ proj
    iso = torch.randn(n, d, device=dev)

    cov_pr_low = calc._cov_pr_rank(low_rank)
    cov_pr_iso = calc._cov_pr_rank(iso)

    b, t = 64, 1
    anchor = torch.randn(b, t, d, device=dev)
    delta = torch.randn(b, t, d, device=dev)
    pred = anchor + delta
    tgt = anchor + delta
    same = calc._transition_alignment_metrics(pred - anchor, tgt - anchor, pred - tgt)

    corr = torch.randn(b, t, d, device=dev)
    shuffled = corr[torch.randperm(b, device=dev)]
    shuf = calc._transition_alignment_metrics(corr, shuffled, corr - shuffled)

    return {
        "cov_pr_rank_low_rank_data": cov_pr_low,
        "cov_pr_rank_isotropic_data": cov_pr_iso,
        "same_transition_cosine_mean": same["cosine_alignment_mean"],
        "same_transition_normalized_error": same["normalized_transition_error"],
        "shuffled_transition_cosine_mean": shuf["cosine_alignment_mean"],
    }
