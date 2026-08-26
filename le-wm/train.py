import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from local_tangent_diagnostics import compute_local_tangent_diagnostics, compute_state_tangent_basis_ablation
from module import ARPredictor, Embedder, MLP, SIGReg, StateConditionedTangentBottleneck, TransitionBottleneck
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack

from transition_geometry_calculator import TransitionGeometryCalculator
# ================================================================================ 
# my part


def _cfg_get(cfg, key: str, default):
    value = OmegaConf.select(cfg, key, default=default)
    return default if value is None else value


def _is_global_zero(module) -> bool:
    trainer = getattr(module, "trainer", None)
    if trainer is not None and hasattr(trainer, "is_global_zero"):
        return bool(trainer.is_global_zero)
    return int(getattr(module, "global_rank", 0)) == 0


def _attach_analysis_metrics(output: dict, metric_prefix: str, metrics: dict):
    for metric_name, metric_value in metrics.items():
        if not torch.is_tensor(metric_value):
            continue
        metric_tensor = metric_value.detach()
        full_name = f"{metric_prefix}/{metric_name}"
        output[full_name] = metric_tensor
        output[f"loss_analysis/{full_name}"] = metric_tensor


def _predictor_kwargs(cfg):
    default = {
        "depth": 6,
        "heads": 16,
        "mlp_dim": 2048,
        "dim_head": 64,
        "dropout": 0.1,
        "emb_dropout": 0.0,
    }
    predictor_cfg = OmegaConf.select(cfg, "predictor", default=None)
    if predictor_cfg is None:
        predictor_cfg = OmegaConf.select(cfg, "model.predictor", default=None)
    if predictor_cfg is None:
        return default
    predictor_dict = OmegaConf.to_container(predictor_cfg, resolve=True)
    for key in ("_target_", "num_frames", "input_dim", "hidden_dim", "output_dim"):
        predictor_dict.pop(key, None)
    return {**default, **predictor_dict}


def _sigreg_kwargs(cfg):
    value = OmegaConf.select(cfg, "loss.sigreg.kwargs", default={"knots": 17, "num_proj": 1024})
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


def _optimizer_cfg(cfg):
    value = OmegaConf.select(cfg, "optimizer", default={"type": "AdamW", "lr": 5e-5, "weight_decay": 1e-3})
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


def _trainer_cfg(cfg):
    value = OmegaConf.select(
        cfg,
        "trainer",
        default={
            "max_epochs": 100,
            "devices": "auto",
            "accelerator": "gpu",
            "precision": "bf16",
            "gradient_clip_val": 1.0,
        },
    )
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


def _wandb_cfg(cfg):
    enabled = bool(OmegaConf.select(cfg, "wandb.enabled", default=False))
    value = OmegaConf.select(cfg, "wandb.config", default={})
    config = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)
    return enabled, config


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = float(_cfg_get(cfg, "loss.sigreg.weight", 0.09))
    enable_transition_geometry = _cfg_get(cfg, "analysis.enable_transition_geometry", False)
    enable_shuffled_action_control = _cfg_get(cfg, "analysis.enable_shuffled_action_control", False)
    enable_local_tangent_diagnostics = bool(_cfg_get(cfg, "analysis.enable_local_tangent_diagnostics", False))
    transition_anchor_mode = str(_cfg_get(cfg, "analysis.transition_anchor_mode", "shifted"))
    transition_eval_slice = str(_cfg_get(cfg, "analysis.transition_eval_slice", "all_shifted"))
    log_every = int(_cfg_get(cfg, "analysis.log_every", 100))
    local_tangent_log_every = int(_cfg_get(cfg, "analysis.local_tangent_log_every", 500))
    no_dynamics = bool(_cfg_get(cfg, "loss.no_dynamics", False))
    shuffled_action_training = bool(_cfg_get(cfg, "loss.shuffled_action_training", False))
    transition_bottleneck_enabled = bool(_cfg_get(cfg, "wm.transition_bottleneck.enabled", False))
    transition_bottleneck_dim = int(_cfg_get(cfg, "wm.transition_bottleneck.dim", 4))
    transition_bottleneck_type = str(_cfg_get(cfg, "wm.transition_bottleneck.type", "linear"))
    transition_bottleneck_type_id = {
        "linear": 0,
        "mlp": 1,
        "state_tangent": 2,
    }.get(transition_bottleneck_type)
    if transition_bottleneck_enabled and transition_bottleneck_type_id is None:
        raise ValueError(
            f"Unsupported transition bottleneck type: {transition_bottleneck_type}. "
            "Expected 'linear', 'mlp', or 'state_tangent'."
        )
    if no_dynamics and shuffled_action_training:
        raise ValueError("Do not enable no_dynamics and shuffled_action_training simultaneously.")
    if shuffled_action_training or enable_shuffled_action_control:
        enable_transition_geometry = True

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    embed_dim = emb.shape[-1]
    if transition_bottleneck_enabled and (
        transition_bottleneck_dim <= 0 or transition_bottleneck_dim > embed_dim
    ):
        raise ValueError(
            f"Invalid transition bottleneck dim={transition_bottleneck_dim}. "
            f"Expected 1 <= k <= embed_dim({embed_dim})."
        )

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    ctx_act_for_pred = ctx_act
    metric_prefix = "analysis/full"
    if shuffled_action_training:
        perm_main = torch.randperm(ctx_act.shape[0], device=ctx_act.device)
        ctx_act_for_pred = ctx_act[perm_main]
        metric_prefix = "analysis/train_shuffled_action"
    if not ctx_act_for_pred.requires_grad:
        ctx_act_for_pred = ctx_act_for_pred.requires_grad_(True)

    pred_emb_raw = self.model.predict(ctx_emb, ctx_act_for_pred)
    if pred_emb_raw.dim() == 2:
        pred_emb_raw = pred_emb_raw.unsqueeze(1)
    elif pred_emb_raw.dim() != 3:
        raise ValueError(f"Expected pred_emb to be 2D or 3D, got shape={tuple(pred_emb_raw.shape)}")
    pred_len = pred_emb_raw.shape[1]
    if transition_anchor_mode == "shifted":
        anchor_emb = emb[:, :pred_len]
        tgt_emb = emb[:, n_preds:n_preds + pred_len]
    elif transition_anchor_mode == "last_context":
        anchor_emb = ctx_emb[:, -1:, :].expand(-1, pred_len, -1)
        tgt_emb = emb[:, ctx_len:ctx_len + pred_len]
    else:
        raise ValueError(
            f"Unsupported cfg.analysis.transition_anchor_mode='{transition_anchor_mode}'. "
            "Expected 'shifted' or 'last_context'."
        )
    assert pred_emb_raw.shape == tgt_emb.shape == anchor_emb.shape, (
        f"Shape mismatch: pred_emb_raw={tuple(pred_emb_raw.shape)}, "
        f"tgt_emb={tuple(tgt_emb.shape)}, anchor_emb={tuple(anchor_emb.shape)}"
    )

    transition_code = None
    delta_raw = None
    delta_bottleneck = None
    tangent_basis = None
    if transition_bottleneck_enabled:
        tb_module = getattr(self.model, "transition_bottleneck", None)
        if tb_module is None:
            raise ValueError("Transition bottleneck is enabled but self.model.transition_bottleneck is not set.")
        delta_raw = pred_emb_raw - anchor_emb
        if transition_bottleneck_type == "state_tangent":
            delta_bottleneck, transition_code, tangent_basis = tb_module(delta_raw, anchor_emb)
        else:
            delta_bottleneck, transition_code = tb_module(delta_raw)
        pred_emb = anchor_emb + delta_bottleneck
    else:
        pred_emb = pred_emb_raw

    if stage == "fit" and (not hasattr(self, "_transition_bottleneck_debug_printed")):
        if _is_global_zero(self):
            print(
                "[TransitionBottleneck debug] "
                f"enabled={transition_bottleneck_enabled}, "
                f"type={transition_bottleneck_type}, "
                f"k={transition_bottleneck_dim}, "
                f"pred_emb_raw.shape={tuple(pred_emb_raw.shape)}, "
                f"anchor_emb.shape={tuple(anchor_emb.shape)}, "
                f"pred_emb.shape={tuple(pred_emb.shape)}"
            )
            if transition_bottleneck_type == "state_tangent" and transition_code is not None:
                print(
                    "[TransitionBottleneck debug] "
                    f"transition_code.shape={tuple(transition_code.shape)}, "
                    f"tangent_basis.shape={tuple(tangent_basis.shape)}"
                )
        self._transition_bottleneck_debug_printed = True

    if transition_eval_slice == "all_shifted":
        pred_eval = pred_emb
        tgt_eval = tgt_emb
        anchor_eval = anchor_emb
        pred_raw_eval = pred_emb_raw
        transition_code_eval = transition_code
        tangent_basis_eval = tangent_basis
    elif transition_eval_slice == "future_last":
        if n_preds != 1:
            raise ValueError(
                f"transition_eval_slice='future_last' expects n_preds=1, got n_preds={n_preds}."
            )
        if pred_emb.shape[1] < 1:
            raise ValueError("transition_eval_slice='future_last' requires at least one predicted step.")
        pred_eval = pred_emb[:, -1:, :]
        anchor_eval = emb[:, ctx_len - 1:ctx_len, :]
        tgt_eval = emb[:, ctx_len:ctx_len + 1, :]
        pred_raw_eval = pred_emb_raw[:, -1:, :]
        transition_code_eval = transition_code[:, -1:, :] if transition_code is not None else None
        tangent_basis_eval = tangent_basis[:, -1:, :, :] if tangent_basis is not None else None
    else:
        raise ValueError(
            f"Unsupported cfg.analysis.transition_eval_slice='{transition_eval_slice}'. "
            "Expected 'all_shifted' or 'future_last'."
        )
    assert pred_eval.shape == tgt_eval.shape == anchor_eval.shape, (
        f"Eval shape mismatch: pred_eval={tuple(pred_eval.shape)}, "
        f"tgt_eval={tuple(tgt_eval.shape)}, anchor_eval={tuple(anchor_eval.shape)}"
    )

    if stage == "fit" and enable_transition_geometry and (not hasattr(self, "_transition_geometry_debug_printed")):
        if _is_global_zero(self):
            print(
                "[TransitionGeometry debug] "
                f"ctx_len={ctx_len}, n_preds={n_preds}, "
                f"emb.shape={tuple(emb.shape)}, ctx_emb.shape={tuple(ctx_emb.shape)}, "
                f"ctx_act.shape={tuple(ctx_act.shape)}, pred_emb.shape={tuple(pred_emb.shape)}, "
                f"tgt_emb.shape={tuple(tgt_emb.shape)}, anchor_emb.shape={tuple(anchor_emb.shape)}, "
                f"transition_eval_slice={transition_eval_slice}, pred_eval.shape={tuple(pred_eval.shape)}"
            )
        self._transition_geometry_debug_printed = True

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    if no_dynamics:
        output["loss"] = lambd * output["sigreg_loss"]
    else:
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    with torch.no_grad():
        pred_loss_eval = (pred_eval - tgt_eval).pow(2).mean()
        copy_loss_eval = (anchor_eval - tgt_eval).pow(2).mean()
        output["loss_analysis/eval/pred_loss_eval"] = pred_loss_eval
        output["loss_analysis/eval/copy_loss_eval"] = copy_loss_eval
        output["loss_analysis/eval/pred_vs_copy_ratio"] = pred_loss_eval / (copy_loss_eval + 1e-8)

        copy_loss_all = (anchor_emb - tgt_emb).pow(2).mean()
        output["loss_analysis/baseline/copy_loss_all"] = copy_loss_all
        output["loss_analysis/baseline/pred_vs_copy_ratio_all"] = output["pred_loss"].detach() / (copy_loss_all + 1e-8)

        if transition_bottleneck_enabled:
            delta_tgt_eval = tgt_eval - anchor_eval
            delta_pred_eval = pred_eval - anchor_eval
            tgt_eval_norm = torch.norm(delta_tgt_eval.detach(), dim=-1)
            pred_eval_norm = torch.norm(delta_pred_eval.detach(), dim=-1)
            output["loss_analysis/transition_bottleneck/delta_pred_to_tgt_norm_ratio_eval"] = (
                pred_eval_norm.mean() / (tgt_eval_norm.mean() + 1e-8)
            )

            delta_raw_eval = pred_raw_eval - anchor_eval
            raw_flat = delta_raw_eval.detach().reshape(-1, delta_raw_eval.shape[-1])
            bottleneck_flat = delta_pred_eval.detach().reshape(-1, delta_pred_eval.shape[-1])
            tgt_flat = delta_tgt_eval.detach().reshape(-1, delta_tgt_eval.shape[-1])
            output["loss_analysis/transition_bottleneck/raw_cosine_eval"] = F.cosine_similarity(
                raw_flat, tgt_flat, dim=-1, eps=1e-8
            ).mean()
            output["loss_analysis/transition_bottleneck/bottleneck_cosine_eval"] = F.cosine_similarity(
                bottleneck_flat, tgt_flat, dim=-1, eps=1e-8
            ).mean()

            code_norm = torch.norm(transition_code.detach(), dim=-1)
            delta_raw_norm = torch.norm(delta_raw.detach(), dim=-1)
            delta_bottleneck_norm = torch.norm(delta_bottleneck.detach(), dim=-1)
            reconstruction_ratio = delta_bottleneck_norm.mean() / (delta_raw_norm.mean() + 1e-8)
            output["loss_analysis/transition_bottleneck/type_id"] = torch.tensor(
                float(transition_bottleneck_type_id), device=emb.device
            )
            output["loss_analysis/transition_bottleneck/dim"] = torch.tensor(
                float(transition_bottleneck_dim), device=emb.device
            )
            output["loss_analysis/transition_bottleneck/code_norm_mean"] = code_norm.mean()
            output["loss_analysis/transition_bottleneck/code_norm_median"] = code_norm.median()
            output["loss_analysis/transition_bottleneck/delta_raw_norm_mean"] = delta_raw_norm.mean()
            output["loss_analysis/transition_bottleneck/delta_bottleneck_norm_mean"] = delta_bottleneck_norm.mean()
            output["loss_analysis/transition_bottleneck/reconstruction_ratio"] = reconstruction_ratio
            output["loss_analysis/transition_bottleneck/delta_bottleneck_to_raw_norm_ratio"] = reconstruction_ratio

            if transition_bottleneck_type == "state_tangent" and tangent_basis is not None:
                basis = tangent_basis.detach().to(dtype=torch.float32)
                basis_flat = basis.reshape(-1, basis.shape[-2], basis.shape[-1])
                basis_col_norm = torch.norm(basis_flat, dim=1)
                basis_col_unit = F.normalize(basis_flat, dim=1, eps=1e-8)
                gram_abs = torch.abs(torch.einsum("bdk,bdl->bkl", basis_col_unit, basis_col_unit))
                k = gram_abs.shape[-1]
                if k > 1:
                    off_diag = ~torch.eye(k, dtype=torch.bool, device=gram_abs.device).unsqueeze(0)
                    basis_col_cos_abs_mean = gram_abs[off_diag.expand_as(gram_abs)].mean()
                else:
                    basis_col_cos_abs_mean = torch.zeros((), device=emb.device)
                output["loss_analysis/transition_bottleneck/basis_norm_mean"] = torch.norm(
                    basis_flat.reshape(basis_flat.shape[0], -1), dim=-1
                ).mean()
                output["loss_analysis/transition_bottleneck/basis_col_norm_mean"] = basis_col_norm.mean()
                output["loss_analysis/transition_bottleneck/basis_col_cos_abs_mean"] = basis_col_cos_abs_mean

            if transition_bottleneck_type == "state_tangent" and transition_code_eval is not None:
                basis_ablation = compute_state_tangent_basis_ablation(
                    code=transition_code_eval,
                    basis=tangent_basis_eval,
                    anchor=anchor_eval,
                    target=tgt_eval,
                )
                for name, value in basis_ablation.items():
                    output[f"loss_analysis/state_tangent_basis_ablation/{name}"] = value

    if stage == "fit":
        if not hasattr(self, "my_total_steps"):
            self.my_total_steps = 0
        self.my_total_steps += 1

        should_log_local_tangent = (
            enable_local_tangent_diagnostics
            and local_tangent_log_every > 0
            and (self.my_total_steps % local_tangent_log_every == 0)
        )
        if should_log_local_tangent:
            delta_tgt_for_local = tgt_eval - anchor_eval
            model_basis_for_local = None
            if transition_bottleneck_type == "state_tangent" and tangent_basis_eval is not None:
                model_basis_for_local = tangent_basis_eval
            local_metrics = compute_local_tangent_diagnostics(
                z=anchor_eval,
                delta=delta_tgt_for_local,
                num_anchors=int(_cfg_get(cfg, "analysis.local_tangent_num_anchors", 128)),
                num_neighbors=int(_cfg_get(cfg, "analysis.local_tangent_num_neighbors", 64)),
                r_basis=int(_cfg_get(cfg, "analysis.local_tangent_basis_dim", 16)),
                model_basis=model_basis_for_local,
            )
            for name, value in local_metrics.items():
                output[f"loss_analysis/local_tangent/{name}"] = value

        should_log_analysis = enable_transition_geometry and log_every > 0 and (self.my_total_steps % log_every == 0)
        if should_log_analysis:
            if not hasattr(self, "transition_geometry_calculator"):
                self.transition_geometry_calculator = TransitionGeometryCalculator(
                    max_jacobian_samples=3,
                    eps=1e-8,
                    min_two_nn_samples=10,
                    enable_standardized_two_nn=True,
                )

            if not ctx_act_for_pred.requires_grad:
                ctx_act_for_pred = ctx_act_for_pred.requires_grad_(True)
            full_metrics = self.transition_geometry_calculator.compute(
                emb=emb,
                pred_emb=pred_eval,
                tgt_emb=tgt_eval,
                act_emb=ctx_act_for_pred,
                ctx_emb=ctx_emb,
                anchor_emb=anchor_eval,
                raw_action=batch.get("action", None),
            )
            _attach_analysis_metrics(output, metric_prefix, full_metrics)

            if metric_prefix == "analysis/full":
                output["mean_rank"] = full_metrics["jacobian_action_emb_pr_rank"].detach()

            shuffle_metrics = None
            if enable_shuffled_action_control and not shuffled_action_training:
                perm = torch.randperm(ctx_act.shape[0], device=ctx_act.device)
                ctx_act_shuf = ctx_act.detach()[perm].clone().requires_grad_(True)
                pred_emb_shuf = self.model.predict(ctx_emb.detach(), ctx_act_shuf)
                if pred_emb_shuf.dim() == 2:
                    pred_emb_shuf = pred_emb_shuf.unsqueeze(1)
                elif pred_emb_shuf.dim() != 3:
                    raise ValueError(
                        f"Expected shuffled pred_emb to be 2D or 3D, got shape={tuple(pred_emb_shuf.shape)}"
                    )
                if transition_eval_slice == "all_shifted":
                    pred_emb_shuf_eval = pred_emb_shuf
                else:
                    pred_emb_shuf_eval = pred_emb_shuf[:, -1:, :]
                if transition_bottleneck_enabled:
                    delta_shuf_raw = pred_emb_shuf_eval - anchor_eval.detach()
                    if transition_bottleneck_type == "state_tangent":
                        delta_shuf_bottleneck, _, _ = self.model.transition_bottleneck(
                            delta_shuf_raw, anchor_eval.detach()
                        )
                    else:
                        delta_shuf_bottleneck, _ = self.model.transition_bottleneck(delta_shuf_raw)
                    pred_emb_shuf_eval = anchor_eval.detach() + delta_shuf_bottleneck
                shuffle_metrics = self.transition_geometry_calculator.compute(
                    emb=emb,
                    pred_emb=pred_emb_shuf_eval,
                    tgt_emb=tgt_eval,
                    act_emb=ctx_act_shuf,
                    ctx_emb=ctx_emb,
                    anchor_emb=anchor_eval,
                    raw_action=batch.get("action", None),
                )
                _attach_analysis_metrics(output, "analysis/shuffle_metric_only", shuffle_metrics)

            if transition_bottleneck_enabled and transition_code is not None:
                with torch.no_grad():
                    code_geom = self.transition_geometry_calculator._geometry_metrics_for_object(
                        transition_code.detach()
                    )
                    for metric_name in ("coord_var_pr_rank", "cov_pr_rank", "id_two_nn"):
                        if metric_name in code_geom:
                            value = code_geom[metric_name]
                            output[f"analysis/full/transition_code/{metric_name}"] = value
                            output[f"loss_analysis/analysis/full/transition_code/{metric_name}"] = value

            if _is_global_zero(self):
                full_dp_id = full_metrics.get("delta_pred/id_two_nn", torch.tensor(float("nan"), device=emb.device))
                full_dt_id = full_metrics.get("delta_tgt/id_two_nn", torch.tensor(float("nan"), device=emb.device))
                full_cos = full_metrics.get(
                    "transition/cosine_alignment_mean", torch.tensor(float("nan"), device=emb.device)
                )
                full_nerr = full_metrics.get(
                    "transition/normalized_transition_error", torch.tensor(float("nan"), device=emb.device)
                )
                msg = (
                    f"[TransitionGeometry step={self.my_total_steps}] "
                    f"full delta_pred ID={full_dp_id.item():.4f}, "
                    f"full delta_tgt ID={full_dt_id.item():.4f}, "
                    f"full cosine={full_cos.item():.4f}, "
                    f"full norm_err={full_nerr.item():.4f}"
                )
                if shuffle_metrics is not None:
                    shuf_cos = shuffle_metrics.get(
                        "transition/cosine_alignment_mean", torch.tensor(float("nan"), device=emb.device)
                    )
                    msg += f", shuffle cosine={shuf_cos.item():.4f}"
                print(msg)

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    if "mean_rank" in output:
        losses_dict[f"{stage}/mean_rank"] = output["mean_rank"]

    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **_predictor_kwargs(cfg),
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    transition_bottleneck_enabled = bool(_cfg_get(cfg, "wm.transition_bottleneck.enabled", False))
    transition_bottleneck_dim = int(_cfg_get(cfg, "wm.transition_bottleneck.dim", 4))
    transition_bottleneck_type = str(_cfg_get(cfg, "wm.transition_bottleneck.type", "linear"))
    if transition_bottleneck_enabled and (
        transition_bottleneck_dim <= 0 or transition_bottleneck_dim > embed_dim
    ):
        raise ValueError(
            f"Invalid transition bottleneck dim={transition_bottleneck_dim}. "
            f"Expected 1 <= k <= embed_dim({embed_dim})."
        )
    if transition_bottleneck_enabled:
        if transition_bottleneck_type in ("linear", "mlp"):
            world_model.transition_bottleneck = TransitionBottleneck(
                input_dim=embed_dim,
                bottleneck_dim=transition_bottleneck_dim,
                bottleneck_type=transition_bottleneck_type,
                hidden_dim=int(_cfg_get(cfg, "wm.transition_bottleneck.hidden_dim", 256)),
                activation=str(_cfg_get(cfg, "wm.transition_bottleneck.activation", "gelu")),
                use_layernorm=bool(_cfg_get(cfg, "wm.transition_bottleneck.use_layernorm", False)),
            )
        elif transition_bottleneck_type == "state_tangent":
            world_model.transition_bottleneck = StateConditionedTangentBottleneck(
                input_dim=embed_dim,
                tangent_dim=transition_bottleneck_dim,
                hidden_dim=int(_cfg_get(cfg, "wm.transition_bottleneck.hidden_dim", 256)),
                code_type=str(_cfg_get(cfg, "wm.transition_bottleneck.code_type", "linear")),
                basis_type=str(_cfg_get(cfg, "wm.transition_bottleneck.basis_type", "mlp")),
                activation=str(_cfg_get(cfg, "wm.transition_bottleneck.activation", "gelu")),
                basis_normalization=str(
                    _cfg_get(cfg, "wm.transition_bottleneck.basis_normalization", "column_norm")
                ),
                use_layernorm=bool(_cfg_get(cfg, "wm.transition_bottleneck.use_layernorm", False)),
            )
        else:
            raise ValueError(
                f"Unsupported transition bottleneck type: {transition_bottleneck_type}. "
                "Expected 'linear', 'mlp', or 'state_tangent'."
            )
    else:
        world_model.transition_bottleneck = None

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": _optimizer_cfg(cfg),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**_sigreg_kwargs(cfg)),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    wandb_enabled, wandb_config = _wandb_cfg(cfg)
    if wandb_enabled:
        logger = WandbLogger(**wandb_config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    # ===============================================================
    rank_store = []
    
    
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.get("output_model_name", "lewm"), epoch_interval=1,
    )

    trainer = pl.Trainer(
        **_trainer_cfg(cfg),
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        # ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
        ckpt_path = None
    )

    manager()
    return


if __name__ == "__main__":
    run()
