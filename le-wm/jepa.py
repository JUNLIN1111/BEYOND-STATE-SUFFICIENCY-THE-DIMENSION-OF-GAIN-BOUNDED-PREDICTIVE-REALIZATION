"""JEPA Implementation"""

import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def _maybe_debug_bottleneck_path(
        self,
        pred_raw: torch.Tensor,
        pred_final: torch.Tensor,
        anchor: torch.Tensor,
        bottleneck_applied: bool,
    ):
        if os.environ.get("DEBUG_BOTTLENECK_PATH") != "1":
            return
        if getattr(self, "_debug_bottleneck_path_printed", False):
            return
        tb = getattr(self, "transition_bottleneck", None)
        raw_norm = torch.norm((pred_raw - anchor).detach().float(), dim=-1).mean()
        final_norm = torch.norm((pred_final - anchor).detach().float(), dim=-1).mean()
        print(
            "[DEBUG_BOTTLENECK_PATH] "
            f"transition_bottleneck_exists={tb is not None}, "
            f"class={tb.__class__.__name__ if tb is not None else 'None'}, "
            f"planning_get_cost_applies_bottleneck={bottleneck_applied}, "
            f"pred_raw.shape={tuple(pred_raw.shape)}, "
            f"pred_final.shape={tuple(pred_final.shape)}, "
            f"mean_norm_raw_minus_anchor={raw_norm.item():.6f}, "
            f"mean_norm_final_minus_anchor={final_norm.item():.6f}",
            flush=True,
        )
        self._debug_bottleneck_path_printed = True

    def _apply_transition_bottleneck_for_rollout(
        self,
        pred_raw: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        tb = getattr(self, "transition_bottleneck", None)
        if tb is None:
            self._maybe_debug_bottleneck_path(pred_raw, pred_raw, anchor, False)
            return pred_raw

        delta_raw = pred_raw - anchor
        if tb.__class__.__name__ == "StateConditionedTangentBottleneck":
            delta_rec, _, _ = tb(delta_raw, anchor)
        else:
            delta_rec, _ = tb(delta_raw)
        pred_final = anchor + delta_rec
        self._maybe_debug_bottleneck_path(pred_raw, pred_final, anchor, True)
        return pred_final

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_raw = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            pred_emb = self._apply_transition_bottleneck_for_rollout(pred_raw, emb_trunc[:, -1:])
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_raw = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        pred_emb = self._apply_transition_bottleneck_for_rollout(pred_raw, emb_trunc[:, -1:])
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        pred_cost_emb = pred_emb[..., -1:, :]
        goal_cost_emb = goal_emb[..., -1:, :].detach()
        cost = F.mse_loss(
            pred_cost_emb,
            goal_cost_emb,
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        if os.environ.get("DEBUG_COST_STATS") == "1" and not getattr(self, "_debug_cost_stats_printed", False):
            pred_norm = torch.norm(pred_cost_emb.detach().float(), dim=-1).mean()
            goal_norm = torch.norm(goal_cost_emb.detach().float(), dim=-1).mean()
            print(
                "[DEBUG_COST_STATS] "
                f"mean_pred_norm={pred_norm.item():.6f}, "
                f"mean_goal_norm={goal_norm.item():.6f}, "
                f"mean_cost={cost.detach().float().mean().item():.6f}",
                flush=True,
            )
            self._debug_cost_stats_printed = True

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
