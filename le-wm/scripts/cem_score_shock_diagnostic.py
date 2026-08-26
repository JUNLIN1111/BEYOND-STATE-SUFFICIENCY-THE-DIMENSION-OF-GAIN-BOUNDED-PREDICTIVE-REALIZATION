from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

from aliasing_experiment import _action_normalization, _make_env, _reset_env_to_state  # noqa: E402
from early_mode_gap_diagnostic import (  # noqa: E402
    EPS,
    _action_sequence_to_raw_future,
    _candidate_pool_path,
    _encode_replay_latents,
    _future_index,
    _group_rows,
    _load_trace_attempts,
    _pool_paths,
    _read_csv,
    _replay_candidate_observations,
    _rollout_predicted_latents,
    _safe_float,
    _safe_int,
    _true_metrics_for_candidate,
)
from planner_success_reference_diagnostic import (  # noqa: E402
    _encode_pixels,
    _extract_pixels_from_obs,
    _find_key,
    _load_model,
    _read_h5_rows,
)
from task_cost import task_cost  # noqa: E402


MODEL_DIMS = {
    "state8": 8,
    "state16": 16,
    "state32": 32,
    "state64": 64,
    "baseline192": 192,
    "state302": 302,
    "state502": 502,
    "global_k32": 32,
    "local_k32": 32,
}


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


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)


def _sqdist(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sum(delta * delta))


def _scaled_sqdist(a: np.ndarray, b: np.ndarray, scale: Optional[np.ndarray]) -> float:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if scale is None:
        return float(np.sum(delta * delta))
    denom = np.asarray(scale, dtype=np.float64).reshape(-1)
    if denom.size != delta.size:
        return float("nan")
    return float(np.sum(delta * delta / np.maximum(denom, EPS)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return float("nan")
    rx = _rankdata(x_arr[mask])
    ry = _rankdata(y_arr[mask])
    if np.std(rx) < EPS or np.std(ry) < EPS:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pairwise_inversion_rate(pred_score: np.ndarray, true_cost: np.ndarray) -> float:
    pred = np.asarray(pred_score, dtype=np.float64)
    true = np.asarray(true_cost, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]
    true = true[mask]
    inversions = 0
    total = 0
    for i in range(pred.size):
        for j in range(i + 1, pred.size):
            true_diff = true[i] - true[j]
            pred_diff = pred[i] - pred[j]
            if abs(true_diff) <= EPS:
                continue
            total += 1
            if true_diff * pred_diff < 0:
                inversions += 1
    return float(inversions / total) if total else float("nan")


def _quantile_threshold(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _read_sorted_h5(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    data_sorted = np.asarray(dataset[sorted_rows])
    data = np.empty_like(data_sorted)
    data[order] = data_sorted
    return data


def _estimate_latent_variance(
    model,
    h5: h5py.File,
    pixels_key: str,
    num_rows: int,
    img_size: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    total = int(h5[pixels_key].shape[0])
    count = min(int(num_rows), total)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(total, size=count, replace=False))
    pixels = _read_sorted_h5(h5[pixels_key], rows)
    with torch.no_grad():
        z = _encode_pixels(model, pixels, img_size, batch_size, device)[:, 0]
    var = np.var(np.asarray(z, dtype=np.float64), axis=0)
    var = np.maximum(var, 1e-8)
    return var.astype(np.float32), {
        "variance_rows": int(count),
        "latent_dim": int(var.shape[0]),
        "latent_variance_mean": float(np.mean(var)),
        "latent_variance_min": float(np.min(var)),
        "latent_variance_max": float(np.max(var)),
    }


def _distance_pack(prefix: str, a: np.ndarray, b: np.ndarray, latent_dim: int, latent_var: Optional[np.ndarray]) -> Dict[str, float]:
    raw = _sqdist(a, b)
    return {
        f"{prefix}_raw_l2sq": raw,
        f"{prefix}_per_dim_l2sq": raw / max(1, int(latent_dim)),
        f"{prefix}_varnorm_l2sq": _scaled_sqdist(a, b, latent_var),
    }


def _cost_from_replay_meta(meta: Dict[str, object], goal_state: np.ndarray) -> float:
    terminal = meta.get("terminal_state")
    if terminal is None:
        return float("nan")
    try:
        return float(task_cost(np.asarray(terminal, dtype=np.float64), goal_state))
    except Exception:
        return float("nan")


def _candidate_indices_for_call(pool: np.lib.npyio.NpzFile, replay_rows: Sequence[Dict[str, str]], topk: int, random_count: int, seed: int) -> np.ndarray:
    costs = np.asarray(pool["terminal_cost_c25"], dtype=np.float64)
    indices = set(np.argsort(costs)[: min(topk, costs.shape[0])].astype(int).tolist())
    for key in ("best_idx_argmin_cost", "selected_candidate_idx"):
        if key in pool.files:
            indices.add(int(np.asarray(pool[key]).reshape(())))
    for row in replay_rows:
        idx = _safe_int(row.get("candidate_idx"))
        if idx >= 0:
            indices.add(idx)
    if random_count > 0 and costs.shape[0] > 0:
        rng = np.random.default_rng(seed)
        random = rng.choice(costs.shape[0], size=min(random_count, costs.shape[0]), replace=False)
        indices.update(int(x) for x in random)
    return np.asarray(sorted(idx for idx in indices if 0 <= idx < costs.shape[0]), dtype=np.int64)


def _append_prefix_rows(
    rows: List[Dict[str, object]],
    episode_id: int,
    mpc_step_idx: int,
    model_name: str,
    latent_dim: int,
    candidate_idx: int,
    candidate_label: str,
    steps: Sequence[int],
    planning_horizon_raw: int,
    predicted: np.ndarray,
    candidate_pos: int,
    real_latents: Dict[int, Dict[int, np.ndarray]],
    replay_meta_by_candidate: Dict[int, Dict[str, object]],
    zg: np.ndarray,
    latent_var: Optional[np.ndarray],
) -> None:
    pred_len = predicted.shape[1]
    for step in steps:
        rollout_idx = 0 if int(step) == 0 else _future_index(pred_len, int(step), planning_horizon_raw)
        z_pred = predicted[candidate_pos, rollout_idx]
        z_actual = real_latents.get(int(candidate_idx), {}).get(int(step))
        if z_actual is None:
            continue
        row = {
            "episode_id": int(episode_id),
            "mpc_step_idx": int(mpc_step_idx),
            "model": model_name,
            "latent_dim": int(latent_dim),
            "candidate_idx": int(candidate_idx),
            "candidate_label": candidate_label,
            "k_raw": int(step),
            "model_rollout_index": int(rollout_idx),
        }
        row.update(_distance_pack("s_pred", z_pred, zg, latent_dim, latent_var))
        row.update(_distance_pack("s_actual", z_actual, zg, latent_dim, latent_var))
        row.update(_distance_pack("e_roll", z_pred, z_actual, latent_dim, latent_var))
        task_cost_by_step = replay_meta_by_candidate.get(int(candidate_idx), {}).get("task_cost_by_step", {})
        if isinstance(task_cost_by_step, dict):
            row["true_task_cost_at_k"] = task_cost_by_step.get(int(step), task_cost_by_step.get(str(int(step)), float("nan")))
        else:
            row["true_task_cost_at_k"] = float("nan")
        for suffix in ("raw_l2sq", "per_dim_l2sq", "varnorm_l2sq"):
            row[f"shock_{suffix}"] = row[f"s_actual_{suffix}"] - row[f"s_pred_{suffix}"]
        rows.append(row)


def _append_path_order_rows(prefix_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in prefix_rows:
        key = (row.get("episode_id"), row.get("mpc_step_idx"), row.get("model"), row.get("candidate_idx"))
        grouped[key].append(row)
    out: List[Dict[str, object]] = []
    for (episode_id, mpc_step_idx, model, candidate_idx), group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item.get("k_raw", 0)))
        reversals = 0
        comparable = 0
        for before, after in zip(ordered, ordered[1:]):
            task_before = _safe_float(before.get("true_task_cost_at_k"))
            task_after = _safe_float(after.get("true_task_cost_at_k"))
            score_before = _safe_float(before.get("s_actual_raw_l2sq"))
            score_after = _safe_float(after.get("s_actual_raw_l2sq"))
            if not all(math.isfinite(v) for v in (task_before, task_after, score_before, score_after)):
                continue
            if task_after < task_before:
                comparable += 1
                reversals += int(score_after > score_before)
        out.append(
            {
                "episode_id": episode_id,
                "mpc_step_idx": mpc_step_idx,
                "model": model,
                "candidate_idx": candidate_idx,
                "path_order_reversal_task_proxy_count": int(reversals),
                "task_improving_prefix_count": int(comparable),
                "path_order_reversal_task_proxy_rate": float(reversals / comparable) if comparable else float("nan"),
                "true_distance_semantics": "task_cost_proxy_not_shortest_path",
            }
        )
    return out


def _summarize_group(rows: List[Dict[str, object]], keys: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for key_values, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        row = {key: value for key, value in zip(keys, key_values)}
        row["num_rows"] = len(group)
        for metric in metrics:
            arr = np.asarray([_safe_float(item.get(metric)) for item in group], dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_ci95_low"] = float(np.mean(finite) - 1.96 * np.std(finite) / math.sqrt(finite.size)) if finite.size else float("nan")
            row[f"{metric}_ci95_high"] = float(np.mean(finite) + 1.96 * np.std(finite) / math.sqrt(finite.size)) if finite.size else float("nan")
        out.append(row)
    return out


def _write_report(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CEM Score-Shock Diagnostic",
        "",
        "This harness is post-hoc only: it reads saved CEM trace files and does not change model, training, or planner behavior.",
        "",
        "## Code Path Audit",
        "",
        "- Encoder: `jepa.py:31`, `JEPA.encode`; pixels -> ViT CLS token -> projector -> `info['emb']`.",
        "- Dynamics predictor: `jepa.py:49`, `JEPA.predict`; autoregressive rollout uses `jepa.py:109`.",
        "- Planner score: `jepa.py:192`, `JEPA.get_cost`; final terminal cost is `criterion` at `jepa.py:162`, summed squared latent MSE to encoded goal.",
        "- CEM/eval entry: `eval.py:100-124`; `stable_worldmodel` CEM solver calls `model.get_cost` through `WorldModelPolicy`.",
        "- PushT reset/clone path: `scripts/aliasing_experiment.py` helpers `_reset_env_to_state`, `_get_env_state`, `_step_env`; trace uses exact reset and validates state when available.",
        "- Latent dim config: `config/train/lewm.yaml:45-50` via `wm.embed_dim`; model module uses `config/train/model/lewm.yaml:11-42`.",
        "- Eval horizon: `config/eval/pusht.yaml:21-24`; horizon=5 blocks, action_block=5 raw steps, so terminal CEM horizon is 25 raw steps.",
        "",
        "## Replay Semantics",
        "",
        "- Environment cloning is implemented as deterministic reset-to-state plus fixed action replay, not a general simulator snapshot API.",
        "- The reset helper validates comparable PushT pose coordinates and refuses true-cost replay if the reset mismatch exceeds tolerance.",
        "- Determinism is assumed conditional on the saved simulator state and fixed actions; this diagnostic exposes `executed_raw_steps`/`done` in the trace outputs so early termination is visible.",
        "- For old traces, `selected_candidate_idx` may be a nearest sampled candidate proxy. New traces also save `solver_return_sequence` in candidate-pool `.npz` files when the solver exposes it.",
        "",
        "## Distance Scales",
        "",
        "Every score is reported as raw squared L2, raw/D, and training-set latent-variance-normalized squared L2. The last form is `sum_j (a_j-b_j)^2 / Var[z_j]` using sampled dataset encodings.",
        "",
        "## True Distance Limitation",
        "",
        "PushT does not provide an exact shortest-path distance in this harness. `true_terminal_task_cost` is the task-cost proxy from `scripts/task_cost.py`, based on block pose alignment, not a graph shortest path.",
        "",
        "## Inputs",
        "",
        f"- trace_dir: `{payload.get('trace_dir')}`",
        f"- dataset: `{payload.get('dataset')}`",
        f"- models: {', '.join(payload.get('models', []))}",
        f"- processed_calls: {payload.get('processed_calls')}",
        f"- skipped_rows: {payload.get('skipped_rows')}",
        "",
        "## Outputs",
        "",
        "- `selected_rollout_prefix_curves.csv`: selected/top/oracle candidate prefix curves `S_pred`, `S_actual`, `E_roll`, and shock.",
        "- `candidate_terminal_scores.csv`: same candidate set, terminal predicted score, actual encoded score, rollout error, task-cost proxy.",
        "- `candidate_ranking_decomposition.csv`: Spearman/pairwise/top-1/oracle-rank/regret for prediction ranking and latent-geometry ranking.",
        "- `failure_classification.csv`: heuristic labels for hallucination, representation collision, search failure, and path-order reversal proxy.",
        "- `path_order_reversal_proxy.csv`: task-cost proxy improves while encoded latent goal score worsens.",
        "- `dimension_diagnostic_summary.csv`: aggregate score shock, ranking, false-perfect, and failure-label rates by model/dimension.",
        "- `figures/`: quick paper-check plots for prefix curves, ranking decomposition, and failure labels.",
        "",
    ]
    path.write_text("\n".join(lines))


def _plot_outputs(output_dir: Path, prefix_rows: List[Dict[str, object]], ranking_rows: List[Dict[str, object]], failure_rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[score_shock] matplotlib unavailable; skipped plots: {exc}", flush=True)
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if prefix_rows:
        metrics = [
            ("s_pred_per_dim_l2sq", "predicted score / D"),
            ("s_actual_per_dim_l2sq", "actual encoded score / D"),
            ("e_roll_per_dim_l2sq", "rollout error / D"),
            ("shock_per_dim_l2sq", "score shock / D"),
        ]
        models = sorted({str(row["model"]) for row in prefix_rows}, key=lambda name: MODEL_DIMS.get(name, 10_000))
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for ax, (metric, title) in zip(axes.ravel(), metrics):
            for model in models:
                model_rows = [row for row in prefix_rows if str(row["model"]) == model and "selected" in str(row.get("candidate_label", ""))]
                by_step: Dict[int, List[float]] = defaultdict(list)
                for row in model_rows:
                    by_step[int(row["k_raw"])].append(_safe_float(row.get(metric)))
                xs = sorted(by_step)
                if not xs:
                    continue
                ys = [float(np.nanmean(by_step[x])) for x in xs]
                ax.plot(xs, ys, marker="o", label=model)
            ax.set_title(title)
            ax.set_xlabel("raw replay step k")
            ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "selected_prefix_score_shock_curves.png", dpi=220)
        fig.savefig(fig_dir / "selected_prefix_score_shock_curves.pdf")
        plt.close(fig)

    if ranking_rows:
        metrics = [
            ("rho_pred_vs_actual_encoded_score", "pred vs actual score Spearman"),
            ("rho_actual_encoded_score_vs_task_cost_proxy", "actual score vs task-cost Spearman"),
            ("pairwise_inversion_pred_vs_task_cost", "pred vs task pairwise inversion"),
            ("pred_top1_task_regret", "pred top-1 task regret"),
        ]
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in ranking_rows:
            grouped[str(row["model"])].append(row)
        models = sorted(grouped, key=lambda name: MODEL_DIMS.get(name, 10_000))
        xs = np.arange(len(models))
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (metric, title) in zip(axes.ravel(), metrics):
            ys = [float(np.nanmean([_safe_float(row.get(metric)) for row in grouped[model]])) for model in models]
            ax.bar(xs, ys, color="#4C78A8")
            ax.set_xticks(xs)
            ax.set_xticklabels(models, rotation=30, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "candidate_ranking_decomposition.png", dpi=220)
        fig.savefig(fig_dir / "candidate_ranking_decomposition.pdf")
        plt.close(fig)

    if failure_rows:
        grouped = defaultdict(list)
        for row in failure_rows:
            grouped[str(row["model"])].append(row)
        models = sorted(grouped, key=lambda name: MODEL_DIMS.get(name, 10_000))
        labels = ["prediction_hallucination", "representation_collision_task_proxy"]
        xs = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        for offset, label in enumerate(labels):
            ys = [float(np.nanmean([_safe_float(row.get(label)) for row in grouped[model]])) for model in models]
            ax.bar(xs + (offset - 0.5) * width, ys, width=width, label=label)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("fraction of audited candidate rows")
        ax.set_title("Failure label rates")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "failure_classification_rates.png", dpi=220)
        fig.savefig(fig_dir / "failure_classification_rates.pdf")
        plt.close(fig)


def _mean_finite(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = np.asarray([_safe_float(row.get(key)) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _quantile_finite(rows: Sequence[Dict[str, object]], key: str, q: float) -> float:
    values = np.asarray([_safe_float(row.get(key)) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def _dimension_summary(
    terminal_rows: List[Dict[str, object]],
    ranking_rows: List[Dict[str, object]],
    failure_rows: List[Dict[str, object]],
    path_order_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    models = sorted({str(row.get("model")) for row in terminal_rows + ranking_rows + failure_rows}, key=lambda name: MODEL_DIMS.get(name, 10_000))
    out: List[Dict[str, object]] = []
    for model in models:
        t_rows = [row for row in terminal_rows if str(row.get("model")) == model]
        r_rows = [row for row in ranking_rows if str(row.get("model")) == model]
        f_rows = [row for row in failure_rows if str(row.get("model")) == model]
        p_rows = [row for row in path_order_rows if str(row.get("model")) == model]
        out.append(
            {
                "model": model,
                "latent_dim": MODEL_DIMS.get(model, _safe_int(t_rows[0].get("latent_dim")) if t_rows else -1),
                "num_terminal_candidate_rows": len(t_rows),
                "num_planning_calls": len(r_rows),
                "mean_terminal_score_shock_per_dim": _mean_finite(t_rows, "terminal_score_shock_per_dim_l2sq"),
                "p95_terminal_score_shock_per_dim": _quantile_finite(t_rows, "terminal_score_shock_per_dim_l2sq", 0.95),
                "mean_terminal_rollout_error_per_dim": _mean_finite(t_rows, "terminal_rollout_error_per_dim_l2sq"),
                "mean_rho_pred_vs_actual_encoded_score": _mean_finite(r_rows, "rho_pred_vs_actual_encoded_score"),
                "mean_rho_actual_encoded_score_vs_task_cost_proxy": _mean_finite(r_rows, "rho_actual_encoded_score_vs_task_cost_proxy"),
                "mean_rho_pred_score_vs_task_cost_proxy": _mean_finite(r_rows, "rho_pred_score_vs_task_cost_proxy"),
                "mean_pairwise_inversion_pred_vs_task_cost": _mean_finite(r_rows, "pairwise_inversion_pred_vs_task_cost"),
                "mean_pairwise_inversion_actual_vs_task_cost": _mean_finite(r_rows, "pairwise_inversion_actual_vs_task_cost"),
                "pred_top1_task_oracle_agreement": _mean_finite(r_rows, "pred_top1_equals_task_oracle"),
                "actual_top1_task_oracle_agreement": _mean_finite(r_rows, "actual_top1_equals_task_oracle"),
                "mean_pred_top1_task_regret": _mean_finite(r_rows, "pred_top1_task_regret"),
                "prediction_hallucination_rate": _mean_finite(f_rows, "prediction_hallucination"),
                "representation_collision_task_proxy_rate": _mean_finite(f_rows, "representation_collision_task_proxy"),
                "false_perfect_task_proxy_rate": _mean_finite(f_rows, "false_perfect_task_proxy"),
                "path_order_reversal_task_proxy_rate": _mean_finite(p_rows, "path_order_reversal_task_proxy_rate"),
                "true_distance_semantics": "task_cost_proxy_not_shortest_path",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc CEM score-shock/ranking diagnostic for LeWM PushT traces.")
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint_object entries.")
    parser.add_argument("--output_dir", default="results/cem_score_shock_diagnostic")
    parser.add_argument("--steps", nargs="*", type=int, default=[0, 5, 10, 15, 20, 25])
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--random_candidates", type=int, default=30)
    parser.add_argument("--max_calls", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--env_name", default="swm/PushT-v1")
    parser.add_argument("--reset_state_tol", type=float, default=1e-3)
    parser.add_argument("--candidate_actions_are_normalized", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    parser.add_argument("--latent_variance_rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hallucination_shock_quantile", type=float, default=0.75)
    parser.add_argument("--collision_actual_quantile", type=float, default=0.10)
    parser.add_argument("--collision_task_quantile", type=float, default=0.75)
    parser.add_argument("--false_perfect_pred_epsilon", type=float, default=float("nan"), help="Raw-L2 predicted terminal score threshold. Default uses 10th percentile.")
    parser.add_argument("--false_perfect_task_epsilon", type=float, default=float("nan"), help="Task-cost threshold. Default uses 75th percentile.")
    args = parser.parse_args()

    if HDF5PLUGIN_AVAILABLE:
        print("[score_shock] hdf5plugin available; compressed HDF5 filters enabled.", flush=True)

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models = _parse_name_paths(args.models)

    pool_rows = _read_csv(trace_dir / "candidate_pool_summary.csv")
    replay_rows_all = _read_csv(trace_dir / "candidate_true_replay_audit.csv")
    if not replay_rows_all:
        replay_rows_all = _read_csv(trace_dir / "true_replay_candidates.csv")
    attempts = _load_trace_attempts(trace_dir)
    pool_rows_by_call = _group_rows(pool_rows, ["episode_id", "mpc_step_idx"])
    replay_rows_by_call = _group_rows(replay_rows_all, ["episode_id", "mpc_step_idx"])
    fallback_pool_paths = _pool_paths(trace_dir)
    if not pool_rows or not replay_rows_by_call:
        raise RuntimeError("Trace dir must contain candidate_pool_summary.csv and candidate_true_replay_audit.csv / true_replay_candidates.csv.")

    prefix_rows: List[Dict[str, object]] = []
    terminal_rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    skip_rows: List[Dict[str, object]] = []
    variance_rows: List[Dict[str, object]] = []
    latent_vars: Dict[str, np.ndarray] = {}

    env = _make_env(args.env_name)
    planning_horizon_raw = max(args.steps)
    processed_calls = 0

    with h5py.File(args.dataset, "r") as h5:
        pixels_key = _find_key(h5, [args.pixels_key, "pixels", "observation/pixels"])
        state_key = _find_key(h5, [args.state_key, "state"])
        action_key = _find_key(h5, [args.action_key, "action"])
        action_mean, action_std = _action_normalization(np.asarray(h5[action_key]).astype(np.float32))

        loaded_models = {}
        for model_name, checkpoint in models.items():
            print(f"[score_shock] loading {model_name}: {checkpoint}", flush=True)
            model = _load_model(checkpoint, device)
            model.eval()
            var, info = _estimate_latent_variance(model, h5, pixels_key, args.latent_variance_rows, args.img_size, args.batch_size, device, args.seed)
            latent_vars[model_name] = var
            info.update({"model": model_name, "checkpoint_object": str(checkpoint)})
            variance_rows.append(info)
            loaded_models[model_name] = model

        call_keys = sorted(replay_rows_by_call)
        if args.max_calls > 0:
            call_keys = call_keys[: args.max_calls]

        for call_number, (episode_id, mpc_step_idx) in enumerate(call_keys, start=1):
            row_group = pool_rows_by_call.get((episode_id, mpc_step_idx), [])
            pool_path = _candidate_pool_path(row_group, fallback_pool_paths, (episode_id, mpc_step_idx))
            if pool_path is None or not pool_path.exists():
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "missing_candidate_pool_npz"})
                continue
            pool = np.load(pool_path, allow_pickle=False)
            if "action_sequence" not in pool.files:
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "pool_missing_action_sequence"})
                continue
            if "current_state" in pool.files and "goal_state" in pool.files:
                start_state = np.asarray(pool["current_state"], dtype=np.float64).reshape(-1)
                goal_state = np.asarray(pool["goal_state"], dtype=np.float64).reshape(-1)
                obs, _info = _reset_env_to_state(env, start_state, goal_state, args.reset_state_tol)
                context_pixels = np.asarray([_extract_pixels_from_obs(env, obs)])
                attempt = attempts.get(episode_id, {})
                goal_row = _safe_int(attempt.get("goal_row"))
                goal_pixels = _read_h5_rows(h5[pixels_key], np.asarray([goal_row], dtype=np.int64)) if goal_row >= 0 else context_pixels
            else:
                attempt = attempts.get(episode_id, {})
                start_row = _safe_int(attempt.get("start_row"))
                goal_row = _safe_int(attempt.get("goal_row"))
                if start_row < 0 or goal_row < 0:
                    skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "missing_start_goal_state"})
                    continue
                start_state = np.asarray(h5[state_key][start_row], dtype=np.float64)
                goal_state = np.asarray(h5[state_key][goal_row], dtype=np.float64)
                context_pixels = _read_h5_rows(h5[pixels_key], np.asarray([start_row], dtype=np.int64))
                goal_pixels = _read_h5_rows(h5[pixels_key], np.asarray([goal_row], dtype=np.int64))

            replay_rows = replay_rows_by_call[(episode_id, mpc_step_idx)]
            candidate_indices = _candidate_indices_for_call(pool, replay_rows, args.topk, args.random_candidates, args.seed + call_number)
            if candidate_indices.size == 0:
                skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "skip_reason": "no_candidate_indices"})
                continue
            action_sequence = np.asarray(pool["action_sequence"], dtype=np.float32)
            pixels_by_candidate: Dict[int, Dict[int, np.ndarray]] = {}
            replay_meta_by_candidate: Dict[int, Dict[str, object]] = {}
            for candidate_idx in candidate_indices:
                try:
                    raw_actions = _action_sequence_to_raw_future(
                        action_sequence[int(candidate_idx)],
                        action_mean,
                        action_std,
                        planning_horizon_raw,
                        args.candidate_actions_are_normalized,
                    )
                    pixels_by_step, replay_meta = _replay_candidate_observations(
                        env,
                        start_state,
                        goal_state,
                        raw_actions,
                        args.steps,
                        args.reset_state_tol,
                    )
                    pixels_by_candidate[int(candidate_idx)] = pixels_by_step
                    replay_meta_by_candidate[int(candidate_idx)] = replay_meta
                except Exception as exc:  # noqa: BLE001
                    skip_rows.append({"episode_id": episode_id, "mpc_step_idx": mpc_step_idx, "candidate_idx": int(candidate_idx), "skip_reason": f"replay_failed: {exc}"})
            candidate_indices = np.asarray([idx for idx in candidate_indices if int(idx) in pixels_by_candidate], dtype=np.int64)
            if candidate_indices.size == 0:
                continue

            true_metrics = _true_metrics_for_candidate(replay_rows)
            start_cost = float(task_cost(start_state, goal_state))
            pool_costs = np.asarray(pool["terminal_cost_c25"], dtype=np.float64)
            selected_idx = int(np.asarray(pool["selected_candidate_idx"]).reshape(())) if "selected_candidate_idx" in pool.files else int(np.nanargmin(pool_costs))
            pred_best_idx = int(np.nanargmin(pool_costs))
            true_cost_by_candidate = {}
            for candidate_idx in candidate_indices:
                meta_cost = _cost_from_replay_meta(replay_meta_by_candidate[int(candidate_idx)], goal_state)
                csv_cost = true_metrics.get(int(candidate_idx), {}).get("true_terminal_task_cost", float("nan"))
                true_cost = csv_cost if math.isfinite(csv_cost) else meta_cost
                true_cost_by_candidate[int(candidate_idx)] = true_cost
            finite_true = {idx: cost for idx, cost in true_cost_by_candidate.items() if math.isfinite(cost)}
            true_best_idx = min(finite_true, key=finite_true.get) if finite_true else -1
            label_by_idx = {int(idx): "candidate" for idx in candidate_indices}
            for idx, label in ((selected_idx, "selected"), (pred_best_idx, "pred_best"), (true_best_idx, "task_best")):
                if idx in label_by_idx:
                    label_by_idx[idx] = label if label_by_idx[idx] == "candidate" else label_by_idx[idx] + "+" + label

            for model_name, model in loaded_models.items():
                print(f"[score_shock] call {call_number}/{len(call_keys)} ep={episode_id} step={mpc_step_idx} model={model_name}", flush=True)
                with torch.no_grad():
                    predicted, z0, _selected_actions = _rollout_predicted_latents(
                        model,
                        context_pixels,
                        action_sequence,
                        candidate_indices,
                        args.img_size,
                        args.batch_size,
                        device,
                    )
                    zg = _encode_pixels(model, goal_pixels, args.img_size, args.batch_size, device)[0, 0]
                    real_latents = _encode_replay_latents(
                        model,
                        pixels_by_candidate,
                        candidate_indices,
                        args.steps,
                        args.img_size,
                        args.batch_size,
                        device,
                    )
                latent_dim = MODEL_DIMS.get(model_name, int(z0.shape[-1]))
                latent_var = latent_vars.get(model_name)
                pred_len = predicted.shape[1]
                terminal_step = int(planning_horizon_raw)
                terminal_rollout_idx = _future_index(pred_len, terminal_step, planning_horizon_raw)
                pos_by_candidate = {int(idx): pos for pos, idx in enumerate(candidate_indices)}
                pred_scores = []
                actual_scores = []
                true_costs = []
                rows_for_model = []

                for candidate_idx in candidate_indices:
                    pos = pos_by_candidate[int(candidate_idx)]
                    z_pred_terminal = predicted[pos, terminal_rollout_idx]
                    z_actual_terminal = real_latents.get(int(candidate_idx), {}).get(terminal_step)
                    if z_actual_terminal is None:
                        continue
                    true_terminal_cost = true_cost_by_candidate.get(int(candidate_idx), float("nan"))
                    true_progress = start_cost - true_terminal_cost if math.isfinite(true_terminal_cost) else float("nan")
                    row = {
                        "episode_id": int(episode_id),
                        "mpc_step_idx": int(mpc_step_idx),
                        "model": model_name,
                        "latent_dim": int(latent_dim),
                        "candidate_idx": int(candidate_idx),
                        "candidate_label": label_by_idx.get(int(candidate_idx), "candidate"),
                        "terminal_k_raw": int(terminal_step),
                        "terminal_rollout_index": int(terminal_rollout_idx),
                        "trace_model_predicted_cost_c25": float(pool_costs[int(candidate_idx)]) if int(candidate_idx) < pool_costs.size else float("nan"),
                        "true_start_task_cost": start_cost,
                        "true_terminal_task_cost": true_terminal_cost,
                        "true_terminal_progress": true_progress,
                        "selected_by_trace": int(int(candidate_idx) == selected_idx),
                        "pred_best_by_trace": int(int(candidate_idx) == pred_best_idx),
                        "task_best_in_audited_set": int(int(candidate_idx) == true_best_idx),
                    }
                    row.update(_distance_pack("pred_terminal_score", z_pred_terminal, zg, latent_dim, latent_var))
                    row.update(_distance_pack("actual_encoded_terminal_score", z_actual_terminal, zg, latent_dim, latent_var))
                    row.update(_distance_pack("terminal_rollout_error", z_pred_terminal, z_actual_terminal, latent_dim, latent_var))
                    for suffix in ("raw_l2sq", "per_dim_l2sq", "varnorm_l2sq"):
                        row[f"terminal_score_shock_{suffix}"] = row[f"actual_encoded_terminal_score_{suffix}"] - row[f"pred_terminal_score_{suffix}"]
                    terminal_rows.append(row)
                    rows_for_model.append(row)
                    pred_scores.append(row["pred_terminal_score_raw_l2sq"])
                    actual_scores.append(row["actual_encoded_terminal_score_raw_l2sq"])
                    true_costs.append(true_terminal_cost)

                for candidate_idx in (selected_idx, pred_best_idx, true_best_idx):
                    if candidate_idx in pos_by_candidate:
                        _append_prefix_rows(
                            prefix_rows,
                            episode_id,
                            mpc_step_idx,
                            model_name,
                            latent_dim,
                            candidate_idx,
                            label_by_idx.get(candidate_idx, "candidate"),
                            args.steps,
                            planning_horizon_raw,
                            predicted,
                            pos_by_candidate[candidate_idx],
                            real_latents,
                            replay_meta_by_candidate,
                            zg,
                            latent_var,
                        )

                if rows_for_model:
                    pred_arr = np.asarray([_safe_float(row["pred_terminal_score_raw_l2sq"]) for row in rows_for_model], dtype=np.float64)
                    actual_arr = np.asarray([_safe_float(row["actual_encoded_terminal_score_raw_l2sq"]) for row in rows_for_model], dtype=np.float64)
                    true_arr = np.asarray([_safe_float(row["true_terminal_task_cost"]) for row in rows_for_model], dtype=np.float64)
                    candidate_arr = np.asarray([_safe_int(row["candidate_idx"]) for row in rows_for_model], dtype=np.int64)
                    finite_mask = np.isfinite(true_arr)
                    pred_top_idx = int(candidate_arr[int(np.nanargmin(pred_arr))]) if pred_arr.size else -1
                    actual_top_idx = int(candidate_arr[int(np.nanargmin(actual_arr))]) if actual_arr.size else -1
                    oracle_pos = int(np.nanargmin(true_arr[finite_mask])) if finite_mask.any() else -1
                    oracle_idx = int(candidate_arr[finite_mask][oracle_pos]) if oracle_pos >= 0 else -1
                    pred_order = np.argsort(pred_arr)
                    oracle_rank = int(np.where(candidate_arr[pred_order] == oracle_idx)[0][0] + 1) if oracle_idx in set(candidate_arr.tolist()) else -1
                    pred_best_true_cost = true_arr[candidate_arr == pred_top_idx][0] if pred_top_idx in set(candidate_arr.tolist()) else float("nan")
                    oracle_true_cost = true_arr[candidate_arr == oracle_idx][0] if oracle_idx in set(candidate_arr.tolist()) else float("nan")
                    ranking_rows.append(
                        {
                            "episode_id": int(episode_id),
                            "mpc_step_idx": int(mpc_step_idx),
                            "model": model_name,
                            "latent_dim": int(latent_dim),
                            "num_candidates_audited": int(len(rows_for_model)),
                            "rho_pred_vs_actual_encoded_score": _spearman(pred_arr, actual_arr),
                            "rho_actual_encoded_score_vs_task_cost_proxy": _spearman(actual_arr, true_arr),
                            "rho_pred_score_vs_task_cost_proxy": _spearman(pred_arr, true_arr),
                            "pairwise_inversion_pred_vs_task_cost": _pairwise_inversion_rate(pred_arr, true_arr),
                            "pairwise_inversion_actual_vs_task_cost": _pairwise_inversion_rate(actual_arr, true_arr),
                            "pred_top1_candidate_idx": pred_top_idx,
                            "actual_encoded_top1_candidate_idx": actual_top_idx,
                            "task_oracle_candidate_idx": oracle_idx,
                            "pred_top1_equals_task_oracle": int(pred_top_idx == oracle_idx),
                            "actual_top1_equals_task_oracle": int(actual_top_idx == oracle_idx),
                            "oracle_rank_by_predicted_score": oracle_rank,
                            "pred_top1_task_regret": float(pred_best_true_cost - oracle_true_cost) if math.isfinite(pred_best_true_cost) and math.isfinite(oracle_true_cost) else float("nan"),
                        }
                    )

            processed_calls += 1

    shock_threshold = _quantile_threshold([row["terminal_score_shock_raw_l2sq"] for row in terminal_rows], args.hallucination_shock_quantile)
    actual_low = _quantile_threshold([row["actual_encoded_terminal_score_raw_l2sq"] for row in terminal_rows], args.collision_actual_quantile)
    task_high = _quantile_threshold([row["true_terminal_task_cost"] for row in terminal_rows], args.collision_task_quantile)
    false_perfect_pred_epsilon = args.false_perfect_pred_epsilon
    if not math.isfinite(false_perfect_pred_epsilon):
        false_perfect_pred_epsilon = _quantile_threshold([row["pred_terminal_score_raw_l2sq"] for row in terminal_rows], 0.10)
    false_perfect_task_epsilon = args.false_perfect_task_epsilon
    if not math.isfinite(false_perfect_task_epsilon):
        false_perfect_task_epsilon = task_high
    for row in terminal_rows:
        shock = _safe_float(row.get("terminal_score_shock_raw_l2sq"))
        actual = _safe_float(row.get("actual_encoded_terminal_score_raw_l2sq"))
        task = _safe_float(row.get("true_terminal_task_cost"))
        pred = _safe_float(row.get("pred_terminal_score_raw_l2sq"))
        label = str(row.get("candidate_label", ""))
        failure_rows.append(
            {
                "episode_id": row["episode_id"],
                "mpc_step_idx": row["mpc_step_idx"],
                "model": row["model"],
                "latent_dim": row["latent_dim"],
                "candidate_idx": row["candidate_idx"],
                "candidate_label": label,
                "prediction_hallucination": int(math.isfinite(shock) and shock > shock_threshold and pred < actual),
                "representation_collision_task_proxy": int(math.isfinite(actual) and math.isfinite(task) and actual <= actual_low and task >= task_high),
                "false_perfect_task_proxy": int(
                    math.isfinite(pred)
                    and math.isfinite(task)
                    and pred < false_perfect_pred_epsilon
                    and task > false_perfect_task_epsilon
                ),
                "candidate_is_predicted_best": int("pred_best" in label),
                "candidate_is_task_best": int("task_best" in label),
                "true_terminal_task_cost": task,
                "pred_terminal_score_raw_l2sq": pred,
                "actual_encoded_terminal_score_raw_l2sq": actual,
                "terminal_score_shock_raw_l2sq": shock,
                "hallucination_shock_threshold": shock_threshold,
                "collision_actual_low_threshold": actual_low,
                "collision_task_high_threshold": task_high,
                "false_perfect_pred_epsilon": false_perfect_pred_epsilon,
                "false_perfect_task_epsilon": false_perfect_task_epsilon,
                "true_distance_semantics": "task_cost_proxy_not_shortest_path",
            }
        )

    path_order_rows = _append_path_order_rows(prefix_rows)
    dimension_summary_rows = _dimension_summary(terminal_rows, ranking_rows, failure_rows, path_order_rows)

    _write_csv(output_dir / "latent_variance_summary.csv", variance_rows)
    _write_csv(output_dir / "selected_rollout_prefix_curves.csv", prefix_rows)
    _write_csv(output_dir / "selected_rollout_prefix_summary.csv", _summarize_group(
        prefix_rows,
        ["model", "latent_dim", "candidate_label", "k_raw"],
        ["s_pred_raw_l2sq", "s_actual_raw_l2sq", "e_roll_raw_l2sq", "shock_raw_l2sq", "s_pred_per_dim_l2sq", "s_actual_per_dim_l2sq", "s_pred_varnorm_l2sq", "s_actual_varnorm_l2sq"],
    ))
    _write_csv(output_dir / "candidate_terminal_scores.csv", terminal_rows)
    _write_csv(output_dir / "candidate_ranking_decomposition.csv", ranking_rows)
    _write_csv(output_dir / "failure_classification.csv", failure_rows)
    _write_csv(output_dir / "path_order_reversal_proxy.csv", path_order_rows)
    _write_csv(output_dir / "dimension_diagnostic_summary.csv", dimension_summary_rows)
    _write_csv(output_dir / "skipped_calls.csv", skip_rows)
    _write_json(output_dir / "run_metadata.json", {
        "trace_dir": str(trace_dir),
        "dataset": str(args.dataset),
        "models": list(models.keys()),
        "processed_calls": processed_calls,
        "skipped_rows": len(skip_rows),
        "distance_scales": ["raw_l2sq", "per_dim_l2sq", "varnorm_l2sq"],
        "true_distance_semantics": "task_cost_proxy_not_shortest_path",
        "planning_horizon_raw": planning_horizon_raw,
        "steps": args.steps,
        "false_perfect_pred_epsilon": false_perfect_pred_epsilon,
        "false_perfect_task_epsilon": false_perfect_task_epsilon,
    })
    _write_report(output_dir / "cem_score_shock_report.md", {
        "trace_dir": str(trace_dir),
        "dataset": str(args.dataset),
        "models": list(models.keys()),
        "processed_calls": processed_calls,
        "skipped_rows": len(skip_rows),
    })
    _plot_outputs(output_dir, prefix_rows, ranking_rows, failure_rows)
    for model in loaded_models.values():
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[score_shock] processed_calls={processed_calls}, wrote outputs under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
