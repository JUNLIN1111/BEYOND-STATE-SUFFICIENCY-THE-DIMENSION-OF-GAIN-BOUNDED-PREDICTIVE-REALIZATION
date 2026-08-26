from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py

from aliasing_experiment import _find_key, _load_model, _preprocess_pixels, _read_rows  # noqa: E402
from task_cost import task_cost  # noqa: E402

EPS = 1e-12
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

COMPACT_COLUMNS = [
    "window_idx",
    "model",
    "latent_dim",
    "task_best_idx",
    "focus_wrong_idx",
    "model_best_idx",
    "model_selects_focus_wrong",
    "model_selects_task_best",
    "task_best_rank_by_model",
    "focus_wrong_rank_by_model",
    "task_value_task_best",
    "task_value_wrong",
    "task_gap",
    "progress_task_best",
    "progress_wrong",
    "progress_gap_task_minus_wrong",
    "terminal_cost_task_best",
    "terminal_cost_wrong",
    "terminal_cost_gap_wrong_minus_task",
    "c25_task",
    "c25_wrong",
    "latent_gap",
    "log_ratio",
    "cos_task",
    "cos_wrong",
    "cos_gap",
]


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


def _write_csv_with_columns(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _load_raw(path: Path) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    required = ["latent_scores", "terminal_latents", "goal_latents", "progress", "terminal_cost"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return {key: np.asarray(raw[key]) for key in raw.files}


def _latent_dim(model_name: str, raw: Dict[str, np.ndarray]) -> int:
    if "terminal_latents" in raw:
        return int(np.asarray(raw["terminal_latents"]).shape[-1])
    return MODEL_DIMS.get(model_name, -1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + EPS)
    return float(np.dot(a, b) / denom)


def _rank_1_based(values: np.ndarray, index: int) -> int:
    order = np.argsort(values)
    return int(np.where(order == int(index))[0][0] + 1)


def _task_scores(raw: Dict[str, np.ndarray], true_metric: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    if true_metric == "progress":
        progress = np.asarray(raw["progress"], dtype=np.float64)
        return progress, progress, False
    if true_metric == "terminal_cost":
        terminal_cost = np.asarray(raw["terminal_cost"], dtype=np.float64)
        return terminal_cost, terminal_cost, True
    raise ValueError(f"Unknown true_metric: {true_metric}")


def _task_best_idx(values: np.ndarray, lower_is_better: bool) -> int:
    return int(np.argmin(values) if lower_is_better else np.argmax(values))


def _task_gap(values: np.ndarray, task_idx: int, wrong_idx: int, lower_is_better: bool) -> float:
    if lower_is_better:
        return float(values[wrong_idx] - values[task_idx])
    return float(values[task_idx] - values[wrong_idx])


def _alignment_report(raws: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, object]:
    names = list(raws)
    ref = raws[names[0]]
    report: Dict[str, object] = {"reference_model": names[0]}
    for name in names[1:]:
        raw = raws[name]
        for key in ("start_rows", "goal_rows", "candidate_future_actions", "original_candidate_indices"):
            if key in ref and key in raw:
                a = np.asarray(ref[key])
                b = np.asarray(raw[key])
                report[f"{name}/{key}_same_shape"] = bool(a.shape == b.shape)
                if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                    report[f"{name}/{key}_max_abs_diff"] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                elif a.shape == b.shape:
                    report[f"{name}/{key}_exact_equal"] = bool(np.array_equal(a, b))
            else:
                report[f"{name}/{key}_available_in_both"] = False
    return report


def _selected_pairs(
    focus_raw: Dict[str, np.ndarray],
    focus_model: str,
    true_metric: str,
    only_focus_errors: bool,
) -> List[Dict[str, object]]:
    true_values, _ranking_values, lower_is_better = _task_scores(focus_raw, true_metric)
    focus_scores = np.asarray(focus_raw["latent_scores"], dtype=np.float64)
    pairs: List[Dict[str, object]] = []
    for window_idx in range(focus_scores.shape[0]):
        task_idx = _task_best_idx(true_values[window_idx], lower_is_better)
        wrong_idx = int(np.argmin(focus_scores[window_idx]))
        focus_error = int(wrong_idx != task_idx)
        if only_focus_errors and not focus_error:
            continue
        gap = _task_gap(true_values[window_idx], task_idx, wrong_idx, lower_is_better)
        pairs.append(
            {
                "window_idx": int(window_idx),
                "task_best_idx": int(task_idx),
                "focus_wrong_idx": int(wrong_idx),
                "focus_rank_error": focus_error,
                "focus_model": focus_model,
                "task_gap": float(gap),
            }
        )
    return pairs


def _delta_thresholds(task_gaps: np.ndarray, small: float) -> Dict[str, float]:
    positive = task_gaps[np.isfinite(task_gaps) & (task_gaps > 0)]
    return {
        "delta0": 0.0,
        "delta_small": float(small),
        "delta_median_positive": float(np.median(positive)) if positive.size else float("nan"),
    }


def _encode_start_latents(
    model_name: str,
    checkpoint: Path,
    raw: Dict[str, np.ndarray],
    dataset: Path,
    pixels_key_arg: str,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    if "start_latents" in raw:
        return np.asarray(raw["start_latents"], dtype=np.float64)
    if "context_latents" in raw:
        context = np.asarray(raw["context_latents"], dtype=np.float64)
        return context[:, -1] if context.ndim == 3 else context
    if "start_rows" not in raw or checkpoint is None or not checkpoint.exists():
        return None
    model = _load_model(checkpoint, device)
    start_rows = np.asarray(raw["start_rows"], dtype=np.int64).reshape(-1)
    chunks = []
    with h5py.File(dataset, "r") as h5, torch.no_grad():
        pixels_key = _find_key(h5, [pixels_key_arg, "observation/pixels"])
        for start in range(0, start_rows.shape[0], batch_size):
            rows = start_rows[start : start + batch_size]
            pixels = _read_rows(h5[pixels_key], rows)[:, None]
            tensor = _preprocess_pixels(pixels, img_size, device)
            emb = model.encode({"pixels": tensor})["emb"][:, 0]
            chunks.append(emb.detach().float().cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[false_shortcut] encoded start latents for {model_name}", flush=True)
    return np.concatenate(chunks, axis=0).astype(np.float64)


def _task_pair_distance(raw: Dict[str, np.ndarray], window_idx: int, wrong_idx: int, task_idx: int) -> float:
    if "terminal_states" not in raw:
        return float("nan")
    states = np.asarray(raw["terminal_states"], dtype=np.float64)
    if states.ndim != 3:
        return float("nan")
    return float(task_cost(states[window_idx, wrong_idx], states[window_idx, task_idx]))


def _window_rows_for_model(
    model_name: str,
    raw: Dict[str, np.ndarray],
    pairs: List[Dict[str, object]],
    thresholds: Dict[str, float],
    true_metric: str,
    start_latents: Optional[np.ndarray],
) -> List[Dict[str, object]]:
    true_values, _ranking_values, lower_is_better = _task_scores(raw, true_metric)
    progress = np.asarray(raw["progress"], dtype=np.float64)
    terminal_cost = np.asarray(raw["terminal_cost"], dtype=np.float64)
    scores = np.asarray(raw["latent_scores"], dtype=np.float64)
    terminals = np.asarray(raw["terminal_latents"], dtype=np.float64)
    goals = np.asarray(raw["goal_latents"], dtype=np.float64)
    latent_dim = _latent_dim(model_name, raw)
    rows: List[Dict[str, object]] = []
    for pair in pairs:
        window_idx = int(pair["window_idx"])
        task_idx = int(pair["task_best_idx"])
        wrong_idx = int(pair["focus_wrong_idx"])
        task_gap = _task_gap(true_values[window_idx], task_idx, wrong_idx, lower_is_better)
        c_task = float(scores[window_idx, task_idx])
        c_wrong = float(scores[window_idx, wrong_idx])
        latent_gap = c_wrong - c_task
        log_ratio = float(np.log((c_wrong + EPS) / (c_task + EPS)))
        z_task = terminals[window_idx, task_idx]
        z_wrong = terminals[window_idx, wrong_idx]
        latent_pair_dist_sq = float(np.sum((z_wrong - z_task) ** 2))
        task_pair_dist = _task_pair_distance(raw, window_idx, wrong_idx, task_idx)
        model_best_idx = int(np.argmin(scores[window_idx]))

        angle_available = start_latents is not None
        cos_task = cos_wrong = cos_gap = float("nan")
        norm_diff = align_diff = score_gap_decomp = decomp_residual = float("nan")
        angle_explained = alignment_dominated = 0
        if angle_available:
            z0 = start_latents[window_idx]
            zg = goals[window_idx]
            q = zg - z0
            d_task = z_task - z0
            d_wrong = z_wrong - z0
            cos_task = _cosine(d_task, q)
            cos_wrong = _cosine(d_wrong, q)
            cos_gap = cos_wrong - cos_task
            norm_diff = float(np.sum(d_wrong ** 2) - np.sum(d_task ** 2))
            align_diff = float(-2.0 * np.dot(q, d_wrong - d_task))
            score_gap_decomp = norm_diff + align_diff
            decomp_residual = float(latent_gap - score_gap_decomp)
            angle_explained = int(latent_gap < 0.0 and cos_gap > 0.0)
            alignment_dominated = int(latent_gap < 0.0 and align_diff < 0.0 and abs(align_diff) > abs(min(norm_diff, 0.0)))

        base = {
            "model": model_name,
            "latent_dim": latent_dim,
            "window_idx": window_idx,
            "true_metric": true_metric,
            "task_best_idx": task_idx,
            "focus_wrong_idx": wrong_idx,
            "model_best_idx": model_best_idx,
            "model_selects_focus_wrong": int(model_best_idx == wrong_idx),
            "model_selects_task_best": int(model_best_idx == task_idx),
            "task_best_rank_by_model": _rank_1_based(scores[window_idx], task_idx),
            "focus_wrong_rank_by_model": _rank_1_based(scores[window_idx], wrong_idx),
            "focus_rank_error": int(pair["focus_rank_error"]),
            "task_gap": float(task_gap),
            "task_value_task_best": float(true_values[window_idx, task_idx]),
            "task_value_wrong": float(true_values[window_idx, wrong_idx]),
            "progress_task_best": float(progress[window_idx, task_idx]),
            "progress_wrong": float(progress[window_idx, wrong_idx]),
            "progress_gap_task_minus_wrong": float(progress[window_idx, task_idx] - progress[window_idx, wrong_idx]),
            "terminal_cost_task_best": float(terminal_cost[window_idx, task_idx]),
            "terminal_cost_wrong": float(terminal_cost[window_idx, wrong_idx]),
            "terminal_cost_gap_wrong_minus_task": float(terminal_cost[window_idx, wrong_idx] - terminal_cost[window_idx, task_idx]),
            "c25_task": c_task,
            "c25_wrong": c_wrong,
            "latent_gap": latent_gap,
            "log_ratio": log_ratio,
            "wrong_win": int(latent_gap < 0.0),
            "latent_pair_dist_sq": latent_pair_dist_sq,
            "task_pair_dist": task_pair_dist,
            "angle_available": bool(angle_available),
            "cos_task": cos_task,
            "cos_wrong": cos_wrong,
            "cos_gap": cos_gap,
            "norm_diff": norm_diff,
            "align_diff": align_diff,
            "score_gap_decomp": score_gap_decomp,
            "decomposition_residual": decomp_residual,
            "angle_explained": angle_explained,
            "alignment_dominated": alignment_dominated,
        }
        for name, threshold in thresholds.items():
            base[f"false_shortcut_{name}"] = int(np.isfinite(threshold) and task_gap > threshold and latent_gap < 0.0)
        rows.append(base)
    return rows


def _summary_rows(rows: List[Dict[str, object]], thresholds: Dict[str, float]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    out = []
    for model, model_rows in grouped.items():
        wrong_rows = [row for row in model_rows if int(row["wrong_win"]) == 1]
        record = {
            "model": model,
            "latent_dim": int(model_rows[0]["latent_dim"]),
            "num_windows": len(model_rows),
            "wrong_win_rate": float(np.mean([float(row["wrong_win"]) for row in model_rows])),
            "median_log_ratio": float(np.nanmedian([float(row["log_ratio"]) for row in model_rows])),
            "mean_log_ratio": float(np.nanmean([float(row["log_ratio"]) for row in model_rows])),
            "mean_task_gap": float(np.nanmean([float(row["task_gap"]) for row in model_rows])),
            "mean_progress_gap_task_minus_wrong": float(np.nanmean([float(row["progress_gap_task_minus_wrong"]) for row in model_rows])),
            "mean_terminal_cost_gap_wrong_minus_task": float(np.nanmean([float(row["terminal_cost_gap_wrong_minus_task"]) for row in model_rows])),
            "mean_latent_gap": float(np.nanmean([float(row["latent_gap"]) for row in model_rows])),
            "mean_cos_gap": float(np.nanmean([float(row["cos_gap"]) for row in model_rows])),
            "angle_explained_rate_among_wrong_wins": float(np.mean([float(row["angle_explained"]) for row in wrong_rows])) if wrong_rows else float("nan"),
            "alignment_dominated_rate_among_wrong_wins": float(np.mean([float(row["alignment_dominated"]) for row in wrong_rows])) if wrong_rows else float("nan"),
            "mean_task_best_rank": float(np.mean([float(row["task_best_rank_by_model"]) for row in model_rows])),
            "mean_focus_wrong_rank": float(np.mean([float(row["focus_wrong_rank_by_model"]) for row in model_rows])),
        }
        for name in thresholds:
            record[f"false_shortcut_rate_{name}"] = float(np.mean([float(row[f"false_shortcut_{name}"]) for row in model_rows]))
        out.append(record)
    return sorted(out, key=lambda row: (int(row["latent_dim"]) if int(row["latent_dim"]) >= 0 else 10_000, str(row["model"])))


def _compact_summary_rows(summary: List[Dict[str, object]]) -> List[Dict[str, object]]:
    columns = [
        "model",
        "latent_dim",
        "num_windows",
        "wrong_win_rate",
        "false_shortcut_rate_delta0",
        "median_log_ratio",
        "mean_log_ratio",
        "mean_task_gap",
        "mean_progress_gap_task_minus_wrong",
        "mean_terminal_cost_gap_wrong_minus_task",
        "mean_latent_gap",
        "mean_cos_gap",
        "mean_task_best_rank",
        "mean_focus_wrong_rank",
    ]
    compact = []
    for row in summary:
        compact.append({column: row.get(column, "") for column in columns})
    return compact


def _plot_outputs(rows: List[Dict[str, object]], summary: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[false_shortcut] matplotlib unavailable; skipping plots.", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [str(row["model"]) for row in summary]
    dims = [int(row["latent_dim"]) for row in summary]
    labels = [f"{model}\nD={dim}" for model, dim in zip(models, dims)]
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    ax.plot(x, [float(row["wrong_win_rate"]) for row in summary], marker="o", label="wrong win")
    key = "false_shortcut_rate_delta0"
    if key in summary[0]:
        ax.plot(x, [float(row[key]) for row in summary], marker="s", label="false shortcut")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("rate")
    ax.set_title("False terminal shortcut rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_false_shortcut_rate_vs_dim.png", dpi=260)
    fig.savefig(output_dir / "fig_false_shortcut_rate_vs_dim.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    data = [[float(row["log_ratio"]) for row in rows if row["model"] == model] for model in models]
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("log((c_wrong+eps)/(c_task+eps))")
    ax.set_title("Latent terminal log ratio")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_latent_logratio_vs_dim.png", dpi=260)
    fig.savefig(output_dir / "fig_latent_logratio_vs_dim.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.4), facecolor="white")
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        ax.scatter(
            [float(row["task_gap"]) for row in model_rows],
            [float(row["log_ratio"]) for row in model_rows],
            s=18,
            alpha=0.65,
            label=model,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle=":")
    ax.set_xlabel("task gap: task_best - wrong")
    ax.set_ylabel("latent log ratio")
    ax.set_title("Task gap vs latent terminal log ratio")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_task_gap_vs_latent_logratio.png", dpi=260)
    fig.savefig(output_dir / "fig_task_gap_vs_latent_logratio.pdf")
    plt.close(fig)

    if any(bool(row["angle_available"]) for row in rows):
        fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
        data = [[float(row["cos_gap"]) for row in rows if row["model"] == model and bool(row["angle_available"])] for model in models]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_ylabel("cos_wrong - cos_task")
        ax.set_title("Angle gap by model")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_angle_gap_vs_dim.png", dpi=260)
        fig.savefig(output_dir / "fig_angle_gap_vs_dim.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="False-terminal-shortcut diagnostic for candidate terminal rankings.")
    parser.add_argument("--raw_pools", nargs="+", required=True, help="NAME=raw_pool.npz")
    parser.add_argument("--focus_model", default="state8")
    parser.add_argument("--true_metric", choices=["progress", "terminal_cost"], default="progress")
    parser.add_argument("--only_focus_errors", action="store_true", help="Only analyze windows where focus model does not pick task-best.")
    parser.add_argument("--small_delta_task", type=float, default=1e-3)
    parser.add_argument("--models", nargs="*", default=[], help="Optional NAME=checkpoint_object.ckpt for z0 angle decomposition.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--pixels_key", default="pixels")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/false_terminal_shortcut")
    args = parser.parse_args()

    raw_paths = _parse_name_paths(args.raw_pools)
    if args.focus_model not in raw_paths:
        raise KeyError(f"--focus_model {args.focus_model!r} is not present in --raw_pools.")
    raws = {name: _load_raw(path) for name, path in raw_paths.items()}
    alignment = _alignment_report(raws)
    pairs = _selected_pairs(raws[args.focus_model], args.focus_model, args.true_metric, args.only_focus_errors)
    if not pairs:
        raise RuntimeError("No selected windows to analyze.")
    thresholds = _delta_thresholds(np.asarray([pair["task_gap"] for pair in pairs], dtype=np.float64), args.small_delta_task)
    model_paths = _parse_name_paths(args.models)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    rows: List[Dict[str, object]] = []
    for model_name, raw in raws.items():
        checkpoint = model_paths.get(model_name)
        start_latents = None
        if checkpoint is not None:
            start_latents = _encode_start_latents(
                model_name,
                checkpoint,
                raw,
                Path(args.dataset),
                args.pixels_key,
                args.img_size,
                args.batch_size,
                device,
            )
        rows.extend(_window_rows_for_model(model_name, raw, pairs, thresholds, args.true_metric, start_latents))

    summary = _summary_rows(rows, thresholds)
    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "false_terminal_shortcut_by_window.csv", rows)
    _write_csv(output_dir / "false_terminal_shortcut_summary.csv", summary)
    _write_csv_with_columns(output_dir / "false_terminal_shortcut_compact.csv", rows, COMPACT_COLUMNS)
    _write_csv(output_dir / "false_terminal_shortcut_compact_summary.csv", _compact_summary_rows(summary))
    with (output_dir / "false_terminal_shortcut_metadata.json").open("w") as file:
        json.dump({"alignment": alignment, "thresholds": thresholds, "args": vars(args)}, file, indent=2)
    _plot_outputs(rows, summary, output_dir / "plots")
    print(f"[false_shortcut] wrote outputs under {output_dir}", flush=True)
    print(f"[false_shortcut] compact table: {output_dir / 'false_terminal_shortcut_compact.csv'}", flush=True)


if __name__ == "__main__":
    main()
