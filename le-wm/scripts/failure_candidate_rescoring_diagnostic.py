from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

EPS = 1e-12


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


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        corr = spearmanr(x, y).correlation
        return float(corr) if corr is not None else float("nan")
    except ImportError:
        xr = np.argsort(np.argsort(x)).astype(np.float64)
        yr = np.argsort(np.argsort(y)).astype(np.float64)
        if np.std(xr) < EPS or np.std(yr) < EPS:
            return float("nan")
        return float(np.corrcoef(xr, yr)[0, 1])


def _rank_1_based(values: np.ndarray, index: int, lower_is_better: bool = True) -> int:
    order = np.argsort(values if lower_is_better else -values)
    return int(np.where(order == int(index))[0][0] + 1)


def _load_raw(path: Path) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    required = ["latent_scores", "progress", "terminal_cost"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return {key: np.asarray(raw[key]) for key in raw.files}


def _latent_dim(raw: Dict[str, np.ndarray]) -> int:
    if "terminal_latents" in raw:
        terminal_latents = np.asarray(raw["terminal_latents"])
        if terminal_latents.ndim >= 1:
            return int(terminal_latents.shape[-1])
    if "goal_latents" in raw:
        goal_latents = np.asarray(raw["goal_latents"])
        if goal_latents.ndim >= 1:
            return int(goal_latents.shape[-1])
    return -1


def _true_scores(raw: Dict[str, np.ndarray], true_metric: str) -> Tuple[np.ndarray, bool]:
    if true_metric == "progress":
        return np.asarray(raw["progress"], dtype=np.float64), False
    if true_metric == "terminal_cost":
        return np.asarray(raw["terminal_cost"], dtype=np.float64), True
    raise ValueError(f"Unknown true_metric: {true_metric}")


def _task_best(true_values: np.ndarray, lower_is_better: bool) -> int:
    return int(np.argmin(true_values) if lower_is_better else np.argmax(true_values))


def _regret(true_values: np.ndarray, selected: int, task_best: int, lower_is_better: bool) -> float:
    if lower_is_better:
        return float(true_values[selected] - true_values[task_best])
    return float(true_values[task_best] - true_values[selected])


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


def _check_shapes(raws: Dict[str, Dict[str, np.ndarray]], true_metric: str) -> Tuple[int, int]:
    shapes = {}
    for name, raw in raws.items():
        latent = np.asarray(raw["latent_scores"])
        true, _lower = _true_scores(raw, true_metric)
        if latent.shape != true.shape:
            raise ValueError(f"{name}: latent_scores shape {latent.shape} != true score shape {true.shape}")
        if latent.ndim != 2:
            raise ValueError(f"{name}: expected latent_scores shape (windows,candidates), got {latent.shape}")
        shapes[name] = latent.shape
    unique = set(shapes.values())
    if len(unique) != 1:
        raise ValueError(f"Raw pools do not have matching (windows,candidates) shapes: {shapes}")
    return next(iter(unique))


def _per_window_rows(
    raws: Dict[str, Dict[str, np.ndarray]],
    focus_model: str,
    true_metric: str,
    topk: int,
) -> List[Dict[str, object]]:
    focus_raw = raws[focus_model]
    true, lower_is_better = _true_scores(focus_raw, true_metric)
    focus_scores = np.asarray(focus_raw["latent_scores"], dtype=np.float64)
    rows: List[Dict[str, object]] = []
    for window_idx in range(focus_scores.shape[0]):
        true_values = true[window_idx].astype(np.float64)
        task_best = _task_best(true_values, lower_is_better)
        focus_best = int(np.argmin(focus_scores[window_idx]))
        focus_rank_error = int(focus_best != task_best)
        focus_regret = _regret(true_values, focus_best, task_best, lower_is_better)
        for model_name, raw in raws.items():
            latent_dim = _latent_dim(raw)
            scores = np.asarray(raw["latent_scores"], dtype=np.float64)[window_idx]
            model_best = int(np.argmin(scores))
            model_regret = _regret(true_values, model_best, task_best, lower_is_better)
            model_rank_error = int(model_best != task_best)
            task_rank = _rank_1_based(scores, task_best, lower_is_better=True)
            focus_rank = _rank_1_based(scores, focus_best, lower_is_better=True)
            score_task = float(scores[task_best])
            score_focus = float(scores[focus_best])
            score_model = float(scores[model_best])
            rows.append(
                {
                    "window_idx": int(window_idx),
                    "model": model_name,
                    "latent_dim": latent_dim,
                    "focus_model": focus_model,
                    "true_metric": true_metric,
                    "num_candidates": int(scores.shape[0]),
                    "task_best_idx": int(task_best),
                    "focus_best_idx": int(focus_best),
                    "model_best_idx": int(model_best),
                    "focus_rank_error": focus_rank_error,
                    "model_rank_error": model_rank_error,
                    "focus_regret": float(focus_regret),
                    "model_regret": float(model_regret),
                    "task_best_rank_by_model": int(task_rank),
                    "focus_best_rank_by_model": int(focus_rank),
                    "topk_hit_task_best": int(task_rank <= min(topk, scores.shape[0])),
                    "model_selects_focus_wrong": int(focus_rank_error and model_best == focus_best),
                    "model_selects_task_best": int(model_best == task_best),
                    "model_prefers_focus_wrong_over_task": int(focus_rank_error and score_focus < score_task),
                    "score_task_best": score_task,
                    "score_focus_best": score_focus,
                    "score_model_best": score_model,
                    "focus_wrong_score_advantage": float(score_task - score_focus),
                    "model_best_score_advantage_over_task": float(score_task - score_model),
                    "task_score_task_best": float(true_values[task_best]),
                    "task_score_focus_best": float(true_values[focus_best]),
                    "task_score_model_best": float(true_values[model_best]),
                    "spearman_latent_vs_task": _spearman(scores, true_values if lower_is_better else -true_values),
                }
            )
    return rows


def _summary_rows(per_window: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in per_window:
        grouped.setdefault(str(row["model"]), []).append(row)
    out = []
    for model, rows in grouped.items():
        focus_errors = [row for row in rows if int(row["focus_rank_error"]) == 1]
        out.append(
            {
                "model": model,
                "focus_model": rows[0]["focus_model"],
                "true_metric": rows[0]["true_metric"],
                "num_windows": len(rows),
                "num_focus_error_windows": len(focus_errors),
                "rank_error_rate": float(np.mean([float(row["model_rank_error"]) for row in rows])),
                "mean_regret": float(np.mean([float(row["model_regret"]) for row in rows])),
                "median_regret": float(np.median([float(row["model_regret"]) for row in rows])),
                "mean_task_best_rank": float(np.mean([float(row["task_best_rank_by_model"]) for row in rows])),
                "median_task_best_rank": float(np.median([float(row["task_best_rank_by_model"]) for row in rows])),
                "topk_hit_task_best_rate": float(np.mean([float(row["topk_hit_task_best"]) for row in rows])),
                "spearman_mean": float(np.nanmean([float(row["spearman_latent_vs_task"]) for row in rows])),
                "on_focus_errors_selects_same_wrong_rate": float(np.mean([float(row["model_selects_focus_wrong"]) for row in focus_errors])) if focus_errors else float("nan"),
                "on_focus_errors_selects_task_best_rate": float(np.mean([float(row["model_selects_task_best"]) for row in focus_errors])) if focus_errors else float("nan"),
                "on_focus_errors_prefers_focus_wrong_over_task_rate": float(np.mean([float(row["model_prefers_focus_wrong_over_task"]) for row in focus_errors])) if focus_errors else float("nan"),
                "on_focus_errors_focus_wrong_score_advantage_mean": float(np.mean([float(row["focus_wrong_score_advantage"]) for row in focus_errors])) if focus_errors else float("nan"),
                "on_focus_errors_task_best_rank_mean": float(np.mean([float(row["task_best_rank_by_model"]) for row in focus_errors])) if focus_errors else float("nan"),
                "on_focus_errors_model_regret_mean": float(np.mean([float(row["model_regret"]) for row in focus_errors])) if focus_errors else float("nan"),
            }
        )
    return out


def _focus_error_example_rows(per_window: List[Dict[str, object]], max_examples: int) -> List[Dict[str, object]]:
    by_window: Dict[int, List[Dict[str, object]]] = {}
    for row in per_window:
        if int(row["focus_rank_error"]) != 1:
            continue
        by_window.setdefault(int(row["window_idx"]), []).append(row)

    selected_windows = sorted(
        by_window,
        key=lambda window: max(float(row["focus_regret"]) for row in by_window[window]),
        reverse=True,
    )
    if max_examples > 0:
        selected_windows = selected_windows[:max_examples]

    out: List[Dict[str, object]] = []
    for window_idx in selected_windows:
        rows = by_window[window_idx]
        focus_row = next(row for row in rows if row["model"] == row["focus_model"])
        for row in sorted(rows, key=lambda item: str(item["model"])):
            score_task = float(row["score_task_best"])
            score_wrong = float(row["score_focus_best"])
            task_progress = float(row["task_score_task_best"])
            wrong_progress = float(row["task_score_focus_best"])
            out.append(
                {
                    "window_idx": window_idx,
                    "model": row["model"],
                    "latent_dim": int(row["latent_dim"]),
                    "focus_model": row["focus_model"],
                    "task_best_idx": int(row["task_best_idx"]),
                    "focus_wrong_idx": int(row["focus_best_idx"]),
                    "model_best_idx": int(row["model_best_idx"]),
                    "task_progress": task_progress,
                    "focus_wrong_progress": wrong_progress,
                    "focus_wrong_regret": float(row["focus_regret"]),
                    "score_task_best": score_task,
                    "score_focus_wrong": score_wrong,
                    "score_wrong_minus_task": score_wrong - score_task,
                    "score_task_minus_wrong": score_task - score_wrong,
                    "task_best_rank_by_model": int(row["task_best_rank_by_model"]),
                    "focus_wrong_rank_by_model": int(row["focus_best_rank_by_model"]),
                    "model_prefers_wrong_over_task": int(row["model_prefers_focus_wrong_over_task"]),
                    "model_selects_focus_wrong": int(row["model_selects_focus_wrong"]),
                    "model_selects_task_best": int(row["model_selects_task_best"]),
                    "focus_model_score_task_best": float(focus_row["score_task_best"]),
                    "focus_model_score_focus_wrong": float(focus_row["score_focus_best"]),
                    "focus_model_score_wrong_minus_task": float(focus_row["score_focus_best"]) - float(focus_row["score_task_best"]),
                }
            )
    return out


def _compact_margin_rows(example_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for row in example_rows:
        rows.append(
            {
                "window_idx": int(row["window_idx"]),
                "model": row["model"],
                "latent_dim": int(row["latent_dim"]),
                "task_best_idx": int(row["task_best_idx"]),
                "focus_wrong_idx": int(row["focus_wrong_idx"]),
                "task_progress": float(row["task_progress"]),
                "focus_wrong_progress": float(row["focus_wrong_progress"]),
                "progress_gap_task_minus_wrong": float(row["task_progress"]) - float(row["focus_wrong_progress"]),
                "score_task_best": float(row["score_task_best"]),
                "score_focus_wrong": float(row["score_focus_wrong"]),
                "score_wrong_minus_task": float(row["score_wrong_minus_task"]),
                "score_task_minus_wrong": float(row["score_task_minus_wrong"]),
                "task_best_rank_by_model": int(row["task_best_rank_by_model"]),
                "focus_wrong_rank_by_model": int(row["focus_wrong_rank_by_model"]),
                "model_prefers_wrong_over_task": int(row["model_prefers_wrong_over_task"]),
            }
        )
    return sorted(rows, key=lambda item: (int(item["window_idx"]), int(item["latent_dim"]), str(item["model"])))


def _write_markdown(path: Path, summary: List[Dict[str, object]], alignment: Dict[str, object], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Failure Candidate Rescoring Diagnostic",
        "",
        "This diagnostic asks whether a low-dimensional focus model ranks task-bad candidates ahead of the task-best candidate, and whether higher-success models assign the same wrong candidates worse scores.",
        "",
        f"- focus model: `{args.focus_model}`",
        f"- true metric: `{args.true_metric}`",
        f"- top-k: `{args.topk}`",
        "",
        "## Summary",
        "",
        "| model | rank error | regret | task-best rank | top-k hit | focus-wrong preferred on focus errors | same wrong selected on focus errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: str(item["model"])):
        lines.append(
            f"| {row['model']} | "
            f"{float(row['rank_error_rate']):.4f} | "
            f"{float(row['mean_regret']):.4f} | "
            f"{float(row['mean_task_best_rank']):.2f} | "
            f"{float(row['topk_hit_task_best_rate']):.4f} | "
            f"{float(row['on_focus_errors_prefers_focus_wrong_over_task_rate']):.4f} | "
            f"{float(row['on_focus_errors_selects_same_wrong_rate']):.4f} |"
        )
    lines.extend(["", "## Alignment checks", "", "```json", json.dumps(alignment, indent=2), "```", ""])
    lines.extend(
        [
            "## Reading guide",
            "",
            "- `focus_wrong_score_advantage = score(task_best) - score(focus_best)`. Positive means the model scores the focus model's wrong candidate better than the task-best candidate.",
            "- A strong failure mechanism is: the focus model has high regret, while a high-SR model does not prefer the same wrong candidate on the focus-error windows.",
            "- This script does not reroll models; it compares saved `latent_scores` from aligned raw pools.",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare how models rescore the same failure candidates.")
    parser.add_argument("--raw_pools", nargs="+", required=True, help="NAME=path_to_raw_pool.npz")
    parser.add_argument("--focus_model", default="state8")
    parser.add_argument("--true_metric", choices=["progress", "terminal_cost"], default="progress")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max_examples", type=int, default=25, help="Number of focus-error windows to include in score example CSV; <=0 writes all.")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence/failure_candidate_rescoring")
    args = parser.parse_args()

    raw_paths = _parse_name_paths(args.raw_pools)
    if args.focus_model not in raw_paths:
        raise KeyError(f"--focus_model {args.focus_model!r} is not present in --raw_pools.")
    raws = {name: _load_raw(path) for name, path in raw_paths.items()}
    num_windows, num_candidates = _check_shapes(raws, args.true_metric)
    print(f"[failure_rescore] loaded {len(raws)} models, windows={num_windows}, candidates={num_candidates}", flush=True)
    alignment = _alignment_report(raws)
    per_window = _per_window_rows(raws, args.focus_model, args.true_metric, args.topk)
    summary = _summary_rows(per_window)
    examples = _focus_error_example_rows(per_window, args.max_examples)
    compact_margins = _compact_margin_rows(examples)

    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "failure_candidate_rescoring_per_window.csv", per_window)
    _write_csv(output_dir / "failure_candidate_rescoring_summary.csv", summary)
    _write_csv(output_dir / "focus_error_score_examples.csv", examples)
    _write_csv(output_dir / "focus_error_margin_by_dim.csv", compact_margins)
    with (output_dir / "failure_candidate_rescoring_alignment.json").open("w") as file:
        json.dump(alignment, file, indent=2)
    _write_markdown(output_dir / "failure_candidate_rescoring_summary.md", summary, alignment, args)
    print(f"[failure_rescore] wrote outputs under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
