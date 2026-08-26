from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _parse_name_paths(items: Iterable[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path)
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as file:
        return list(csv.DictReader(file))


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


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _attempt_success_map(trace_dir: Path) -> Dict[int, int]:
    path = trace_dir / "trace_run_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    attempts = payload.get("metrics", {}).get("attempts", [])
    out = {}
    for item in attempts:
        try:
            out[int(item["trace_episode_id"])] = int(bool(item["success"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _trace_rows(model: str, trace_dir: Path, low_threshold: float, high_threshold: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    full = _read_csv(trace_dir / "full_episode_trace.csv")
    failures = _read_csv(trace_dir / "planning_call_failure_types.csv")
    solver = _read_csv(trace_dir / "solver_return_replay.csv")
    success_map = _attempt_success_map(trace_dir)
    full_by_episode: Dict[int, List[Dict[str, str]]] = {}
    for row in full:
        full_by_episode.setdefault(_safe_int(row.get("episode_id")), []).append(row)
    solver_by_call = {
        (_safe_int(row.get("episode_id")), _safe_int(row.get("mpc_step_idx"))): row
        for row in solver
        if str(row.get("candidate_type", "")) == "solver_return"
    }
    failure_by_call = {
        (_safe_int(row.get("episode_id")), _safe_int(row.get("mpc_step_idx"))): row
        for row in failures
    }

    call_rows: List[Dict[str, object]] = []
    for row in full:
        episode = _safe_int(row.get("episode_id"))
        step = _safe_int(row.get("mpc_step_idx"))
        key = (episode, step)
        failure = failure_by_call.get(key, {})
        solver_row = solver_by_call.get(key, {})
        call_rows.append(
            {
                "model": model,
                "episode_id": episode,
                "mpc_step_idx": step,
                "episode_success": success_map.get(episode, _safe_int(row.get("episode_final_success"))),
                "current_task_cost": _safe_float(row.get("current_task_cost")),
                "best_predicted_cost": _safe_float(row.get("best_predicted_cost")),
                "elite_mean_predicted_cost": _safe_float(row.get("elite_mean_predicted_cost")),
                "solver_return_predicted_cost": _safe_float(row.get("solver_return_predicted_cost")),
                "selected_first_block_true_progress": _safe_float(row.get("executed_first_block_true_progress")),
                "task_cost_after_first_block": _safe_float(row.get("task_cost_after_first_block")),
                "first_block_failure": _safe_int(failure.get("first_block_failure"), int(_safe_float(row.get("executed_first_block_true_progress")) < 0.0)),
                "support_failure": _safe_int(failure.get("support_failure")),
                "ranking_failure": _safe_int(failure.get("ranking_failure")),
                "imagination_failure": _safe_int(failure.get("imagination_failure")),
                "mean_sequence_failure": _safe_int(failure.get("mean_sequence_failure")),
                "solver_return_true_progress_25": _safe_float(solver_row.get("true_terminal_progress")),
                "solver_return_true_terminal_task_cost": _safe_float(solver_row.get("true_terminal_task_cost")),
                "solver_return_executed_raw_steps": _safe_int(solver_row.get("executed_raw_steps")),
            }
        )

    jump_rows: List[Dict[str, object]] = []
    for episode, rows in full_by_episode.items():
        rows = sorted(rows, key=lambda item: _safe_int(item.get("mpc_step_idx")))
        for prev, cur in zip(rows[:-1], rows[1:]):
            prev_best = _safe_float(prev.get("best_predicted_cost"))
            cur_best = _safe_float(cur.get("best_predicted_cost"))
            jump_rows.append(
                {
                    "model": model,
                    "episode_id": episode,
                    "episode_success": success_map.get(episode, _safe_int(prev.get("episode_final_success"))),
                    "from_mpc_step_idx": _safe_int(prev.get("mpc_step_idx")),
                    "to_mpc_step_idx": _safe_int(cur.get("mpc_step_idx")),
                    "best_predicted_cost_before": prev_best,
                    "best_predicted_cost_after": cur_best,
                    "cost_jump": cur_best - prev_best,
                    "catastrophic_jump": int(prev_best < low_threshold and cur_best > high_threshold),
                }
            )
    return call_rows, jump_rows


def _summary_rows(call_rows: List[Dict[str, object]], jump_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    models = sorted({str(row["model"]) for row in call_rows})
    out = []
    for model in models:
        calls = [row for row in call_rows if row["model"] == model]
        jumps = [row for row in jump_rows if row["model"] == model]
        success_values = [_safe_int(row["episode_success"]) for row in calls if _safe_int(row["episode_success"]) in (0, 1)]

        def arr(rows: List[Dict[str, object]], key: str) -> np.ndarray:
            return np.asarray([_safe_float(row.get(key)) for row in rows], dtype=np.float64)

        def binary_mean(key: str) -> float:
            values = np.asarray([_safe_int(row.get(key)) for row in calls if _safe_int(row.get(key)) in (0, 1)], dtype=np.float64)
            return float(np.mean(values)) if values.size else float("nan")

        out.append(
            {
                "model": model,
                "num_planning_calls": len(calls),
                "num_episodes": len({row["episode_id"] for row in calls}),
                "success_episode_frac": float(np.mean(success_values)) if success_values else float("nan"),
                "first_block_failure_rate": binary_mean("first_block_failure"),
                "support_failure_rate": binary_mean("support_failure"),
                "ranking_failure_rate": binary_mean("ranking_failure"),
                "imagination_failure_rate": binary_mean("imagination_failure"),
                "mean_sequence_failure_rate": binary_mean("mean_sequence_failure"),
                "mean_selected_first_block_true_progress": float(np.nanmean(arr(calls, "selected_first_block_true_progress"))),
                "median_selected_first_block_true_progress": float(np.nanmedian(arr(calls, "selected_first_block_true_progress"))),
                "mean_solver_return_true_progress_25": float(np.nanmean(arr(calls, "solver_return_true_progress_25"))),
                "median_solver_return_true_progress_25": float(np.nanmedian(arr(calls, "solver_return_true_progress_25"))),
                "mean_cost_jump": float(np.nanmean(arr(jumps, "cost_jump"))) if jumps else float("nan"),
                "median_cost_jump": float(np.nanmedian(arr(jumps, "cost_jump"))) if jumps else float("nan"),
                "catastrophic_jump_rate": float(np.nanmean([_safe_int(row["catastrophic_jump"]) for row in jumps])) if jumps else float("nan"),
            }
        )
    return out


def _plot(call_rows: List[Dict[str, object]], jump_rows: List[Dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[cem_compare] matplotlib unavailable; skipping plots.", flush=True)
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in call_rows})

    def grouped_values(rows: List[Dict[str, object]], key: str):
        return [[_safe_float(row.get(key)) for row in rows if row["model"] == model] for model in models]

    for rows, key, title, filename in (
        (jump_rows, "cost_jump", "Best predicted cost jump", "cost_jump_by_model"),
        (call_rows, "selected_first_block_true_progress", "Selected first-block true progress", "first_block_progress_by_model"),
    ):
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(max(5.5, len(models) * 0.9), 3.8), facecolor="white")
        ax.boxplot(grouped_values(rows, key), labels=models, showfliers=False)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{filename}.png", dpi=260)
        fig.savefig(fig_dir / f"{filename}.pdf")
        plt.close(fig)

    categories = ("support_failure", "ranking_failure", "imagination_failure", "first_block_failure")
    x = np.arange(len(models))
    width = min(0.18, 0.75 / len(categories))
    fig, ax = plt.subplots(figsize=(max(6.0, len(models) * 1.1), 4.0), facecolor="white")
    for idx, category in enumerate(categories):
        vals = []
        for model in models:
            subset = [_safe_int(row.get(category)) for row in call_rows if row["model"] == model and _safe_int(row.get(category)) in (0, 1)]
            vals.append(float(np.mean(subset)) if subset else float("nan"))
        ax.bar(x + (idx - (len(categories) - 1) / 2) * width, vals, width=width, label=category)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("rate")
    ax.set_title("Failure category rates")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "failure_category_counts.png", dpi=260)
    fig.savefig(fig_dir / "failure_category_counts.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    for model in models:
        for episode in sorted({row["episode_id"] for row in call_rows if row["model"] == model})[:3]:
            rows = sorted([row for row in call_rows if row["model"] == model and row["episode_id"] == episode], key=lambda row: row["mpc_step_idx"])
            ax.plot([row["mpc_step_idx"] for row in rows], [row["best_predicted_cost"] for row in rows], marker="o", alpha=0.75, label=f"{model} ep{episode}")
    ax.set_xlabel("MPC step")
    ax.set_ylabel("best predicted cost")
    ax.set_title("Representative episode timelines")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "representative_episode_timelines.png", dpi=260)
    fig.savefig(fig_dir / "representative_episode_timelines.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CEM trace failure/success diagnostics across PushT models.")
    parser.add_argument("--trace_dirs", nargs="+", required=True, help="NAME=trace_dir")
    parser.add_argument("--output_dir", default="results/cem_trace_comparison")
    parser.add_argument("--low_threshold", type=float, default=0.05)
    parser.add_argument("--high_threshold", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dirs = _parse_name_paths(args.trace_dirs)
    call_rows: List[Dict[str, object]] = []
    jump_rows: List[Dict[str, object]] = []
    for model, trace_dir in trace_dirs.items():
        calls, jumps = _trace_rows(model, trace_dir, args.low_threshold, args.high_threshold)
        call_rows.extend(calls)
        jump_rows.extend(jumps)
        print(f"[cem_compare] {model}: calls={len(calls)} jumps={len(jumps)}", flush=True)
    summary = _summary_rows(call_rows, jump_rows)
    _write_csv(output_dir / "planning_call_metrics.csv", call_rows)
    _write_csv(output_dir / "cost_jump_metrics.csv", jump_rows)
    _write_csv(output_dir / "state8_state16_baseline_summary.csv", summary)
    _write_json(output_dir / "cem_trace_comparison_metadata.json", {"args": vars(args)})
    _plot(call_rows, jump_rows, output_dir)
    print(f"[cem_compare] wrote outputs under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
