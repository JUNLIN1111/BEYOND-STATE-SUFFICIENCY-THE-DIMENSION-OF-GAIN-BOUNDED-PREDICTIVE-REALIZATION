from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as file:
        return list(csv.DictReader(file))


def _float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(row: Dict[str, str], key: str, default: int = -1) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _group(rows: List[Dict[str, str]], keys: List[str]) -> Dict[tuple, List[Dict[str, str]]]:
    out: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        out.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    return out


def _save(fig, path_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=260)
    fig.savefig(path_base.with_suffix(".pdf"))


def _plot_cost_hist(trace_dir: Path, figure_dir: Path, max_plots: int) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(trace_dir / "cem_iteration_summary.csv")
    by_step = _group(rows, ["episode_id", "mpc_step_idx"])
    for count, (key, group) in enumerate(sorted(by_step.items(), key=lambda item: (int(item[0][0]), int(item[0][1])))):
        if count >= max_plots:
            break
        iters = sorted({_int(row, "cem_iter") for row in group})
        ncols = min(3, max(1, len(iters)))
        nrows = int(np.ceil(len(iters) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), facecolor="white", squeeze=False)
        all_costs = []
        loaded = {}
        for cem_iter in iters:
            match = [row for row in group if _int(row, "cem_iter") == cem_iter][0]
            path = Path(match.get("iteration_cost_npz", ""))
            if path.exists():
                values = np.load(path)["terminal_cost_c25"].astype(np.float64)
                loaded[cem_iter] = values[np.isfinite(values)]
                all_costs.append(loaded[cem_iter])
        bins = 40
        if all_costs:
            pooled = np.concatenate(all_costs)
            bins = np.linspace(np.nanmin(pooled), np.nanmax(pooled), 40) if np.nanmax(pooled) > np.nanmin(pooled) else 20
        for axis, cem_iter in zip(axes.reshape(-1), iters):
            values = loaded.get(cem_iter)
            match = [row for row in group if _int(row, "cem_iter") == cem_iter][0]
            if values is not None and values.size:
                axis.hist(values, bins=bins, alpha=0.78)
            axis.axvline(_float(match, "best_cost"), color="tab:red", linewidth=1.4, label="best")
            axis.axvline(_float(match, "elite_max_cost"), color="tab:orange", linestyle="--", linewidth=1.1, label="elite max")
            axis.set_title(f"iter {cem_iter}")
            axis.set_xlabel("terminal cost")
            axis.set_ylabel("count")
            axis.grid(alpha=0.2)
        for axis in axes.reshape(-1)[len(iters):]:
            axis.axis("off")
        axes.reshape(-1)[0].legend(frameon=False)
        fig.suptitle(f"CEM terminal-cost histograms ep {key[0]} step {key[1]}")
        _save(fig, figure_dir / f"cem_cost_hist_by_iteration_ep{key[0]}_step{key[1]}")
        plt.close(fig)


def _plot_best_over_iterations(trace_dir: Path, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(trace_dir / "cem_iteration_summary.csv")
    by_step = _group(rows, ["episode_id", "mpc_step_idx"])
    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    for key, group in sorted(by_step.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        iters = sorted({_int(row, "cem_iter") for row in group})
        best = [_float([row for row in group if _int(row, "cem_iter") == cem_iter][0], "best_cost") for cem_iter in iters]
        ax.plot(iters, best, alpha=0.45, linewidth=1.2)
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("best predicted terminal cost")
    ax.set_title("Best CEM cost over iterations")
    ax.grid(alpha=0.25)
    _save(fig, figure_dir / "cem_best_cost_over_iterations")
    plt.close(fig)


def _plot_pred_vs_true(trace_dir: Path, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(trace_dir / "true_replay_candidates.csv")
    if not rows:
        return
    x = np.asarray([_float(row, "predicted_terminal_cost_c25") for row in rows], dtype=np.float64)
    y = np.asarray([_float(row, "true_terminal_progress") for row in rows], dtype=np.float64)
    selected = np.asarray([_int(row, "selected_flag", 0) == 1 for row in rows])
    fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="white")
    ax.scatter(x[~selected], y[~selected], s=14, alpha=0.35, label="candidate")
    ax.scatter(x[selected], y[selected], s=48, color="tab:red", label="selected")
    ax.set_xlabel("predicted terminal cost")
    ax.set_ylabel("true progress")
    ax.set_title("Predicted cost vs true replay progress")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, figure_dir / "candidate_pred_cost_vs_true_progress")
    plt.close(fig)


def _plot_timeline(trace_dir: Path, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(trace_dir / "episode_timeline.csv")
    by_ep = _group(rows, ["episode_id"])
    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor="white")
    for key, group in sorted(by_ep.items(), key=lambda item: int(item[0][0])):
        group = sorted(group, key=lambda row: _int(row, "mpc_step_idx"))
        x = [_int(row, "mpc_step_idx") for row in group]
        y = [_float(row, "task_cost") for row in group]
        ax.plot(x, y, marker="o", linewidth=1.2, label=f"ep {key[0]}")
    ax.set_xlabel("MPC step")
    ax.set_ylabel("true task cost before planning")
    ax.set_title("Episode task-cost timeline")
    ax.grid(alpha=0.25)
    if len(by_ep) <= 8:
        ax.legend(frameon=False, ncol=2)
    _save(fig, figure_dir / "episode_timeline_task_cost")
    plt.close(fig)


def _plot_angle(trace_dir: Path, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(trace_dir / "candidate_angle_scores.csv")
    if not rows:
        return
    cos = np.asarray([_float(row, "cos_to_goal_direction") for row in rows], dtype=np.float64)
    selected = np.asarray([_int(row, "is_selected_final", 0) == 1 for row in rows])
    fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor="white")
    ax.hist(cos[np.isfinite(cos)], bins=40, alpha=0.75, label="all candidates")
    if np.any(selected):
        ax.hist(cos[selected & np.isfinite(cos)], bins=20, alpha=0.75, label="selected")
    ax.set_xlabel("cos(candidate displacement, goal direction)")
    ax.set_ylabel("count")
    ax.set_title("Candidate angle distribution")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    _save(fig, figure_dir / "angle_histogram")
    plt.close(fig)


def _plot_selected_vs_truebest(trace_dir: Path, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(trace_dir / "trace_failure_analysis.csv")
    rows = [row for row in rows if row.get("analysis_status") == "true_replay_available"]
    if not rows:
        return
    x = np.arange(len(rows))
    pred_selected = [_float(row, "predicted_cost_of_selected") for row in rows]
    pred_true = [_float(row, "predicted_cost_of_true_best") for row in rows]
    true_selected = [_float(row, "true_progress_of_selected") for row in rows]
    true_best = [_float(row, "true_progress_of_true_best") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), facecolor="white")
    width = 0.38
    axes[0].bar(x - width / 2, pred_selected, width=width, label="selected")
    axes[0].bar(x + width / 2, pred_true, width=width, label="true-best")
    axes[0].set_title("Predicted cost")
    axes[1].bar(x - width / 2, true_selected, width=width, label="selected")
    axes[1].bar(x + width / 2, true_best, width=width, label="true-best")
    axes[1].set_title("True progress")
    for ax in axes:
        ax.set_xlabel("planning step")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    _save(fig, figure_dir / "selected_vs_truebest_bar")
    plt.close(fig)


def _plot_rank_true_best(trace_dir: Path, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(trace_dir / "trace_failure_analysis.csv")
    rows = [row for row in rows if row.get("analysis_status") == "true_replay_available"]
    if not rows:
        return
    by_ep = _group(rows, ["episode_id"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="white")
    for key, group in sorted(by_ep.items(), key=lambda item: int(item[0][0])):
        group = sorted(group, key=lambda row: _int(row, "mpc_step_idx"))
        ax.plot([_int(row, "mpc_step_idx") for row in group], [_float(row, "rank_of_true_best_by_predicted_cost") for row in group], marker="o", label=f"ep {key[0]}")
    ax.set_xlabel("MPC step")
    ax.set_ylabel("rank of true-best by predicted cost")
    ax.set_title("True-best predicted rank over time")
    ax.grid(alpha=0.25)
    if len(by_ep) <= 8:
        ax.legend(frameon=False, ncol=2)
    _save(fig, figure_dir / "rank_of_true_best_over_time")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LeWorldModel CEM trace outputs.")
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--max_step_plots", type=int, default=8)
    args = parser.parse_args()
    trace_dir = Path(args.trace_dir)
    figure_dir = trace_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_cost_hist(trace_dir, figure_dir, args.max_step_plots)
    _plot_best_over_iterations(trace_dir, figure_dir)
    _plot_pred_vs_true(trace_dir, figure_dir)
    _plot_selected_vs_truebest(trace_dir, figure_dir)
    _plot_timeline(trace_dir, figure_dir)
    _plot_angle(trace_dir, figure_dir)
    _plot_rank_true_best(trace_dir, figure_dir)
    print(f"[plot_cem_trace] wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
