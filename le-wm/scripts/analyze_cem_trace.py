from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


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


def _analyze_predicted_only(trace_dir: Path) -> List[Dict[str, object]]:
    pools = _read_csv(trace_dir / "candidate_pool_summary.csv")
    exec_rows = _read_csv(trace_dir / "executed_step_trace.csv")
    angles = _read_csv(trace_dir / "candidate_angle_scores.csv")
    by_step = _group(pools, ["episode_id", "mpc_step_idx"])
    exec_by_step = {
        (row.get("episode_id", ""), row.get("mpc_step_idx", "")): row
        for row in exec_rows
    }
    angle_by_step = _group(angles, ["episode_id", "mpc_step_idx"])
    rows: List[Dict[str, object]] = []
    for key, group in sorted(by_step.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        costs = np.asarray([_float(row, "terminal_cost_c25") for row in group], dtype=np.float64)
        ranks = np.asarray([_int(row, "rank_by_terminal_cost") for row in group], dtype=np.int64)
        selected = [row for row in group if _int(row, "is_selected_final", 0) == 1]
        selected_row = selected[0] if selected else min(group, key=lambda row: _float(row, "terminal_cost_c25"))
        selected_idx = _int(selected_row, "candidate_idx")
        angle_rows = angle_by_step.get(key, [])
        selected_angle = [row for row in angle_rows if _int(row, "candidate_idx") == selected_idx]
        cos_selected = _float(selected_angle[0], "cos_to_goal_direction") if selected_angle else float("nan")
        row_out = {
            "episode_id": int(key[0]),
            "mpc_step_idx": int(key[1]),
            "analysis_status": "predicted_only_no_true_replay",
            "candidate_count": int(len(group)),
            "best_predicted_cost": float(np.nanmin(costs)),
            "median_predicted_cost": float(np.nanmedian(costs)),
            "mean_predicted_cost": float(np.nanmean(costs)),
            "selected_candidate_idx": int(selected_idx),
            "selected_predicted_cost": float(_float(selected_row, "terminal_cost_c25")),
            "selected_rank_by_predicted_cost": int(_int(selected_row, "rank_by_terminal_cost")),
            "selected_cos_to_goal_direction": cos_selected,
            "candidate_support_failure": "",
            "terminal_score_ranking_failure": "",
            "imagination_mismatch_failure": "",
            "replan_recovery": "",
            "compounding_failure": "",
            "note": "True candidate replay was not available, so failure categories A-C are not assigned.",
        }
        exec_row = exec_by_step.get(key)
        if exec_row:
            row_out.update(
                {
                    "true_task_cost_before_execution": _float(exec_row, "true_task_cost_before_execution"),
                    "true_task_cost_after_execution": _float(exec_row, "true_task_cost_after_execution"),
                    "true_progress_delta_after_first_block": _float(exec_row, "true_progress_delta"),
                    "progress_improved_after_first_block": exec_row.get("progress_improved_after_first_block", ""),
                }
            )
        rows.append(row_out)
    return rows


def _analyze_true_replay(trace_dir: Path, progress_threshold: float) -> List[Dict[str, object]]:
    replay = _read_csv(trace_dir / "true_replay_candidates.csv")
    if not replay:
        return []
    by_step = _group(replay, ["episode_id", "mpc_step_idx"])
    rows: List[Dict[str, object]] = []
    for key, group in sorted(by_step.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        pred = np.asarray([_float(row, "predicted_terminal_cost_c25") for row in group], dtype=np.float64)
        true_progress = np.asarray([_float(row, "true_terminal_progress") for row in group], dtype=np.float64)
        selected_mask = np.asarray([_int(row, "selected_flag", 0) == 1 for row in group])
        true_good = true_progress >= progress_threshold
        true_best_idx = int(np.nanargmax(true_progress))
        selected_idx = int(np.where(selected_mask)[0][0]) if np.any(selected_mask) else int(np.nanargmin(pred))
        rank_true_best_by_pred = int(np.where(np.argsort(pred) == true_best_idx)[0][0] + 1)
        candidate_support_failure = not bool(np.any(true_good))
        terminal_score_ranking_failure = bool(np.any(true_good) and rank_true_best_by_pred > 1)
        imagination_mismatch_failure = bool(pred[selected_idx] <= np.nanpercentile(pred, 10) and true_progress[selected_idx] < progress_threshold)
        rows.append(
            {
                "episode_id": int(key[0]),
                "mpc_step_idx": int(key[1]),
                "analysis_status": "true_replay_available",
                "candidate_count": int(len(group)),
                "candidate_support_failure": int(candidate_support_failure),
                "terminal_score_ranking_failure": int(terminal_score_ranking_failure),
                "imagination_mismatch_failure": int(imagination_mismatch_failure),
                "rank_of_true_best_by_predicted_cost": rank_true_best_by_pred,
                "predicted_cost_of_true_best": float(pred[true_best_idx]),
                "predicted_cost_of_selected": float(pred[selected_idx]),
                "true_progress_of_true_best": float(true_progress[true_best_idx]),
                "true_progress_of_selected": float(true_progress[selected_idx]),
                "rank_regret": float(true_progress[true_best_idx] - true_progress[selected_idx]),
            }
        )
    return rows


def _write_summary(trace_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        text = "# CEM Trace Failure Summary\n\nNo trace rows were available.\n"
    else:
        statuses = {}
        for row in rows:
            statuses[str(row.get("analysis_status", "unknown"))] = statuses.get(str(row.get("analysis_status", "unknown")), 0) + 1
        true_replay = [row for row in rows if row.get("analysis_status") == "true_replay_available"]
        predicted = [row for row in rows if row.get("analysis_status") == "predicted_only_no_true_replay"]
        lines = [
            "# CEM Trace Failure Summary",
            "",
            f"Planning steps analyzed: {len(rows)}",
            "",
            "## Status counts",
            "",
        ]
        for key, value in statuses.items():
            lines.append(f"- `{key}`: {value}")
        if true_replay:
            lines.extend(
                [
                    "",
                    "## True replay categories",
                    "",
                    f"- candidate support failures: {sum(int(row.get('candidate_support_failure', 0)) for row in true_replay)}",
                    f"- terminal-score ranking failures: {sum(int(row.get('terminal_score_ranking_failure', 0)) for row in true_replay)}",
                    f"- imagination mismatch failures: {sum(int(row.get('imagination_mismatch_failure', 0)) for row in true_replay)}",
                ]
            )
        if predicted:
            selected_costs = [_float({k: str(v) for k, v in row.items()}, "selected_predicted_cost") for row in predicted]
            lines.extend(
                [
                    "",
                    "## Predicted-only trace",
                    "",
                    "True candidate replay was not available, so support/ranking/imagination categories are intentionally left unassigned.",
                    f"- mean selected predicted cost: {float(np.nanmean(selected_costs)):.6g}",
                    f"- median selected predicted cost: {float(np.nanmedian(selected_costs)):.6g}",
                ]
            )
        text = "\n".join(lines) + "\n"
    (trace_dir / "trace_failure_summary.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LeWorldModel CEM trace outputs.")
    parser.add_argument("--trace_dir", required=True)
    parser.add_argument("--progress_threshold", type=float, default=0.01)
    args = parser.parse_args()
    trace_dir = Path(args.trace_dir)
    true_rows = _analyze_true_replay(trace_dir, args.progress_threshold)
    rows = true_rows if true_rows else _analyze_predicted_only(trace_dir)
    _write_csv(trace_dir / "trace_failure_analysis.csv", rows)
    _write_summary(trace_dir, rows)
    print(f"[analyze_cem_trace] wrote {trace_dir / 'trace_failure_analysis.csv'}")
    print(f"[analyze_cem_trace] wrote {trace_dir / 'trace_failure_summary.md'}")


if __name__ == "__main__":
    main()

