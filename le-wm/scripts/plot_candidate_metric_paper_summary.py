from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

MODEL_ORDER = ["state8", "state16", "state32", "state64", "baseline192"]
MODEL_DIMS = {"state8": 8, "state16": 16, "state32": 32, "state64": 64, "baseline192": 192}
METRICS = [
    ("candidate_false_shortcut_rate", "candidate_false_shortcut_vs_dim_n100", "Candidate false shortcut rate", 0.05),
    ("candidate_goal_metric_spearman", "candidate_goal_spearman_vs_dim_n100", "Spearman(graph distance, latent score)", None),
    ("candidate_pairwise_rank_acc", "candidate_pairwise_rank_acc_vs_dim_n100", "Pairwise rank accuracy", 0.5),
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _paper_rows(rows: List[Dict[str, str]], q_far: float, q_near: float) -> List[Dict[str, object]]:
    out = []
    for model in MODEL_ORDER:
        for metric, _filename, _ylabel, _hline in METRICS:
            match = next(
                (
                    row for row in rows
                    if row.get("model") == model
                    and row.get("metric") == metric
                    and abs(_float(row, "q_far") - q_far) < 1e-9
                    and abs(_float(row, "q_near") - q_near) < 1e-9
                ),
                None,
            )
            if match is None:
                continue
            out.append(
                {
                    "model": model,
                    "latent_dim": MODEL_DIMS[model],
                    "metric": metric,
                    "mean": _float(match, "mean"),
                    "ci95_low": _float(match, "ci95_low"),
                    "ci95_high": _float(match, "ci95_high"),
                    "std": _float(match, "std"),
                    "num_windows": int(_float(match, "num_windows")) if math.isfinite(_float(match, "num_windows")) else "",
                    "q_far": q_far,
                    "q_near": q_near,
                }
            )
    return out


def _plot_main(rows: List[Dict[str, object]], plots_dir: Path, d_refs: List[int]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    for metric, filename, ylabel, hline in METRICS:
        selected = [row for row in rows if row["metric"] == metric]
        if not selected:
            continue
        xs = np.asarray([float(row["latent_dim"]) for row in selected])
        ys = np.asarray([float(row["mean"]) for row in selected])
        lows = np.asarray([float(row["ci95_low"]) for row in selected])
        highs = np.asarray([float(row["ci95_high"]) for row in selected])
        labels = [str(row["model"]) for row in selected]
        fig, ax = plt.subplots(figsize=(5.7, 3.6), facecolor="white")
        ax.errorbar(xs, ys, yerr=np.vstack([ys - lows, highs - ys]), marker="o", linewidth=2, capsize=4)
        for x, y, label in zip(xs, ys, labels):
            ax.text(x, y, f" {label}", fontsize=8, va="center")
        if hline is not None:
            ax.axhline(hline, color="0.35", linestyle="--", linewidth=1, label=f"reference={hline:g}")
        for d_ref, color in zip(d_refs, ["tab:orange", "tab:green", "tab:red"]):
            ax.axvline(d_ref, color=color, linestyle=":", linewidth=1, label=f"d={d_ref}")
        ax.set_xscale("log")
        ax.set_xlabel("latent dimension")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{filename}.png", dpi=260)
        fig.savefig(plots_dir / f"{filename}.pdf")
        plt.close(fig)


def _write_interpretation(path: Path, rows: List[Dict[str, object]]) -> None:
    by_metric = {(row["model"], row["metric"]): row for row in rows}
    state8 = by_metric.get(("state8", "candidate_false_shortcut_rate"), {})
    baseline = by_metric.get(("baseline192", "candidate_false_shortcut_rate"), {})
    ratio = float("nan")
    if state8 and baseline and float(baseline["mean"]) > 0:
        ratio = float(state8["mean"]) / float(baseline["mean"])
    lines = [
        "# Candidate Future Metric Summary, 100 windows",
        "",
        "This is mechanism evidence for the plannable-latent-dimension story.",
        "",
        f"- state8 mean false shortcut rate: {state8.get('mean', 'NA')}",
        f"- baseline192 mean false shortcut rate: {baseline.get('mean', 'NA')}",
        f"- state8 / baseline192 false-shortcut ratio: {ratio:.2f}x" if math.isfinite(ratio) else "- state8 / baseline192 false-shortcut ratio: NA",
        "",
        "Interpretation: state8 has the highest mean false shortcut rate, while 16D and above form a better plateau. Confidence intervals overlap, so this should not be phrased as a statistical-significance claim. The result supports the mechanism-level claim that too-small latent width can produce more graph-far / score-near candidate errors; it does not imply a universal monotonic relationship between nominal dimension and planning performance.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _threshold_summary(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        if row.get("model") not in MODEL_ORDER:
            continue
        out.append(
            {
                "model": row.get("model"),
                "latent_dim": MODEL_DIMS.get(row.get("model", ""), ""),
                "q_far": _float(row, "q_far"),
                "q_near": _float(row, "q_near"),
                "mean": _float(row, "mean") if "mean" in row else _float(row, "candidate_false_shortcut_rate"),
                "ci95_low": _float(row, "ci95_low"),
                "ci95_high": _float(row, "ci95_high"),
                "relative_to_random": _float(row, "relative_to_random"),
                "relative_increase_vs_baseline192": _float(row, "relative_increase_vs_baseline192"),
                "num_windows": _float(row, "num_windows"),
            }
        )
    return out


def _plot_threshold(rows: List[Dict[str, object]], plots_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    combos = sorted({(float(row["q_far"]), float(row["q_near"])) for row in rows})
    fig, axes = plt.subplots(len(combos), 1, figsize=(6.2, max(2.4, 2.0 * len(combos))), sharex=True, facecolor="white")
    if len(combos) == 1:
        axes = [axes]
    for ax, (q_far, q_near) in zip(axes, combos):
        selected = [row for row in rows if abs(float(row["q_far"]) - q_far) < 1e-9 and abs(float(row["q_near"]) - q_near) < 1e-9]
        selected.sort(key=lambda row: float(row["latent_dim"]))
        xs = np.asarray([float(row["latent_dim"]) for row in selected])
        ys = np.asarray([float(row["mean"]) for row in selected])
        lows = np.asarray([float(row["ci95_low"]) for row in selected])
        highs = np.asarray([float(row["ci95_high"]) for row in selected])
        ax.errorbar(xs, ys, yerr=np.vstack([ys - lows, highs - ys]), marker="o", linewidth=1.8, capsize=3)
        ax.axhline(q_near, color="0.35", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_ylabel(f"q_far={q_far:g}\nq_near={q_near:g}")
        ax.grid(alpha=0.20)
    axes[-1].set_xlabel("latent dimension")
    fig.supylabel("candidate false shortcut rate")
    fig.tight_layout()
    fig.savefig(plots_dir / "candidate_threshold_sweep_false_shortcuts_n100.png", dpi=260)
    fig.savefig(plots_dir / "candidate_threshold_sweep_false_shortcuts_n100.pdf")
    plt.close(fig)


def _write_threshold_interpretation(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Candidate Threshold Sweep Summary",
        "",
        "This sweep checks whether the state8 false-shortcut trend depends on the single default choice q_far=0.75 and q_near=0.05.",
        "",
        "Interpret state8 against higher-D models across the threshold grid. Use 'robust across thresholds' only if state8 is consistently worse across the reasonable settings; otherwise phrase the result as directionally supportive but threshold-sensitive.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready candidate-future metric summaries and plots.")
    parser.add_argument("--summary_ci_csv", default="rollout_results/plannable_dim_evidence/trained_latent_metric_n100/candidate_future_metric_summary_with_ci.csv")
    parser.add_argument("--threshold_ci_csv", default="rollout_results/plannable_dim_evidence/trained_latent_metric_n100/candidate_future_threshold_sweep_with_ci.csv")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence")
    parser.add_argument("--q_far", type=float, default=0.75)
    parser.add_argument("--q_near", type=float, default=0.05)
    parser.add_argument("--d_refs", default="96,163,302")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    summaries_dir = output_dir / "summaries"
    summary_rows = _read_csv(Path(args.summary_ci_csv))
    if not summary_rows:
        raise FileNotFoundError(f"No rows found in {args.summary_ci_csv}")
    paper_rows = _paper_rows(summary_rows, args.q_far, args.q_near)
    d_refs = [int(item) for item in str(args.d_refs).replace(",", " ").split() if item.strip()]
    _write_csv(summaries_dir / "candidate_future_metric_paper_table_n100.csv", paper_rows)
    _plot_main(paper_rows, plots_dir, d_refs)
    _write_interpretation(summaries_dir / "candidate_future_metric_interpretation_n100.md", paper_rows)

    threshold_rows = _threshold_summary(_read_csv(Path(args.threshold_ci_csv)))
    if threshold_rows:
        _write_csv(summaries_dir / "candidate_threshold_sweep_summary_n100.csv", threshold_rows)
        _plot_threshold(threshold_rows, plots_dir)
        _write_threshold_interpretation(summaries_dir / "candidate_threshold_sweep_interpretation_n100.md", threshold_rows)
    else:
        print(f"[candidate_paper] warning: no threshold rows found in {args.threshold_ci_csv}")
    print(f"[candidate_paper] wrote summaries to {summaries_dir}")
    print(f"[candidate_paper] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
