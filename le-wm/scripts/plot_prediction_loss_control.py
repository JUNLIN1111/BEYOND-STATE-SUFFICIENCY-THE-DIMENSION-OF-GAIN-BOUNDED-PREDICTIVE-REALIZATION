from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


MAIN_MODELS = ["state8", "state16", "state32", "state64", "baseline192", "state302", "state502"]
APPENDIX_MODELS = ["state8", "state16", "state32", "state64", "baseline192", "global_k32", "local_k32"]
LATENT_DIMS = {
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
SUCCESS_RATE = {
    "state8": 36,
    "state16": 72,
    "state32": 78,
    "state64": 90,
    "baseline192": 96,
    "global_k32": "",
    "local_k32": "",
    "state302": "",
    "state502": "",
}
MODEL_COLORS = {
    "state8": "#4E79A7",
    "state16": "#59A14F",
    "state32": "#F28E2B",
    "state64": "#B07AA1",
    "baseline192": "#E15759",
    "state302": "#76B7B2",
    "state502": "#EDC948",
    "global_k32": "#7F7F7F",
    "local_k32": "#000000",
}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _float(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _infer_model(run_name: str) -> str | None:
    name = run_name.lower()
    if "global" in name and "k32" in name:
        return "global_k32"
    if ("state_tangent" in name or "local_tangent" in name or "local" in name) and "k32" in name:
        return "local_k32"
    if "state502" in name:
        return "state502"
    if "state302" in name:
        return "state302"
    if "state64" in name:
        return "state64"
    if "state32" in name:
        return "state32"
    if "state16" in name:
        return "state16"
    if "state8" in name:
        return "state8"
    if "baseline_no_bottleneck" in name or "baseline192" in name:
        return "baseline192"
    return None


def _load_wandb_wide(path: Path, metric_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    rows = _read_csv(path)
    if not rows:
        raise FileNotFoundError(f"No rows found in {path}")
    columns = rows[0].keys()
    step_col = "trainer/global_step" if "trainer/global_step" in columns else next(iter(columns))
    series: Dict[str, Dict[str, List[float]]] = {}
    for col in columns:
        if not col.endswith(f" - {metric_name}"):
            continue
        if col.endswith("__MIN") or col.endswith("__MAX"):
            continue
        run_name = col[: -len(f" - {metric_name}")]
        model = _infer_model(run_name)
        if model is None:
            print(f"[pred_loss_plot] warning: could not infer model from column {col}; skipping")
            continue
        if model in series:
            print(f"[pred_loss_plot] warning: duplicate series for {model}; keeping first and skipping {run_name}")
            continue
        series[model] = {"step": [], "loss": [], "run_name": run_name}
        for row in rows:
            step = _float(row.get(step_col))
            value = _float(row.get(col))
            if math.isfinite(step) and math.isfinite(value):
                series[model]["step"].append(step)
                series[model]["loss"].append(value)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for model, data in series.items():
        step = np.asarray(data["step"], dtype=np.float64)
        loss = np.asarray(data["loss"], dtype=np.float64)
        order = np.argsort(step)
        out[model] = {"step": step[order], "loss": loss[order], "run_name": data["run_name"]}
    return out


def _ema(y: np.ndarray, alpha: float) -> np.ndarray:
    if y.size == 0:
        return y
    out = np.empty_like(y, dtype=np.float64)
    out[0] = y[0]
    for idx in range(1, y.size):
        out[idx] = alpha * y[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def _truncate(step: np.ndarray, loss: np.ndarray, max_step: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = step <= max_step
    return step[mask], loss[mask]


def _plot_loss(
    series: Dict[str, Dict[str, np.ndarray]],
    models: Sequence[str],
    output_base: Path,
    title: str,
    max_step: float,
    smoothing_alpha: float,
    include_raw: bool,
    dashed_models: Iterable[str] = (),
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc
    dashed = set(dashed_models)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.7), facecolor="white")
    plotted = []
    for model in models:
        if model not in series:
            print(f"[pred_loss_plot] warning: no loss curve for {model}; skipping")
            continue
        step, loss = _truncate(series[model]["step"], series[model]["loss"], max_step)
        if step.size == 0:
            continue
        color = MODEL_COLORS.get(model)
        linestyle = "--" if model in dashed else "-"
        if include_raw:
            ax.plot(step, loss, color=color, linewidth=0.7, alpha=0.18)
        ax.plot(step, _ema(loss, smoothing_alpha), color=color, linewidth=2.0, linestyle=linestyle, label=model)
        plotted.append(model)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel("prediction loss")
    ax.set_xlim(0, max_step)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if plotted:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def _last_window_stats(step: np.ndarray, loss: np.ndarray, max_step: float, last_steps: float) -> Tuple[float, float]:
    step, loss = _truncate(step, loss, max_step)
    if step.size == 0:
        return float("nan"), float("nan")
    threshold = max(float(np.max(step)) - last_steps, float(np.min(step)))
    values = loss[step >= threshold]
    if values.size == 0:
        values = loss[-min(20, loss.size):]
    return float(np.nanmean(values)), float(np.nanstd(values))


def _candidate_metrics(candidate_csv: Path) -> Dict[str, Dict[str, float]]:
    rows = _read_csv(candidate_csv)
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        model = row.get("model")
        metric = row.get("metric")
        if not model or not metric:
            continue
        out.setdefault(model, {})[metric] = _float(row.get("mean"))
    return out


def _make_table(
    series: Dict[str, Dict[str, np.ndarray]],
    candidate: Dict[str, Dict[str, float]],
    models: Sequence[str],
    max_step: float,
    last_steps: float,
) -> List[Dict[str, object]]:
    rows = []
    for model in models:
        if model not in series:
            continue
        mean, std = _last_window_stats(series[model]["step"], series[model]["loss"], max_step, last_steps)
        note = "bottleneck_control" if model in ("global_k32", "local_k32") else "no_bottleneck"
        metrics = candidate.get(model, {})
        rows.append(
            {
                "model": model,
                "latent_dim": LATENT_DIMS.get(model, ""),
                "final_pred_loss_mean_last_5k_steps": mean,
                "final_pred_loss_std_last_5k_steps": std,
                "candidate_false_shortcut_rate": metrics.get("candidate_false_shortcut_rate", ""),
                "candidate_goal_metric_spearman": metrics.get("candidate_goal_metric_spearman", ""),
                "pairwise_rank_acc": metrics.get("candidate_pairwise_rank_acc", ""),
                "SR": SUCCESS_RATE.get(model, ""),
                "notes": note,
            }
        )
    return rows


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), int(x.size)
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman, int(x.size)


def _correlation_rows(table: List[Dict[str, object]]) -> List[Dict[str, object]]:
    pairs = [
        ("final_pred_loss_mean_last_5k_steps", "candidate_false_shortcut_rate"),
        ("final_pred_loss_mean_last_5k_steps", "SR"),
        ("candidate_false_shortcut_rate", "SR"),
        ("candidate_goal_metric_spearman", "SR"),
    ]
    out = []
    for x_key, y_key in pairs:
        rows = [row for row in table if row.get("notes") == "no_bottleneck"]
        x = np.asarray([_float(row.get(x_key)) for row in rows], dtype=np.float64)
        y = np.asarray([_float(row.get(y_key)) for row in rows], dtype=np.float64)
        pearson, spearman, n = _pearson_spearman(x, y)
        out.append({"x": x_key, "y": y_key, "pearson": pearson, "spearman": spearman, "n": n, "scope": "no_bottleneck"})
    return out


def _write_interpretation(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Prediction Loss Negative Control",
                "",
                "This experiment is a negative control for the plannable-latent-geometry story.",
                "",
                "If prediction-loss curves are similar across latent widths while candidate-level false shortcuts, ranking metrics, or closed-loop success differ, then prediction loss alone does not explain planning performance.",
                "",
                "The intended conclusion is not that prediction loss is irrelevant. Prediction accuracy is necessary for a useful world model. The narrower claim is that one-step latent prediction loss is insufficient as a diagnostic for distance-based latent planning, because the planner also depends on whether Euclidean latent distance metricizes transition-to-goal geometry.",
                "",
                "Suggested paper sentence:",
                "",
                "> Although all models achieve comparable latent prediction losses, their candidate-level transition false shortcuts and closed-loop performance differ substantially. This suggests that one-step predictive accuracy is not sufficient to diagnose whether the latent metric is suitable as a planner cost.",
            ]
        )
        + "\n"
    )


def _default_loss_csv() -> str:
    candidates = [
        Path("rollout_results/plannable_dim_evidence/prediction_loss_control/wandb_export.csv"),
        Path("rollout_results/prediction_loss/wandb_export.csv"),
        Path("/home/junlin/下载/wandb_export_2026-06-15T01_14_24.432+00_00.csv"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prediction-loss negative-control figures.")
    parser.add_argument("--loss_csv", default=_default_loss_csv(), help="W&B wide CSV containing '<run> - fit/pred_loss' columns.")
    parser.add_argument("--metric_name", default="fit/pred_loss")
    parser.add_argument("--candidate_metric_ci_csv", default="rollout_results/plannable_dim_evidence/trained_latent_metric_n100/candidate_future_metric_summary_with_ci.csv")
    parser.add_argument("--output_dir", default="rollout_results/plannable_dim_evidence/plots")
    parser.add_argument("--summary_dir", default="rollout_results/plannable_dim_evidence/summaries")
    parser.add_argument("--max_step", type=float, default=40000)
    parser.add_argument("--last_steps", type=float, default=5000)
    parser.add_argument("--smoothing_alpha", type=float, default=0.08)
    parser.add_argument("--no_raw", action="store_true")
    args = parser.parse_args()

    if not args.loss_csv:
        raise FileNotFoundError("No --loss_csv provided and no default W&B export CSV found.")
    series = _load_wandb_wide(Path(args.loss_csv), args.metric_name)
    candidate = _candidate_metrics(Path(args.candidate_metric_ci_csv))
    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_dir)
    _plot_loss(
        series,
        MAIN_MODELS,
        output_dir / "prediction_loss_main_dim_sweep",
        "Prediction loss: no-bottleneck dimension sweep",
        args.max_step,
        args.smoothing_alpha,
        not args.no_raw,
    )
    _plot_loss(
        series,
        APPENDIX_MODELS,
        output_dir / "prediction_loss_with_bottleneck_controls",
        "Prediction loss with bottleneck controls",
        args.max_step,
        args.smoothing_alpha,
        not args.no_raw,
        dashed_models=("global_k32", "local_k32"),
    )
    table = _make_table(series, candidate, sorted(set(MAIN_MODELS + APPENDIX_MODELS), key=lambda m: (m in ("global_k32", "local_k32"), LATENT_DIMS.get(m, 10**9), m)), args.max_step, args.last_steps)
    _write_csv(summary_dir / "prediction_loss_control_table.csv", table)
    _write_csv(summary_dir / "prediction_loss_control_correlations.csv", _correlation_rows(table))
    _write_interpretation(summary_dir / "prediction_loss_control_interpretation.md")
    print(f"[pred_loss_plot] wrote {output_dir / 'prediction_loss_main_dim_sweep.png'}")
    print(f"[pred_loss_plot] wrote {output_dir / 'prediction_loss_with_bottleneck_controls.png'}")
    print(f"[pred_loss_plot] wrote {summary_dir / 'prediction_loss_control_table.csv'}")


if __name__ == "__main__":
    main()
