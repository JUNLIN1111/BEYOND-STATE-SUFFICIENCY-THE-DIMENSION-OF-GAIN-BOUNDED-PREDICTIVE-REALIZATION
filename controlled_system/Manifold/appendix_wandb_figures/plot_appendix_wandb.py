#!/usr/bin/env python3
"""Generate supplementary W&B appendix figures from cleaned CSV exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


DIM_ORDER = [8, 16, 32, 64, 96, 192]
BASELINE_LABELS = {
    8: "State-8",
    16: "State-16",
    32: "State-32",
    64: "State-64",
    96: "State-96",
    192: "Baseline-192",
}
METHOD_LABELS = ["State-32", "State-tangent (k=32)", "Global-linear (k=32)"]

DIM_COLORS = {
    8: "#4E79A7",
    16: "#F28E2B",
    32: "#59A14F",
    64: "#E15759",
    96: "#B07AA1",
    192: "#9C755F",
}
METHOD_STYLES = {
    "State-32": {"color": DIM_COLORS[32], "linestyle": "-", "marker": "o", "hatch": ""},
    "State-tangent (k=32)": {
        "color": DIM_COLORS[32],
        "linestyle": "--",
        "marker": "s",
        "hatch": "///",
    },
    "Global-linear (k=32)": {
        "color": DIM_COLORS[32],
        "linestyle": ":",
        "marker": "^",
        "hatch": "\\\\\\",
    },
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeseries",
        type=Path,
        default=Path("/home/junlin/下载/appendix_wandb_timeseries_long.csv"),
        help="Cleaned long-form W&B timeseries CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/home/junlin/下载/appendix_wandb_plateau_summary.csv"),
        help="Cleaned plateau summary CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for PDF/PNG outputs.",
    )
    return parser.parse_args()


def load_inputs(timeseries_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.read_csv(timeseries_path)
    summary = pd.read_csv(summary_path)
    ts = ts.loc[ts["group"] != "incomplete_smoke"].copy()
    summary = summary.loc[summary["group"] != "incomplete_smoke"].copy()
    return ts, summary


def save_figure(fig: mpl.figure.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
    )


def style_axis(ax: mpl.axes.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.grid(True, which="minor", color="#ECECEC", linewidth=0.4, alpha=0.6)
    ax.tick_params(length=3, width=0.8)


def format_training_step_axis(ax: mpl.axes.Axes) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(50_000))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if value == 0 else f"{value / 1000:.0f}k")
    )


def smoothed_curve(frame: pd.DataFrame) -> pd.DataFrame:
    curve = frame.sort_values("global_step").copy()
    curve["plot_value"] = curve["value"].rolling(
        window=51, center=True, min_periods=1
    ).median()
    return curve


def baseline_summary(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    frame = summary.loc[
        (summary["group"] == "dimension_baseline") & (summary["metric"] == metric)
    ].copy()
    frame["latent_dim"] = frame["latent_dim"].astype(int)
    frame = frame.sort_values("latent_dim")
    return frame


def method_summary(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    state32 = summary.loc[
        (summary["group"] == "dimension_baseline")
        & (summary["metric"] == metric)
        & (summary["label"] == "State-32")
    ]
    methods = summary.loc[
        (summary["group"] == "method_ablation") & (summary["metric"] == metric)
    ]
    frame = pd.concat([state32, methods], ignore_index=True)
    frame["label"] = pd.Categorical(frame["label"], categories=METHOD_LABELS, ordered=True)
    return frame.sort_values("label")


def plot_training_convergence(ts: pd.DataFrame, outdir: Path) -> None:
    pred = ts.loc[ts["metric"] == "fit/pred_loss"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)

    ax = axes[0]
    for dim in DIM_ORDER:
        label = BASELINE_LABELS[dim]
        frame = pred.loc[(pred["group"] == "dimension_baseline") & (pred["label"] == label)]
        if frame.empty:
            continue
        curve = smoothed_curve(frame)
        ax.plot(
            curve["global_step"],
            curve["plot_value"],
            color=DIM_COLORS[dim],
            linewidth=1.3,
            label=label,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Prediction loss")
    ax.set_yscale("log")
    format_training_step_axis(ax)
    ax.legend(frameon=False, ncol=2, columnspacing=0.9, handlelength=1.8)
    panel_label(ax, "(a)")
    style_axis(ax)

    ax = axes[1]
    for label in METHOD_LABELS:
        if label == "State-32":
            frame = pred.loc[
                (pred["group"] == "dimension_baseline") & (pred["label"] == label)
            ]
        else:
            frame = pred.loc[(pred["group"] == "method_ablation") & (pred["label"] == label)]
        if frame.empty:
            continue
        curve = smoothed_curve(frame)
        style = METHOD_STYLES[label]
        ax.plot(
            curve["global_step"],
            curve["plot_value"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("Training step")
    ax.set_yscale("log")
    format_training_step_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "(b)")
    style_axis(ax)

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, outdir, "fig_appendix_training_convergence")


def plot_local_geometry(summary: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("fit/mean_rank", "Reported mean rank"),
        (
            "fit/loss_analysis/local_tangent/top_eig_fraction_mean",
            "Top-eigenvalue fraction",
        ),
        (
            "fit/loss_analysis/local_tangent/cov_pr_rank_mean",
            "Local covariance\nparticipation-ratio rank",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55))
    for ax, (metric, ylabel), letter in zip(axes, metrics, ["(a)", "(b)", "(c)"]):
        frame = baseline_summary(summary, metric)
        ax.plot(
            frame["latent_dim"],
            frame["tail10_median"],
            color="#222222",
            linewidth=1.0,
            zorder=1,
        )
        ax.scatter(
            frame["latent_dim"],
            frame["tail10_median"],
            c=[DIM_COLORS[int(dim)] for dim in frame["latent_dim"]],
            s=28,
            edgecolor="#222222",
            linewidth=0.5,
            zorder=2,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(DIM_ORDER)
        ax.set_xticklabels([str(dim) for dim in DIM_ORDER])
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
        style_axis(ax)
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, outdir, "fig_appendix_local_geometry")


def plot_transition_geometry(summary: pd.DataFrame, outdir: Path) -> None:
    full_metric = "fit/loss_analysis/analysis/full/transition/delta_norm_ratio_p90"
    shuffle_metric = (
        "fit/loss_analysis/analysis/shuffle_metric_only/transition/delta_norm_ratio_p90"
    )
    align_metric = (
        "fit/loss_analysis/analysis/full/transition/cosine_alignment_median"
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))

    ax = axes[0]
    for metric, label, color, linestyle, marker in [
        (full_metric, "Full transition", "#222222", "-", "o"),
        (shuffle_metric, "Shuffled transition", "#666666", "--", "s"),
    ]:
        frame = baseline_summary(summary, metric)
        ax.plot(
            frame["latent_dim"],
            frame["tail10_median"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=color,
            linewidth=1.3,
            markersize=4,
            label=label,
        )
    ax.axhline(1.0, color="#B0B0B0", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIM_ORDER)
    ax.set_xticklabels([str(dim) for dim in DIM_ORDER])
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("90th-percentile\ndelta-norm ratio")
    ax.legend(frameon=False)
    panel_label(ax, "(a)")
    style_axis(ax)

    ax = axes[1]
    frame = baseline_summary(summary, align_metric)
    ax.plot(
        frame["latent_dim"],
        frame["tail10_median"],
        color="#222222",
        linewidth=1.1,
        zorder=1,
    )
    ax.scatter(
        frame["latent_dim"],
        frame["tail10_median"],
        c=[DIM_COLORS[int(dim)] for dim in frame["latent_dim"]],
        s=30,
        edgecolor="#222222",
        linewidth=0.5,
        zorder=2,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIM_ORDER)
    ax.set_xticklabels([str(dim) for dim in DIM_ORDER])
    ax.set_ylim(0.90, 1.00)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Median cosine alignment")
    panel_label(ax, "(b)")
    style_axis(ax)

    fig.tight_layout(w_pad=1.3)
    save_figure(fig, outdir, "fig_appendix_transition_geometry")


def plot_method_ablation(summary: pd.DataFrame, outdir: Path) -> None:
    panels = [
        ("fit/pred_loss", "Prediction loss", "log"),
        (
            "fit/loss_analysis/local_tangent/top_eig_fraction_mean",
            "Top-eigenvalue fraction",
            "linear",
        ),
        (
            "fit/loss_analysis/local_tangent/cov_pr_rank_mean",
            "Local covariance PR rank",
            "linear",
        ),
        (
            "fit/loss_analysis/analysis/full/transition/delta_norm_ratio_p90",
            "90th-percentile\ndelta-norm ratio",
            "linear",
        ),
        (
            "fit/loss_analysis/analysis/full/transition/cosine_alignment_median",
            "Median cosine alignment",
            "linear",
        ),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(8.2, 2.65))
    x = np.arange(len(METHOD_LABELS))

    for ax, (metric, ylabel, yscale), letter in zip(
        axes, panels, ["(a)", "(b)", "(c)", "(d)", "(e)"]
    ):
        frame = method_summary(summary, metric)
        values = frame.set_index("label").loc[METHOD_LABELS, "tail10_median"]
        bars = ax.bar(
            x,
            values,
            width=0.62,
            color=DIM_COLORS[32],
            edgecolor="#222222",
            linewidth=0.6,
        )
        for bar, label in zip(bars, METHOD_LABELS):
            bar.set_hatch(METHOD_STYLES[label]["hatch"])
        if metric.endswith("delta_norm_ratio_p90"):
            ax.axhline(1.0, color="#B0B0B0", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        panel_label(ax, letter)
        style_axis(ax, grid=True)

    handles = [
        mpl.patches.Patch(
            facecolor=DIM_COLORS[32],
            edgecolor="#222222",
            hatch=METHOD_STYLES[label]["hatch"],
            label=label,
        )
        for label in METHOD_LABELS
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.9)
    save_figure(fig, outdir, "fig_appendix_method_ablation")


def write_captions(outdir: Path) -> None:
    captions = """# Appendix W&B Figure Captions

**Figure A1. Training convergence.** Prediction loss curves are shown with a centred rolling median over 51 logged points. Lines stop at their actual final logged training step, without extrapolation or forward-filling. Panel (a) shows baseline latent-dimension runs. The State-8 prediction-loss export uses run ID `baseline_state8_no_bottleneck_TBA`, whereas the other State-8 metrics use the fresh-seed run; both are labelled State-8. Panel (b) compares State-32, State-tangent (k=32), and Global-linear (k=32). Training lengths differ, so this comparison is descriptive rather than a controlled convergence-speed comparison.

**Figure A2. Local representation geometry.** Plateau statistics are tail10 medians over the final 10% of logged points for baseline dimension runs only. Larger ambient dimensions are associated with lower top-eigenvalue concentration and higher local effective rank.

**Figure A3. Full versus shuffled transition geometry.** Plateau statistics are tail10 medians for baseline dimension runs only. Full transition pairings remain much closer to a delta-norm ratio of 1 than shuffled-transition controls across dimensions; directional alignment is shown only for the full transition pairing.

**Figure A4. Structured predictor ablation.** Plateau statistics are tail10 medians. State-tangent (k=32) has prediction loss similar to State-32, a lower top-eigenvalue fraction, a higher local participation-ratio rank, and the best directional alignment. Global-linear (k=32) has a norm ratio close to 1 but poorer directional alignment and higher prediction loss, indicating that matching transition magnitude alone is not sufficient; transition direction also matters.
"""
    (outdir / "appendix_wandb_captions.md").write_text(captions, encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    ts, summary = load_inputs(args.timeseries, args.summary)
    plot_training_convergence(ts, args.outdir)
    plot_local_geometry(summary, args.outdir)
    plot_transition_geometry(summary, args.outdir)
    plot_method_ablation(summary, args.outdir)
    write_captions(args.outdir)


if __name__ == "__main__":
    main()
