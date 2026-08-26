from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


MAP_ORDER = [
    "path_0",
    "one_t_junction",
    "comb_4",
    "comb_8",
    "comb_12",
    "double_comb_12",
    "hierarchical_tree_8",
]
DIM_ORDER = [2, 4, 8, 16, 32, 64, 128]


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def mean_ci(group: pd.DataFrame, metric: str) -> Dict[str, float]:
    values = group[metric].astype(float).dropna().to_numpy()
    n = int(values.size)
    if n == 0:
        return {
            f"{metric}_mean": float("nan"),
            f"{metric}_std": float("nan"),
            f"{metric}_sem": float("nan"),
            f"{metric}_ci95_low": float("nan"),
            f"{metric}_ci95_high": float("nan"),
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    delta = 1.96 * sem
    return {
        f"{metric}_mean": mean,
        f"{metric}_std": std,
        f"{metric}_sem": sem,
        f"{metric}_ci95_low": mean - delta,
        f"{metric}_ci95_high": mean + delta,
    }


def aggregate_by_seed(df: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in df.groupby(list(group_cols), sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for metric in metrics:
            if metric in group:
                row.update(mean_ci(group, metric))
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, path: Path, columns: Sequence[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    present = [col for col in columns if col in df.columns]
    latex = df[present].to_latex(index=False, escape=True, float_format=lambda value: f"{value:.3g}")
    latex = latex.replace("\\begin{tabular}", f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
    latex = latex.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}")
    path.write_text(latex)


def _sort_map_dim(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["map"] = out["map"].astype(str)
    out["_map_order"] = out["map"].map({name: idx for idx, name in enumerate(MAP_ORDER)})
    if "m" in out:
        out["_dim_order"] = out["m"].map({dim: idx for idx, dim in enumerate(DIM_ORDER)})
        out = out.sort_values(["_map_order", "_dim_order", "seed"] if "seed" in out else ["_map_order", "_dim_order"])
        out = out.drop(columns=["_map_order", "_dim_order"])
    else:
        out = out.sort_values(["_map_order"]).drop(columns=["_map_order"])
    return out.reset_index(drop=True)


def spearman_bootstrap(df: pd.DataFrame, complexity_col: str, metric_col: str, rng: np.random.Generator, reps: int) -> Dict[str, float]:
    sub = df[[complexity_col, metric_col]].dropna()
    if len(sub) < 3 or sub[complexity_col].nunique() < 2 or sub[metric_col].nunique() < 2:
        return {"spearman": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rho = float(sub[complexity_col].rank().corr(sub[metric_col].rank()))
    values = []
    arr = sub.to_numpy(dtype=np.float64)
    for _ in range(int(reps)):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        sample_df = pd.DataFrame(sample, columns=[complexity_col, metric_col])
        if sample_df[complexity_col].nunique() < 2 or sample_df[metric_col].nunique() < 2:
            continue
        values.append(float(sample_df[complexity_col].rank().corr(sample_df[metric_col].rank())))
    if not values:
        return {"spearman": rho, "ci95_low": float("nan"), "ci95_high": float("nan")}
    return {"spearman": rho, "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975))}


def build_tables(diag: pd.DataFrame, complexity: pd.DataFrame, out_dir: Path, bootstrap_reps: int) -> Dict[str, pd.DataFrame]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    complexity = complexity.rename(columns={"map_name": "map"}).copy()
    table1_cols = [
        "map",
        "num_states",
        "physical_state_dimension",
        "branch_excess",
        "junction_count",
        "dead_end_count",
        "cycle_rank",
        "graph_diameter",
        "mean_shortest_path",
    ]
    complexity["physical_state_dimension"] = 2
    table1 = _sort_map_dim(complexity[table1_cols])

    table2 = diag.rename(columns={"map_name": "map"}).copy()
    table2 = table2[
        [
            "map",
            "m",
            "seed",
            "current_reconstruction_rmse",
            "exact_current_state_accuracy",
            "exact_grid_cell_accuracy",
            "current_state_decoding_mistakes",
            "latent_collision_count",
            "min_pair_distance",
            "pair_distance_p1",
            "pair_distance_p5",
            "pair_distance_median",
            "pair_distance_p99",
        ]
    ]
    table2 = _sort_map_dim(table2)

    table3 = diag.rename(columns={"map_name": "map"}).copy()
    table3 = table3[
        [
            "map",
            "m",
            "seed",
            "q95_r_true",
            "q99_r_true",
            "max_r_true",
            "participation_effective_rank",
            "entropy_effective_rank",
            "k95_variance",
            "k99_variance",
        ]
    ]
    table3 = _sort_map_dim(table3)

    table4 = diag.rename(columns={"map_name": "map"}).copy()
    table4 = table4[
        [
            "map",
            "m",
            "seed",
            "decoded_next_rmse",
            "decoded_next_accuracy",
            "mean_raw_latent_pred_l2",
            "q95_raw_latent_pred_l2",
            "q99_raw_latent_pred_l2",
            "mean_raw_latent_pred_l2_over_sqrt_m",
            "per_coordinate_pred_mse",
            "q99_norm_pair_error_now",
            "q99_norm_pair_error_exclude_bottom1_denom",
            "q99_norm_pair_error_exclude_bottom5_denom",
        ]
    ]
    table4 = _sort_map_dim(table4)

    summary_metrics = [
        "q99_r_true",
        "decoded_next_rmse",
        "nearest_state_next_accuracy",
        "q99_norm_pair_error_now",
        "q99_norm_pair_error_exclude_bottom5_denom",
        "exact_current_state_accuracy",
        "decoded_next_accuracy",
        "per_coordinate_pred_mse",
        "participation_effective_rank",
    ]
    by_dim = aggregate_by_seed(diag.rename(columns={"map_name": "map"}), ["map", "m"], summary_metrics)
    by_dim = _sort_map_dim(by_dim)

    table5_rows: List[Dict[str, object]] = []
    for map_name, group in by_dim.groupby("map", sort=False):
        m2 = group[group["m"] == 2]
        higher = group[group["m"] > 2]
        if m2.empty:
            continue
        best_geo = higher.loc[higher["q99_r_true_mean"].idxmin()] if not higher.empty else group.loc[group["q99_r_true_mean"].idxmin()]
        best_pred = group.loc[group["decoded_next_rmse_mean"].idxmin()]
        row = {
            "map": map_name,
            "current_state_accuracy_at_m2": float(m2.iloc[0]["exact_current_state_accuracy_mean"]),
            "q99_r_true_at_m2": float(m2.iloc[0]["q99_r_true_mean"]),
            "best_q99_r_true": float(best_geo["q99_r_true_mean"]),
            "best_dimension": int(best_geo["m"]),
            "geometric_benefit": float(m2.iloc[0]["q99_r_true_mean"] - best_geo["q99_r_true_mean"]),
            "decoded_next_error_at_m2": float(m2.iloc[0]["decoded_next_rmse_mean"]),
            "best_decoded_next_error": float(best_pred["decoded_next_rmse_mean"]),
            "best_prediction_dimension": int(best_pred["m"]),
        }
        for dim in [16, 64, 128]:
            selected = group[group["m"] == dim]
            row[f"q99_r_true_m2_minus_m{dim}"] = (
                float(m2.iloc[0]["q99_r_true_mean"] - selected.iloc[0]["q99_r_true_mean"]) if not selected.empty else float("nan")
            )
        table5_rows.append(row)
    table5 = _sort_map_dim(pd.DataFrame(table5_rows))
    table5 = table5.merge(
        complexity[
            [
                "map",
                "branch_excess",
                "junction_count",
                "dead_end_count",
                "graph_diameter",
                "mean_shortest_path",
            ]
        ],
        on="map",
        how="left",
    )

    merged = by_dim.merge(complexity, on="map", how="left")
    rng = np.random.default_rng(0)
    corr_rows: List[Dict[str, object]] = []
    for m, group in merged.groupby("m", sort=False):
        for complexity_col in ["branch_excess", "junction_count", "dead_end_count", "graph_diameter", "mean_shortest_path"]:
            stats = spearman_bootstrap(group, complexity_col, "q99_r_true_mean", rng, bootstrap_reps)
            corr_rows.append({"m": int(m), "complexity": complexity_col, **stats})
    benefit = table5.copy()
    for complexity_col in ["branch_excess", "junction_count", "dead_end_count", "graph_diameter", "mean_shortest_path"]:
        stats = spearman_bootstrap(benefit, complexity_col, "geometric_benefit", rng, bootstrap_reps)
        corr_rows.append({"m": "benefit", "complexity": complexity_col, **stats})
    correlations = pd.DataFrame(corr_rows)

    outputs = {
        "table1_map_properties": table1,
        "table2_state_sufficiency": table2,
        "table3_learned_geometry": table3,
        "table4_prediction": table4,
        "table5_map_level_summary": table5,
        "summary_by_map_dim": by_dim,
        "complexity_correlations": correlations,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)

    write_latex(table1, tables_dir / "table1_map_properties.tex", table1_cols, "Map properties.", "tab:maze-map-properties")
    write_latex(
        table2,
        tables_dir / "table2_state_sufficiency.tex",
        ["map", "m", "seed", "current_reconstruction_rmse", "exact_current_state_accuracy", "latent_collision_count", "min_pair_distance"],
        "Current-state sufficiency diagnostics.",
        "tab:maze-state-sufficiency",
    )
    write_latex(
        table3,
        tables_dir / "table3_learned_geometry.tex",
        ["map", "m", "seed", "q95_r_true", "q99_r_true", "max_r_true", "participation_effective_rank", "k95_variance", "k99_variance"],
        "Learned high-tail expansion burden and representation usage.",
        "tab:maze-learned-geometry",
    )
    write_latex(
        table4,
        tables_dir / "table4_prediction.tex",
        ["map", "m", "seed", "decoded_next_rmse", "nearest_state_next_accuracy", "mean_raw_latent_pred_l2", "per_coordinate_pred_mse", "q99_norm_pair_error_now"],
        "Prediction diagnostics.",
        "tab:maze-prediction",
    )
    write_latex(
        table5,
        tables_dir / "table5_map_level_summary.tex",
        list(table5.columns),
        "Map-level summary.",
        "tab:maze-map-summary",
    )
    return outputs


def _plot_line_with_ci(ax, df: pd.DataFrame, metric: str, label_col: str = "map") -> None:
    for map_name, group in df.groupby(label_col, sort=False):
        group = group.sort_values("m")
        y = group[f"{metric}_mean"].astype(float).to_numpy()
        lo = group[f"{metric}_ci95_low"].astype(float).to_numpy()
        hi = group[f"{metric}_ci95_high"].astype(float).to_numpy()
        x = group["m"].astype(int).to_numpy()
        ax.plot(x, y, marker="o", lw=1.4, ms=3.5, label=map_name)
        ax.fill_between(x, lo, hi, alpha=0.16, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIM_ORDER)
    ax.set_xticklabels([str(dim) for dim in DIM_ORDER])


def make_figures(tables: Dict[str, pd.DataFrame], out_dir: Path) -> List[Path]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()
    by_dim = tables["summary_by_map_dim"]
    table5 = tables["table5_map_level_summary"]
    correlations = tables["complexity_correlations"]
    paths: List[Path] = []

    fig, ax = plt.subplots(figsize=(5.4, 3.2), facecolor="white")
    _plot_line_with_ci(ax, by_dim, "exact_current_state_accuracy")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("exact current-state decoding accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        path = figures_dir / f"figure_a_state_sufficiency.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.4), facecolor="white")
    _plot_line_with_ci(ax, by_dim, "q99_r_true")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("learned high-tail expansion burden")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        path = figures_dir / f"figure_b_expansion_burden.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor="white")
    ax.scatter(table5["branch_excess"], table5["geometric_benefit"], s=24, color="#2f6f9f")
    for _, row in table5.iterrows():
        ax.annotate(str(row["map"]), (row["branch_excess"], row["geometric_benefit"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    benefit_corr = correlations[(correlations["m"].astype(str) == "benefit") & (correlations["complexity"] == "branch_excess")]
    if not benefit_corr.empty:
        rho = float(benefit_corr.iloc[0]["spearman"])
        ax.text(0.02, 0.96, f"Spearman rho={rho:.2f}", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel("branch excess")
    ax.set_ylabel("q99_r_true(m=2) - min_{m>2} q99_r_true")
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        path = figures_dir / f"figure_c_complexity_benefit.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.2), facecolor="white")
    focus = by_dim[by_dim["map"].isin(["path_0", "comb_12", "hierarchical_tree_8"])]
    _plot_line_with_ci(ax, focus, "decoded_next_rmse")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("decoded next-state RMSE")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        path = figures_dir / f"figure_d_decoded_next_prediction.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.8), facecolor="white")
    _plot_line_with_ci(axes[0, 0], by_dim, "q99_norm_pair_error_now")
    axes[0, 0].set_title("Original normalized pair error")
    _plot_line_with_ci(axes[0, 1], by_dim, "q99_norm_pair_error_exclude_bottom5_denom")
    axes[0, 1].set_title("Excluding bottom 5% denominators")
    _plot_line_with_ci(axes[0, 2], by_dim, "per_coordinate_pred_mse")
    axes[0, 2].set_title("Per-coordinate prediction MSE")
    _plot_line_with_ci(axes[1, 0], by_dim, "participation_effective_rank")
    axes[1, 0].set_title("Effective rank")
    _plot_line_with_ci(axes[1, 1], by_dim, "decoded_next_accuracy")
    axes[1, 1].set_title("Decoded next-state accuracy")
    _plot_line_with_ci(axes[1, 2], by_dim, "decoded_next_rmse")
    axes[1, 2].set_title("Decoded next-state RMSE")
    for ax in axes.flat:
        ax.set_xlabel("latent dimension")
    axes[0, 0].legend(frameon=False, fontsize=5, ncol=2)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        path = figures_dir / f"appendix_diagnostics.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication tables and figures for the maze predictive-geometry experiment.")
    parser.add_argument("--experiment-dir", default="results/maze_predictive_geometry_decay100k_v1")
    parser.add_argument("--diagnostics-csv", default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    diagnostics_csv = Path(args.diagnostics_csv) if args.diagnostics_csv else experiment_dir / "checkpoint_diagnostics.csv"
    complexity_csv = experiment_dir / "map_complexity.csv"
    if not diagnostics_csv.exists():
        raise FileNotFoundError(f"Missing diagnostics CSV: {diagnostics_csv}")
    if not complexity_csv.exists():
        raise FileNotFoundError(f"Missing map complexity CSV: {complexity_csv}")
    diag = pd.read_csv(diagnostics_csv)
    if "decoded_next_accuracy" not in diag.columns and "nearest_state_next_accuracy" in diag.columns:
        diag["decoded_next_accuracy"] = diag["nearest_state_next_accuracy"]
    complexity = pd.read_csv(complexity_csv)
    tables = build_tables(diag, complexity, experiment_dir, args.bootstrap_reps)
    paths = make_figures(tables, experiment_dir)
    print(f"wrote tables to {experiment_dir / 'tables'}")
    print("wrote figures:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
