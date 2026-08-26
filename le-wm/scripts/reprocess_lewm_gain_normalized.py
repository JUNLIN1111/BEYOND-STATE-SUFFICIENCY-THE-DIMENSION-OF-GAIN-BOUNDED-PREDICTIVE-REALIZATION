from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_EPSILONS = (1e-6, 1e-4, 1e-2)


def _parse_float_list(value: str | None, default: Sequence[float]) -> List[float]:
    if value is None or str(value).strip() == "":
        return list(default)
    return [float(item) for item in str(value).replace(",", " ").split()]


def _read_pair_metrics(path: Path):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _quantile(values, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _pearson(x, y) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(x, y) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return float("nan")
    return _pearson(_rankdata(x_arr), _rankdata(y_arr))


def _column_for(frame, base: str, epsilon: float) -> str | None:
    candidates = [
        f"{base}_eps_{epsilon:g}",
        f"{base}_eps_{epsilon}",
    ]
    if math.isclose(epsilon, float(frame["epsilon"].iloc[0]), rel_tol=0.0, abs_tol=1e-15) and base in frame.columns:
        candidates.insert(0, base)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _model_sort_key(row: Dict[str, object]) -> tuple:
    try:
        dim = int(float(row["latent_dim"]))
    except Exception:  # noqa: BLE001
        dim = 10**9
    return dim, str(row.get("model", ""))


def _summarize(frame, epsilons: Sequence[float]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    required = {"model", "latent_dimension", "checkpoint", "d_now", "d_next_true", "d_next_pred", "deficit", "pair_error"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pair metrics missing required columns: {missing}")

    for (model, latent_dim), group in frame.groupby(["model", "latent_dimension"], sort=False):
        for epsilon in epsilons:
            r_true_col = _column_for(group, "r_true", epsilon)
            r_model_col = _column_for(group, "r_model", epsilon)
            if r_true_col is None:
                r_true = group["d_next_true"].to_numpy(dtype=np.float64) / np.maximum(
                    group["d_now"].to_numpy(dtype=np.float64),
                    float(epsilon),
                )
            else:
                r_true = group[r_true_col].to_numpy(dtype=np.float64)
            if r_model_col is None:
                r_model = group["d_next_pred"].to_numpy(dtype=np.float64) / np.maximum(
                    group["d_now"].to_numpy(dtype=np.float64),
                    float(epsilon),
                )
            else:
                r_model = group[r_model_col].to_numpy(dtype=np.float64)

            d_now = group["d_now"].to_numpy(dtype=np.float64)
            d_next_true = group["d_next_true"].to_numpy(dtype=np.float64)
            deficit = group["deficit"].to_numpy(dtype=np.float64)
            pair_error = group["pair_error"].to_numpy(dtype=np.float64)

            ratio_deficit = np.maximum(r_true - r_model, 0.0)
            norm_pair_error_now = pair_error / np.maximum(d_now, float(epsilon))
            norm_pair_error_next = pair_error / np.maximum(d_next_true, float(epsilon))
            relative_deficit_next = deficit / np.maximum(d_next_true, float(epsilon))

            rows.append(
                {
                    "model": model,
                    "latent_dim": int(latent_dim),
                    "checkpoint": str(group["checkpoint"].iloc[0]),
                    "epsilon": float(epsilon),
                    "num_pairs": int(len(group)),
                    "q95_ratio_deficit": _quantile(ratio_deficit, 0.95),
                    "q99_ratio_deficit": _quantile(ratio_deficit, 0.99),
                    "q999_ratio_deficit": _quantile(ratio_deficit, 0.999),
                    "median_norm_pair_error_now": _quantile(norm_pair_error_now, 0.5),
                    "q95_norm_pair_error_now": _quantile(norm_pair_error_now, 0.95),
                    "q99_norm_pair_error_now": _quantile(norm_pair_error_now, 0.99),
                    "median_norm_pair_error_next": _quantile(norm_pair_error_next, 0.5),
                    "q95_norm_pair_error_next": _quantile(norm_pair_error_next, 0.95),
                    "q99_norm_pair_error_next": _quantile(norm_pair_error_next, 0.99),
                    "median_relative_deficit_next": _quantile(relative_deficit_next, 0.5),
                    "q95_relative_deficit_next": _quantile(relative_deficit_next, 0.95),
                    "q99_relative_deficit_next": _quantile(relative_deficit_next, 0.99),
                    "median_d_now": _quantile(d_now, 0.5),
                    "q99_d_now": _quantile(d_now, 0.99),
                    "median_d_next_true": _quantile(d_next_true, 0.5),
                    "q99_d_next_true": _quantile(d_next_true, 0.99),
                    "q99_r_true": _quantile(r_true, 0.99),
                    "q99_r_model": _quantile(r_model, 0.99),
                    "spearman_corr_ratio_deficit_norm_error_now": _spearman(ratio_deficit, norm_pair_error_now),
                    "pearson_corr_ratio_deficit_norm_error_now": _pearson(ratio_deficit, norm_pair_error_now),
                }
            )
    return sorted(rows, key=_model_sort_key)


def _plot_gain_ratios(path: Path, rows: List[Dict[str, object]], primary_epsilon: float) -> None:
    import matplotlib.pyplot as plt

    data = [row for row in rows if math.isclose(float(row["epsilon"]), primary_epsilon, rel_tol=0.0, abs_tol=1e-15)]
    if not data:
        return
    x = np.asarray([float(row["latent_dim"]) for row in data])
    y_true = np.asarray([float(row["q99_r_true"]) for row in data])
    y_model = np.asarray([float(row["q99_r_model"]) for row in data])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(x, y_true, marker="o", label="q99 r_true")
    ax.plot(x, y_model, marker="s", label="q99 r_model")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("q99 transition gain")
    ax.set_title(f"Empirical vs model transition gain, epsilon={primary_epsilon:g}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)


def _plot_normalized_deficit_error(path: Path, rows: List[Dict[str, object]], frame, primary_epsilon: float) -> None:
    import matplotlib.pyplot as plt

    data = [row for row in rows if math.isclose(float(row["epsilon"]), primary_epsilon, rel_tol=0.0, abs_tol=1e-15)]
    if not data:
        return
    x = np.asarray([float(row["latent_dim"]) for row in data])
    q99_ratio_deficit = np.asarray([float(row["q99_ratio_deficit"]) for row in data])
    q99_norm_error = np.asarray([float(row["q99_norm_pair_error_now"]) for row in data])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    axes[0].plot(x, q99_ratio_deficit, marker="o", color="tab:red")
    axes[0].set_xlabel("latent dimension")
    axes[0].set_ylabel("q99 ratio_deficit")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, q99_norm_error, marker="o", color="tab:purple")
    axes[1].set_xlabel("latent dimension")
    axes[1].set_ylabel("q99 pair_error / max(d_now, eps)")
    axes[1].grid(alpha=0.25)

    rng = np.random.default_rng(0)
    plot_frame = frame
    if len(plot_frame) > 30000:
        plot_frame = plot_frame.iloc[rng.choice(len(plot_frame), size=30000, replace=False)]
    for (model, latent_dim), group in plot_frame.groupby(["model", "latent_dimension"], sort=False):
        r_true_col = _column_for(group, "r_true", primary_epsilon)
        r_model_col = _column_for(group, "r_model", primary_epsilon)
        if r_true_col is None:
            r_true = group["d_next_true"].to_numpy(dtype=np.float64) / np.maximum(group["d_now"].to_numpy(dtype=np.float64), primary_epsilon)
        else:
            r_true = group[r_true_col].to_numpy(dtype=np.float64)
        if r_model_col is None:
            r_model = group["d_next_pred"].to_numpy(dtype=np.float64) / np.maximum(group["d_now"].to_numpy(dtype=np.float64), primary_epsilon)
        else:
            r_model = group[r_model_col].to_numpy(dtype=np.float64)
        ratio_deficit = np.maximum(r_true - r_model, 0.0)
        norm_error = group["pair_error"].to_numpy(dtype=np.float64) / np.maximum(
            group["d_now"].to_numpy(dtype=np.float64),
            primary_epsilon,
        )
        axes[2].scatter(ratio_deficit / 2.0, norm_error, s=6, alpha=0.18, linewidths=0, label=f"{int(latent_dim)}D")
    limit = axes[2].get_xlim()[1]
    axes[2].plot([0, limit], [0, limit], color="black", lw=1, alpha=0.6)
    axes[2].set_xlabel("ratio_deficit / 2")
    axes[2].set_ylabel("normalized pair_error")
    axes[2].grid(alpha=0.25)
    axes[2].legend(markerscale=2, fontsize=7)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocess LeWorldModel transition-gain outputs into scale-free diagnostics.")
    parser.add_argument("--pair_metrics", default="outputs/lewm_gain/pair_metrics.parquet")
    parser.add_argument("--output_dir", default="outputs/lewm_gain")
    parser.add_argument("--epsilon", "--epsilons", dest="epsilons", default="1e-6,1e-4,1e-2")
    args = parser.parse_args()

    pair_path = Path(args.pair_metrics)
    output_dir = Path(args.output_dir)
    epsilons = _parse_float_list(args.epsilons, DEFAULT_EPSILONS)
    frame = _read_pair_metrics(pair_path)
    rows = _summarize(frame, epsilons)

    summary_path = output_dir / "summary_normalized.csv"
    _write_csv(summary_path, rows)
    primary_epsilon = epsilons[0]
    _plot_gain_ratios(output_dir / "lewm_gain_ratios", rows, primary_epsilon)
    _plot_normalized_deficit_error(output_dir / "lewm_normalized_deficit_error", rows, frame, primary_epsilon)

    print(f"[lewm_gain_norm] read {pair_path}", flush=True)
    print(f"[lewm_gain_norm] wrote {summary_path}", flush=True)
    print(f"[lewm_gain_norm] wrote {output_dir / 'lewm_gain_ratios.pdf'}", flush=True)
    print(f"[lewm_gain_norm] wrote {output_dir / 'lewm_normalized_deficit_error.pdf'}", flush=True)


if __name__ == "__main__":
    main()
