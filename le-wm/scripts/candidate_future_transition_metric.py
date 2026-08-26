from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py
import numpy as np

EPS = 1e-12
MAIN_MODELS = ["state8", "state16", "state32", "state64", "baseline192", "state302", "state502"]
MODEL_DIMS = {"state8": 8, "state16": 16, "state32": 32, "state64": 64, "baseline192": 192, "state302": 302, "state502": 502}


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
    with path.open() as file:
        return list(csv.DictReader(file))


def _parse_float_list(text: str) -> List[float]:
    return [float(item) for item in str(text).replace(",", " ").split() if item.strip()]


def _read_h5_rows(dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique_rows])[inverse]


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


def _pairwise_rank_acc(true_cost: np.ndarray, score: np.ndarray) -> float:
    i, j = np.triu_indices(true_cost.shape[0], k=1)
    true_diff = true_cost[i] - true_cost[j]
    score_diff = score[i] - score[j]
    valid = (np.abs(true_diff) > EPS) & (np.abs(score_diff) > EPS)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sign(true_diff[valid]) == np.sign(score_diff[valid])))


def _pairwise_rank_count(true_cost: np.ndarray, score: np.ndarray) -> int:
    i, j = np.triu_indices(true_cost.shape[0], k=1)
    true_diff = true_cost[i] - true_cost[j]
    score_diff = score[i] - score[j]
    return int(np.sum((np.abs(true_diff) > EPS) & (np.abs(score_diff) > EPS)))


def _candidate_metric_alias(true_cost: np.ndarray, score: np.ndarray, q_true: float, q_score: float) -> Tuple[float, float, float]:
    i, j = np.triu_indices(true_cost.shape[0], k=1)
    true_gap = np.abs(true_cost[i] - true_cost[j])
    score_gap = np.abs(score[i] - score[j])
    true_thr = float(np.quantile(true_gap, q_true))
    score_thr = float(np.quantile(score_gap, q_score))
    true_distinct = true_gap >= true_thr
    score_alias = score_gap <= score_thr
    return (
        float(np.mean(score_alias[true_distinct])) if np.any(true_distinct) else float("nan"),
        true_thr,
        score_thr,
    )


def _nearest_landmarks(states: np.ndarray, X_landmark: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("scipy is required for nearest-landmark graph approximation.") from exc
    flat = states.reshape(-1, states.shape[-1]).astype(np.float64)
    dim = min(flat.shape[-1], mean.shape[-1], X_landmark.shape[-1])
    z = (flat[:, :dim] - mean.reshape(-1)[:dim]) / np.maximum(std.reshape(-1)[:dim], 1e-8)
    tree = cKDTree(X_landmark[:, :dim])
    _dist, idx = tree.query(z, k=1)
    return np.asarray(idx, dtype=np.int64).reshape(states.shape[:-1])


def _load_goal_states(raw: np.lib.npyio.NpzFile, dataset: str, state_key: str) -> np.ndarray:
    if "goal_states" in raw:
        return np.asarray(raw["goal_states"], dtype=np.float64)
    if "goal_rows" not in raw:
        raise KeyError("raw npz must contain goal_states or goal_rows.")
    if not dataset:
        raise ValueError("--dataset is required when raw npz contains goal_rows but not goal_states.")
    with h5py.File(dataset, "r") as h5:
        key = state_key if state_key in h5 else "state"
        return np.asarray(_read_h5_rows(h5[key], np.asarray(raw["goal_rows"], dtype=np.int64)), dtype=np.float64)


def _window_rows(
    model_name: str,
    window_idx: int,
    graph_cost: np.ndarray,
    latent_score: np.ndarray,
    q_far: float,
    q_near: float,
) -> List[Dict[str, object]]:
    graph_far_thr = float(np.quantile(graph_cost, q_far))
    score_near_thr = float(np.quantile(latent_score, q_near))
    graph_far = graph_cost >= graph_far_thr
    score_near = latent_score <= score_near_thr
    metric_alias, true_gap_thr, score_gap_thr = _candidate_metric_alias(graph_cost, latent_score, q_far, q_near)
    return [
        {
            "model": model_name,
            "window": window_idx,
            "metric": "candidate_goal_metric_spearman",
            "value": _spearman(latent_score, graph_cost),
        },
        {
            "model": model_name,
            "window": window_idx,
            "metric": "candidate_pairwise_rank_acc",
            "value": _pairwise_rank_acc(graph_cost, latent_score),
        },
        {
            "model": model_name,
            "window": window_idx,
            "metric": "candidate_false_shortcut_rate",
            "value": float(np.mean(score_near[graph_far])) if np.any(graph_far) else float("nan"),
        },
        {
            "model": model_name,
            "window": window_idx,
            "metric": "candidate_metric_alias_rate",
            "value": metric_alias,
        },
        {"model": model_name, "window": window_idx, "metric": "graph_far_threshold", "value": graph_far_thr},
        {"model": model_name, "window": window_idx, "metric": "latent_score_near_threshold", "value": score_near_thr},
        {"model": model_name, "window": window_idx, "metric": "pairwise_graph_gap_threshold", "value": true_gap_thr},
        {"model": model_name, "window": window_idx, "metric": "pairwise_score_gap_threshold", "value": score_gap_thr},
    ]


def _window_metrics(graph_cost: np.ndarray, latent_score: np.ndarray, q_far: float, q_near: float) -> Dict[str, object]:
    graph_far_thr = float(np.quantile(graph_cost, q_far))
    score_near_thr = float(np.quantile(latent_score, q_near))
    graph_far = graph_cost >= graph_far_thr
    score_near = latent_score <= score_near_thr
    metric_alias, true_gap_thr, score_gap_thr = _candidate_metric_alias(graph_cost, latent_score, q_far, q_near)
    return {
        "candidate_goal_metric_spearman": _spearman(latent_score, graph_cost),
        "candidate_pairwise_rank_acc": _pairwise_rank_acc(graph_cost, latent_score),
        "candidate_false_shortcut_rate": float(np.mean(score_near[graph_far])) if np.any(graph_far) else float("nan"),
        "candidate_metric_alias_rate": metric_alias,
        "graph_far_threshold": graph_far_thr,
        "latent_score_near_threshold": score_near_thr,
        "pairwise_graph_gap_threshold": true_gap_thr,
        "pairwise_score_gap_threshold": score_gap_thr,
        "num_candidates": int(graph_cost.shape[0]),
        "num_graph_far_events": int(np.sum(graph_far)),
        "num_false_shortcut_events": int(np.sum(score_near & graph_far)),
        "num_pairwise_rank_pairs": _pairwise_rank_count(graph_cost, latent_score),
    }


def _bootstrap(values: List[float], samples: int, seed: int) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "std": float("nan")}
    if samples <= 0 or arr.size == 1:
        mean = float(np.mean(arr))
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "std": float(np.std(arr))}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(samples, arr.size))
    boot = np.mean(arr[idx], axis=1)
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "std": float(np.std(arr)),
    }


def _save_summary(path: Path, model_name: str, rows: List[Dict[str, object]]) -> None:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["metric"]), []).append(float(row["value"]))
    summary = {"model": model_name}
    for metric, values in grouped.items():
        summary[metric] = float(np.nanmean(values))
    existing: List[Dict[str, object]] = []
    if path.exists():
        with path.open() as file:
            existing = [dict(row) for row in csv.DictReader(file)]
        existing = [row for row in existing if row.get("model") != model_name]
    existing.append(summary)
    _write_csv(path, existing)


def _upsert_rows(path: Path, new_rows: List[Dict[str, object]], keys: Tuple[str, ...]) -> List[Dict[str, object]]:
    existing = [dict(row) for row in _read_csv(path)]
    def key_of(row: Dict[str, object]) -> Tuple[str, ...]:
        return tuple(str(row.get(key, "")) for key in keys)
    new_keys = {key_of(row) for row in new_rows}
    out = [row for row in existing if key_of(row) not in new_keys]
    out.extend(new_rows)
    _write_csv(path, out)
    return out


def _save_main_ci(
    path: Path,
    model_name: str,
    latent_dim: int,
    per_window: List[Dict[str, object]],
    q_far: float,
    q_near: float,
    bootstrap_samples: int,
    seed: int,
) -> None:
    metrics = ["candidate_false_shortcut_rate", "candidate_goal_metric_spearman", "candidate_pairwise_rank_acc"]
    rows = []
    for metric in metrics:
        stats = _bootstrap([float(row[metric]) for row in per_window], bootstrap_samples, seed)
        rows.append(
            {
                "model": model_name,
                "latent_dim": latent_dim,
                "metric": metric,
                "q_far": q_far,
                "q_near": q_near,
                "num_windows": len(per_window),
                **stats,
            }
        )
    _upsert_rows(path, rows, ("model", "metric", "q_far", "q_near"))


def _refresh_baseline_relative(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    baseline = {}
    for row in rows:
        if str(row.get("model")) == "baseline192":
            baseline[(str(row.get("q_far")), str(row.get("q_near")))] = float(row.get("candidate_false_shortcut_rate", "nan"))
    for row in rows:
        q_near = float(row.get("q_near", "nan"))
        rate = float(row.get("candidate_false_shortcut_rate", "nan"))
        base = baseline.get((str(row.get("q_far")), str(row.get("q_near"))), float("nan"))
        row["random_independent_baseline"] = q_near
        row["relative_to_random"] = rate / q_near if q_near > 0 else float("nan")
        row["relative_increase_vs_baseline192"] = rate / base if base > 0 else float("nan")
    return rows


def _save_threshold_sweep(
    summary_path: Path,
    ci_path: Path,
    model_name: str,
    latent_dim: int,
    windows: List[Tuple[np.ndarray, np.ndarray]],
    q_far_values: List[float],
    q_near_values: List[float],
    bootstrap_samples: int,
    seed: int,
) -> None:
    summary_rows = []
    ci_rows = []
    for q_far in q_far_values:
        for q_near in q_near_values:
            metrics = [_window_metrics(graph_cost, score, q_far, q_near) for graph_cost, score in windows]
            values = [float(row["candidate_false_shortcut_rate"]) for row in metrics]
            rate = float(np.nanmean(values))
            stats = _bootstrap(values, bootstrap_samples, seed)
            base = {
                "model": model_name,
                "latent_dim": latent_dim,
                "q_far": q_far,
                "q_near": q_near,
                "candidate_false_shortcut_rate": rate,
                "random_independent_baseline": q_near,
                "relative_to_random": rate / q_near if q_near > 0 else float("nan"),
                "relative_increase_vs_baseline192": float("nan"),
                "num_windows": len(metrics),
                "num_candidates": int(np.mean([row["num_candidates"] for row in metrics])),
                "num_pairs_or_events": int(np.sum([row["num_graph_far_events"] for row in metrics])),
            }
            summary_rows.append(base)
            ci_rows.append({**base, **stats})
    all_summary = _refresh_baseline_relative(_upsert_rows(summary_path, summary_rows, ("model", "q_far", "q_near")))
    _write_csv(summary_path, all_summary)
    all_ci = _refresh_baseline_relative(_upsert_rows(ci_path, ci_rows, ("model", "q_far", "q_near")))
    _write_csv(ci_path, all_ci)


def _float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _latent_dim_for_row(row: Dict[str, str]) -> float:
    model = row.get("model", "")
    if model in MODEL_DIMS:
        return float(MODEL_DIMS[model])
    return _float(row, "latent_dim")


def _plot_candidate_ci(summary_ci_csv: Path, plot_dir: Path, main_q_far: float, main_q_near: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = [
        row for row in _read_csv(summary_ci_csv)
        if row.get("model") in MAIN_MODELS and abs(_float(row, "q_far") - main_q_far) < 1e-9 and abs(_float(row, "q_near") - main_q_near) < 1e-9
    ]
    if not rows:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename, ylabel, hline in [
        ("candidate_false_shortcut_rate", "candidate_false_shortcut_vs_dim.png", "Candidate false shortcut rate", main_q_near),
        ("candidate_goal_metric_spearman", "candidate_goal_spearman_vs_dim.png", "Spearman(graph distance, latent score)", None),
        ("candidate_pairwise_rank_acc", "candidate_pairwise_rank_acc_vs_dim.png", "Pairwise rank accuracy", 0.5),
    ]:
        selected = [row for row in rows if row.get("metric") == metric]
        selected.sort(key=_latent_dim_for_row)
        if not selected:
            continue
        xs = np.asarray([_float(row, "latent_dim") for row in selected], dtype=float)
        ys = np.asarray([_float(row, "mean") for row in selected], dtype=float)
        lows = np.asarray([_float(row, "ci95_low") for row in selected], dtype=float)
        highs = np.asarray([_float(row, "ci95_high") for row in selected], dtype=float)
        labels = [row["model"] for row in selected]
        plt.figure(figsize=(7.4, 4.8))
        plt.errorbar(xs, ys, yerr=np.vstack([ys - lows, highs - ys]), marker="o", linewidth=2, capsize=4)
        for x, y, label in zip(xs, ys, labels):
            plt.text(x, y, f" {label}", fontsize=8, va="center")
        if hline is not None:
            plt.axhline(hline, color="0.35", linestyle="--", linewidth=1, label=f"random={hline:g}")
        for d, color in [(96, "tab:orange"), (163, "tab:green"), (302, "tab:red")]:
            plt.axvline(d, color=color, linestyle=":", linewidth=1, label=f"d{d}")
        plt.xscale("log")
        plt.xlabel("latent dimension")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure candidate-future latent metric fidelity against graph/reachability distance.")
    parser.add_argument("--raw_npz", required=True)
    parser.add_argument("--graph_distance_cache", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--q_far", type=float, default=0.75)
    parser.add_argument("--q_near", type=float, default=0.05)
    parser.add_argument("--q_far_values", default="0.70,0.75,0.80")
    parser.add_argument("--q_near_values", default="0.01,0.05,0.10")
    parser.add_argument("--latent_dim", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--summary_ci_csv", default="")
    parser.add_argument("--threshold_sweep_csv", default="")
    parser.add_argument("--threshold_sweep_ci_csv", default="")
    parser.add_argument("--plot_dir", default="")
    args = parser.parse_args()

    raw = np.load(args.raw_npz)
    cache = np.load(args.graph_distance_cache)
    required = ["terminal_states", "terminal_latents", "goal_latents"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{args.raw_npz} missing keys: {missing}")
    terminal_states = np.asarray(raw["terminal_states"], dtype=np.float64)
    terminal_latents = np.asarray(raw["terminal_latents"], dtype=np.float64)
    goal_latents = np.asarray(raw["goal_latents"], dtype=np.float64)
    goal_states = _load_goal_states(raw, args.dataset, args.state_key)

    X_landmark = np.asarray(cache["X_landmark"], dtype=np.float64)
    D_graph = np.asarray(cache["D_landmark"], dtype=np.float64)
    mean = np.asarray(cache["feature_mean"], dtype=np.float64)
    std = np.asarray(cache["feature_std"], dtype=np.float64)
    terminal_lm = _nearest_landmarks(terminal_states, X_landmark, mean, std)
    goal_lm = _nearest_landmarks(goal_states, X_landmark, mean, std).reshape(-1)

    rows: List[Dict[str, object]] = []
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    per_window: List[Dict[str, object]] = []
    for window_idx in range(terminal_states.shape[0]):
        latent_score = np.sum((terminal_latents[window_idx] - goal_latents[window_idx][None, :]) ** 2, axis=-1)
        graph_cost = D_graph[terminal_lm[window_idx], goal_lm[window_idx]]
        windows.append((graph_cost, latent_score))
        per_window.append(_window_metrics(graph_cost, latent_score, args.q_far, args.q_near))
        rows.extend(_window_rows(args.model_name, window_idx, graph_cost, latent_score, args.q_far, args.q_near))

    latent_dim = args.latent_dim or MODEL_DIMS.get(args.model_name, int(terminal_latents.shape[-1]))
    _write_csv(Path(args.output_csv), rows)
    if args.summary_csv:
        _save_summary(Path(args.summary_csv), args.model_name, rows)
        summary_dir = Path(args.summary_csv).parent
    else:
        summary_dir = Path(args.output_csv).parent
    summary_ci_csv = Path(args.summary_ci_csv) if args.summary_ci_csv else summary_dir / "candidate_future_metric_summary_with_ci.csv"
    threshold_sweep_csv = Path(args.threshold_sweep_csv) if args.threshold_sweep_csv else summary_dir / "candidate_future_threshold_sweep.csv"
    threshold_sweep_ci_csv = Path(args.threshold_sweep_ci_csv) if args.threshold_sweep_ci_csv else summary_dir / "candidate_future_threshold_sweep_with_ci.csv"
    plot_dir = Path(args.plot_dir) if args.plot_dir else summary_dir.parent / "plots"
    _save_main_ci(
        summary_ci_csv,
        args.model_name,
        latent_dim,
        per_window,
        args.q_far,
        args.q_near,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    _save_threshold_sweep(
        threshold_sweep_csv,
        threshold_sweep_ci_csv,
        args.model_name,
        latent_dim,
        windows,
        _parse_float_list(args.q_far_values),
        _parse_float_list(args.q_near_values),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    if args.model_name != "global_k32":
        _plot_candidate_ci(summary_ci_csv, plot_dir, args.q_far, args.q_near)
    json_path = Path(args.output_csv).with_suffix(".json")
    with json_path.open("w") as file:
        json.dump({"model": args.model_name, "raw_npz": args.raw_npz, "graph_distance_cache": args.graph_distance_cache}, file, indent=2)
    print(f"[candidate_metric] wrote {args.output_csv}")
    print(f"[candidate_metric] wrote {summary_ci_csv}")
    print(f"[candidate_metric] wrote {threshold_sweep_csv}")
    print(f"[candidate_metric] wrote {threshold_sweep_ci_csv}")


if __name__ == "__main__":
    main()
