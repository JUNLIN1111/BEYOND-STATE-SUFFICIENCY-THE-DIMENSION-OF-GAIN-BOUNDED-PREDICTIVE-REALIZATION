from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def _read_rows(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with Path(path).open() as file:
            rows.extend(csv.DictReader(file))
    return rows


def _read_and_maybe_aggregate_rows(paths: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with Path(path).open() as file:
            file_rows = list(csv.DictReader(file))
        if file_rows and "K_gamma_mean" not in file_rows[0]:
            file_rows = _aggregate_subset_rows(file_rows)
        rows.extend(file_rows)
    return rows


def _float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _aggregate_subset_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    group_keys = ["model", "effective_dim", "N", "gamma", "subset_mode"]
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)

    aggregate_rows: List[Dict[str, str]] = []
    for key_values, group in grouped.items():
        out = {key: value for key, value in zip(group_keys, key_values)}
        metric_keys = sorted(set().union(*(row.keys() for row in group)) - set(group_keys))
        for metric_key in metric_keys:
            values = [_float(row, metric_key) for row in group]
            values = [value for value in values if math.isfinite(value)]
            if values:
                out[f"{metric_key}_mean"] = str(sum(values) / len(values))
        aggregate_rows.append(out)
    return aggregate_rows


def _close(value: float, target: float, tol: float = 1e-8) -> bool:
    return math.isfinite(value) and abs(value - target) <= tol


def _find_rho_key(rows: Sequence[Dict[str, str]], prefix: str, rho: float) -> str:
    desired = f"{prefix}_{rho:g}_mean"
    if rows and desired in rows[0]:
        return desired
    candidates = sorted({key for row in rows for key in row if key.startswith(f"{prefix}_") and key.endswith("_mean")})
    best_key = ""
    best_dist = float("inf")
    for key in candidates:
        raw = key.removeprefix(f"{prefix}_").removesuffix("_mean")
        try:
            value = float(raw)
        except ValueError:
            continue
        dist = abs(value - rho)
        if dist < best_dist:
            best_key = key
            best_dist = dist
    if not best_key:
        raise KeyError(f"Could not find a column like {prefix}_<rho>_mean.")
    return best_key


def _filter_rows(rows: Sequence[Dict[str, str]], gamma: float) -> List[Dict[str, str]]:
    filtered = [row for row in rows if _close(_float(row, "gamma"), gamma, tol=1e-7)]
    if not filtered:
        raise ValueError(f"No rows found for gamma={gamma}.")
    return filtered


def _model_order(model: str) -> Tuple[int, str]:
    order = {
        "state8": 0,
        "state16": 1,
        "state32": 2,
        "state64": 3,
        "baseline192": 4,
        "global_k32": 5,
        "local_k32": 6,
    }
    return order.get(model, 100), model


def _group_by_model(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        model = row.get("model", "unknown")
        groups.setdefault(model, []).append(row)
    return groups


def _plot_lines(
    rows: Sequence[Dict[str, str]],
    y_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8.5, 5.2))
    for model, group in sorted(_group_by_model(rows).items(), key=lambda item: _model_order(item[0])):
        points = sorted((_float(row, "K_gamma_mean"), _float(row, y_key)) for row in group)
        points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
        if not points:
            continue
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", linewidth=2, label=model)
    plt.title(title)
    plt.xlabel("K_gamma mean")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _select_scatter_rows(rows: Sequence[Dict[str, str]], scatter_n: Optional[int]) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for model, group in _group_by_model(rows).items():
        if scatter_n is not None:
            candidates = [row for row in group if int(round(_float(row, "N"))) == scatter_n]
        else:
            max_n = max(_float(row, "N") for row in group)
            candidates = [row for row in group if _close(_float(row, "N"), max_n, tol=1e-7)]
        if not candidates:
            continue
        selected.append(sorted(candidates, key=lambda row: _float(row, "K_gamma_mean"))[-1])
    return selected


def _parse_fixed_json_specs(specs: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected fixed JSON spec as model=path, got: {spec}")
        model, path = spec.split("=", 1)
        out[model.strip()] = Path(path)
    return out


def _parse_rename_specs(specs: Sequence[str]) -> Dict[str, str]:
    renames: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected rename spec as old=new, got: {spec}")
        old, new = spec.split("=", 1)
        renames[old.strip()] = new.strip()
    return renames


def _rename_models(rows: Sequence[Dict[str, str]], specs: Sequence[str]) -> List[Dict[str, str]]:
    renames = _parse_rename_specs(specs)
    if not renames:
        return list(rows)
    out: List[Dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        model = copied.get("model", "")
        if model in renames:
            copied["model"] = renames[model]
        out.append(copied)
    return out


def _fixed_json_scatter_rows(specs: Sequence[str], gamma: float, rho: float) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model, path in _parse_fixed_json_specs(specs).items():
        with path.open() as file:
            data = json.load(file)
        aggregate = data["aggregate"]
        gamma_key = str(gamma)
        alias_key = f"gamma={gamma},rho={rho}"
        config = data.get("config", {})
        rows.append(
            {
                "model": model,
                "effective_dim": str(data.get("model", {}).get("latent_dim", "")),
                "N": str(config.get("num_candidates", "")),
                "gamma": gamma_key,
                "K_gamma_mean": str(aggregate["K_gamma"][gamma_key]["mean"]),
                f"norm_geom_alias_rho_{rho:g}_mean": str(aggregate["alias_norm"][alias_key]["mean"]),
                "spearman_mean": str(aggregate["auxiliary"]["mean_spearman"]),
            }
        )
    return rows


def _override_rows_by_model(base_rows: Sequence[Dict[str, str]], override_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    override_models = {row.get("model", "") for row in override_rows}
    return [row for row in base_rows if row.get("model", "") not in override_models] + list(override_rows)


def _parse_success_rates(value: str) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    if not value:
        return rates
    for item in value.split(","):
        if not item.strip():
            continue
        name, raw = item.split("=", 1)
        rates[name.strip()] = float(raw)
    return rates


def _plot_scatter(
    rows: Sequence[Dict[str, str]],
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y_values_override: Optional[Dict[str, float]] = None,
) -> None:
    plt.figure(figsize=(6.5, 5.2))
    for row in rows:
        model = row.get("model", "unknown")
        x = _float(row, x_key)
        y = y_values_override[model] if y_values_override and model in y_values_override else _float(row, y_key)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        plt.scatter([x], [y], s=70)
        plt.annotate(model, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trained-model Vary-K aliasing summaries.")
    parser.add_argument("--aggregate_csvs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="rollout_results/trained_varyK_plots")
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--scatter_N", type=int, default=1024)
    parser.add_argument(
        "--fixed_jsons",
        nargs="*",
        default=[],
        help="Optional fixed-N JSON overrides for scatter points, as model=path.",
    )
    parser.add_argument(
        "--rename_models",
        nargs="*",
        default=[],
        help="Optional model label rewrites for CSV rows, as old=new.",
    )
    parser.add_argument(
        "--success_rates",
        default="",
        help="Optional comma list, e.g. baseline192=0.96,state8=0.42,global_k32=0.08,local_k32=0.94",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _filter_rows(_rename_models(_read_and_maybe_aggregate_rows(args.aggregate_csvs), args.rename_models), args.gamma)
    alias_key = _find_rho_key(rows, "norm_geom_alias_rho", args.rho)
    scatter_rows = _select_scatter_rows(rows, args.scatter_N)
    if args.fixed_jsons:
        scatter_rows = _override_rows_by_model(
            scatter_rows,
            _fixed_json_scatter_rows(args.fixed_jsons, args.gamma, args.rho),
        )
    success_rates = _parse_success_rates(args.success_rates)

    _plot_lines(
        rows,
        alias_key,
        f"normalized geometric aliasing, rho={args.rho:g}",
        f"Trained models: aliasing pressure, gamma={args.gamma:g}",
        output_dir / "trained_varyK_norm_alias_vs_Kgamma.png",
    )
    _plot_lines(
        rows,
        "pairwise_rank_acc_mean",
        "pairwise rank accuracy",
        f"Trained models: ranking vs K_gamma, gamma={args.gamma:g}",
        output_dir / "trained_varyK_pairwise_rank_acc_vs_Kgamma.png",
    )
    _plot_lines(
        rows,
        "spearman_mean",
        "Spearman",
        f"Trained models: Spearman vs K_gamma, gamma={args.gamma:g}",
        output_dir / "trained_varyK_spearman_vs_Kgamma.png",
    )
    _plot_scatter(
        scatter_rows,
        alias_key,
        "spearman_mean",
        f"normalized aliasing, rho={args.rho:g}",
        "Spearman",
        f"Aliasing vs Spearman, gamma={args.gamma:g}, N={args.scatter_N}",
        output_dir / "trained_aliasing_vs_spearman.png",
    )
    if success_rates:
        _plot_scatter(
            scatter_rows,
            "spearman_mean",
            "",
            "Spearman",
            "eval success rate",
            f"Spearman vs eval SR, gamma={args.gamma:g}, N={args.scatter_N}",
            output_dir / "trained_spearman_vs_eval_sr.png",
            y_values_override=success_rates,
        )
    else:
        print("[plot] skipped Spearman-vs-SR scatter; pass --success_rates to enable it.")

    print(f"[plot] wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
