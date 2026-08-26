from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ["state8", "state16", "state32", "state64", "baseline192"]
DIM_ORDER = [8, 16, 32, 64, 192]
EPSILON = "1e-06"


CHECKPOINTS = {
    "state8": "/home/jw3425/.stable_worldmodel/pusht/state8_baseline_object.ckpt",
    "state16": "/home/jw3425/.stable_worldmodel/pusht/state16_baseline_object.ckpt",
    "state32": "/home/jw3425/.stable_worldmodel/pusht/state32_baseline_object.ckpt",
    "state64": "/home/jw3425/.stable_worldmodel/pusht/state64_baseline_object.ckpt",
    "baseline192": "/home/jw3425/.stable_worldmodel/pusht/baseline_object.ckpt",
}


def _float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _normalise_epsilon(value: str) -> str:
    return f"{float(value):.0e}" if value else value


def _load_geometry_summary(path: Path, epsilon: str) -> Tuple[List[Dict[str, object]], str]:
    if path.exists():
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        filtered = [row for row in rows if _normalise_epsilon(row.get("epsilon", "")) == _normalise_epsilon(epsilon)]
        by_model = {row["model"]: row for row in filtered}
        missing = [model for model in MODEL_ORDER if model not in by_model]
        if missing:
            raise RuntimeError(f"{path} is missing geometry rows for: {missing}")
        out = []
        for model in MODEL_ORDER:
            row = by_model[model]
            out.append(
                {
                    "model_name": model,
                    "latent_dim": int(float(row["latent_dim"])),
                    "q99_r_true": _float(row, "q99_r_true"),
                    "q99_r_model": _float(row, "q99_r_model"),
                    "q99_expansion_shortfall": _float(row, "q99_ratio_deficit"),
                    "q99_normalized_pair_error": _float(row, "q99_norm_pair_error_now"),
                    "geometry_source": str(path),
                }
            )
        return out, str(path)

    fallback_path = Path("scripts/plot_lewm_dimension_gain_diagnostics_v2.py")
    if not fallback_path.exists():
        raise FileNotFoundError(
            f"Missing {path}, and fallback source {fallback_path} does not exist. "
            "Provide outputs/lewm_gain/summary_normalized.csv."
        )
    tree = ast.parse(fallback_path.read_text())
    fallback_rows = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FALLBACK_ROWS":
                    fallback_rows = ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "FALLBACK_ROWS":
            fallback_rows = ast.literal_eval(node.value)
    if fallback_rows is None:
        raise RuntimeError(f"Could not recover FALLBACK_ROWS from {fallback_path}.")
    by_model = {row["model"]: row for row in fallback_rows}
    missing = [model for model in MODEL_ORDER if model not in by_model]
    if missing:
        raise RuntimeError(f"{fallback_path} fallback rows are missing: {missing}")
    out = []
    for model in MODEL_ORDER:
        row = by_model[model]
        out.append(
            {
                "model_name": model,
                "latent_dim": int(row["latent_dim"]),
                "q99_r_true": float(row["q99_r_true"]),
                "q99_r_model": float(row["q99_r_model"]),
                "q99_expansion_shortfall": float(row["q99_ratio_deficit"]),
                "q99_normalized_pair_error": float(row["q99_norm_pair_error_now"]),
                "geometry_source": str(fallback_path) + "::FALLBACK_ROWS",
            }
        )
    return out, str(fallback_path) + "::FALLBACK_ROWS"


def _load_success(path: Path) -> Tuple[Dict[str, Dict[str, object]], str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PushT success table: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    by_model = {row["model"]: row for row in rows if row.get("model") in MODEL_ORDER}
    missing = [model for model in MODEL_ORDER if model not in by_model]
    if missing:
        raise RuntimeError(f"{path} is missing success rows for: {missing}")
    out = {}
    for model in MODEL_ORDER:
        row = by_model[model]
        if not row.get("SR_100"):
            raise RuntimeError(f"{path} row for {model} has no SR_100 value.")
        out[model] = {
            "success_rate": float(row["SR_100"]) / 100.0,
            "success_rate_percent": float(row["SR_100"]),
            "success_metric_key": "SR_100",
            "success_source": str(path),
        }
    return out, str(path)


def _merge_rows(geometry_rows: List[Dict[str, object]], success_rows: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for row in geometry_rows:
        model = str(row["model_name"])
        payload = dict(row)
        payload.update(success_rows[model])
        payload["checkpoint"] = CHECKPOINTS.get(model, "")
        payload["success_num_episodes"] = 50
        payload["success_episode_count_source"] = "config/eval/pusht.yaml::eval.num_eval"
        rows.append(payload)
    dims = [int(row["latent_dim"]) for row in rows]
    if dims != DIM_ORDER:
        raise RuntimeError(f"Expected latent dims {DIM_ORDER}, got {dims}")
    return rows


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "latent_dim",
        "model_name",
        "q99_r_true",
        "q99_r_model",
        "q99_expansion_shortfall",
        "q99_normalized_pair_error",
        "success_rate",
        "success_rate_percent",
        "success_num_episodes",
        "success_metric_key",
        "checkpoint",
        "geometry_source",
        "success_source",
        "success_episode_count_source",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_provenance(
    rows: List[Dict[str, object]],
    path: Path,
    geometry_source: str,
    success_source: str,
    env_name: str,
) -> None:
    lines = [
        f"# {env_name} LeWorldModel main figure provenance",
        "",
        "## Plotted sources",
        f"- Geometry diagnostics: `{geometry_source}`.",
        f"- Task success values: `{success_source}`, column `SR_100`.",
        "- Success episode count: `config/eval/pusht.yaml`, `eval.num_eval=50`; the success table stores the percentage value, not per-episode records.",
        "- PushT success/cost code checked against `scripts/task_cost.py`: block pose only, using block xy and wrapped block angle; pusher position and velocity are ignored.",
        "- Geometry run protocol from saved transition-gain script/results: 2000 transitions, 50000 pairs, epsilon 1e-6, approximate matched 5-step action blocks with normalized tolerance 1.5, shared pair IDs across dimensions, eval mode, no retraining, no triangle-bound violations.",
        "",
        "## Checkpoint mapping",
    ]
    for row in rows:
        lines.append(f"- {row['model_name']} ({row['latent_dim']}D): `{row['checkpoint']}`")
    lines += [
        "",
        "## Exact plotted rows",
        "",
        "| model | dim | q99 r_true | q99 shortfall | q99 norm error | success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['latent_dim']} | "
            f"{row['q99_r_true']:.12g} | {row['q99_expansion_shortfall']:.12g} | "
            f"{row['q99_normalized_pair_error']:.12g} | {row['success_rate_percent']:.3g}% |"
        )
    lines += [
        "",
        "## Caption",
        (
            "Independently trained PushT LeWorldModel latent-width configurations. "
            "Panel (a) reports the 99th percentile of the encoded true transition expansion; "
            "panel (b) reports the 99th percentile of the realised expansion shortfall; "
            "panel (c) reports the 99th percentile scale-normalised pair prediction error; "
            "panel (d) reports PushT task success. Geometry diagnostics use matched continuous "
            "5-step action blocks with tolerance 1.5 and the same sampled transition and pair IDs "
            "across dimensions. These are sampled high-tail diagnostics rather than estimates of "
            "the global supremum. The latent-width sweep also changes predictor/projector width "
            "through `wm.embed_dim`, so the figure should not be read as isolating nominal dimension alone."
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def _style_axes(ax: plt.Axes, panel: str, title: str) -> None:
    ax.text(-0.16, 1.08, panel, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.set_title(title, loc="left", pad=7)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.8)
    ax.set_xticks(range(len(DIM_ORDER)))
    ax.set_xticklabels([str(dim) for dim in DIM_ORDER])
    ax.set_xlabel("Latent dimension")


def _plot(rows: List[Dict[str, object]], output_prefix: Path, env_name: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
        }
    )
    x = list(range(len(rows)))
    colour = "#2c6fbb"
    accent = "#b44d2a"
    success_colour = "#2b8c62"
    line_kwargs = dict(linewidth=2.15, markersize=6.0, marker="o", markeredgecolor="white", markeredgewidth=0.8)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.7), constrained_layout=True)
    ax = axes[0, 0]
    _style_axes(ax, "(a)", "True transition expansion")
    ax.plot(x, [row["q99_r_true"] for row in rows], color=colour, **line_kwargs)
    ax.axhline(1.0, color="#737373", linestyle=(0, (2, 2)), linewidth=1.0)
    ax.text(0.02, 0.08, "non-expansive", transform=ax.transAxes, color="#666666", fontsize=8.5)
    ax.set_ylabel(r"$Q_{0.99}(r^{\mathrm{true}})$")
    ax.set_ylim(0.98, max(row["q99_r_true"] for row in rows) * 1.04)

    ax = axes[0, 1]
    _style_axes(ax, "(b)", "Realised expansion shortfall")
    ax.plot(x, [row["q99_expansion_shortfall"] for row in rows], color=accent, **line_kwargs)
    ax.set_ylabel(r"$Q_{0.99}([r^{\mathrm{true}}-r^{\mathrm{model}}]_+)$")
    ax.set_ylim(0, max(row["q99_expansion_shortfall"] for row in rows) * 1.18)

    ax = axes[1, 0]
    _style_axes(ax, "(c)", "Normalised prediction error")
    ax.plot(x, [row["q99_normalized_pair_error"] for row in rows], color="#5a4fa3", **line_kwargs)
    ax.set_ylabel(r"$Q_{0.99}(\varepsilon_{\mathrm{pair}}/d_{\mathrm{now}})$")
    ax.set_ylim(0, max(row["q99_normalized_pair_error"] for row in rows) * 1.18)

    ax = axes[1, 1]
    _style_axes(ax, "(d)", f"{env_name} task success")
    ax.plot(x, [row["success_rate_percent"] for row in rows], color=success_colour, **line_kwargs)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make the final PushT LeWM main-result figure.")
    parser.add_argument("--geometry_summary", default="outputs/lewm_gain/summary_normalized.csv")
    parser.add_argument(
        "--success_table",
        default="rollout_results/plannable_dim_evidence/summaries/model_evidence_table.csv",
    )
    parser.add_argument("--epsilon", default=EPSILON)
    parser.add_argument("--output_prefix", default="figures/lewm_pusht_main_v3")
    parser.add_argument("--env_name", default="PushT")
    args = parser.parse_args()

    geometry_rows, geometry_source = _load_geometry_summary(Path(args.geometry_summary), args.epsilon)
    success_rows, success_source = _load_success(Path(args.success_table))
    rows = _merge_rows(geometry_rows, success_rows)

    output_prefix = Path(args.output_prefix)
    _plot(rows, output_prefix, args.env_name)
    _write_csv(rows, output_prefix.with_name(output_prefix.name + "_data.csv"))
    _write_provenance(
        rows,
        output_prefix.with_name(output_prefix.name + "_provenance.md"),
        geometry_source,
        success_source,
        args.env_name,
    )

    print(f"[plot_lewm_main_v3] wrote {output_prefix.with_suffix('.pdf')}")
    print(f"[plot_lewm_main_v3] wrote {output_prefix.with_suffix('.png')}")
    print(f"[plot_lewm_main_v3] wrote {output_prefix.with_name(output_prefix.name + '_data.csv')}")
    print(f"[plot_lewm_main_v3] wrote {output_prefix.with_name(output_prefix.name + '_provenance.md')}")


if __name__ == "__main__":
    main()
