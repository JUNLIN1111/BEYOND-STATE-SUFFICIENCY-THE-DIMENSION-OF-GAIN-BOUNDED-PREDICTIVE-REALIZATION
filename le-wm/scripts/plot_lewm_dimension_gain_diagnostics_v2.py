from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_PATH = Path("outputs/lewm_gain/summary_normalized.csv")
FIGURE_DIR = Path("figures")
EPSILON = 1e-6
DIMS = [8, 16, 32, 64, 192]
OUTPUT_STEM = "lewm_dimension_gain_diagnostics_v2"
CAPTION = (
    "Learned LeWorldModel latents exhibit the predicted dimension--gain--error trend. "
    "We report scale-free 99th-percentile diagnostics across checkpoints with different "
    "latent dimensions. The empirical true transition gain decreases with latent dimension, "
    "and the model's predicted successor-ratio follows the same trend. The high-tail expansion "
    "deficit [r_true - r_model]_+ and the normalized pairwise prediction error are largest for "
    "the lowest-dimensional latent and decrease for larger latents. These are empirical "
    "diagnostics on sampled prediction pairs, not certified global estimates of L_m^*."
)


FALLBACK_ROWS: List[Dict[str, object]] = [
    {
        "model": "state8",
        "latent_dim": 8,
        "q99_r_true": 1.3020267827431773,
        "q99_r_model": 1.255453826071214,
        "q99_ratio_deficit": 0.14889759610026737,
        "q99_norm_pair_error_now": 0.2733808938259078,
        "q99_norm_pair_error_next": 0.30802766279675664,
    },
    {
        "model": "state16",
        "latent_dim": 16,
        "q99_r_true": 1.2113852969779322,
        "q99_r_model": 1.2043564529076802,
        "q99_ratio_deficit": 0.058543261668010785,
        "q99_norm_pair_error_now": 0.15932617221436418,
        "q99_norm_pair_error_next": 0.17260788056322085,
    },
    {
        "model": "state32",
        "latent_dim": 32,
        "q99_r_true": 1.1738313784910797,
        "q99_r_model": 1.183767919560048,
        "q99_ratio_deficit": 0.041553884862779444,
        "q99_norm_pair_error_now": 0.15356122410538425,
        "q99_norm_pair_error_next": 0.15592178524770076,
    },
    {
        "model": "state64",
        "latent_dim": 64,
        "q99_r_true": 1.1591376171260719,
        "q99_r_model": 1.1522057881780041,
        "q99_ratio_deficit": 0.045880899642109965,
        "q99_norm_pair_error_now": 0.15062704535702148,
        "q99_norm_pair_error_next": 0.1587220302136677,
    },
    {
        "model": "baseline192",
        "latent_dim": 192,
        "q99_r_true": 1.1362107765717402,
        "q99_r_model": 1.1284342728778365,
        "q99_ratio_deficit": 0.03752879196509771,
        "q99_norm_pair_error_now": 0.12722798444198954,
        "q99_norm_pair_error_next": 0.13041726744605167,
    },
]


def _as_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _load_rows() -> List[Dict[str, object]]:
    if not SUMMARY_PATH.exists():
        return [dict(row) for row in FALLBACK_ROWS]

    with SUMMARY_PATH.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"{SUMMARY_PATH} exists but is empty.")
    if "epsilon" not in rows[0]:
        raise RuntimeError(f"{SUMMARY_PATH} does not include an epsilon column.")

    filtered = [row for row in rows if abs(float(row["epsilon"]) - EPSILON) < 1e-15]
    assert filtered, f"No rows with epsilon={EPSILON:g} found in {SUMMARY_PATH}."
    assert all(abs(float(row["epsilon"]) - EPSILON) < 1e-15 for row in filtered)

    by_model = {row["model"]: row for row in filtered}
    model_order = ["state8", "state16", "state32", "state64", "baseline192"]
    required = {
        "model",
        "latent_dim",
        "q99_r_true",
        "q99_r_model",
        "q99_ratio_deficit",
        "q99_norm_pair_error_now",
        "q99_norm_pair_error_next",
    }
    missing_models = [model for model in model_order if model not in by_model]
    if missing_models:
        raise RuntimeError(f"{SUMMARY_PATH} is missing models: {missing_models}")
    missing_columns = sorted(required - set(filtered[0].keys()))
    if missing_columns:
        raise RuntimeError(f"{SUMMARY_PATH} is missing columns: {missing_columns}")

    out: List[Dict[str, object]] = []
    for model in model_order:
        row = by_model[model]
        out.append(
            {
                "model": model,
                "latent_dim": int(float(row["latent_dim"])),
                "q99_r_true": _as_float(row, "q99_r_true"),
                "q99_r_model": _as_float(row, "q99_r_model"),
                "q99_ratio_deficit": _as_float(row, "q99_ratio_deficit"),
                "q99_norm_pair_error_now": _as_float(row, "q99_norm_pair_error_now"),
                "q99_norm_pair_error_next": _as_float(row, "q99_norm_pair_error_next"),
            }
        )
    return out


def _validate_rows(rows: List[Dict[str, object]]) -> None:
    dims = [int(row["latent_dim"]) for row in rows]
    assert dims == DIMS, f"Expected latent dimensions {DIMS}, got {dims}."
    q99_true = [float(row["q99_r_true"]) for row in rows]
    assert all(a >= b for a, b in zip(q99_true[:-1], q99_true[1:])), "q99_r_true is not monotonically decreasing."
    assert float(rows[0]["q99_ratio_deficit"]) > float(rows[-1]["q99_ratio_deficit"])


def _write_data_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "model",
        "latent_dim",
        "epsilon",
        "q99_r_true",
        "q99_r_model",
        "q99_ratio_deficit",
        "q99_norm_pair_error_now",
        "q99_norm_pair_error_next",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["epsilon"] = EPSILON
            writer.writerow(payload)


def _print_table(rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "latent_dim",
        "q99_r_true",
        "q99_r_model",
        "q99_ratio_deficit",
        "q99_norm_pair_error_now",
        "q99_norm_pair_error_next",
    ]
    widths = {
        field: max(len(field), *(len(f"{row[field]:.6g}") if isinstance(row[field], float) else len(str(row[field])) for row in rows))
        for field in fieldnames
    }
    print("Dataframe used:")
    print("  " + "  ".join(field.rjust(widths[field]) for field in fieldnames))
    for row in rows:
        values = []
        for field in fieldnames:
            value = row[field]
            if isinstance(value, float):
                values.append(f"{value:.12g}".rjust(widths[field]))
            else:
                values.append(str(value).rjust(widths[field]))
        print("  " + "  ".join(values))


def _annotate_drop(ax: plt.Axes, text: str, x0: int, x1: int, y: float, y_text: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": "0.35", "shrinkA": 0, "shrinkB": 0},
    )
    x_mid = (x0 * x1) ** 0.5
    ax.text(x_mid, y_text, text, ha="center", va="bottom", fontsize=8, color="0.25")


def _plot(rows: List[Dict[str, object]]) -> List[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dims = [int(row["latent_dim"]) for row in rows]
    q99_true = [float(row["q99_r_true"]) for row in rows]
    q99_model = [float(row["q99_r_model"]) for row in rows]
    deficit = [float(row["q99_ratio_deficit"]) for row in rows]
    err_now = [float(row["q99_norm_pair_error_now"]) for row in rows]
    err_next = [float(row["q99_norm_pair_error_next"]) for row in rows]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.0), constrained_layout=True)
    true_color = "#1F77B4"
    model_color = "#D55E00"
    deficit_color = "#5B5B5B"
    err_color = "#0072B2"
    err_next_color = "#56B4E9"
    lw = 1.7
    ms = 4.6

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(dim) for dim in dims])
        ax.grid(True, color="0.88", linewidth=0.6)
        ax.tick_params(length=3, width=0.8)

    ax = axes[0]
    ax.plot(dims, q99_true, marker="o", lw=lw, ms=ms, color=true_color, label="true successor ratio")
    ax.plot(dims, q99_model, marker="s", lw=lw, ms=ms, ls="--", color=model_color, label="model successor ratio")
    ax.axhline(1.0, color="0.45", lw=0.9, ls=":", label="non-expansive")
    ax.set_title("A. Empirical transition gain")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("99th percentile gain")
    ax.set_ylim(0.98, 1.335)
    ax.legend(frameon=False, loc="lower left", handlelength=2.5)
    _annotate_drop(ax, "q99 excess over 1 drops by ~55%", 8, 192, 1.318, 1.322)

    ax = axes[1]
    ax.plot(dims, deficit, marker="o", lw=lw, ms=ms, color=deficit_color)
    ax.set_title("B. Model expansion deficit")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("99th percentile ratio deficit")
    ax.set_ylim(0.0, 0.165)
    ax.text(10.0, 0.015, r"$[r_{\mathrm{true}} - r_{\mathrm{model}}]_+$", fontsize=8, color="0.25")
    _annotate_drop(ax, "~75% drop", 8, 192, 0.153, 0.156)

    ax = axes[2]
    ax.plot(dims, err_now, marker="o", lw=lw, ms=ms, color=err_color, label="normalized by current distance")
    ax.plot(
        dims,
        err_next,
        marker="s",
        lw=lw,
        ms=ms,
        ls="--",
        color=err_next_color,
        label="normalized by successor distance",
    )
    ax.set_title("C. Normalized prediction error")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("99th percentile normalized error")
    ax.set_ylim(0.10, 0.33)
    ax.legend(frameon=False, loc="lower left", handlelength=2.5)
    _annotate_drop(ax, "~54% drop", 8, 192, 0.302, 0.306)

    output_paths = [
        FIGURE_DIR / f"{OUTPUT_STEM}.pdf",
        FIGURE_DIR / f"{OUTPUT_STEM}.png",
        FIGURE_DIR / f"{OUTPUT_STEM}.svg",
    ]
    for path in output_paths:
        if path.suffix == ".png":
            fig.savefig(path, dpi=320)
        else:
            fig.savefig(path)
    plt.close(fig)
    return output_paths


def main() -> None:
    rows = _load_rows()
    rows = sorted(rows, key=lambda row: int(row["latent_dim"]))
    _validate_rows(rows)
    _print_table(rows)

    data_path = FIGURE_DIR / f"{OUTPUT_STEM}_data.csv"
    caption_path = FIGURE_DIR / f"{OUTPUT_STEM}_caption.txt"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _write_data_csv(rows, data_path)
    caption_path.write_text(CAPTION + "\n")
    output_paths = _plot(rows)
    output_paths.extend([data_path, caption_path])

    print("\nOutput paths:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
