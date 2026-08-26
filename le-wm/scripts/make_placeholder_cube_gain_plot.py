from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Hypothetical Cube values for draft-only plotting.
# These follow the monotone PushT trend but are NOT measured results.
ROWS = [
    {"model": "state8", "latent_dim": 8, "q99_r_true": 1.26, "q99_r_model": 1.22, "q99_ratio_deficit": 0.125, "q99_norm_pair_error_now": 0.240},
    {"model": "state16", "latent_dim": 16, "q99_r_true": 1.19, "q99_r_model": 1.17, "q99_ratio_deficit": 0.070, "q99_norm_pair_error_now": 0.175},
    {"model": "state32", "latent_dim": 32, "q99_r_true": 1.15, "q99_r_model": 1.14, "q99_ratio_deficit": 0.050, "q99_norm_pair_error_now": 0.155},
    {"model": "state64", "latent_dim": 64, "q99_r_true": 1.13, "q99_r_model": 1.12, "q99_ratio_deficit": 0.040, "q99_norm_pair_error_now": 0.140},
    {"model": "baseline192", "latent_dim": 192, "q99_r_true": 1.10, "q99_r_model": 1.095, "q99_ratio_deficit": 0.030, "q99_norm_pair_error_now": 0.115},
]


def write_csv(out_dir: Path) -> Path:
    out = out_dir / "cube_hypothetical_summary_normalized.csv"
    fieldnames = ["model", "latent_dim", "epsilon", "q99_r_true", "q99_r_model", "q99_ratio_deficit", "q99_norm_pair_error_now", "source_note"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ROWS:
            payload = dict(row)
            payload["epsilon"] = "1e-06"
            payload["source_note"] = "HYPOTHETICAL_PLACEHOLDER_NOT_MEASURED"
            writer.writerow(payload)
    return out


def plot_gain(out_dir: Path) -> None:
    dims = [r["latent_dim"] for r in ROWS]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(dims, [r["q99_r_true"] for r in ROWS], marker="o", label=r"$Q_{.99}(r^{true})$")
    ax.plot(dims, [r["q99_r_model"] for r in ROWS], marker="s", label=r"$Q_{.99}(r^{model})$")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("q99 transition gain")
    ax.set_title("Cube transition gain (hypothetical placeholder)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.5, 0.03, "PLACEHOLDER — NOT MEASURED", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="crimson", alpha=0.85)
    fig.tight_layout()
    fig.savefig(out_dir / "cube_hypothetical_gain_ratios.pdf")
    fig.savefig(out_dir / "cube_hypothetical_gain_ratios.png", dpi=260)
    plt.close(fig)


def plot_deficit(out_dir: Path) -> None:
    dims = [r["latent_dim"] for r in ROWS]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    axes[0].plot(dims, [r["q99_ratio_deficit"] for r in ROWS], marker="o", color="tab:red")
    axes[0].set_xlabel("latent dimension")
    axes[0].set_ylabel(r"$Q_{.99}([r^{true}-r^{model}]_+)$")
    axes[0].grid(alpha=0.25)
    axes[1].plot(dims, [r["q99_norm_pair_error_now"] for r in ROWS], marker="o", color="tab:purple")
    axes[1].set_xlabel("latent dimension")
    axes[1].set_ylabel(r"$Q_{.99}(\epsilon_{pair}/d_{now})$")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Cube normalized deficit/error (hypothetical placeholder)", y=1.02)
    fig.text(0.5, 0.01, "PLACEHOLDER — NOT MEASURED", ha="center", va="bottom", fontsize=9, color="crimson", alpha=0.85)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "cube_hypothetical_normalized_deficit_error.pdf")
    fig.savefig(out_dir / "cube_hypothetical_normalized_deficit_error.png", dpi=260)
    plt.close(fig)


def main() -> None:
    out_dir = Path("outputs/lewm_gain_cube_hypothetical")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(out_dir)
    plot_gain(out_dir)
    plot_deficit(out_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'cube_hypothetical_gain_ratios.pdf'}")
    print(f"wrote {out_dir / 'cube_hypothetical_normalized_deficit_error.pdf'}")


if __name__ == "__main__":
    main()
