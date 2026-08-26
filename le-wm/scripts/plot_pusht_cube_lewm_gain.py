from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_summary(path: Path, env_name: str, epsilon: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open() as file:
        for row in csv.DictReader(file):
            if row.get("epsilon") != epsilon:
                continue
            row = dict(row)
            row["env"] = env_name
            row["latent_dim"] = int(float(row["latent_dim"]))
            for key in [
                "q99_r_true",
                "q99_r_model",
                "q99_ratio_deficit",
                "q99_norm_pair_error_now",
            ]:
                row[key] = float(row[key])
            rows.append(row)
    return sorted(rows, key=lambda item: int(item["latent_dim"]))


def _plot_panel(ax, rows: List[Dict[str, object]], env_name: str) -> None:
    dims = np.asarray([row["latent_dim"] for row in rows], dtype=float)
    ax.plot(dims, [row["q99_r_true"] for row in rows], marker="o", label=r"$Q_{.99}(r^{true})$")
    ax.plot(dims, [row["q99_r_model"] for row in rows], marker="s", label=r"$Q_{.99}(r^{model})$")
    ax.plot(dims, [row["q99_ratio_deficit"] for row in rows], marker="^", label=r"$Q_{.99}([r^{true}-r^{model}]_+)$")
    ax.plot(dims, [row["q99_norm_pair_error_now"] for row in rows], marker="d", label=r"$Q_{.99}(\epsilon_{pair}/d_{now})$")
    ax.set_title(env_name)
    ax.set_xlabel("latent dimension")
    ax.grid(alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PushT/Cube LeWorldModel transition-gain summaries.")
    parser.add_argument("--pusht_summary", default="outputs/lewm_gain/summary_normalized.csv")
    parser.add_argument("--cube_summary", required=True)
    parser.add_argument("--epsilon", default="1e-06")
    parser.add_argument("--output_dir", default="outputs/lewm_gain")
    args = parser.parse_args()

    pusht_path = Path(args.pusht_summary)
    cube_path = Path(args.cube_summary)
    if not pusht_path.exists():
        raise FileNotFoundError(f"Missing PushT summary: {pusht_path}")
    if not cube_path.exists():
        raise FileNotFoundError(f"Missing Cube summary: {cube_path}")

    pusht_rows = _read_summary(pusht_path, "PushT", args.epsilon)
    cube_rows = _read_summary(cube_path, "Cube", args.epsilon)
    if not pusht_rows:
        raise ValueError(f"No PushT rows with epsilon={args.epsilon}")
    if not cube_rows:
        raise ValueError(f"No Cube rows with epsilon={args.epsilon}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=False)
    _plot_panel(axes[0], pusht_rows, "PushT")
    _plot_panel(axes[1], cube_rows, "Cube")
    axes[0].set_ylabel("q99 diagnostic value")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "lewm_pusht_cube_gain_summary"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=260)
    plt.close(fig)
    print(f"[plot_pusht_cube] wrote {out.with_suffix('.pdf')}", flush=True)
    print(f"[plot_pusht_cube] wrote {out.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
