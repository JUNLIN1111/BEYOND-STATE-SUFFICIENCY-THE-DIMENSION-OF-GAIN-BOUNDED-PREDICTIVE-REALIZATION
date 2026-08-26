"""Audit controlled finite transition-system experiment artifacts.

This script reads completed full-run outputs and cached figure data only. It
does not run embedding optimization or predictor training.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import scipy
import torch

import dimension_gain_experiments as dg


E1_DIR = Path("outputs/dimension_gain_e1_full")
E1_RESULTS_PATH = E1_DIR / "e1_full_results.csv"
E1_SUMMARY_PATH = E1_DIR / "e1_full_summary.csv"
E1_THRESHOLDS_PATH = E1_DIR / "e1_full_thresholds.csv"
E1_METADATA_PATH = E1_DIR / "e1_full_metadata.json"
E1_EMBEDDING_DIR = E1_DIR / "embeddings"
E2_DIR = Path("outputs/e2_complete_transposition_scaling")
E2_RESULTS_PATH = E2_DIR / "e2_scaling_results.csv"
E2_SUMMARY_PATH = E2_DIR / "e2_scaling_summary.csv"
E2_METADATA_PATH = E2_DIR / "e2_scaling_metadata.json"
RESULT_FIGURE_MANIFEST_PATH = Path("figures/results_figure_manifest.json")
ORACLE_CANDIDATES_PATH = Path("figures/results_n8_oracle_error_all_candidates.csv")
ORACLE_MONOTONE_PATH = Path("figures/results_n8_oracle_error_monotone.csv")
MECHANISM_TABLE_PATH = Path("figures/mechanism_prediction_error_table_monotone.csv")
OUTPUT_TABLE_PATH = Path("figures/controlled_finite_experiment_machine_table.csv")
OUTPUT_N8_PATH = Path("figures/controlled_finite_experiment_n8_values.csv")
OUTPUT_CYCLE_PATH = Path("figures/controlled_finite_experiment_cycle_n4_m1.json")
OUTPUT_REPORT_PATH = Path("figures/controlled_finite_experiment_audit.md")
OUTPUT_MANIFEST_PATH = Path("figures/controlled_finite_experiment_audit_manifest.json")
SMOKE_TOKEN = "dimension_gain_smoke"

PUBLIC_LABELS = {
    dg.SYSTEM_CYCLE: "Cycle",
    dg.SYSTEM_ADJACENT: "Adjacent pair actions",
    dg.SYSTEM_ALLSWAP: "All-pair actions",
}
SYSTEM_ORDER = [dg.SYSTEM_CYCLE, dg.SYSTEM_ADJACENT, dg.SYSTEM_ALLSWAP]


def assert_no_smoke_path(path: Path) -> None:
    assert SMOKE_TOKEN not in str(path.resolve())


def read_csv(path: Path) -> pd.DataFrame:
    assert_no_smoke_path(path)
    return pd.read_csv(path)


def load_json(path: Path) -> Dict[str, Any]:
    assert_no_smoke_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pairwise_witness(z: torch.Tensor, system: dg.PreparedSystem) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    assert system.transitions_np is not None
    for action_idx, transition in enumerate(system.transitions_np):
        for i_tensor, j_tensor in zip(system.pair_i, system.pair_j):
            i = int(i_tensor.item())
            j = int(j_tensor.item())
            si = int(transition[i])
            sj = int(transition[j])
            d_now = float(torch.linalg.norm(z[i] - z[j]).item())
            d_next = float(torch.linalg.norm(z[si] - z[sj]).item())
            ratio = d_next / d_now
            if best is None or ratio > float(best["ratio"]):
                best = {
                    "action_index": action_idx,
                    "current_pair": [i, j],
                    "successor_pair_ordered": [si, sj],
                    "successor_pair_unordered": sorted([si, sj]),
                    "d_now": d_now,
                    "d_next": d_next,
                    "ratio": ratio,
                }
    if best is None:
        raise RuntimeError("No witness found")
    return best


def build_machine_table(
    summary: pd.DataFrame,
    results: pd.DataFrame,
    oracle_monotone: pd.DataFrame,
    oracle_candidates: pd.DataFrame,
) -> pd.DataFrame:
    random_rows = results[results["run_type"] == "random_optimization"].copy()
    valid_counts = (
        random_rows.groupby(["system", "n", "m"])["best_required_gain"]
        .apply(lambda values: int(np.isfinite(values.astype(float)).sum()))
        .rename("num_valid_restarts")
        .reset_index()
    )

    best_candidate_rows: List[Dict[str, Any]] = []
    for (system_label, m), group in oracle_candidates.groupby(["system", "source_m"], sort=True):
        optimized = group[group["run_type"] == "random optimization"].copy()
        if len(optimized) == 0:
            continue
        row = optimized.sort_values(["exact_hard_required_gain", "seed"]).iloc[0]
        best_candidate_rows.append(
            {
                "system_public": str(system_label),
                "n": int(row["n"]),
                "m": int(m),
                "oracle_mse_at_L1_for_best_optimized_seed": float(row["oracle_mse_at_L1"]),
                "oracle_best_optimized_seed": "" if pd.isna(row["seed"]) else int(row["seed"]),
                "oracle_best_optimized_hard_gain": float(row["exact_hard_required_gain"]),
            },
        )
    best_candidate = pd.DataFrame(best_candidate_rows)

    oracle_profile = oracle_monotone.rename(
        columns={
            "system": "system_public",
            "target_m": "m",
            "selected_oracle_mse_at_L1": "oracle_mse_at_L1_monotone_profile",
            "selected_source_m": "oracle_selected_source_m",
            "selected_seed": "oracle_selected_seed",
            "selected_exact_hard_required_gain": "oracle_selected_hard_gain",
            "selected_run_type": "oracle_selected_run_type",
            "selected_init_kind": "oracle_selected_init_kind",
        },
    )[
        [
            "system_public",
            "n",
            "m",
            "oracle_mse_at_L1_monotone_profile",
            "oracle_selected_source_m",
            "oracle_selected_seed",
            "oracle_selected_hard_gain",
            "oracle_selected_run_type",
            "oracle_selected_init_kind",
        ]
    ]

    table = summary.merge(valid_counts, on=["system", "n", "m"], how="left")
    table["system_public"] = table["system"].map(PUBLIC_LABELS)
    table = table.merge(oracle_profile, on=["system_public", "n", "m"], how="left")
    table = table.merge(best_candidate, on=["system_public", "n", "m"], how="left")
    table = table[
        [
            "system_public",
            "system",
            "n",
            "m",
            "best_optimized_gain",
            "median_optimized_gain",
            "optimized_gain_q1",
            "optimized_gain_q3",
            "optimized_gain_iqr",
            "analytic_construction_gain",
            "analytic_construction_kind",
            "best_available_upper_bound",
            "best_solution_source",
            "theoretical_L1_threshold",
            "optimized_L1_feasible",
            "analytic_L1_feasible",
            "best_available_L1_feasible",
            "oracle_mse_at_L1_monotone_profile",
            "oracle_selected_source_m",
            "oracle_selected_seed",
            "oracle_selected_hard_gain",
            "oracle_selected_run_type",
            "oracle_selected_init_kind",
            "oracle_mse_at_L1_for_best_optimized_seed",
            "oracle_best_optimized_seed",
            "oracle_best_optimized_hard_gain",
            "num_valid_restarts",
        ]
    ].sort_values(["system_public", "n", "m"])
    table.to_csv(OUTPUT_TABLE_PATH, index=False, float_format="%.15g")
    return table


def cycle_n4_m1_details(results: pd.DataFrame) -> Dict[str, Any]:
    rows = results[
        (results["system"] == dg.SYSTEM_CYCLE)
        & (results["n"].astype(int) == 4)
        & (results["m"].astype(int) == 1)
        & (results["run_type"] == "random_optimization")
    ].copy()
    row = rows.sort_values(["best_required_gain", "seed"]).iloc[0]
    seed = int(row["seed"])
    path = E1_EMBEDDING_DIR / f"e1_full_Cycle_n4_m1_seed{seed}.pt"
    assert_no_smoke_path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    z = checkpoint["Z"].to(dtype=dg.DTYPE, device=dg.CPU)
    system = dg.prepare_system(dg.SYSTEM_CYCLE, 4, build_successor_pairs=True)
    gain = dg.hard_required_gain(z, system)
    min_distance, max_distance = dg.pairwise_distance_stats(z, system)
    witness = pairwise_witness(z, system)
    details = {
        "system": "Cycle",
        "internal_system_id": dg.SYSTEM_CYCLE,
        "n": 4,
        "m": 1,
        "selected_seed": seed,
        "checkpoint_path": str(path),
        "saved_best_Z": [[float(value) for value in row_values] for row_values in z.tolist()],
        "mean": [float(value) for value in z.mean(dim=0).tolist()],
        "min_pairwise_distance": float(min_distance),
        "max_pairwise_distance": float(max_distance),
        "unrounded_required_gain": float(gain),
        "summary_best_available_upper_bound": float(row["best_required_gain"]),
        "witness": witness,
        "heatmap_display_label_with_two_decimals": f"{float(gain):.2f}",
        "diagnosis": "rounding of a value slightly larger than one",
    }
    OUTPUT_CYCLE_PATH.write_text(json.dumps(details, indent=2, sort_keys=True), encoding="utf-8")
    return details


def build_n8_values(table: pd.DataFrame) -> pd.DataFrame:
    rows = table[table["n"].astype(int) == 8].copy()
    rows = rows[
        [
            "system_public",
            "m",
            "best_optimized_gain",
            "median_optimized_gain",
            "analytic_construction_gain",
            "best_available_upper_bound",
            "best_solution_source",
            "oracle_mse_at_L1_monotone_profile",
            "oracle_selected_source_m",
            "oracle_selected_seed",
            "oracle_selected_hard_gain",
            "oracle_mse_at_L1_for_best_optimized_seed",
            "oracle_best_optimized_seed",
        ]
    ].sort_values(["system_public", "m"])
    rows.to_csv(OUTPUT_N8_PATH, index=False, float_format="%.15g")
    return rows


def figure_provenance_markdown() -> str:
    return """
| Figure | Recommended filename(s) | Generating script | Inputs | Data source |
|---|---|---|---|---|
| 1. E1 best/median frontiers | `figures/results_e1_frontier_grid.pdf` or `.png`/`.svg` | `make_paper_result_figures.py::plot_e1_frontier_grid`, lines 98-214 | `outputs/dimension_gain_e1_full/e1_full_summary.csv`; `outputs/dimension_gain_e1_full/e1_full_results.csv` (`read_e1_summary`, lines 66-73; `read_e1_results`, lines 76-83) | Raw random seed rows plus aggregated summary columns; no manual numeric arrays. |
| 2. E1 gain heatmap | `figures/results_e1_gain_heatmaps.pdf` or `.png`/`.svg` | `make_paper_result_figures.py::plot_e1_heatmaps`, lines 217-264 | `outputs/dimension_gain_e1_full/e1_full_summary.csv` | Aggregated `best_available_upper_bound`; cell text is rounded with `f"{value:.2f}"` at line 239. |
| 3. All-pair scaling | `figures/results_e2_all_pair_scaling.pdf` or `.png`/`.svg` | `make_paper_result_figures.py::plot_e2_scaling`, lines 454-508 | `outputs/e2_complete_transposition_scaling/e2_scaling_summary.csv` (`read_e2_summary`, lines 86-88) | E2 aggregated CSV: analytic lower, explicit grid, and optimized markers where run; no manual numeric arrays. |
| 4. n=8 gain/error curves | `figures/results_n8_gain_error_curves.pdf` or `.png`/`.svg` | `make_paper_result_figures.py::plot_n8_gain_error_curves`, lines 386-451 | E1 summary plus cached/generated `figures/results_n8_oracle_error_all_candidates.csv` and `figures/results_n8_oracle_error_monotone.csv` | Gain panel from E1 summary; error panel from finite-state SLSQP oracle profile. |
| 5. Clean mechanism schematic | `figures/mechanism_paper_9panel_clean.pdf` or `.png`/`.svg` | `make_paper_mechanism_clean_and_table.py::make_clean_mechanism`, lines 73-136; schematic coordinates from `make_paper_mechanism_9panel.py::schematic_points`, lines 119-136 | `figures/mechanism_prediction_error_table_monotone.csv` for separate numeric table | Schematic coordinates, not raw checkpoint coordinates; numeric table from completed E1 embeddings and SLSQP oracle. |
"""


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".15g") -> str:
    """Small markdown formatter to avoid a runtime dependency on tabulate."""

    columns = [str(column) for column in df.columns]

    def format_value(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    rows = [[format_value(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(columns[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(columns))
    ]
    header = "| " + " | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
    separator = "| " + " | ".join("-" * max(3, widths[idx]) for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_report(
    summary: pd.DataFrame,
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    e1_metadata: Dict[str, Any],
    e2_metadata: Dict[str, Any],
    table: pd.DataFrame,
    n8_values: pd.DataFrame,
    cycle_details: Dict[str, Any],
) -> None:
    n_values = sorted(int(value) for value in summary["n"].unique())
    m_by_n = {
        int(n): sorted(int(value) for value in summary[summary["n"].astype(int) == int(n)]["m"].unique())
        for n in n_values
    }
    e2 = read_csv(E2_SUMMARY_PATH)
    results_random = results[results["run_type"] == "random_optimization"].copy()
    analytic = results[results["run_type"] == "analytic_construction"].copy()
    current_cpu_model = "not recovered"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                current_cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    requested_rows = n8_values[
        ((n8_values["m"].astype(int).isin([1, 2, 7])))
        | (
            (n8_values["m"].astype(int) == 6)
            & (n8_values["system_public"].isin(["Adjacent pair actions", "All-pair actions"]))
        )
    ].copy()
    requested_markdown = dataframe_to_markdown(requested_rows, floatfmt=".15g")
    threshold_markdown = dataframe_to_markdown(thresholds, floatfmt=".15g")
    cycle_z = json.dumps(cycle_details["saved_best_Z"], indent=2)
    e2_preview = dataframe_to_markdown(
        e2[
            [
                "n",
                "m",
                "analytic_lower_bound",
                "explicit_grid_gain",
                "optimized_best_gain",
                "grid_to_lower_ratio",
                "optimized_to_lower_ratio",
                "num_seeds",
            ]
        ].head(12),
        floatfmt=".8g",
    )

    methods_text = (
        "The experiment directly optimizes finite latent coordinates for deterministic "
        "controlled transition systems, without training an encoder, a learned predictor, "
        "a world model, a policy, or an RL benchmark. For each system, the labelled state "
        "set is `{0,...,n-1}` and every action is an explicit deterministic map on these "
        "labels. The reported gain is the exact hard maximum over actions and unordered "
        "state pairs of successor distance divided by current distance. The optimized "
        "variable is a free `n x m` coordinate matrix `Z` in `torch.float64` on CPU. "
        "Training uses a smooth log-sum-exp surrogate of log distance ratios; the hard "
        "maximum is recomputed for reporting and checkpoint selection. After initialization "
        "and after each Adam update, coordinates are centered by subtracting the row mean "
        "and scaled so the minimum nonzero pairwise distance equals one. Collisions below "
        "`1e-10`, nonfinite losses, or failed projections trigger an internal restart, but "
        "the summary does not remove non-converged seed rows. E1 evaluates Cycle, Adjacent "
        "pair actions, and All-pair actions for `n in {4,6,8,10}` and all `m=1,...,n-1`, "
        "with ten seeds per configuration and 15000 steps per seed. Analytic constructions "
        "are evaluated separately: regular polygons for Cycle when `m>=2`, regular simplexes "
        "for the pair-action families when `m>=n-1`, and grid constructions for All-pair "
        "actions. The n=8 oracle MSE curves fix stored E1 embeddings and solve a finite-state "
        "constrained prediction problem at gain budget `L=1`."
    )
    results_text = (
        "The full E1 output contains 720 random optimization rows and 52 analytic "
        "construction rows, summarized into 72 family-state-dimension configurations. "
        "The qualitative pattern is stable across the saved results: Cycle reaches a "
        "gain-one construction in two dimensions via the regular polygon, whereas the "
        "two pair-action families reach the analytic gain-one construction at dimension "
        "`n-1` via a regular simplex. At `n=8`, the best available gains are "
        "`2.34594912688328` for Cycle at `m=1`, `1.0000000000000004` for Cycle at `m=2`, "
        "`1.560477598995624` for Adjacent pair actions at `m=2`, and "
        "`2.32477037253896` for All-pair actions at `m=2`. At `m=7`, all three are at "
        "gain one up to floating-point tolerance when analytic constructions are included. "
        "The heatmap cell that visually reads as Cycle `n=4,m=1` equal to `1.00` is a "
        "rounding artifact: the raw best value is `1.0022480814457853`, the run is not "
        "converged to gain one, and the saved coordinates have minimum pairwise distance "
        "one. E2 extends only the All-pair action family to `n in {8,12,16,24,32,64,128}` "
        "and `m in {1,2,3,4}`; it plots the stored analytic lower bound, grid upper "
        "construction, and optimized markers only for `n<=32`. The oracle MSE plots are "
        "finite-state optimization diagnostics for fixed embeddings, not training losses."
    )

    report = f"""# Controlled Finite Transition-System Experiment Audit

Generated by `make_controlled_finite_experiment_audit.py`. This audit reads completed full-run outputs only; it does not run optimization or predictor training.

## A. Figure and Data Provenance
{figure_provenance_markdown()}

The plotting manifest at `figures/results_figure_manifest.json` records `used_smoke_outputs=false`, the E1 summary/results inputs, the cached n=8 oracle CSVs, and the result-figure outputs. There is no `.tex` paper file in this repository that proves final LaTeX inclusion; the filenames above are therefore the evidence-based filenames present in the repo.

## B. Transition-Family Definitions

- Cycle: `dimension_gain_experiments.py::cycle_system`, lines 115-120, returns one action with `T[0,i]=(i+1) mod n`. Yes, it is one action implementing an n-cycle.
- Adjacent pair actions: `dimension_gain_experiments.py::adjacent_swap_system`, lines 101-112. For action `k`, the row sends `k` to `k+1` and `k+1` to `k`, with all other states fixed; there are `n-1` actions.
- All-pair actions: `dimension_gain_experiments.py::allswap_system`, lines 123-135. For each unordered pair `p<q`, one action sends `p` to `q` and `q` to `p`, with all others fixed; there are `n(n-1)/2` actions.
- Determinism and labels: transition tables are integer arrays checked by `assert_valid_transition_table` and `assert_permutation_rows`, lines 172-193. All systems use labelled states `0,...,n-1`.

## C. Sweep Design

- E1 state counts: `{n_values}`, from `run_e1_full.py::N_VALUES`, line 36, and metadata `outputs/dimension_gain_e1_full/e1_full_metadata.json`.
- E1 dimensions: `{m_by_n}`, from the loop `for m in range(1, n)` in `run_e1_full.py`, line 572.
- E1 random restarts: `{int(e1_metadata["random_optimization_seeds_per_configuration"])}` per `(family,n,m)`, seeds `{e1_metadata["random_seed_values"]}`, from metadata and `run_e1_full.py`, lines 573-584.
- The same seed numbers are reused across systems and dimensions because every configuration loops over `range(config.seeds)`; see `run_e1_full.py`, line 573.
- E2 state counts: `{e2_metadata["n_values"]}`; E2 dimensions: `{e2_metadata["m_values"]}`; E2 random seeds per optimized point: `{e2_metadata["seeds"]}`. Optimization is run only when `n<=32` unless `--run-large-optimization` is passed; see `run_e2_complete_transposition_scaling.py`, lines 189-204.

## D. Latent-Coordinate Optimization

- Optimized variable: `raw_z = torch.nn.Parameter(z0.clone())`, shape `(system.n,m)`, in `dimension_gain_experiments.py::optimize_embedding`, lines 548-583.
- Training objective: smooth log-sum-exp over log distance ratios, `logsumexp(beta*(log(succ+eps)-log(base+eps)))/beta`, in `smooth_logmax_generic_loss`, lines 326-340. The all-pair fast E2 objective uses row-aspect log-sum-exp in lines 343-355.
- Reported hard gain: `hard_required_gain`, lines 315-323, calls exact generic max lines 256-292 or exact all-pair row-aspect formula lines 295-312.
- Centering and scale: `centered_min_distance_normalize`, lines 424-441, subtracts row mean and divides by the minimum pairwise distance. Yes, the minimum pairwise distance is fixed to one after projection.
- Collision handling: no anti-collision loss term. `COLLISION_TOL=1e-10` at line 44; hard gain has no epsilon denominator and returns `inf` on zero base distance, lines 266-267 and 286-287. The smooth surrogate uses `eps=1e-12` from metadata, added inside the log at line 339.
- Optimizer/settings: Adam and cosine LR scheduler in `make_optimizer`, lines 468-479; E1 full metadata records lr `{e1_metadata["learning_rate"]}`, steps `{e1_metadata["optimization_steps_per_seed"]}`, beta values `{e1_metadata["beta_values"]}`, beta boundaries `{e1_metadata["beta_boundaries"]}`, gradient clip `{e1_metadata["gradient_clip_norm"]}`, patience `{e1_metadata["patience"]}`, dtype `{e1_metadata["dtype"]}`, CPU only `{e1_metadata["cpu_only"]}`.
- Best/median: `summarize_full_e1`, lines 144-168, takes the minimum and median of `best_required_gain` over the ten random rows for the same `(system,n,m)`.
- Failed/non-converged rows: no row-level filtering is applied before summary beyond grouping random rows; see `run_e1_full.py`, lines 618-627. Non-converged rows remain included. Internal restarts after nonfinite loss or projection failure are counted inside a row; see `dimension_gain_experiments.py`, lines 603-639.

## E. Analytic Values and Constructions

- Analytic markers in E1 come from `run_e1_full.py::analytic_constructions`, lines 56-66, and are evaluated without optimization by `evaluate_analytic_construction`, lines 69-113.
- Cycle: regular polygon from `dimension_gain_experiments.py::regular_polygon`, lines 387-393, padded when `m>=2`; analytic gain-one threshold is `m=2`, from `theoretical_l1_threshold`, lines 206-213.
- Adjacent pair actions: analytic gain-one threshold is `m=n-1`, implemented with `regular_simplex`, lines 371-384, selected in `run_e1_full.py`, lines 62-63.
- All-pair actions: E1 has grid markers for every `m` and simplex markers at `m>=n-1`, from `run_e1_full.py`, lines 62-65.
- E2 formulas actually plotted: lower bound `max(1,(n^(1/m)-1)/2)` in `run_e2_complete_transposition_scaling.py::analytic_lower_bound`, lines 117-119; grid upper `sqrt(m)*(ceil(n^(1/m))-1)` in lines 121-122; normalized grid hard gain in lines 125-131.
- Gap panel: `grid_to_lower_ratio = explicit_grid_gain / analytic_lower_bound` and `optimized_to_lower_ratio = optimized_best_gain / analytic_lower_bound`, in lines 230-240. The plot shows the grid ratio as a line and optimized ratios as markers; it does not take the smaller of the two.

## F. Cycle n=4,m=1 Heatmap Issue

Diagnosis: rounding. The heatmap text uses `f"{{value:.2f}}"` at `make_paper_result_figures.py`, line 239, so `{cycle_details["unrounded_required_gain"]:.15g}` is displayed as `{cycle_details["heatmap_display_label_with_two_decimals"]}`. The raw summary row is `outputs/dimension_gain_e1_full/e1_full_summary.csv`, line 50; the best raw seed row is `e1_full_results.csv`, line 11.

It is not caused by latent collision, wrong min-distance normalization, denominator clipping, failed aggregation, label/index misalignment, or a gain-one run. The selected run has `converged=False`, `stopped_reason=max_steps`, minimum pairwise distance `{cycle_details["min_pairwise_distance"]:.15g}`, and maximum pairwise distance `{cycle_details["max_pairwise_distance"]:.15g}`.

Saved best coordinates:

```json
{cycle_z}
```

Witness ratio: action `{cycle_details["witness"]["action_index"]}`, current pair `{cycle_details["witness"]["current_pair"]}`, successor pair `{cycle_details["witness"]["successor_pair_ordered"]}`, `d_now={cycle_details["witness"]["d_now"]:.15g}`, `d_next={cycle_details["witness"]["d_next"]:.15g}`, ratio `{cycle_details["witness"]["ratio"]:.15g}`.

The figure should be regenerated or annotated with more precision if it will be used to distinguish exact gain one from values slightly above one. Corrected value: `{cycle_details["unrounded_required_gain"]:.15g}`.

## G. Oracle Prediction-Error Experiment

- Mathematical definition for one fixed embedding `Z`, action `a`, and budget `L`: minimize `(1/n) sum_i ||Y_i - Z[T_a(i)]||_2^2` over predicted successor coordinates `Y in R^(n x m)` subject to `||Y_i-Y_j||_2 <= L ||Z_i-Z_j||_2` for every unordered pair `{{i,j}}`. The reported oracle MSE averages this optimum over actions.
- Variables: one free predicted successor vector per state for each fixed action; not a neural network and not a linear map. In the SLSQP implementation this is flattened `y`, `make_mechanism_prediction_arrows.py`, lines 224-260.
- Constraints: all unordered state pairs under each action; SLSQP constraints are lines 262-282; averaging over actions is lines 299-321.
- Loss/normalization: sum squared coordinate residuals divided by `n`, line 256, then averaged across actions at line 309. It is raw coordinate MSE after the embedding has been centered and min-distance normalized; it is not further scale-normalized.
- Solver settings for Figure 4: SciPy SLSQP, `ftol=1e-10`, `maxiter=1000`, deterministic initialization at lines 248-250, no learning rate, no restarts, no random seeds. If the exact target already satisfies the budget, MSE is returned as `0.0`, lines 237-246.
- Displayed zeros: exact feasible checks return `0.0`; numerical values below formatting precision can also display as zero in tables. The result-curve plot uses `symlog` with `linthresh=1e-4`, in `make_paper_result_figures.py`, line 428, so zero is plottable.
- The separate `run_e3_oracle_mse.py` is a cvxpy/SOCP version, but this repository only contains `outputs/e3_oracle_mse/README.md`, not completed E3 CSV outputs. Figure 4 uses the SLSQP cached CSVs, not `run_e3_oracle_mse.py`.
- The oracle-MSE plot is a finite-state diagnostic for fixed embeddings. It tests the same obstruction direction qualitatively, but it is not the theorem's exact dimension-gain profile: it depends on selected stored embeddings, the `L=1` budget, and the raw coordinate MSE objective.

## H. Representative Latent Configurations

- Current clean mechanism figure: `make_paper_mechanism_clean_and_table.py`, lines 73-136, imports schematic geometry from `make_paper_mechanism_9panel.py::schematic_points`, lines 119-136. Therefore the displayed coordinates are schematic, not actual optimized coordinates.
- Edges in the clean mechanism row represent transition actions: Cycle draws all 8 directed edges; Adjacent pair actions draw all 7 neighboring-pair bidirectional edges; All-pair actions draw all 28 unordered-pair bidirectional edges. See `make_paper_mechanism_9panel.py::transition_edges` and `draw_transitions`, lines 189-258.
- Actual checkpoint-coordinate alternatives: `make_mechanism_dimension_distribution.py`, lines 1-7 and 165-204, loads completed E1 checkpoints and records actual embeddings; `make_mechanism_dimension_transition_arrows.py`, lines 174-226 and 241-275, evaluates padded source dimensions and selected checkpoint-derived coordinates.
- For the current clean mechanism figure, state labels are the same labels `0,...,7`, but they do not correspond to actual optimized coordinates. For the actual checkpoint-coordinate figures, labels do correspond to loaded embeddings after centering/min-distance normalization; 3D views are camera projections for visualization.

## I. Exact Reported Numbers

Machine-readable full table written to `{OUTPUT_TABLE_PATH}`.

Requested n=8 values:

{requested_markdown}

Threshold table:

{threshold_markdown}

## J. Reproducibility

- E1 saved metadata: Python `{e1_metadata["python_version"].replace(chr(10), " ")}`, PyTorch `{e1_metadata["pytorch_version"]}`, NumPy `{e1_metadata["numpy_version"]}`, pandas `{e1_metadata["pandas_version"]}`, matplotlib `{e1_metadata["matplotlib_version"]}`, platform `{e1_metadata["platform"]}`.
- E2 saved metadata: Python `{e2_metadata["python_version"].replace(chr(10), " ")}`, PyTorch `{e2_metadata["pytorch_version"]}`, NumPy `{e2_metadata["numpy_version"]}`, pandas `{e2_metadata["pandas_version"]}`, matplotlib `{e2_metadata["matplotlib_version"]}`, platform `{e2_metadata["platform"]}`.
- Current environment also has SciPy `{scipy.__version__}` for the SLSQP oracle. Current host CPU visible now: `{current_cpu_model}` with `os.cpu_count()` reported by Python as `{os.cpu_count()}`. The exact CPU model used when the saved E1/E2 runs were created is not stored in metadata, so it cannot be recovered without logs.
- Approximate E1/E2 runtime cannot be recovered from saved metadata; scripts printed elapsed seconds (`run_e1_full.py`, lines 643-645; `run_e2_complete_transposition_scaling.py`, lines 475-477), but those logs are not present in the saved outputs.

Reproduction commands:

```bash
/home/junlin/miniconda3/envs/SemNev/bin/python run_e1_full.py --output-dir outputs/dimension_gain_e1_full --steps 15000 --seeds 10 --patience 15001
/home/junlin/miniconda3/envs/SemNev/bin/python run_e2_complete_transposition_scaling.py --output-dir outputs/e2_complete_transposition_scaling --n-values 8,12,16,24,32,64,128 --m-values 1,2,3,4 --seeds 5 --steps 15000
/home/junlin/miniconda3/envs/SemNev/bin/python make_paper_result_figures.py
/home/junlin/miniconda3/envs/SemNev/bin/python make_mechanism_dimension_transition_arrows.py
/home/junlin/miniconda3/envs/SemNev/bin/python make_paper_mechanism_clean_and_table.py
/home/junlin/miniconda3/envs/SemNev/bin/python make_controlled_finite_experiment_audit.py
```

The cvxpy E3 command exists but did not produce completed CSV outputs in this repository:

```bash
/home/junlin/miniconda3/envs/SemNev/bin/python run_e3_oracle_mse.py
```

## E2 Preview

{e2_preview}

## Methods-Ready Description

{methods_text}

## Results-Ready Description

{results_text}
"""
    OUTPUT_REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    torch.set_default_dtype(dg.DTYPE)
    for path in [
        E1_RESULTS_PATH,
        E1_SUMMARY_PATH,
        E1_THRESHOLDS_PATH,
        E1_METADATA_PATH,
        E2_RESULTS_PATH,
        E2_SUMMARY_PATH,
        E2_METADATA_PATH,
        ORACLE_CANDIDATES_PATH,
        ORACLE_MONOTONE_PATH,
        MECHANISM_TABLE_PATH,
    ]:
        assert_no_smoke_path(path)

    summary = read_csv(E1_SUMMARY_PATH)
    results = read_csv(E1_RESULTS_PATH)
    thresholds = read_csv(E1_THRESHOLDS_PATH)
    e1_metadata = load_json(E1_METADATA_PATH)
    e2_metadata = load_json(E2_METADATA_PATH)
    oracle_candidates = read_csv(ORACLE_CANDIDATES_PATH)
    oracle_monotone = read_csv(ORACLE_MONOTONE_PATH)

    table = build_machine_table(summary, results, oracle_monotone, oracle_candidates)
    n8_values = build_n8_values(table)
    cycle_details = cycle_n4_m1_details(results)
    write_report(summary, results, thresholds, e1_metadata, e2_metadata, table, n8_values, cycle_details)

    manifest = {
        "used_smoke_outputs": False,
        "inputs": [
            str(E1_RESULTS_PATH),
            str(E1_SUMMARY_PATH),
            str(E1_THRESHOLDS_PATH),
            str(E1_METADATA_PATH),
            str(E2_RESULTS_PATH),
            str(E2_SUMMARY_PATH),
            str(E2_METADATA_PATH),
            str(ORACLE_CANDIDATES_PATH),
            str(ORACLE_MONOTONE_PATH),
            str(MECHANISM_TABLE_PATH),
        ],
        "outputs": [
            str(OUTPUT_TABLE_PATH),
            str(OUTPUT_N8_PATH),
            str(OUTPUT_CYCLE_PATH),
            str(OUTPUT_REPORT_PATH),
        ],
        "row_counts": {
            "e1_results": int(len(results)),
            "e1_summary": int(len(summary)),
            "oracle_candidates": int(len(oracle_candidates)),
            "oracle_monotone": int(len(oracle_monotone)),
            "machine_table": int(len(table)),
        },
        "python_version_now": sys.version,
        "platform_now": platform.platform(),
    }
    OUTPUT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("Wrote audit artifacts:")
    for path in [
        OUTPUT_TABLE_PATH,
        OUTPUT_N8_PATH,
        OUTPUT_CYCLE_PATH,
        OUTPUT_REPORT_PATH,
        OUTPUT_MANIFEST_PATH,
    ]:
        print(path.resolve())
    print()
    print("Cycle n=4,m=1 corrected gain:", f"{cycle_details['unrounded_required_gain']:.15g}")
    print("Valid restart count values:", sorted(table["num_valid_restarts"].dropna().astype(int).unique().tolist()))


if __name__ == "__main__":
    main()
