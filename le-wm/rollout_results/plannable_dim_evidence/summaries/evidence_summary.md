# Plannable Latent Dimension Evidence

## Main Claim

In reward-free / goal-conditioned latent planning, latent distance is the planner cost. The representation therefore needs more than decodable future structure: transition/reachability structure must be encoded in the latent metric itself.

## How To Read This Package

- `spectra/`: graph-distance MDS spectra and `d_plan(q)` estimates.
- `false_shortcuts/`: low-dimensional MDS embeddings and graph-far/latent-near shortcut rates.
- `trained_latent_metric/`: learned latent Euclidean distances compared against graph/reachability distances.
- `compression_metric/`: post-hoc projection of baseline latents as a geometry-loss control.
- `prediction_loss_control/`: prediction-loss summaries when available.
- `toy/`: decodable-but-not-plannable toy examples.

## Cautious Interpretation

`d_plan(q)` is a geometry-retention curve, not a magic exact latent dimension. Average stress can be low in small dimension while planner-facing false shortcuts remain high. The key diagnostic is whether graph-far futures contract into latent-near pairs, because those are exactly the false shortcuts that a distance-based planner can confuse.

## Outputs

- Model evidence table: `rollout_results/plannable_dim_evidence/summaries/model_evidence_table.csv`
- Main evidence-chain plot: `plots/evidence_chain_by_latent_dim.png`

## What Remains Weak

This package estimates metric embeddability and compares it against cached model diagnostics. It does not prove optimal representation width, and downstream closed-loop success still depends on policy/CEM details, action distribution, and model rollout quality.
