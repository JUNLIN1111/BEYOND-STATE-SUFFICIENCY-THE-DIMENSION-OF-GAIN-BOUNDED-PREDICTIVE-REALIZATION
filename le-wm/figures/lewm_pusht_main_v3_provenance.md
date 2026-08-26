# PushT LeWorldModel main figure provenance

## Plotted sources
- Geometry diagnostics: `scripts/plot_lewm_dimension_gain_diagnostics_v2.py::FALLBACK_ROWS`.
- Task success values: `rollout_results/plannable_dim_evidence/summaries/model_evidence_table.csv`, column `SR_100`.
- Success episode count: `config/eval/pusht.yaml`, `eval.num_eval=50`; the success table stores the percentage value, not per-episode records.
- PushT success/cost code checked against `scripts/task_cost.py`: block pose only, using block xy and wrapped block angle; pusher position and velocity are ignored.
- Geometry run protocol from saved transition-gain script/results: 2000 transitions, 50000 pairs, epsilon 1e-6, approximate matched 5-step action blocks with normalized tolerance 1.5, shared pair IDs across dimensions, eval mode, no retraining, no triangle-bound violations.

## Checkpoint mapping
- state8 (8D): `/home/jw3425/.stable_worldmodel/pusht/state8_baseline_object.ckpt`
- state16 (16D): `/home/jw3425/.stable_worldmodel/pusht/state16_baseline_object.ckpt`
- state32 (32D): `/home/jw3425/.stable_worldmodel/pusht/state32_baseline_object.ckpt`
- state64 (64D): `/home/jw3425/.stable_worldmodel/pusht/state64_baseline_object.ckpt`
- baseline192 (192D): `/home/jw3425/.stable_worldmodel/pusht/baseline_object.ckpt`

## Exact plotted rows

| model | dim | q99 r_true | q99 shortfall | q99 norm error | success |
|---|---:|---:|---:|---:|---:|
| state8 | 8 | 1.30202678274 | 0.1488975961 | 0.273380893826 | 36% |
| state16 | 16 | 1.21138529698 | 0.058543261668 | 0.159326172214 | 72% |
| state32 | 32 | 1.17383137849 | 0.0415538848628 | 0.153561224105 | 78% |
| state64 | 64 | 1.15913761713 | 0.0458808996421 | 0.150627045357 | 90% |
| baseline192 | 192 | 1.13621077657 | 0.0375287919651 | 0.127227984442 | 96% |

## Caption
Independently trained PushT LeWorldModel latent-width configurations. Panel (a) reports the 99th percentile of the encoded true transition expansion; panel (b) reports the 99th percentile of the realised expansion shortfall; panel (c) reports the 99th percentile scale-normalised pair prediction error; panel (d) reports PushT task success. Geometry diagnostics use matched continuous 5-step action blocks with tolerance 1.5 and the same sampled transition and pair IDs across dimensions. These are sampled high-tail diagnostics rather than estimates of the global supremum. The latent-width sweep also changes predictor/projector width through `wm.embed_dim`, so the figure should not be read as isolating nominal dimension alone.
