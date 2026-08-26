Transition gain implementation note
===================================

Scope
-----

This diagnostic is evaluation-only. It does not modify training, CEM planning,
control evaluation, or the world-model architecture.

2026-07-26 update: the diagnostic now also evaluates the latent predictor
`g_m` by comparing true successor expansion against predicted successor
expansion. The same transition IDs, pair IDs, and action blocks are reused for
all checkpoint dimensions.

Repository inspection
---------------------

1. Encoder class
   - `jepa.py` defines `JEPA`.
   - `JEPA.encode()` converts `info["pixels"]` to float, flattens batch/time,
     applies the image encoder, takes the CLS token, and applies `self.projector`.
   - The latent used by the planner is `info["emb"]`.

2. Latent dimension configuration
   - Default training config: `config/train/lewm.yaml`, `wm.embed_dim`.
   - Model components in `config/train/model/lewm.yaml` use `${wm.embed_dim}` for
     predictor, projector, action encoder, and prediction projection dimensions.
   - Existing scripts also infer latent dimension from encoded output shape when
     checkpoint names are not enough.

3. Existing trained checkpoints
   - Checkpoints are not hard-coded by this diagnostic. The script accepts
     `NAME=PATH` entries and evaluates only paths that exist.
   - Known local naming convention from prior runs includes `state8`, `state16`,
     `state32`, `state64`, and `baseline192`; the script does not fabricate
     missing dimensions.

4. Evaluation replay buffer / transition dataset
   - PushT transitions are read from an HDF5 dataset, normally
     `/tmp/pusht_expert_train.h5`.
   - Expected keys are `pixels`, `action`, `episode_idx`, and `step_idx`.
   - Transitions are constructed only within the same episode.

5. Action space
   - PushT actions are continuous 2D raw actions in the dataset.
   - Because actions are continuous, arbitrary actions are not treated as equal.

6. One-step actions versus action blocks
   - Training uses `config/train/data/pusht.yaml` with `frameskip=5`.
   - Eval uses `config/eval/pusht.yaml` with `plan_config.action_block=5`.
   - Therefore one learned model transition corresponds to a complete 5-raw-step
     action block, not one raw simulator action.

7. Observation preprocessing
   - The diagnostic reuses the repository helper in
     `scripts/planner_success_reference_diagnostic.py`.
   - Images are converted to float CHW, scaled to `[0,1]` if needed, resized to
     `img_size=224`, and normalized with ImageNet mean/std before `model.encode`.

Same-action rule used here
--------------------------

The default same-action mode is exact action-block matching:

1. Build each transition from row `t` to row `t + frameskip` in the same episode.
2. Normalize raw actions using dataset mean/std, matching the rollout diagnostic
   path, then flatten the model action block
   `[a_t, a_{t+1}, ..., a_{t+frameskip-1}]`.
3. Group transitions by the exact flattened action-block bytes.
4. Sample pairs only from the same exact action-block group.

If exact continuous-action matches are too rare, the script has an explicit
opt-in approximation:

`--action_matching_mode approximate --action_matching_threshold <value>`.

Approximate mode normalizes action blocks using sampled transition-block
mean/std and samples pairs whose normalized action-block L2 distance is below
the threshold. The action distance is always logged. This mode is intentionally
not the default.

The replay-buffer fallback is used here because no simulator clone/restore API
is invoked by this script. It keeps the same sampled transition IDs, pair IDs,
and action blocks across all checkpoint dimensions.

Distance convention
-------------------

The reported gain uses Euclidean latent distances:

`r_true = ||phi(x_i^+) - phi(x_j^+)||_2 / max(||phi(x_i) - phi(x_j)||_2, epsilon)`.

The predictor is evaluated with the repository rollout path:

1. encode the full predictor history context;
2. encode the matching history action blocks with `model.action_encoder`;
3. call `model.predict`;
4. apply `transition_bottleneck` exactly as rollout/planning does, if present.

The script reports:

- `r_true`;
- `r_model`;
- `deficit = max(d_next_true - d_next_pred, 0)`;
- `pair_error = sqrt((e_i^2 + e_j^2) / 2)`;
- `lower_bound = deficit / 2`;
- triangle-bound violation rate under `--triangle_tol`.

The script also reports the fraction of sampled pairs with `d_now < epsilon`
for each requested epsilon.
