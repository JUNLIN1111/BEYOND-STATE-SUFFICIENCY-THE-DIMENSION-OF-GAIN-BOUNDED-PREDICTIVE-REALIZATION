# Plannable Latent Dimension Evidence Package

## 1. Theory message

Distance-based latent planners require transition structure to be represented in the latent metric, not merely decoded by downstream dynamics. If the planner cost is Euclidean distance to a latent goal, then transition-far candidate futures that become latent-near create false shortcuts.

## 2. Diagnostic

We build an empirical transition graph from feasible transition edges in offline trajectories, compute shortest-path transition/reachability distances, double-center the distance matrix, and inspect the positive MDS spectrum. `D_plan(q)` is a planner-facing spectral diagnostic: it estimates the Euclidean latent width needed to preserve q fraction of task-induced transition-distance energy.

This is not claimed to be an exact optimal dimension.

## 3. Difference from Isomap

Classical Isomap uses observation-space similarity to build a kNN graph and recover data-manifold geodesics. Here, the primary edges are feasible transition edges from trajectories. kNN edges, when used, are a stitching / robustness device rather than the definition of transition geometry.

## 4. Controlled toy

The controlled toy shows a 2D physical layout with nearby states in adjacent corridors whose transition path must go through a junction. Physical proximity is therefore not transition proximity, and low-dimensional embeddings can create graph-far / Euclidean-near false shortcuts.

Outputs: `toy/controlled_toy_summary.md` and `plots/controlled_toy_*`.

## 5. PushT graph sensitivity

_Not available yet._

Interpretation: exact values can depend on graph construction, but the key question is whether reasonable transition graphs remain far above physical state dimension (7) and local intrinsic dimension (~2). `knn_only_k10` should be treated as an Isomap-style ablation.

## 6. PushT sampling convergence

_Not available yet._

If D_plan grows with sample size, it should be reported as a sampled empirical estimate. The diagnostic remains useful if values consistently stay far above physical/local dimension.

## 7. Candidate mechanism validation

| model | latent_dim | metric | mean | ci95_low | ci95_high |
|---|---|---|---|---|---|
| state8 | 8 | candidate_false_shortcut_rate | 0.026107459118571034 | 0.010957655921950466 | 0.046333441036295255 |
| state8 | 8 | candidate_goal_metric_spearman | 0.29605937649437425 | 0.07473300759206639 | 0.4838571148847636 |
| state8 | 8 | candidate_pairwise_rank_acc | 0.6075406513550947 | 0.5121371979632536 | 0.6877072508407819 |
| state16 | 16 | candidate_false_shortcut_rate | 0.021460571185099402 | 0.00535475941385803 | 0.042409568755514805 |
| state16 | 16 | candidate_goal_metric_spearman | 0.367777818687672 | 0.16217469027969755 | 0.5197903248288218 |
| state16 | 16 | candidate_pairwise_rank_acc | 0.6375964233881813 | 0.5461146843307458 | 0.7072430396420077 |
| state32 | 32 | candidate_false_shortcut_rate | 0.02016624055284282 | 0.004504051985866072 | 0.04056023518937751 |
| state32 | 32 | candidate_goal_metric_spearman | 0.35190807316079276 | 0.15339815351580524 | 0.5060939481693584 |
| state32 | 32 | candidate_pairwise_rank_acc | 0.6311471886296836 | 0.5402523244148392 | 0.7010491582884443 |
| state64 | 64 | candidate_false_shortcut_rate | 0.019530341646014963 | 0.0032644103356890457 | 0.04191132311804058 |
| state64 | 64 | candidate_goal_metric_spearman | 0.37421713147914903 | 0.17739713525882164 | 0.5298890030898199 |
| state64 | 64 | candidate_pairwise_rank_acc | 0.6410078040758782 | 0.5527724270149303 | 0.7114260683003024 |
| baseline192 | 192 | candidate_false_shortcut_rate | 0.01869136881674846 | 0.003432352803067134 | 0.039363969087216615 |
| baseline192 | 192 | candidate_goal_metric_spearman | 0.37650295719571286 | 0.1806047062631531 | 0.5300567289985931 |
| baseline192 | 192 | candidate_pairwise_rank_acc | 0.6424216965278062 | 0.5530769696166853 | 0.71182283750715 |

This is mechanism evidence. The intended claim is that too-small latent width can increase graph-far / score-near candidate errors. Confidence intervals should be respected; do not claim statistical significance if they overlap.

## 8. Prediction-loss negative control

| model | latent_dim |
|---|---|
| state8 | 8 |
| state16 | 16 |
| state32 | 32 |
| state64 | 64 |
| baseline192 | 192 |
| global_k32 | 32 |
| local_k32 | 32 |

Prediction accuracy is necessary but not sufficient. Comparable one-step latent prediction losses can coexist with different candidate-level false shortcuts, ranking fidelity, and closed-loop success.

## 9. Optional second task

_Not available yet._

The second-task spectrum is an offline diagnostic only; it is not closed-loop Cube evaluation.

## 10. Limitations

- Graph construction matters.
- Offline coverage matters.
- `D_plan(q)` is a tradeoff curve, not an exact optimal latent dimension.
- The scope is distance-based latent planning; the claim should not be directly generalized to critic-based world models such as Dreamer.

## Recommended wording

> Plannable latent dimension is a planner-facing spectral diagnostic. It estimates the Euclidean latent width needed to represent task-induced transition geometry as a metric for distance-based planning.
