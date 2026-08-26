# PushT Graph-Method Comparison

This table separates graph-construction choices from the downstream spectral diagnostic.

| method | status | similarity creates transition edges? | LCC | components | d90 | d95 | d99 | negative energy | semantics |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| temporal_only | missing | no |  |  |  |  |  |  | observed adjacent transitions only; no cross-trajectory stitching |
| temporal_plus_knn_k5 | missing | yes |  |  |  |  |  |  | temporal edges plus kNN state-similarity edges |
| temporal_plus_knn_k10 | missing | yes |  |  |  |  |  |  | temporal edges plus kNN state-similarity edges |
| temporal_plus_knn_k20 | missing | yes |  |  |  |  |  |  | temporal edges plus kNN state-similarity edges |
| knn_only_k10 | missing | yes |  |  |  |  |  |  | Isomap-style state-similarity graph; appendix/sensitivity baseline |
| kmeans_quotient | missing | no |  |  |  |  |  |  | cluster states by k-means; quotient edges induced only by temporal transitions |
| fixed_radius_quotient | missing | no |  |  |  |  |  |  | cluster states by radius cover; quotient edges induced only by temporal transitions |
| kcenter_quotient | missing | no |  |  |  |  |  |  | cluster states by farthest-point centers; quotient edges induced only by temporal transitions |

## Reading the comparison

- `temporal_only` is the cleanest transition graph but is usually disconnected across offline trajectories, so its spectrum can describe only small local components.
- `temporal_plus_knn` and `knn_only` are sensitivity baselines: they allow state-space proximity to create transition edges, which is exactly the possible false-shortcut issue.
- Quotient methods use state similarity only to aggregate nearby samples; all graph edges come from observed temporal transitions.
- Exact D_plan values can move with graph construction. The useful question is whether conservative transition-based quotients remain far above raw physical state dimension and local intrinsic dimension.
