# Parallel-Road Graph-Construction Sanity Check

This appendix toy isolates the graph-construction failure mode: physical proximity does not imply transition reachability.

| method | d_est(s,g) | d_true(s,g) | false shortcut rate | spurious edge rate |
|---|---:|---:|---:|---:|
| oracle_graph | 37.0 | 37.0 | 0.000 | 0.000 |
| knn_only | 0.16 | 37.0 | 0.046 | 0.568 |
| temporal_plus_knn | 0.16 | 37.0 | 0.046 | 0.568 |
| fixed_radius_quotient | 37.0 | 37.0 | 0.000 | 0.000 |

Interpretation:
- kNN-only fails because physical proximity is incorrectly treated as transition reachability.
- temporal+knn can also fail when the kNN shortcut enters the shortest path.
- fixed-radius quotient succeeds here when the radius is smaller than the corridor gap, because similarity is used only for aggregation and edges come only from observed temporal transitions.
