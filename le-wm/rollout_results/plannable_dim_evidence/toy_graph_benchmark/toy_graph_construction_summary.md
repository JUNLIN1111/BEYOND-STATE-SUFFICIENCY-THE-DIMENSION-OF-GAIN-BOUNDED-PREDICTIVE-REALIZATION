# Toy Graph Construction Benchmark

This benchmark compares graph-construction methods in a toy road environment where physical proximity can create false shortcuts.

The oracle road graph defines true transition distance. Synthetic trajectories are generated only along feasible road edges, and graph methods are evaluated by how well their estimated shortest-path distances recover the oracle distances.

| method | components | spurious edge rate | missing edge rate | Spearman | MAE | rel err | d_est(s,g) | d_true(s,g) | false shortcut | d90/d95/d99 | spectral err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_graph | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 37.0 | 37.0 | 0.000 | 3/4/5 | 0.000 |
| knn_only | 1 | 0.568 | 0.000 | 0.717 | 2.690 | 0.141 | 0.16 | 37.0 | 0.046 | 2/2/3 | 0.562 |
| temporal_only | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 37.0 | 37.0 | 0.000 | 3/4/5 | 0.000 |
| temporal_plus_knn | 1 | 0.568 | 0.000 | 0.717 | 2.690 | 0.141 | 0.16 | 37.0 | 0.046 | 2/2/3 | 0.562 |
| kmeans_quotient | 1 | 0.471 | 0.090 | 0.711 | 3.078 | 0.172 | 2.0 | 37.0 | 0.056 | 2/2/4 | 0.587 |
| fixed_radius_quotient | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 37.0 | 37.0 | 0.000 | 3/4/5 | 0.000 |
| kcenter_quotient | 1 | 0.457 | 0.011 | 0.702 | 2.591 | 0.133 | 0.0 | 37.0 | 0.076 | 2/2/3 | 0.549 |

Notes:
- `oracle_graph` is the gold-standard transition metric on sampled road states.
- `knn_only` is expected to create false shortcut edges because physical closeness does not imply transition reachability.
- `temporal_only` can miss feasible edges if the synthetic trajectory set does not observe them.
- Quotient methods use physical similarity only to aggregate states, then use observed temporal transitions as graph edges.

This benchmark is for method selection and debugging. Do not turn it directly into paper conclusions without checking the generated plots.
