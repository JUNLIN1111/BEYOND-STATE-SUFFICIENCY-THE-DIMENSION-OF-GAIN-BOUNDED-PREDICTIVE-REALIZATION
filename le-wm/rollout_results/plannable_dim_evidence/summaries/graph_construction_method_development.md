# Graph-Construction Method Development

This note records why the plannable-dimension diagnostic moved from raw temporal+KNN graphs toward clustered transition quotients.

## Main-paper toy

The main conceptual toy is now the symmetric crossroad. Its endpoint transition-distance matrix is exactly a four-point equidistant metric, so the positive MDS rank is 3 even though the physical layout is 2D. This cleanly shows that transition distance can require more Euclidean directions than physical coordinates.

## Appendix graph-construction sanity check

The parallel-road toy stays as an appendix sanity check. It is useful for graph construction because adjacent corridors create physical-near but transition-far states. KNN-only and temporal+KNN graphs can add false shortcut edges across the gap; quotient graphs avoid this by not treating state similarity as a transition edge.

## Why not KNN-only?

KNN-only is an Isomap-style baseline, not our transition graph. It can be informative as an ablation, but it directly encodes the assumption under question: nearby observations are nearby under transition reachability.

## Why temporal-only is insufficient

Temporal-only edges are conceptually clean because every edge is an observed feasible one-step transition. In offline multi-episode data, however, temporal-only graphs are usually disconnected, so the resulting spectrum describes only small connected components rather than the global planning geometry.

## Why temporal+KNN was a first approximation

Temporal+KNN connects disconnected offline trajectories, but it does so by adding state-similarity edges as if they were transition edges. That makes it a useful sensitivity baseline but a weaker main diagnostic for a paper about false shortcuts.

## Clustered transition quotient

The quotient graph keeps the useful part of similarity without letting it define reachability. Similarity is used only to aggregate near-duplicate states into clusters or landmarks. The quotient graph adds an edge between two clusters only if the dataset contains an observed temporal transition crossing those clusters.

This should be described as an estimator for transition distances used in the plannable-dimension diagnostic, not as a new state-abstraction algorithm.

## Recommended PushT reporting

- Use clustered quotient graphs as the main PushT transition-distance diagnostic once the fixed-radius and k-center sensitivity curves are stable.
- Keep temporal+KNN and KNN-only as graph-construction sensitivity or appendix baselines.
- Report exact D_plan values cautiously: they depend on graph construction and sampling resolution.
- Emphasize the qualitative question: do conservative transition-based graphs still produce D_plan far above raw 7D physical state and local intrinsic dimension near 2?

## Caveats

Observed temporal transitions reflect dataset coverage. A missing quotient edge does not prove a transition is impossible. Physical-near / transition-far pairs should therefore be called empirical shortcut candidates rather than hard impossibility certificates.
