# Predictive-realization experiment notes

## E3 bounded-gain oracle MSE dependency

The E3 oracle experiment solves one convex second-order cone problem per
transition action. It requires `cvxpy` and an SOCP solver.

Install the dependency stack with either:

```bash
pip install cvxpy clarabel ecos scs
```

or:

```bash
conda install -c conda-forge cvxpy clarabel ecos scs
```

Do not replace E3 with penalty-trained neural predictors; the experiment is
defined as a convex finite-state oracle problem.
