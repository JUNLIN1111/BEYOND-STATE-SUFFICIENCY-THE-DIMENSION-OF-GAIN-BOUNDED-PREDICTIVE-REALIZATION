# E3 bounded-gain oracle MSE

This experiment requires `cvxpy` because it solves one convex SOCP per
transition action. If the dependency is missing, install it with:

```bash
pip install cvxpy clarabel ecos scs
# or
conda install -c conda-forge cvxpy clarabel ecos scs
```

Do not substitute this experiment with neural-network predictor training.
