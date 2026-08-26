from __future__ import annotations

import math
import numpy as np

from cem_score_shock_diagnostic import (
    _append_path_order_rows,
    _distance_pack,
    _pairwise_inversion_rate,
    _rankdata,
    _scaled_sqdist,
    _spearman,
)


def run_sanity_checks() -> None:
    x = np.asarray([1.0, 3.0, 3.0, 10.0])
    ranks = _rankdata(x)
    assert np.allclose(ranks, [1.0, 2.5, 2.5, 4.0]), ranks

    assert math.isclose(_spearman([1, 2, 3], [10, 20, 30]), 1.0)
    assert math.isclose(_spearman([1, 2, 3], [30, 20, 10]), -1.0)
    assert math.isnan(_spearman([1, 1, 1], [1, 2, 3]))

    pred = np.asarray([0.1, 0.2, 0.3])
    true = np.asarray([0.1, 0.3, 0.2])
    assert math.isclose(_pairwise_inversion_rate(pred, true), 1.0 / 3.0)

    a = np.asarray([1.0, 2.0])
    b = np.asarray([3.0, 6.0])
    var = np.asarray([2.0, 8.0])
    assert math.isclose(_scaled_sqdist(a, b, var), 4.0 / 2.0 + 16.0 / 8.0)
    pack = _distance_pack("d", a, b, 2, var)
    assert math.isclose(pack["d_raw_l2sq"], 20.0)
    assert math.isclose(pack["d_per_dim_l2sq"], 10.0)
    assert math.isclose(pack["d_varnorm_l2sq"], 4.0)

    proxy_rows = [
        {"episode_id": 0, "mpc_step_idx": 0, "model": "m", "candidate_idx": 1, "k_raw": 0, "true_task_cost_at_k": 2.0, "s_actual_raw_l2sq": 5.0},
        {"episode_id": 0, "mpc_step_idx": 0, "model": "m", "candidate_idx": 1, "k_raw": 5, "true_task_cost_at_k": 1.0, "s_actual_raw_l2sq": 6.0},
        {"episode_id": 0, "mpc_step_idx": 0, "model": "m", "candidate_idx": 1, "k_raw": 10, "true_task_cost_at_k": 0.5, "s_actual_raw_l2sq": 4.0},
    ]
    reversal = _append_path_order_rows(proxy_rows)[0]
    assert reversal["task_improving_prefix_count"] == 2
    assert reversal["path_order_reversal_task_proxy_count"] == 1
    assert math.isclose(reversal["path_order_reversal_task_proxy_rate"], 0.5)
    print("cem_score_shock metric sanity checks passed")


if __name__ == "__main__":
    run_sanity_checks()
