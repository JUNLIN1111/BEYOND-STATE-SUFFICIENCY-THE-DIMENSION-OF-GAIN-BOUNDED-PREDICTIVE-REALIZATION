from __future__ import annotations

import math
import numpy as np

from eval_transition_gain import (
    _compute_ratios,
    _l2_distance,
    _pair_arrays,
    _quantile,
    _quantile_label,
    _sample_action_matched_pairs,
    _sample_exact_action_pairs,
)


def run_sanity_checks() -> None:
    a = np.asarray([[0.0, 0.0], [3.0, 4.0]])
    b = np.asarray([[0.0, 0.0], [0.0, 0.0]])
    dist = _l2_distance(a, b)
    assert np.allclose(dist, [0.0, 5.0]), dist

    ratios = _compute_ratios(np.asarray([0.0, 2.0]), np.asarray([1.0, 6.0]), epsilon=0.5)
    assert np.allclose(ratios, [2.0, 3.0]), ratios

    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(_quantile(values, 0.5), 2.5)
    assert math.isclose(_quantile(values, 0.75), 3.25)
    assert _quantile_label(0.95) == "q95"
    assert _quantile_label(0.99) == "q99"
    assert _quantile_label(0.999) == "q999"

    action_blocks = np.asarray(
        [
            [0.0, 0.0],
            [0.01, 0.01],
            [10.0, 10.0],
            [10.01, 10.01],
        ],
        dtype=np.float32,
    )
    left1, right1, dist1 = _sample_action_matched_pairs(action_blocks, 8, 0.05, seed=123)
    left2, right2, dist2 = _sample_action_matched_pairs(action_blocks, 8, 0.05, seed=123)
    assert np.array_equal(left1, left2)
    assert np.array_equal(right1, right2)
    assert np.allclose(dist1, dist2)
    assert np.all(left1 != right1)
    assert np.all(dist1 <= 0.05 + 1e-8)

    action_ids = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.int64)
    exact_left, exact_right, exact_dist = _sample_exact_action_pairs(action_ids, 12, seed=9)
    assert np.all(exact_left != exact_right)
    assert np.all(action_ids[exact_left] == action_ids[exact_right])
    assert np.allclose(exact_dist, 0.0)

    z_now = np.asarray([[0.0], [1.0]])
    z_next_true = np.asarray([[0.0], [4.0]])
    z_next_pred = np.asarray([[0.0], [2.0]])
    arrays = _pair_arrays(
        np.asarray([0]),
        np.asarray([1]),
        z_now,
        z_next_true,
        z_next_pred,
        epsilon=1e-6,
    )
    assert math.isclose(float(arrays["d_now"][0]), 1.0)
    assert math.isclose(float(arrays["d_next_true"][0]), 4.0)
    assert math.isclose(float(arrays["d_next_pred"][0]), 2.0)
    assert math.isclose(float(arrays["deficit"][0]), 2.0)
    assert math.isclose(float(arrays["lower_bound"][0]), 1.0)
    assert float(arrays["pair_error"][0]) >= float(arrays["lower_bound"][0])

    try:
        _sample_action_matched_pairs(action_blocks, 4, 1e-8, seed=0, max_attempt_multiplier=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected no pairs for too-small action threshold.")

    print("transition_gain metric sanity checks passed")


if __name__ == "__main__":
    run_sanity_checks()
