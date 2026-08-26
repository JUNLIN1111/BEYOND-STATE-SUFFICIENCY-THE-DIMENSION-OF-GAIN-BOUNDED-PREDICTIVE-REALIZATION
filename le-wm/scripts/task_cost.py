from __future__ import annotations

import numpy as np


IMAGE_SIZE = 512.0


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def block_pose_cost(final_state: np.ndarray, goal_state: np.ndarray) -> float:
    """Continuous PushT block-pose proxy cost.

    PushT state layout:
      state[0:2] = agent / pusher position
      state[2:4] = block position
      state[4]   = block angle in [0, 2*pi]
      state[5:7] = agent velocity
    """
    final_state = np.asarray(final_state, dtype=np.float64).reshape(-1)
    goal_state = np.asarray(goal_state, dtype=np.float64).reshape(-1)
    if final_state.shape[0] < 5 or goal_state.shape[0] < 5:
        raise ValueError(
            f"PushT block-pose cost expects state dim >= 5, got {final_state.shape[0]} and {goal_state.shape[0]}"
        )

    pos_dist = np.linalg.norm(final_state[2:4] - goal_state[2:4]) / IMAGE_SIZE
    angle_diff = wrap_to_pi(final_state[4] - goal_state[4])
    angle_dist = abs(angle_diff) / np.pi
    return float(pos_dist + angle_dist)


def task_cost(final_state: np.ndarray, goal_state: np.ndarray) -> float:
    """Return scalar PushT task cost; lower means closer to the goal.

    This implements a continuous block-pose fallback:
        ||block_xy - goal_block_xy|| / 512 + |wrap(theta - theta_goal)| / pi

    It intentionally ignores agent position and agent velocity because PushT
    success is task-grounded in block pose alignment.
    """
    return block_pose_cost(final_state, goal_state)
