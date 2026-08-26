from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from task_cost import task_cost  # noqa: E402


def _episode_first_last_rows(h5: h5py.File, max_episodes: int):
    if "ep_offset" in h5 and "ep_len" in h5:
        offsets = np.asarray(h5["ep_offset"]).reshape(-1)
        lengths = np.asarray(h5["ep_len"]).reshape(-1)
        for episode_id, (offset, length) in enumerate(zip(offsets[:max_episodes], lengths[:max_episodes])):
            if length <= 1:
                continue
            yield int(episode_id), int(offset), int(offset + length - 1)
        return

    episode_key = "episode_idx" if "episode_idx" in h5 else "ep_idx"
    if episode_key not in h5 or "step_idx" not in h5:
        raise KeyError("Expected either ep_offset/ep_len or episode_idx(or ep_idx)/step_idx in HDF5.")
    episode_idx = np.asarray(h5[episode_key]).reshape(-1)
    step_idx = np.asarray(h5["step_idx"]).reshape(-1)
    for episode_id in np.unique(episode_idx)[:max_episodes]:
        rows = np.where(episode_idx == episode_id)[0]
        rows = rows[np.argsort(step_idx[rows])]
        if len(rows) <= 1:
            continue
        yield int(episode_id), int(rows[0]), int(rows[-1])


def main():
    parser = argparse.ArgumentParser(description="Sanity-check PushT task_cost on expert trajectories.")
    parser.add_argument("--dataset", default="/tmp/pusht_expert_train.h5")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--state_key", default="state")
    parser.add_argument("--goal_state_key", default="goal_state")
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as h5:
        if args.state_key not in h5:
            raise KeyError(f"State key '{args.state_key}' not found. Available keys: {list(h5.keys())}")
        states = h5[args.state_key]
        has_goal_state = args.goal_state_key in h5
        goal_states = h5[args.goal_state_key] if has_goal_state else None

        print(f"dataset={args.dataset}")
        print(f"state shape={states.shape}")
        print(f"using explicit goal_state key={has_goal_state}")
        print("episode | start_row | final_row | start_cost | final_cost | improvement")
        print("-" * 82)

        printed = 0
        for episode_id, start_row, final_row in _episode_first_last_rows(h5, args.num_episodes):
            start_state = np.asarray(states[start_row])
            final_state = np.asarray(states[final_row])
            goal_state = np.asarray(goal_states[start_row] if has_goal_state else states[final_row])
            start_cost = task_cost(start_state, goal_state)
            final_cost = task_cost(final_state, goal_state)
            print(
                f"{episode_id:>7} | "
                f"{start_row:>9} | "
                f"{final_row:>9} | "
                f"{start_cost:>10.5f} | "
                f"{final_cost:>10.5f} | "
                f"{start_cost - final_cost:>11.5f}"
            )
            printed += 1
            if printed >= args.num_episodes:
                break


if __name__ == "__main__":
    main()
