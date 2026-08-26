import gymnasium as gym
import metaworld
import numpy as np
import time
import argparse
# [
#   # -------------------
#   # Robot Hand (4 dims)
#   # -------------------
#   0:  end_effector_x (ee x position)
#   1:  end_effector_y (ee y position)
#   2:  end_effector_z (ee z position)
#   3:  gripper_openness (normalized 0~1: 0=closed, 1=open)

#   # -------------------
#   # Object 0 (7 dims)
#   # -------------------
#   4:  object0_x
#   5:  object0_y
#   6:  object0_z
#   7:  object0_qw (quaternion w)
#   8:  object0_qx (quaternion x)
#   9:  object0_qy (quaternion y)
#   10: object0_qz (quaternion z)

#   # -------------------
#   # Object 1 (7 dims)
#   # -------------------
#   11: object1_x
#   12: object1_y
#   13: object1_z
#   14: object1_qw (quaternion w)
#   15: object1_qx (quaternion x)
#   16: object1_qy (quaternion y)
#   17: object1_qz (quaternion z)
# ]

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="random", help="Choose 'random' for random actions or 'expert' for expert policy actions")
if __name__ == "__main__":
    env = gym.make("Meta-World/MT1", env_name="reach-v3",render_mode="human" )
    observation, info = env.reset()
    args = parser.parse_args()

    for t in range(10000):

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        # time.sleep(0.2) 

        goal_pos = observation[35:38]  # Assuming the goal position is at indices 35, 36, 37

        if truncated:
            env.reset()
            print("[Info] Reset here")
    env.close()