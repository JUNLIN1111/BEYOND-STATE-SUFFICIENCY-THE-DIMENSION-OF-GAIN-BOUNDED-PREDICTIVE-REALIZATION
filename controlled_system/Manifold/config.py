from dataclasses import dataclass
import itertools

# Apple Gripper and MetaWorld Configurations
@dataclass
class AppleConfig:
    class PlannerConfig:
        class astar_planner:
            max_nodes: int = 4000
            goal_threshold: float = 0.15
            # action here is 2 d:
            # dim1: velocity of ee
            # dim2: gripper open/close (1 or -1)
            action_space_A_star_2d: list = [
                (-2.0, 0),
                (-2.0, 1),
                (-1.0, 0),
                (-1.0, 1),
                (-0.5, 0),
                (-0.5, 1),
                (0.5, 0),
                (0.5, 1),
                (1.0, 0),
                (1.0, 1),
                (2.0, 0),
                (2.0, 1)
            ]
            action_space_A_star_1d: list = [(-2.0,), (-1.0,), (-0.5,), (0.5,), (1.0,), (2.0,)]

    class ModelConfig:
        latent_dim: int = 2
        use_vector_field:bool = True
        obs_dim = latent_dim # both is 2 here
        action_dim: int = 1
        hidden_dim: int = 64
        learning_rate: float = 1e-3
        epochs: int = 1000
        batch_size: int = 256
        encoder_type: str = "Dreamerv3"  # "Conv" or "Dreamerv3" or "identity"
        n_samples: int = 3000
        data_type: str = "state"
        lambda_dyn: float = 1
        lambda_laminar: float = 1
        latent_mode = "vector_field" if use_vector_field else "normal_dynamics"   
        whether_norm:bool = False
        # latent_mode: str = "direct_dynamics"
        # "vector_field" or "direct_dynamics"
        render_mode = "Black" # "Black" or "RGB"
        in_channel:int = 1 if render_mode == "Black" else 3
        save_rank = True
        

    class VisualizerConfig:
        def get_points():
            points:dict  = {
            "start": [8.0, 2.0],
            "goal": [6.0, 7.0],
            "bound1": [10.0, 10.0],
            "bound2": [-10.0, -10.0],
            "bound3": [10.0, -10.0],
            "bound4": [-10.0, 10.0]
        }
            return points
        action_space_2d: list = [
                (-2.0, 0),
                (-2.0, 1),
                (-1.0, 0),
                (-1.0, 1),
                (1.0, 0),
                (1.0, 1),
                (2.0, 0),
                (2.0, 1)
            ]
        action_space_1d: list = [(-2.0,),(-0.5,),(0.0,),(0.5,),(2.0,)]
        visualize_index_0 = 0 # choose 2 from 18 dimension,and the choise is above
        visualize_index_1 = 1
        index_dict = {
            0: "apple_pos",
            1: "arm_pos",
        }
# ====================================================
#           Overall Config
# ====================================================
@dataclass
class MetaWorldConfig:
    class PlannerConfig:
        class astar_planner:
            max_nodes: int = 100
            goal_threshold: float = 0.15
            values = [-1.0, -0.5, 0.0, 0.5, 1.0]

            # 生成 5^4 = 625 种discrete组合
            # Probaly explore here because the action space is too large for A*
            action_space_A_star_4d: list = list(itertools.product(values, repeat=4))
            # Sample from expert policy data
            # however the orinal obs is 39 dim, we only use the first 18 dim as the latent state, and the rest 21 dim are masked out as zeros since they are not relevant to the dynamics of the task. The action is 4 dim: [ee_vel_x, ee_vel_y, ee_vel_z, gripper_open_close]
            start: list = [4.31506138e-03, 6.01806846e-01, 1.94536858e-01, 1.00000000e+00,
                            6.74411620e-02, 6.54416656e-01, 1.99935718e-02, -6.16408551e-06,
                            -2.46283731e-06, -9.75341444e-10, 1.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 4.73982371e-03, 6.01394155e-01,
                            1.95107714e-01, 1.00000000e+00, 6.74410013e-02, 6.54416108e-01,
                            2.00000000e-02, 7.62471520e-06, -6.50832487e-06, -1.03485659e-09,
                            1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            -3.33797077e-02, 8.39528808e-01, 1.08390510e-01]

            goal: list = [-2.80518093e-02, 7.97393110e-01, 1.22479684e-01, 9.96649523e-01,
                            6.74417433e-02, 6.54416925e-01, 1.99732912e-02, -1.28435387e-05,
                            1.20823304e-05, -9.98171897e-10, 1.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, -2.76610416e-02, 7.95127812e-01,
                            1.23190796e-01, 9.96642764e-01, 6.74415753e-02, 6.54416747e-01,
                            1.99732932e-02, -8.36150411e-06, 7.85580152e-06, -1.00320213e-09,
                            1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            -3.33797077e-02, 8.39528808e-01, 1.08390510e-01]

    class ModelConfig:
        fully_markov: bool = False
        latent_dim: int = 18
        action_dim: int = 4
        hidden_dim: int = 128
        learning_rate: float = 1e-3
        epochs: int = 5000
        batch_size: int = 256
        data_type: str = "state" # state or visual
        encoder_type: str = "MLP"  # "simple" or "Dreamerv3" or "identity" or MLP
        n_samples: int = 5000
        lambda_dyn: float = 1
        lambda_laminar: float = 1
        latent_mode = "vector_field"
        whether_norm:bool = False
        obs_dim:int = 39 
        queue_len:int = 4 # Dreamerv3 is 4? not sure

        in_channel:int = 3


    class VisualizerConfig:
        # This is suggested range from ChatGPT 
        # I check with env.py I think it is quite reasonable 
        state_space_range = [
                [-0.5, 0.5],   # 0: end_effector_x  (机械臂末端 X)
                [0.3, 1.0],    # 1: end_effector_y  (机械臂末端 Y)
                [0.0, 0.5],    # 2: end_effector_z  (机械臂末端 Z)

                [-1.0, 1.0],   # 3: gripper_openness (夹爪开合程度)

                [-0.5, 0.5],   # 4: object1_x (物体1位置 X)
                [0.3, 1.0],    # 5: object1_y (物体1位置 Y)
                [0.0, 0.3],    # 6: object1_z (物体1位置 Z)

                [-1.0, 1.0],   # 7: object1_quat_w (四元数 w)
                [-1.0, 1.0],   # 8: object1_quat_x
                [-1.0, 1.0],   # 9: object1_quat_y
                [-1.0, 1.0],   # 10: object1_quat_z

                # Object 2 position, if there is no object 2, it gonna be 0
                [-0.5, 0.5],   # 11: object2_x 
                [0.3, 1.0],    # 12: object2_y
                [0.0, 0.3],    # 13: object2_z

                [-1.0, 1.0],   # 14: object2_quat_w
                [-1.0, 1.0],   # 15: object2_quat_x
                [-1.0, 1.0],   # 16: object2_quat_y
                [-1.0, 1.0],   # 17: object2_quat_z
                # Repeat again 
                [-0.5, 0.5],   # 18: end_effector_x  (机械臂末端 X)
                [0.3, 1.0],    # 19: end_effector_y  (机械臂末端 Y)
                [0.0, 0.5],    # 20: end_effector_z  (机械臂末端 Z)

                [-1.0, 1.0],   # 21: gripper_openness (夹爪开合程度)

                [-0.5, 0.5],   # 22: object1_x (物体1位置 X)
                [0.3, 1.0],    # 23: object1_y (物体1位置 Y)
                [0.0, 0.3],    # 24: object1_z (物体1位置 Z)

                [-1.0, 1.0],   # 25: object1_quat_w (四元数 w)
                [-1.0, 1.0],   # 26: object1_quat_x
                [-1.0, 1.0],   # 27: object1_quat_y
                [-1.0, 1.0],   # 28: object1_quat_z

                # Object 2 position, if there is no object 2, it gonna be 0
                [-0.5, 0.5],   # 29: object2_x 
                [0.3, 1.0],    # 30: object2_y
                [0.0, 0.3],    # 31: object2_z

                [-1.0, 1.0],   # 32: object2_quat_w
                [-1.0, 1.0],   # 33: object2_quat_x
                [-1.0, 1.0],   # 34: object2_quat_y
                [-1.0, 1.0],   # 35: object2_quat_z

                # Goal pos ""Retrieves goal position from mujoco properties or instance vars.
                # which return Flat array (3 elements) representing the goal position
                [-0.5, 0.5],   # 36: (goal pos X)
                [0.3, 1.0],    # 37: (goal pos Y)
                [0.0, 0.3]     # 38  (goal pos Z)


            ]
        index_dict = {
            # 当前状态 0~17
            0: "end_effector_x",
            1: "end_effector_y",
            2: "end_effector_z",
            3: "gripper_openness",
            4: "object1_x",
            5: "object1_y",
            6: "object1_z",
            7: "object1_quat_w",
            8: "object1_quat_x",
            9: "object1_quat_y",
            10: "object1_quat_z",
            11: "object2_x",
            12: "object2_y",
            13: "object2_z",
            14: "object2_quat_w",
            15: "object2_quat_x",
            16: "object2_quat_y",
            17: "object2_quat_z",

            18: "prev_end_effector_x",
            19: "prev_end_effector_y",
            20: "prev_end_effector_z",
            21: "prev_gripper_openness",
            22: "prev_object1_x",
            23: "prev_object1_y",
            24: "prev_object1_z",
            25: "prev_object1_quat_w",
            26: "prev_object1_quat_x",
            27: "prev_object1_quat_y",
            28: "prev_object1_quat_z",
            29: "prev_object2_x",
            30: "prev_object2_y",
            31: "prev_object2_z",
            32: "prev_object2_quat_w",
            33: "prev_object2_quat_x",
            34: "prev_object2_quat_y",
            35: "prev_object2_quat_z",

            36: "goal_pos_x",
            37: "goal_pos_y",
            38: "goal_pos_z"
        }

        # [1,1,1,1] [1,1,1,0] [1,1,1-1]
        visualize_index_0 = 0 # choose 2 from 18 dimension,and the choise is above
        visualize_index_1 = 4 
        def get_points():
            return {
                "start": MetaWorldConfig.PlannerConfig.astar_planner.start,
                "goal": MetaWorldConfig.PlannerConfig.astar_planner.goal,
            }
        
        number = [-1,0,1]
        action_space:list = list(itertools.product(number, repeat=4))