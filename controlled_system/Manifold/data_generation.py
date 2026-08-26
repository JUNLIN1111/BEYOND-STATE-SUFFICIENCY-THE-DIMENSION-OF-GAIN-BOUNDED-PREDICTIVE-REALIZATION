import numpy as np
import torch
import metaworld
import gymnasium as gym

class AppleGripperDataGenerator:
    """
        z[0]: arm_x, (both in range [-10, 10])
        z[1]: apple_x (both in range [-10, 10])
        a[0]: arm_velocity ([-2, 2])
        a[1]: gripper_action (0 or 1, 0=open, 1=close) 

        Generate synthetic data for training the model. Each sample consists of a current latent state z_t, an action a_t, and the next latent state z_next.
        The dynamics are defined as follows:
        - The first dimension of the latent state (z[0]) is influenced by the action[0] directly.
        - The second dimension of the latent state (z[1]) is influenced by the action only if the first dimension is close to the second dimension (|z[0] - z[1]| < 1.0) and the action[1] is on
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dim = self.cfg.ModelConfig.action_dim
        self.latent_dim = self.cfg.ModelConfig.latent_dim
        self.n_samples = self.cfg.ModelConfig.n_samples
        self.data_type = self.cfg.ModelConfig.data_type
        self.whether_norm = self.cfg.ModelConfig.whether_norm
        self.render_mode = self.cfg.ModelConfig.render_mode 
        print(f"[Info]: Datagenerator initialized withaction_dim: {self.action_dim}, latent_dim: {self.latent_dim}, n_samples: {self.n_samples}, data_type: {self.data_type}")
    def render_frame(self,z):
        if self.render_mode == "Black":
            # Black and grey render 
            frame = np.zeros((64, 64))
            # Robot arm z[0] is represented as a vertical line, and the apple z[1] is represented as another vertical line. The closer they are, the more likely the apple will be influenced by the arm's action.
            arm_x = int(np.clip((z[0] + 10) * 3.1, 2, 62))
            frame[30:34, arm_x-2:arm_x+2] = 1.0
            apple_x = int(np.clip((z[1] + 10) * 3.1, 2, 62))
            frame[36:40, apple_x-2:apple_x+2] = 0.8
            frame = frame[None, :] # there is no unsqueeze in numpy
            return frame
        
        elif self.render_mode == "RGB":
            # 3 channel image
            frame = np.zeros((3, 64, 64), dtype=np.float32)
            
            # 1. 背景填充浅灰色
            frame[:, :, :] = 0.2

            # 2. 画地板（底部一横条）
            frame[:, 50:, :] = 0.4
            
            # 获取坐标（完全沿用你的公式）
            arm_x = int(np.clip((z[0] + 10) * 3.1, 2, 62))
            apple_x = int(np.clip((z[1] + 10) * 3.1, 2, 62))

            # 机械臂杆子（白色竖条）
            frame[0:3, 20:50, arm_x-2:arm_x+2] = 1.0  # RGB全1 = 白色
            
            # 机械臂抓手（左右小方块）
            frame[0:3, 45:50, arm_x-4:arm_x-2] = 0.9  # 左边抓手
            frame[0:3, 45:50, arm_x+2:arm_x+4] = 0.9  # 右边抓手
            
            # 机械臂底座（红色）
            frame[0, 50:54, arm_x-3:arm_x+3] = 1.0   # 红通道
            frame[1, 50:54, arm_x-3:arm_x+3] = 0.2   # 绿通道
            frame[2, 50:54, arm_x-3:arm_x+3] = 0.2   # 蓝通道

            cx, cy = apple_x, 30  # 苹果中心坐标
            r = 3  # 苹果半径
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        x = cx + dx
                        y = cy + dy
                        if 0 <= x < 64 and 0 <= y < 64:
                            # 苹果主体：红色
                            frame[0, y, x] = 0.9
                            frame[1, y, x] = 0.2
                            frame[2, y, x] = 0.2
            
            # 苹果梗（绿色）
            frame[1, cy-4:cy-2, cx:cx+1] = 0.8

            return frame


    def generate_data(self):
        """
        n_samples: number of samples to generate
        action_dim: dimension of the action space (1 or 2)
        latent_dim: dimension of the latent space (currently fixed to 2 for this dataset)

        return ((batch_size, channel_dim, H, W), (batch_size, action_dim), (batch_size, channel_dim, H, W)) if data_type is "visual"
        return ((batch_size, latent_dim), (batch_size, action_dim), (batch_size, latent_dim)) if data_type is "non_visual"
        """
        # Generate random latent states and actions, then compute the next latent state according to the defined dynamics. Finally, render images if data_type is "visual" or return raw states if data_type is "non_visual".
        z_t = (torch.rand(self.n_samples, 2) * 20) - 10 
        if self.action_dim == 1:
            a_t = (torch.rand(self.n_samples, 1) * 2 - 1) * 2.0
            z_next = z_t.clone()
            z_next[:, 0] += a_t[:, 0] 
            dist = torch.abs(z_t[:, 0] - z_t[:, 1])
            mask = (dist < 1.0).float()
            z_next[:, 1] += a_t[:, 0] * mask 
        elif self.action_dim == 2: 
            a_t = torch.zeros(self.n_samples, 2)
            a_t[:, 0] = (torch.rand(self.n_samples) * 2 - 1) * 2.0
            a_t[:, 1] = torch.randint(0, 2, (self.n_samples,)).float()
            z_next = z_t.clone()
            z_next[:, 0] += a_t[:, 0] # arm_x is always influenced by action[0]
            dist = torch.abs(z_t[:, 0] - z_t[:, 1])
            mask = ((dist < 1.0) & (a_t[:, 1] == 1)).float() # apple_x is influenced by action[0] only if arm_x is close to apple_x and gripper_action is 1 (close)
            z_next[:, 1] += a_t[:, 0] * mask   # apple_x is influenced by action[0] only if arm_x is close to apple_x and gripper_action is 1 (close)
        
        # According to data_type, we can return either visual data (images) or non-visual data (latent states and actions)
        if self.data_type == "visual":
            img_t_list, img_next_list,a_t_list = [], [],[]
            for i in range(self.n_samples):
                img_t_list.append(self.render_frame(z_t[i].numpy()))
                img_next_list.append(self.render_frame(z_next[i].numpy()))
                a_t_list.append(a_t[i].numpy())
            print(f"[INFO] data shape [0] is{np.array(img_t_list).shape}")
            return (torch.from_numpy(np.array(img_t_list)).float(),
                    torch.from_numpy(np.array(img_next_list)).float(),
                    torch.from_numpy(np.array(a_t_list)).float())
        else:   
            obs_t_list,obs_next_list,a_t_list = [],[],[]
            for i in range(self.n_samples):
                obs_t_list.append(z_t[i].numpy())
                obs_next_list.append(z_next[i].numpy())
                a_t_list.append(a_t[i].numpy())
            return (torch.from_numpy(np.array(obs_t_list)).float(),
                    torch.from_numpy(np.array(obs_next_list)).float(),
                    torch.from_numpy(np.array(a_t_list)).float())

class MetaWorldDataGenerator:
    """
    Placeholder for a more complex data generator that interacts with the MetaWorld environment to collect real trajectories of the robotic arm and apple. This would involve resetting the environment, taking random actions, and recording the resulting states and images.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.whether_norm = self.cfg.ModelConfig.whether_norm

        print(f"[Info]: MetaWorldDataGenerator initialized with config: {self.cfg}")     
        
        self.env = gym.make("Meta-World/MT1", env_name="reach-v3",render_mode='rgb_array')
        observation, info = self.env.reset()
        print(f"[Info]: MetaWorld environment reset successful. Initial observation shape: {observation}")
    
    def generate_data(self):
        """
        What MetaWorld environment returns in the observation:

        """
        if self.cfg.ModelConfig.data_type == "state":
            observation, info = self.env.reset() 
        else:
            observation, info = self.env.reset() 
            img = self.env.render()
            img = np.transpose(img, (2, 0, 1))
        fully_markov = self.cfg.ModelConfig.fully_markov
        print(f"debug self.cfg.ModelConfig.data_type: {self.cfg.ModelConfig.data_type}")
        obs_t_list,obs_next_list,a_t_list = [],[],[]
        
        # ==================== Visual =======================
        if self.cfg.ModelConfig.data_type == "visual":
            for t in range(self.cfg.ModelConfig.n_samples):
                action = self.env.action_space.sample()
                obs_t_list.append(img)
                a_t_list.append(action)
                observation, reward, terminated, truncated, info = self.env.step(action)
                img = self.env.render()
                img = np.transpose(img, (2, 0, 1)) # there is no permute for numpy
                obs_next_list.append(img)    
                if terminated or truncated:
                    observation, info = self.env.reset()            
            self.env.close()
        # ================= state ===========================
        else:
            for t in range(self.cfg.ModelConfig.n_samples):
                action = self.env.action_space.sample()
                obs_t_list.append(observation)
                a_t_list.append(action)
                observation, reward, terminated, truncated, info = self.env.step(action)
                obs_next_list.append(observation)
                if terminated or truncated:
                    observation, info = self.env.reset()
                    print(f"Terminated at{t}")
                    # break
                    # print(f"shape of new observation after reset: {observation.shape}")
            self.env.close()
            # obs_t_list = [x*100 for x in obs_t_list] 
            # obs_next_list = [x*100 for x in obs_next_list] 
            # a_t_list = [x*100 for x in a_t_list]
            if self.whether_norm:
                obs_t_np = np.array(obs_t_list)
                obs_next_np = np.array(obs_next_list)
                actions_np = np.array(a_t_list)
                # --- DATA NORMALIZATION ---
                # We calculate stats based on the starting observations (obs_t)
                # and apply them to both obs_t and obs_next to keep the feature space consistent.
                obs_mean = obs_t_np.mean(axis=0)
                obs_std = obs_t_np.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero
                
                action_mean = actions_np.mean(axis=0)
                action_std = actions_np.std(axis=0) + 1e-8

                # Apply Z-score: (x - mean) / std
                obs_t_norm = (obs_t_np - obs_mean) / obs_std
                obs_next_norm = (obs_next_np - obs_mean) / obs_std
                actions_norm = (actions_np - action_mean) / action_std
                return (torch.from_numpy(obs_t_norm).float(),
                torch.from_numpy(obs_next_norm).float(),
                torch.from_numpy(actions_norm).float())
            
        print(f"[Info] shape of obs_t_list: {np.array(obs_t_list).shape}, shape of obs_next_list: {np.array(obs_next_list).shape}, shape of a_t_list: {np.array(a_t_list).shape}")
        return (torch.from_numpy(np.array(obs_t_list)).float(),
                torch.from_numpy(np.array(obs_next_list)).float(),
                torch.from_numpy(np.array(a_t_list)).float())       
    
    def render_frame(self,obs):
        """
        We actually dont need render ourself in MetaWorld
        """
        pass

    