import torch
import heapq
import numpy as np
from config import MetaWorldConfig

class LatentPlanner:
    def __init__(self, model,cfg):
        self.model = model
        self.cfg = cfg 
        self.max_nodes = self.cfg.PlannerConfig.astar_planner.max_nodes
        self.goal_threshold = self.cfg.PlannerConfig.astar_planner.goal_threshold
        if self.cfg.ModelConfig.action_dim == 2:
            self.action_space = self.cfg.PlannerConfig.astar_planner.action_space_A_star_2d
        elif self.cfg.ModelConfig.action_dim == 4:
            self.action_space = self.cfg.PlannerConfig.astar_planner.action_space_A_star_4d
        else:
            self.action_space = self.cfg.PlannerConfig.astar_planner.action_space_A_star_1d

    def planner_astar(self,start, goal):
        """
        A* search in the latent space to find a sequence of actions that takes us from the start image to the goal image.
        The heuristic is the L2 distance in the latent space to the goal.
        It is a offline planner that uses the learned dynamics model (vector field network) to predict the next latent state given the current latent state and an action.
        
        Args:
            start_img: Tensor of shape (64, 64) representing the starting image.
            goal_img: Tensor of shape (64, 64) representing the goal image.
            max_nodes: Maximum number of nodes to explore in the A* search to prevent infinite loops.
        """
        self.model.eval()

        if self.cfg.ModelConfig.data_type == "visual":
            with torch.no_grad():
                z_start = self.model.encoder(start.unsqueeze(0)).squeeze(0)
                z_goal = self.model.encoder(goal.unsqueeze(0)).squeeze(0)
        elif isinstance(self.cfg,MetaWorldConfig) and self.cfg.ModelConfig.data_type=="state":
            with torch.no_grad():
                z_start = self.model.encoder(start)
                z_goal = self.model.encoder(goal)
        else:
            z_start = start.squeeze()
            z_goal = goal.squeeze()
        def h(z): return torch.norm(z - z_goal).item() # heuristic: L2 distance to goal in latent space


        start_tuple = tuple(z_start.numpy())
        pq = [(h(z_start), 0, start_tuple, [start_tuple], [])]
        visited = {}
        
        
        print(f"\n[A* 诊断]: Start from {np.round(start_tuple, 2)} -> Goal {np.round(z_goal.numpy(), 2)}")
        print(f"[Info] Initial distance: {h(z_start):.4f}")
        print(f"[Info] Current Encoder is {self.cfg.ModelConfig.encoder_type}")

        while pq and len(visited) < self.max_nodes:
            f, g, curr_z_tuple, z_path, a_path = heapq.heappop(pq)
            curr_z = torch.tensor(curr_z_tuple).float()
            
            dist = h(curr_z)
            if len(visited) % 500 == 0:
                print(f"[Info]:Explore 节点 {len(visited):4d} | 当前距离 {dist:.4f}")

            # 核心修复：调小阈值
            if dist < 0.15: 
                print(f"===>[Successfully Planned]！总步数: {len(a_path)}")
                for action in a_path:
                    print(f"动作: {action}")
                return z_path, a_path, z_start, z_goal
            
            state_key = tuple(np.round(curr_z_tuple, 2))
            if state_key in visited: continue
            visited[state_key] = g
            
            for a_val in self.action_space:
                with torch.no_grad():
                    a_in = torch.tensor(a_val).float()
                    # print(f"[debug] shape of curr_z: {curr_z.shape}, shape of a_in: {a_in.shape}")
                    if self.cfg.ModelConfig.latent_mode == "vector_field":
                        dz = self.model.vf_net(torch.cat([curr_z, a_in], dim=-1))
                        next_z = curr_z + dz.squeeze(0)
                    else:
                        next_z = self.model.vf_net(torch.cat([curr_z, a_in], dim=-1)).squeeze(0)
                    next_tuple = tuple(next_z.numpy())
                    heapq.heappush(pq, (g + 1 + h(next_z), g + 1, next_tuple, 
                                        z_path + [next_tuple], a_path + [a_val]))
                    
        print("!!! A* 失败：未能在限制内到达终点。")
        for action in a_path:
            print(f"动作: {action}")
        return z_path, a_path, z_start, z_goal