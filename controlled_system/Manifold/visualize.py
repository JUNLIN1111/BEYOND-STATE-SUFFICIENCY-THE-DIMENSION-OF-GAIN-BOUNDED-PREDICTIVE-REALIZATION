import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_generation import AppleGripperDataGenerator,MetaWorldDataGenerator
from planner import LatentPlanner
from config import AppleConfig,MetaWorldConfig

class Visualizer:
    def __init__(self, model,cfg):
        self.model = model
        self.cfg = cfg
        self.data_type = self.cfg.ModelConfig.data_type
        self.points = self.cfg.VisualizerConfig.get_points()
        self.data_gen = AppleGripperDataGenerator(self.cfg) if isinstance(self.cfg,AppleConfig) else MetaWorldDataGenerator(self.cfg)
        def to_tensor(img):
            return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        
        if isinstance(self.cfg,AppleConfig):
            self.visualize_index_0 = 0
            self.visualize_index_1 = 1
        elif isinstance(self.cfg,MetaWorldConfig):
            self.visualize_index_0 = self.cfg.VisualizerConfig.visualize_index_0
            self.visualize_index_1 = self.cfg.VisualizerConfig.visualize_index_1
            
    # def visualize_mean_rank(self,rank_data_set):

    def build_bounds(self):
        """
        I need to build the bounds after training otherwise
        The random encoder will be used for bound building
        """
        if isinstance(self.cfg,AppleConfig) and self.cfg.ModelConfig.data_type == "visual":
            bounds = [torch.tensor(self.data_gen.render_frame(self.points[k])) for k in ["bound1", "bound2", "bound3", "bound4"]]
        if isinstance(self.cfg,MetaWorldConfig):
            bounds = self.integrate_bounds(self.cfg.VisualizerConfig.state_space_range[self.visualize_index_0],self.cfg.VisualizerConfig.state_space_range[self.visualize_index_0])
        print(f"Find the encoder type here:{self.cfg.ModelConfig.encoder_type}")
        if isinstance(self.cfg,MetaWorldConfig):
            self.z_bounds = [torch.tensor(k).float().numpy() for k in bounds]
        elif self.data_type == "visual":
            print(f"shape of bounds[0]:{bounds[0].shape}")
            self.z_bounds = [self.model.encoder(b.float().unsqueeze(0)).detach().numpy()[0] for b in bounds]
        else:
            self.z_bounds = [torch.tensor(self.points[k]).float().numpy() for k in ["bound1", "bound2", "bound3", "bound4"]]  
        self.column = max(abs(z[0]) for z in self.z_bounds)
        self.row = max(abs(z[1]) for z in self.z_bounds)


    
    def integrate_bounds(self,first_group,second_group):
        range_1 = max(first_group)
        range_2 = max(second_group)
        return [[-range_1,-range_2],[-range_1,range_2],[range_1,-range_2],[range_1,range_2]]
    
    def visualize_multi_action_fields(self, res=40):
        """
        Generates a grid of streamplot visualizations with global labels
        and larger individual plots.
        """
        self.model.eval()
        
        # 1. Get actions from Config
        if self.cfg.ModelConfig.action_dim == 1:
            self.action_space = self.cfg.VisualizerConfig.action_space_1d
        elif self.cfg.ModelConfig.action_dim == 4:
            self.action_space = self.cfg.VisualizerConfig.action_space
        else:
            self.action_space = self.cfg.VisualizerConfig.action_space_2d
        
        # Select actions to display
        display_actions = self.action_space[::max(1, len(self.action_space)//6)] if len(self.action_space) > 6 else self.action_space
        num_plots = len(display_actions)
        print(f"display_actions:{display_actions}")
        # 2. Setup Grid: 2 columns wide, rows calculated automatically
        cols = 2
        rows = (num_plots + cols - 1) // cols
        
        # Increase figsize for much larger graphs
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows), sharex=True, sharey=True)
        axes = axes.flatten() # Flatten to iterate easily

        self.build_bounds()
        # Define Latent Space Grid
        x_range = np.linspace(-self.column,self.column,res)
        y_range = np.linspace(-self.row, self.row, res)

        X, Y = np.meshgrid(x_range, y_range)
        
        z_grid = torch.zeros((res**2, self.cfg.ModelConfig.latent_dim)) 
        z_grid[:, self.visualize_index_0] = torch.from_numpy(X.flatten()).float()
        z_grid[:, self.visualize_index_1] = torch.from_numpy(Y.flatten()).float()

        # 3. Plotting loop
        for i in range(len(axes)):
            ax = axes[i]
            if i < num_plots:
                a_tuple = display_actions[i]
                action_tensor = torch.tensor([a_tuple]).float()
                
                with torch.no_grad():
                    inputs = torch.cat([z_grid, action_tensor.repeat(res**2, 1)], dim=-1)
                    if self.cfg.ModelConfig.latent_mode == "vector_field":
                        dz = self.model.vf_net(inputs).numpy()
                        if i == 2:
                            print(f"[Info z is {inputs}]")
                            print(f"[info] dz is {dz} when we use vector field")
                    else:
                        z_nxt = self.model.vf_net(inputs).numpy()
                        dz = z_nxt - z_grid.numpy()
                        if i == 2:
                            print(f"[Info z is {inputs}]")
                            print(f"[info] dz is {dz} when we use normal dynamics")
                DX = dz[:, self.visualize_index_0].reshape(res, res)
                DY = dz[:, self.visualize_index_1].reshape(res, res)
                mag = np.sqrt(DX**2 + DY**2)
                print(f"[Debug] Action: {a_tuple} | Max Vector Magnitude: {mag.max():.4f}")
                # Streamplot
                ax.streamplot(X, Y, DX, DY, color=mag, cmap='viridis', 
                                     linewidth=1.2, density=1.1, arrowsize=1.3)
                
                # Reference diagonal
                ax.plot([-self.column, self.column], [-self.row, self.row], color='gray', linestyle='--', alpha=0.5)
                ax.set_title(f"Action (a) = {a_tuple}", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.2)
            else:
                # Hide empty subplots if num_plots is odd
                ax.axis('off')

        # 4. Global Labels (One X and one Y for the whole figure)
        if isinstance(self.cfg,AppleConfig):
            fig.supxlabel("(z0): Hopefully Arm Latent", fontsize=14)
            fig.supylabel("(z1): Hopefully Apple Latent", fontsize=14)
        else:
            fig.supxlabel(f"(z0): {self.cfg.VisualizerConfig.index_dict[self.cfg.VisualizerConfig.visualize_index_0]}", fontsize=14)
            fig.supylabel(f"(z1): {self.cfg.VisualizerConfig.index_dict[self.cfg.VisualizerConfig.visualize_index_1]}", fontsize=14)
        
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.97]) # Add padding for global labels
        plt.show()
        fig.savefig("multi_action_fields.png", dpi=300, bbox_inches='tight')
    
    def visualize_eff_rank(self,store_loss,store_rank):
        fig, ax = plt.subplots(2, 1, figsize=(6,8))

        # ===== loss =====
        ax[0].plot(store_loss)
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epoch")

        # ===== rank =====
        ax[1].plot(store_rank)
        ax[1].set_title("Soft Rank")
        ax[1].set_xlabel("Epoch")

        plt.tight_layout()
        fig.savefig("Loss and Rank.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_astar_distribution(self):
        p = LatentPlanner(self.model,cfg=self.cfg)
        self.model.eval()

        # ---------------------------------------------------------
        # 1. Apple Gripper World
        # ---------------------------------------------------------
        self.points = self.cfg.VisualizerConfig.get_points()
        rendered = {k: self.data_gen.render_frame(v) for k, v in self.points.items()}

        if self.cfg.ModelConfig.data_type == "visual":
            s = torch.from_numpy(rendered["start"]).float()
            g = torch.from_numpy(rendered["goal"]).float()
        else:
            s = torch.tensor(self.points["start"]).float()
            g = torch.tensor(self.points["goal"]).float()

        def to_tensor(img):
            return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        
        z_path, a_path, z_s, z_g = p.planner_astar(s, g)
        # ---------------------------------------------------------
        # 2. Fig1 reconstruction and A* path
        # ---------------------------------------------------------
        fig1 = plt.figure(figsize=(18, 9))
        ax1 = fig1.add_subplot(2, 3, 1)
        res = 20
        self.build_bounds()
        #  latent 空间网格（自动适配边界，无需手动写范围）
        x, y = np.meshgrid(np.linspace(-self.column, self.column, res), np.linspace(-self.row, self.row, res))
        z_grid = torch.zeros((res**2, self.cfg.ModelConfig.latent_dim))
        z_grid[:, 0], z_grid[:, 1] = torch.from_numpy(x.flatten()), torch.from_numpy(y.flatten())
        if isinstance(self.cfg,MetaWorldConfig):
            action = torch.tensor([[1,1,1,1]])
        elif self.cfg.ModelConfig.action_dim == 1:
            action = torch.tensor([[2.0]])  
        else:
            action = torch.tensor([[2.0,1]])  
        with torch.no_grad():
            if self.cfg.ModelConfig.latent_mode == "vector_field":
                dz = self.model.vf_net(torch.cat([z_grid, action.repeat(res**2, 1)], dim=-1))
            else:
                z_nxt = self.model.vf_net(torch.cat([z_grid, action.repeat(res**2, 1)], dim=-1))
                dz = z_nxt - z_grid

        ax1.quiver(x, y, dz[:, self.visualize_index_0], dz[:, self.visualize_index_1], color='blue', alpha=0.2)
        p_np = np.array(z_path)
        ax1.plot(p_np[:, self.visualize_index_0], p_np[:, self.visualize_index_1], 'ro-', markersize=3, label='A* Path')
        ax1.scatter(z_s[self.visualize_index_0], z_s[self.visualize_index_1], c='green', s=100, label='Start')
        ax1.scatter(z_g[self.visualize_index_0], z_g[self.visualize_index_1], c='red', marker='x', s=100, label='Goal')
        ax1.set_title(f"Latent Trajectory & Vector Field(action={action.squeeze().tolist()})")
        ax1.legend()

        # subplot: Reconstruction Check (GT vs Predicted)
        if self.cfg.ModelConfig.data_type == "visual":
            random_idx = np.random.randint(0, self.cfg.ModelConfig.batch_size)

            img_t, img_next, a_t = self.data_gen.generate_data()
            img_visual = img_t[random_idx].unsqueeze(0)
            img_visual_next = img_next[random_idx].unsqueeze(0)
            at_visual = a_t[random_idx].unsqueeze(0)
            print(f"[Info] Shape of img_t: {img_visual.shape}, Shape of a_t: {at_visual.shape}")
            with torch.no_grad():
                _, z_next_p = self.model(img_visual, at_visual)
                recon = self.model.decoder(z_next_p)
            ax2 = fig1.add_subplot(2, 3, 2);ax3 = fig1.add_subplot(2, 3, 3);
            if self.cfg.ModelConfig.in_channel==1:
                ax2.imshow(img_visual_next.squeeze(), cmap='gray'); ax2.set_title("GT Next Frame")
                ax3 = fig1.add_subplot(2, 3, 3); ax3.imshow(recon.squeeze(), cmap='gray'); ax3.set_title("Model Prediction")
            elif self.cfg.ModelConfig.in_channel == 3:
                ax2.imshow(img_visual_next.squeeze().permute(1, 2, 0).cpu().numpy()); ax2.set_title("GT Next Frame")
                ax3.imshow(recon.squeeze().permute(1, 2, 0).cpu().numpy()); ax3.set_title("Model Prediction")
            else:
                raise ValueError("[Error] You can only use 1 or 3 channels")
        
        if len(z_path) > 1  and self.cfg.ModelConfig.data_type == "visual":
            show_num = min(5, len(z_path))
            indices = np.linspace(0, len(z_path)-1, show_num, dtype=int)
            for i, idx in enumerate(indices):
                ax = fig1.add_subplot(2, 5, 5 + i + 1)
                zv = torch.tensor(z_path[idx]).unsqueeze(0).float()
                with torch.no_grad():
                    im = self.model.decoder(zv).numpy()
                if self.cfg.ModelConfig.in_channel == 1:
                    im = im.squeeze()
                    ax.imshow(im, cmap='gray')
                else:
                    im = im.squeeze(0).permute(1, 2, 0)
                    ax.imshow(im)
        fig1.savefig("Prediction_result", dpi=300, bbox_inches='tight')
        # ---------------------------------------------------------
        # 3. Latent distribution
        # ---------------------------------------------------------
        # 1. Basic Network
        steps = 15
        visualize_1 = np.linspace(-self.column, self.column, steps)
        print(f"The value of colume:{self.column}")
        visualize_2 = np.linspace(-self.row, self.row, steps)
        print(f"Shape of visualize_1: {visualize_1.shape}")
        latent_grid = np.zeros((steps, steps, 2))
    

        print("[Info] 扫描物理空间映射中...")
        with torch.no_grad():
            # 扫描基础网格
            for i, vis_2 in enumerate(visualize_2):
                for j, vis_1 in enumerate(visualize_1):
                    if self.cfg.ModelConfig.data_type == "visual":
                        img = torch.from_numpy(self.data_gen.render_frame([vis_1, vis_2])).float().unsqueeze(0)
                        latent_grid[i, j] = self.model.encoder(img).squeeze(0).numpy()
                    else:
                        obs = torch.tensor([vis_1, vis_2]).float().unsqueeze(0)
                        latent_grid[i, j] = self.model.encoder(obs).squeeze(0).numpy()

        # 展开基础网格数据
        flat_latent = latent_grid.reshape(-1, 2)
        fir_coords_2d, sec_coords_2d = np.meshgrid(visualize_1, visualize_2)
        flat_fir_pos = fir_coords_2d.flatten()
        flat_sec_pos = sec_coords_2d.flatten()

        # 绘图：白底、简洁风格
        fig2, (ax_arm, ax_apple) = plt.subplots(1, 2, figsize=(20, 8))
        
        for ax, color_data, title, label_txt in zip(
            [ax_arm, ax_apple], 
            [flat_fir_pos, flat_sec_pos], 
            [f"Heatmap: {self.cfg.VisualizerConfig.index_dict[self.cfg.VisualizerConfig.visualize_index_0]}",
            f"Heatmap: {self.cfg.VisualizerConfig.index_dict[self.cfg.VisualizerConfig.visualize_index_1]}"],
            [f"{self.cfg.VisualizerConfig.index_dict[self.cfg.VisualizerConfig.visualize_index_0]}", 
            f"{self.cfg.VisualizerConfig.index_dict[self.cfg.VisualizerConfig.visualize_index_1]}"]
        ):
            # 1. 绘制 15x15 的灰色参考连线
            for i in range(steps):
                ax.plot(latent_grid[i, :, 0], latent_grid[i, :, 1], color='gray', lw=0.5, alpha=0.2)
                ax.plot(latent_grid[:, i, 0], latent_grid[:, i, 1], color='gray', lw=0.5, alpha=0.2)
            
            # 2. 绘制基础网格点 (带黑边)
            sc = ax.scatter(flat_latent[:, 0], flat_latent[:, 1], 
                            c=color_data, cmap='viridis', s=30,
                            edgecolors='black', linewidths=0.5, alpha=0.7, zorder=2)
            
            
            # 4. 侧边栏与辅助线
            plt.colorbar(sc, ax=ax).set_label(label_txt)
            ax.set_title(title)
            ax.set_xlabel("Latent Dimension 1 ($z_1$)")
            ax.set_ylabel("Latent Dimension 2 ($z_2$)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_facecolor('white')

        plt.tight_layout()
        fig2.savefig("latent_space_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()