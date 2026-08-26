# ==========================================
#  Definitiion of the VisualCausalFlow model
#  JUnlin 2026-04-20
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.func import jacrev, vmap



class VisualCausalFlow(nn.Module):
    def __init__(self, cfg):
        """
        Here we define a model that encodes 64x64 images into a latent space, predicts the next latent state given an action, and decodes back to images. 
        The encoder can be a simple CNN or a more complex Dreamerv3-like architecture.
        And the vector field network takes the current latent state and action to predict the change in latent state, which is crucial for planning in the latent space.
        """
        super().__init__()
        self.cfg = cfg
        self.obs_dim = cfg.ModelConfig.obs_dim
        self.latent_dim = cfg.ModelConfig.latent_dim
        self.action_dim = cfg.ModelConfig.action_dim
        self.encoder_type = cfg.ModelConfig.encoder_type
        self.decoder_type = cfg.ModelConfig.encoder_type # Normally we want the decoder to mirror the encoder's architecture
        self.lambda_dyn = cfg.ModelConfig.lambda_dyn
        self.lambda_laminar = cfg.ModelConfig.lambda_laminar
        # 1. Encoder
        print(f"[Info] in_channel is {self.cfg.ModelConfig.in_channel}")
        if self.encoder_type == "identity":
            self.encoder = nn.Identity()  # which used for state input, not image
        elif self.encoder_type == "Conv":
            self.encoder = nn.Sequential(
                nn.Conv2d(self.cfg.ModelConfig.in_channel, 16,4,stride=2, padding=1), nn.LeakyReLU(),
                nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.LeakyReLU(),
                nn.Flatten(), nn.Linear(32 * 16 * 16, self.latent_dim)
            )
        elif self.encoder_type == "Dreamerv3":
            self.encoder = nn.Sequential(
                nn.Conv2d(self.cfg.ModelConfig.in_channel, 32, 4, stride=2), nn.ReLU(),    # 64x64 -> 31x31
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),   # 31x31 -> 14x14
                nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),  # 14x14 -> 6x6
                nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(), # 6x6 -> 2x2
                nn.Flatten(),
                nn.Linear(256 * 2 * 2, self.latent_dim)           # Final bottleneck
            )
        elif self.encoder_type == "VAE":
            raise NotImplementedError("VAE encoder is not implemented yet. Please choose 'simple' or 'Dreamerv3' or 'identity'.")
        elif self.encoder_type == "MLP": # which is learned from dreamer v3 
            self.encoder = nn.Sequential(
                nn.Linear(self.obs_dim, 512),nn.LayerNorm(512), # It seems that to use LayerNorm
                nn.SiLU(),
                nn.Linear(512, 512),nn.LayerNorm(512),nn.SiLU(),
                nn.Linear(512, 512),nn.LayerNorm(512),nn.SiLU(),
                nn.Linear(512, self.latent_dim)
            )
        elif self.encoder_type == "RecurrentConv": #RNN only support for the conv now
            pass
        # 2. ======================= Vector Field Network =============================
        self.vf_net = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, 128), nn.Tanh(),
            # The input to the vector field network is the concatenation of the latent state and the action
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, self.latent_dim)
        )
        # 3. ======================= Decoder  ==========================
        if self.encoder_type == "Conv":
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 32 * 16 * 16),
                nn.Unflatten(1, (32, 16, 16)),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.LeakyReLU(),
                nn.ConvTranspose2d(16, self.cfg.ModelConfig.in_channel, 4, stride=2, padding=1), nn.Sigmoid()
            )
        elif self.encoder_type == "identity":
            self.decoder = nn.Identity()  # which used for state input, not image

        elif self.encoder_type == "Dreamerv3":
            self.decoder = nn.Sequential(
            # Start: Input is [Batch, latent_dim]
            nn.Linear(self.latent_dim, 256 * 2 * 2),
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, self.cfg.ModelConfig.in_channel, 4, stride=2, padding=1), nn.Sigmoid()
        )
        elif self.encoder_type == "MLP":
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 512),nn.LayerNorm(512), # It seems that to use LayerNorm
                nn.SiLU(),
                nn.Linear(512, 512),nn.LayerNorm(512),nn.SiLU(),
                nn.Linear(512, 512),nn.LayerNorm(512),nn.SiLU(),
                nn.Linear(512, self.obs_dim)
            )

    def forward(self, obs, a):
        """
        obs:(batch_size,obs_dim)
        a:(batch,action_dim)
        """
        z = self.encoder(obs)
        if self.cfg.ModelConfig.latent_mode == "vector_field":
            dz = self.vf_net(torch.cat([z, a], dim=-1))
            return z, z + dz
        else:
            z_nxt = self.vf_net(torch.cat([z, a], dim=-1))
            return z, z_nxt

    def compute_soft_rank(self, batch):
        """
        使用 torch.func (vmap + jacrev) 向量化高效计算雅可比和有效秩
        """
        obs_t, a_t, obs_next = batch
        
        # 1. 定义一个针对「单个样本」的前向映射函数
        # 这样做是为了配合 vmap 自动处理 batch 维度
        def single_forward(single_obs, single_a):
            # 内部前向传播
            z_t = self.encoder(single_obs.unsqueeze(0)).squeeze(0)
            z_outcome = self.vf_net(torch.cat([z_t, single_a], dim=-1).unsqueeze(0)).squeeze(0)
            
            if self.cfg.ModelConfig.latent_mode == "vector_field":
                return z_outcome + z_t
            else:
                return z_outcome

        # 2. 使用 jacrev 计算对 single_a 的雅可比，并用 vmap 自动并行化整个 batch
        # argnums=1 表示对第二个参数 single_a 求导
        # J_batch 的 shape 将会是: [batch_size, latent_dim, a_dim]
        jacobian_fn = vmap(jacrev(single_forward, argnums=1))
        
        # 运行此函数时，建议包裹 no_grad（如果此 rank 只是为了 log，不需要反向传播）
        # 如果你的 Soft Rank 要作为 Loss 的一部分参与反向传播，请去掉下面这行 context manager
        with torch.no_grad():
            J_batch = jacobian_fn(obs_t, a_t)
            
            # 3. 向量化计算每个样本的奇异值
            # torch.linalg.svdvals 支持 batch 操作
            sigma = torch.linalg.svdvals(J_batch) # shape: [batch_size, min(latent_dim, a_dim)]
            
            # 4. 向量化计算 Effective Rank
            sum_sigma = sigma.sum(dim=-1)
            sum_sigma_sq = sigma.pow(2).sum(dim=-1)
            
            eff_rank_batch = (sum_sigma ** 2) / (sum_sigma_sq + 1e-8)
            
            # 5. 求 Batch 均值
            return eff_rank_batch.mean()

    
    def compute_loss(self, batch,epoch):
        """
        batch: Current A tuple which include (obs_t, a_t, obs_next)
        # loss = a * −lnpϕ​(x∣z,h) + β * MSE(z_pred, z_gt)
        # We minimize the in information content of the predicted image given the latent state (reconstruction loss) 
        # Also ensure that the predicted next latent state is close to the ground truth next latent state (MSE loss).
        """
        obs_t, a_t, obs_next = batch
        
        # 1. Encode 2 images to latent space
        z_t = self.encoder(obs_t)
        z_next_real = self.encoder(obs_next).detach()
        # print(f"[Info] The encoder type is {self.cfg.ModelConfig.encoder_type}")
        # print(f"[Info] data shpe of z_t is {z_t.shape}")
        # calculate the predicted next latent state
        # print(f"[Info] Latent mode is {self.cfg.ModelConfig.latent_mode}")
        if self.cfg.ModelConfig.latent_mode == "vector_field":
            dz = self.vf_net(torch.cat([z_t, a_t], dim=-1))
            z_next_pred = z_t + dz
        else:
            z_next_pred = self.vf_net(torch.cat([z_t, a_t], dim=-1))
        
        # 2. 基础损失：重构 + 动力学
        if self.encoder_type == "MLP":
            recon = self.decoder(z_next_pred)
            recon_loss = nn.functional.mse_loss(recon, obs_next)
            dyn_loss = nn.functional.mse_loss(z_next_pred, z_next_real)
        elif self.encoder_type != "identity":  # only compute reconstruction loss if we have a decoder (i.e., not for state input)
            recon = self.decoder(z_next_pred) # we only use bse loss when we have image to recon
            recon_loss = nn.functional.binary_cross_entropy(recon, obs_next)
            dyn_loss = nn.functional.mse_loss(z_next_pred, z_next_real)
        else:
            recon_loss = 0.0
            dyn_loss = nn.functional.mse_loss(z_next_pred, z_next_real)
        # 3. 硬核改进：Laminar Consistency (层流一致性)
        laminar_loss = 0.0
        # 总损失封装
        if epoch % 200 == 0: print(f"Epoch {epoch} | recon_loss: {recon_loss:.4f} and the dyn_loss is {dyn_loss:.4f}")
        total_loss = recon_loss + self.lambda_dyn * dyn_loss + self.lambda_laminar * laminar_loss
        
        return total_loss

    def _adaptive_local_smooth_loss(self, z_t, a_t, dz, n_neighbors=8, eps=0.05):
        """
        Currently not used in the main loss, 
        but you can call this function from compute_loss() to add an additional regularization term 
        that encourages the vector field to be locally smooth around the training samples. 
        This can help improve generalization and make the learned dynamics more robust to small perturbations in the latent space.
        """
        # 构造虚拟邻居
        # z_t: [B, D] -> [B * n_neighbors, D]
        batch_size = z_t.size(0)
        z_t_rep = z_t.repeat_interleave(n_neighbors, dim=0)
        a_t_rep = a_t.repeat_interleave(n_neighbors, dim=0)
        
        # 加上微小扰动
        z_prime = z_t_rep + torch.randn_like(z_t_rep) * eps
        
        # 计算邻居的位移
        dz_prime = self.vf_net(torch.cat([z_prime, a_t_rep], dim=-1))
        
        # 你的核心逻辑：计算相似度权重
        # 距离越近，权重越高
        dist_z = torch.norm(z_t_rep - z_prime, dim=-1)
        weight = torch.exp(-dist_z) 
        
        # 预测位移的差异
        dz_diff = torch.norm(dz.repeat_interleave(n_neighbors, dim=0) - dz_prime, dim=-1)
        
        # 加权损失
        return torch.mean(weight * dz_diff)


