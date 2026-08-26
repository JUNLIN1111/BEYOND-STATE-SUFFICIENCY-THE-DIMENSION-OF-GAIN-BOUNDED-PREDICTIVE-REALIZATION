import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Sequence Data Generation
# ==========================================
def render_frame(z):
    """ Robot pos: z[0], 
        apple pos: z[1]
    """
    frame = np.zeros((64, 64))
    arm_x = int(np.clip((z[0] + 10) * 3.2, 5, 58))
    frame[30:34, arm_x-2:arm_x+2] = 1.0
    apple_x = int(np.clip((z[1] + 10) * 3.2, 5, 58))
    frame[36:40, apple_x-2:apple_x+2] = 0.8
    return frame

def generate_sequence_data(n_sequences=500, seq_len=15):
    """Generates a sequence of frames: Arm moves, then potentially moves apple."""
    seq_imgs, seq_actions = [], []
    for _ in range(n_sequences):
        z = (np.random.rand(2) * 20) - 10
        imgs, actions = [], []
        for _ in range(seq_len):
            a = (np.random.rand(1) * 2 - 1) * 2.0
            imgs.append(render_frame(z))
            actions.append(a)
            # Physics update
            z_next = z.copy()
            z_next[0] += a[0]
            if abs(z[0] - z[1]) < 1.0: z_next[1] += a[0]
            z = z_next
        seq_imgs.append(imgs)
        seq_actions.append(actions)
    
    return (torch.tensor(np.array(seq_imgs)).float().unsqueeze(2), 
            torch.tensor(np.array(seq_actions)).float())

# ==========================================
# 2. Recurrent Visual Model (RSSM-lite)
# ==========================================
class RecurrentCausalFlow(nn.Module):
    def __init__(self, latent_dim=8, action_dim=1, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder: Image -> Stochastic features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2), nn.LeakyReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.LeakyReLU(),
            nn.Flatten(), nn.Linear(32 * 14 * 14, latent_dim)
        )

        # RNN: History (h_t, z_t, a_t) -> h_{t+1}
        self.rnn = nn.GRUCell(latent_dim+ action_dim, hidden_dim)

        # Transition: h_{t+1} -> Predicted z_{t+1} (Prior)
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder: (h_t, z_t) -> Reconstructed Image
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 32 * 16 * 16),
            nn.Unflatten(1, (32, 16, 16)),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def get_initial_state(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim), 
                torch.zeros(batch_size, self.latent_dim))

    def forward(self, obs_seq, action_seq):
        """Processes a sequence of observations and actions."""
        batch_size, seq_len, _, _, _ = obs_seq.shape
        h, z = self.get_initial_state(batch_size)
        h, z = h.to(obs_seq.device), z.to(obs_seq.device)

        recon_loss = 0
        kl_loss = 0

        for t in range(seq_len):
            # 1. Recurrent Update: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
            # Note: At t=0, z and a are zeros or initial values
            inputs = torch.cat([z, action_seq[:, t]], dim=-1)
            h = self.rnn(inputs, h)

            # 2. Prior: Predict z from history (h)
            z_prior = self.transition(h)

            # 3. Posterior: Get z from current observation
            z_post = self.encoder(obs_seq[:, t])

            # 4. Losses
            recon = self.decoder(torch.cat([h, z_post], dim=-1))
            recon_loss += nn.functional.binary_cross_entropy(recon, obs_seq[:, t])
            kl_loss += nn.functional.mse_loss(z_prior, z_post.detach()) # Consistency loss

            z = z_post # Teacher forcing with observations during training

        return recon_loss / seq_len, kl_loss / seq_len

# ==========================================
# 3. Planning in the RNN Latent Space
# ==========================================
def rnn_planner(model, start_img, goal_img, steps=10):
    """Imagination: Planning by rolling out the RNN without new images."""
    model.eval()
    with torch.no_grad():
        h, z = model.get_initial_state(1)
        z = model.encoder(start_img.unsqueeze(0).unsqueeze(0))
        z_goal = model.encoder(goal_img.unsqueeze(0).unsqueeze(0))
        
        planned_imgs = []
        curr_z = z
        curr_h = h

        # Simple Greedy Search (Instead of A* for brevity in RNN rollout)
        for _ in range(steps):
            best_a = None
            min_dist = float('inf')
            
            for a_val in [-2.0, 0.0, 2.0]:
                a_in = torch.tensor([[a_val]]).float()
                test_h = model.rnn(torch.cat([curr_z, a_in], dim=-1), curr_h)
                test_z = model.transition(test_h)
                
                dist = torch.norm(test_z - z_goal).item()
                if dist < min_dist:
                    min_dist = dist
                    best_a = (test_h, test_z, a_val)
            
            curr_h, curr_z, _ = best_a
            recon = model.decoder(torch.cat([curr_h, curr_z], dim=-1))
            planned_imgs.append(recon.squeeze().numpy())
            
        return planned_imgs

# ==========================================
# 4. Training Loop
# ==========================================
if __name__ == "__main__":
    model = RecurrentCausalFlow()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate sequence data (Batch, Time, C, H, W)
    imgs, actions = generate_sequence_data(n_sequences=400, seq_len=10)
    
    print("Training RNN World Model...")
    for epoch in range(501):
        optimizer.zero_grad()
        # Random batch
        idx = torch.randint(0, 400, (32,))
        r_loss, k_loss = model(imgs[idx], actions[idx])
        total_loss = r_loss + 2.0 * k_loss
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Recon Loss: {r_loss.item():.4f} | Consistency: {k_loss.item():.4f}")

    # Visualization of "Imagination"
    s_img = torch.from_numpy(render_frame([0.0, 5.0])).float()
    g_img = torch.from_numpy(render_frame([5.0, 5.0])).float()
    plan = rnn_planner(model, s_img, g_img)
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(plan[i*2], cmap='gray')
        axes[i].set_title(f"Step {i*2}")
    plt.show()
