import argparse
from Model import VisualCausalFlow
import os
from config import AppleConfig, MetaWorldConfig
import torch
import torch.nn as nn
import torch.optim as optim
from visualize import Visualizer
from data_generation import AppleGripperDataGenerator, MetaWorldDataGenerator

def get_args():
    parser = argparse.ArgumentParser(description="ACLF Latent Planning")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--data_type", type=str, default="visual", choices=["visual", "state"])
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--load", action="store_true", help="Whether to load a pre-trained model")
    parser.add_argument("--env", type=str, default="AppleGripper", choices=["AppleGripper", "MetaWorld"], help="Which environment to use for data generation")
    return parser.parse_args()
if __name__ == "__main__":
    args = get_args()
    
    if args.env == "AppleGripper":
        print(f"[Info]: Using AppleGripper environment for data generation.")
        cfg = AppleConfig()
        cfg.ModelConfig.action_dim = args.action_dim
        cfg.ModelConfig.data_type = args.data_type
        data_gen = AppleGripperDataGenerator(cfg=cfg)
    else:
        print(f"[Info]: Using MetaWorld environment for data generation.")
        cfg = MetaWorldConfig()
        cfg.ModelConfig.data_type = args.data_type
        data_gen = MetaWorldDataGenerator(cfg=cfg)

    cfg.ModelConfig.epochs = args.epochs  
    
    print(f"dataType is{cfg.ModelConfig.data_type}")
    if cfg.ModelConfig.data_type == "visual" and cfg.ModelConfig.encoder_type == "identity":
        raise ValueError("Identity encoder cannot be used with image data. Please choose a different encoder type.")
    if cfg.ModelConfig.data_type == "state" and isinstance(cfg,AppleConfig):
        cfg.ModelConfig.encoder_type = "identity"
    model = VisualCausalFlow(cfg=cfg)
    vis = Visualizer(model, cfg=cfg) 


    # ==========================================
    #                 Training Loop
    # ==========================================
    if args.load and os.path.exists("aclf_model.pth"):
        model.load_state_dict(torch.load("aclf_model.pth"))
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        it, inxt, at = data_gen.generate_data()
        store_rank = [];store_loss = []
        print(f"[INfo] shape of it is{it.shape}, and shpe of at is {at.shape}")
        print(f"[Info] The encoder type is{cfg.ModelConfig.encoder_type}")
        crit = nn.BCELoss()
        # print(f"[Info] The encoder type is {self.cfg.ModelConfig.encoder_type}")
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            idx = torch.randint(0, cfg.ModelConfig.n_samples, (cfg.ModelConfig.batch_size,))
            batch = (it[idx], at[idx], inxt[idx])
            if cfg.ModelConfig.save_rank:
                rank = model.compute_soft_rank(batch=batch).item()
                store_rank.append(rank)
            loss = model.compute_loss(batch,epoch)
            store_loss.append(loss.item())
            loss.backward(); optimizer.step() # backpropagation
        torch.save(model.state_dict(), "aclf_model.pth")
    if cfg.ModelConfig.latent_dim == 2:
        vis.visualize_astar_distribution()
        vis.visualize_multi_action_fields()
    if cfg.ModelConfig.save_rank:
        vis.visualize_eff_rank(store_loss=store_loss,store_rank=store_rank)