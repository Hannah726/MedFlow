import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

from model.ditfm import MedFlowDiT
from model.fm_criterion import FlowMatchingEngine


class MedFlowDataset(Dataset):
    def __init__(self, latent_path, time_path, split_path, split_type='train'):

        self.latents = np.load(latent_path, mmap_mode='r')
        self.times = np.load(time_path, mmap_mode='r')
        
        # Expecting columns: [subject_id, hadm_id, seed0, seed1, ...]
        split_df = pd.read_csv(split_path)
    
        self.indices = split_df[split_df['seed0'] == split_type].index.tolist()
        
        print(f">>> MedFlow Dataset: {split_type} split loaded with {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map the dataset index to the global matrix index
        global_idx = self.indices[idx]
        
        x_latent = torch.from_numpy(self.latents[global_idx]).float()
        x_time = torch.from_numpy(self.times[global_idx]).float()
        
        # Concatenate into (243, 9)
        return torch.cat([x_latent, x_time], dim=-1)

def train():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 200
    lr = 2e-4  # Slightly higher LR often works better for Flow Matching
    hidden_size = 512
    
    DATA_DIR = "data/processed_12"
    LATENT_DATA = os.path.join(DATA_DIR, "mimiciv_hi_code.npy")
    TIME_DATA = os.path.join(DATA_DIR, "mimiciv_con_time_12.npy")
    SPLIT_CSV = os.path.join(DATA_DIR, "mimiciv_split.csv")
    
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dataset = MedFlowDataset(LATENT_DATA, TIME_DATA, SPLIT_CSV, split_type='train')
    val_dataset = MedFlowDataset(LATENT_DATA, TIME_DATA, SPLIT_CSV, split_type='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    model = MedFlowDiT(hidden_size=hidden_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
 # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        train_losses = []
        
        for batch in train_pbar:
            x_1 = batch.to(device)
            # Placeholder for AdaLN condition (Age, Sex, etc.)
            conds = torch.zeros(x_1.shape[0], hidden_size).to(device)
            
            optimizer.zero_grad()
            loss = FlowMatchingEngine.train_loss(model, x_1, conds)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({"loss": f"{np.mean(train_losses):.6f}"})
        
    
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_1 = batch.to(device)
                conds = torch.zeros(x_1.shape[0], hidden_size).to(device)
                loss = FlowMatchingEngine.train_loss(model, x_1, conds)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f">>> Epoch {epoch+1} Summary: Train Loss: {np.mean(train_losses):.6f} | Val Loss: {avg_val_loss:.6f}")
        
        scheduler.step()

    
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"medflow_epoch_{epoch+1}_val_{avg_val_loss:.4f}.pth")
            torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    train()