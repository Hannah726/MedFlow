import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional, Dict
import json


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        lr = float(config.training.lr) if isinstance(config.training.lr, str) else config.training.lr
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=config.training.weight_decay
        )
        
        total_steps = len(train_loader) * config.training.num_epochs
        # 处理 min_lr 可能是字符串的情况，如果 scheduler 不存在则使用默认值
        if hasattr(config.training, 'scheduler') and hasattr(config.training.scheduler, 'min_lr'):
            min_lr = config.training.scheduler.min_lr
            min_lr = float(min_lr) if isinstance(min_lr, str) else min_lr
        else:
            min_lr = 1e-6  # 默认最小学习率
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=int(total_steps * 0.8),
            eta_min=min_lr
        )
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.paths.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        
        self.use_classifier_free = (
            hasattr(config.training, 'classifier_free_dropout') and
            config.training.classifier_free_dropout > 0
        )
        self.cf_dropout = getattr(config.training, 'classifier_free_dropout', 0.0)
    
    def flow_matching_loss(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        mse = (v_pred - v_target) ** 2
        mse_per_event = mse.mean(dim=-1)
        
        # Mask and average over valid events
        masked_mse = mse_per_event * mask.float()

        # Average per sequence
        B = mask.shape[0]
        loss_per_seq = masked_mse.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B,)
        
        # Average over batch
        loss = loss_per_seq.mean()
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif isinstance(batch[key], dict):
                for k in batch[key]:
                    if isinstance(batch[key][k], torch.Tensor):
                        batch[key][k] = batch[key][k].to(self.device)
        
        z_data = self.model.encode_data(batch)
        
        B = z_data.shape[0]
        s = torch.rand(B, device=self.device)
        z_0 = torch.randn_like(z_data)
        
        s_expand = s.view(B, 1, 1)
        z_s = (1 - s_expand) * z_0 + s_expand * z_data
        
        v_target = z_data - z_0
        
        conditions = batch.get('conditions', None)
        
        if self.use_classifier_free and conditions is not None:
            if torch.rand(1).item() < self.cf_dropout:
                conditions = None
        
        v_pred = self.model(z_s, s, batch['mask'], conditions)
        
        loss = self.flow_matching_loss(v_pred, v_target, batch['mask'])

        # print for checking
        if self.global_step % 1000 == 0:
            avg_len = batch['mask'].sum().item() / B
            raw_mse = (v_pred - v_target).pow(2).mean().item()
            print(f"\n[TRAIN DEBUG @ step {self.global_step}]")
            print(f"  Batch size: {B}")
            print(f"  Avg seq len: {avg_len:.1f}")
            print(f"  Raw MSE: {raw_mse:.6f}")
            print(f"  Loss: {loss.item():.6f}")
            
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
                elif isinstance(batch[key], dict):
                    for k in batch[key]:
                        if isinstance(batch[key][k], torch.Tensor):
                            batch[key][k] = batch[key][k].to(self.device)
            
            z_data = self.model.encode_data(batch)
            
            B = z_data.shape[0]
            s = torch.rand(B, device=self.device)
            z_0 = torch.randn_like(z_data)
            
            s_expand = s.view(B, 1, 1)
            z_s = (1 - s_expand) * z_0 + s_expand * z_data
            
            v_target = z_data - z_0
            
            conditions = batch.get('conditions', None)
            v_pred = self.model(z_s, s, batch['mask'], conditions)
            
            loss = self.flow_matching_loss(v_pred, v_target, batch['mask'])
            
            total_loss += loss.item() * B
            total_samples += B

        print(f"\n[VAL DEBUG]")
        print(f"  Avg seq len: {np.mean(all_lens):.1f}")
        print(f"  Avg loss per batch: {np.mean(all_losses):.6f}")
        print(f"  Final weighted loss: {total_loss / total_samples:.6f}")
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def train_epoch(self):
        """Train for one epoch"""
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}/{self.config.training.num_epochs}')
        
        for batch in pbar:
            loss = self.train_step(batch)
            B = batch['mask'].shape[0]  # batch size
            total_loss += loss * B
            total_samples += B
            
            if self.global_step % self.config.training.log_every == 0:
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training: {self.config.experiment.name}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.config.training.num_epochs}")
        print(f"Batch size: {self.config.data.batch_size}")
        print(f"Learning rate: {self.config.training.lr}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            if (epoch + 1) % self.config.training.eval_every == 0:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                
                print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best.pt')
                    print(f"  → Saved best model (val_loss={val_loss:.4f})")
            else:
                # 如果没有验证，添加 None 以保持列表长度一致
                self.val_losses.append(None)
            
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')
            
            self.save_training_history()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best val loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        val_losses_clean = [loss for loss in self.val_losses if loss is not None]
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': val_losses_clean,
            'val_losses_all': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.epoch + 1,
            'global_steps': self.global_step
        }
        
        path = self.log_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)