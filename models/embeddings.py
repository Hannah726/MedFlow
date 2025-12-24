import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class EventEncoder(nn.Module):
    """
    Encodes event sequences by fusing input, type, and digit-place embeddings.
    Each event is represented as a sequence of 128 indices that are embedded and aggregated.
    """
    
    def __init__(
        self,
        vocab_size_input: int,
        vocab_size_type: int,
        vocab_size_dpe: int,
        d_model: int,
        dropout: float = 0.1,
        fusion_method: str = 'sum'
    ):
        """
        Args:
            vocab_size_input: Size of input vocabulary
            vocab_size_type: Size of type vocabulary
            vocab_size_dpe: Size of digit-place encoding vocabulary
            d_model: Output embedding dimension
            dropout: Dropout probability
            fusion_method: How to fuse embeddings ('sum', 'concat', 'multihead')
        """
        super().__init__()
        
        self.d_model = d_model
        self.fusion_method = fusion_method
        
        if fusion_method == 'sum':
            self.embed_input = nn.Embedding(vocab_size_input, d_model)
            self.embed_type = nn.Embedding(vocab_size_type, d_model)
            self.embed_dpe = nn.Embedding(vocab_size_dpe, d_model)
            self.output_proj = None
            
        elif fusion_method == 'concat':
            emb_dim = d_model // 3
            self.embed_input = nn.Embedding(vocab_size_input, emb_dim)
            self.embed_type = nn.Embedding(vocab_size_type, emb_dim)
            self.embed_dpe = nn.Embedding(vocab_size_dpe, emb_dim)
            self.output_proj = nn.Linear(emb_dim * 3, d_model)
            
        elif fusion_method == 'multihead':
            head_dim = d_model // 3
            self.embed_input = nn.Embedding(vocab_size_input, head_dim)
            self.embed_type = nn.Embedding(vocab_size_type, head_dim)
            self.embed_dpe = nn.Embedding(vocab_size_dpe, head_dim)
            
            self.input_proj = nn.Linear(head_dim, d_model)
            self.type_proj = nn.Linear(head_dim, d_model)
            self.dpe_proj = nn.Linear(head_dim, d_model)
            self.output_proj = None
            
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_idx: torch.Tensor,
        type_idx: torch.Tensor,
        dpe_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_idx: (B, L, 128) input indices
            type_idx: (B, L, 128) type indices
            dpe_idx: (B, L, 128) digit-place indices
        
        Returns:
            event_emb: (B, L, d_model) aggregated event embeddings
        """
        B, L, seq_len = input_idx.shape
        
        input_flat = input_idx.reshape(B * L, seq_len)
        type_flat = type_idx.reshape(B * L, seq_len)
        dpe_flat = dpe_idx.reshape(B * L, seq_len)
        
        e_input = self.embed_input(input_flat)
        e_type = self.embed_type(type_flat)
        e_dpe = self.embed_dpe(dpe_flat)
        
        if self.fusion_method == 'sum':
            fused = e_input + e_type + e_dpe
            event_emb = fused.mean(dim=1)
            
        elif self.fusion_method == 'concat':
            fused = torch.cat([e_input, e_type, e_dpe], dim=-1)
            event_emb = fused.mean(dim=1)
            event_emb = self.output_proj(event_emb)
            
        elif self.fusion_method == 'multihead':
            e_input_agg = e_input.mean(dim=1)
            e_type_agg = e_type.mean(dim=1)
            e_dpe_agg = e_dpe.mean(dim=1)
            
            proj_input = self.input_proj(e_input_agg)
            proj_type = self.type_proj(e_type_agg)
            proj_dpe = self.dpe_proj(e_dpe_agg)
            
            event_emb = proj_input + proj_type + proj_dpe
        
        event_emb = event_emb.reshape(B, L, -1)
        event_emb = self.dropout(event_emb)
        
        return event_emb


class SinusoidalTimeEncoder(nn.Module):
    """
    Sinusoidal positional encoding for continuous time values.
    Maps continuous time in [-1, 1] to multi-frequency sinusoidal features.
    """
    
    def __init__(self, d_time: int = 16, max_period: float = 10000.0):
        """
        Args:
            d_time: Output time embedding dimension (must be even)
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        
        assert d_time % 2 == 0, "d_time must be even"
        
        self.d_time = d_time
        self.max_period = max_period
        
        half_dim = d_time // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, L, 1) continuous time values in [-1, 1]
        
        Returns:
            time_emb: (B, L, d_time) sinusoidal time embeddings
        """
        t = t.squeeze(-1)
        
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        
        sin_part = torch.sin(args)
        cos_part = torch.cos(args)
        
        time_emb = torch.cat([sin_part, cos_part], dim=-1)
        
        return time_emb


class LearnableTimeEncoder(nn.Module):
    """
    Learnable MLP-based time encoder as an alternative to sinusoidal encoding.
    """
    
    def __init__(self, d_time: int = 16, hidden_dim: int = 64):
        """
        Args:
            d_time: Output time embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.d_time = d_time
        
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_time)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, L, 1) continuous time values in [-1, 1]
        
        Returns:
            time_emb: (B, L, d_time) learned time embeddings
        """
        return self.net(t)


class ConditionEncoder(nn.Module):
    """
    Encodes patient-level conditions (gender, age, diagnosis) for conditional generation.
    """
    
    def __init__(
        self,
        num_diag_codes: int,
        d_cond: int = 32,
        d_output: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            num_diag_codes: Number of unique diagnosis codes
            d_cond: Dimension for each condition type embedding
            d_output: Output condition embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_gender = nn.Embedding(2, d_cond)
        
        self.embed_age = nn.Sequential(
            nn.Linear(1, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond)
        )
        
        self.embed_diag = nn.Embedding(num_diag_codes, d_cond)
        
        self.proj = nn.Sequential(
            nn.Linear(d_cond * 3, d_output),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_output, d_output)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, conditions: dict) -> torch.Tensor:
        """
        Args:
            conditions: dict with keys:
                - 'gender': (B,) LongTensor in {0, 1}
                - 'age': (B,) FloatTensor
                - 'diagnosis': (B, max_diags) LongTensor
                - 'diag_mask': (B, max_diags) BoolTensor
        
        Returns:
            cond_emb: (B, d_output) condition embedding
        """
        gender_emb = self.embed_gender(conditions['gender'])
        
        age = conditions['age'].unsqueeze(-1)
        age_emb = self.embed_age(age).squeeze(-1)
        
        diag_embs = self.embed_diag(conditions['diagnosis'])
        diag_mask = conditions['diag_mask'].unsqueeze(-1).float()
        
        diag_emb = (diag_embs * diag_mask).sum(dim=1) / diag_mask.sum(dim=1).clamp(min=1)
        
        cond_emb = torch.cat([gender_emb, age_emb, diag_emb], dim=-1)
        cond_emb = self.proj(cond_emb)
        
        return cond_emb