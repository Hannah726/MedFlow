import torch
import torch.nn as nn
from typing import Optional, Dict
from models.embeddings import EventEncoder, SinusoidalTimeEncoder, ConditionEncoder
from models.transformer import FlowTransformer


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        vocab_size_input: int,
        vocab_size_type: int,
        vocab_size_dpe: int,
        d_model: int = 128,
        d_time: int = 16,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_flash: bool = True,
        fusion_method: str = 'sum',
        use_conditions: bool = False,
        num_diag_codes: int = 1000,
        d_cond: int = 32
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_time = d_time
        self.d_joint = d_model + d_time
        self.use_conditions = use_conditions
        
        self.event_encoder = EventEncoder(
            vocab_size_input=vocab_size_input,
            vocab_size_type=vocab_size_type,
            vocab_size_dpe=vocab_size_dpe,
            d_model=d_model,
            dropout=dropout,
            fusion_method=fusion_method
        )
        
        self.time_encoder = SinusoidalTimeEncoder(d_time=d_time)
        
        self.transformer = FlowTransformer(
            d_model=self.d_joint,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            use_flash=use_flash,
            causal=False
        )
        
        if use_conditions:
            self.condition_encoder = ConditionEncoder(
                num_diag_codes=num_diag_codes,
                d_cond=d_cond,
                d_output=self.d_joint,
                dropout=dropout
            )
        else:
            self.condition_encoder = None
    
    def encode_data(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        event_emb = self.event_encoder(
            batch['input'],
            batch['type'],
            batch['dpe']
        )
        
        time_emb = self.time_encoder(batch['time'])
        
        z = torch.cat([event_emb, time_emb], dim=-1)
        
        return z
    
    def forward(
        self,
        z_s: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor,
        conditions: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if conditions is not None and self.condition_encoder is not None:
            cond_emb = self.condition_encoder(conditions)
            z_s = z_s + cond_emb.unsqueeze(1)
        
        v = self.transformer(z_s, s, mask)
        
        return v
    
    def get_num_params(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_module_params(self) -> Dict[str, int]:
        """Return parameter counts per module"""
        params = {
            'event_encoder': sum(p.numel() for p in self.event_encoder.parameters()),
            'time_encoder': sum(p.numel() for p in self.time_encoder.parameters()),
            'transformer': sum(p.numel() for p in self.transformer.parameters()),
        }
        
        if self.condition_encoder is not None:
            params['condition_encoder'] = sum(
                p.numel() for p in self.condition_encoder.parameters()
            )
        
        params['total'] = self.get_num_params()
        return params