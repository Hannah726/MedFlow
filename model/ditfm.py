import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp
from .time_encoder import FourierTimeEncoder

class MedFlowBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=hidden_size * 4)
        
        # AdaLN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c is the modulation signal from conditional Embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        x = x + gate_msa.unsqueeze(1) * self.attn(self.modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MedFlowDiT(nn.Module):
    def __init__(self, hidden_size=512, cond_dim=128):
        super().__init__()
        # First dimension: Fourier time encoding
        self.time_encoder = FourierTimeEncoder(hidden_size)
        # Semantic dimension: 8-dimensional latent code projection
        self.event_embed = nn.Linear(8, hidden_size)
        
        self.joint_proj = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList([MedFlowBlock(hidden_size, 8, hidden_size) for _ in range(12)])
        
        # Encoding for flow time step t (scalar between 0-1)
        self.flow_t_embed = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.final_layer = nn.Linear(hidden_size, 9) # Predict 9-dimensional vector field

    def forward(self, x_t, flow_t, cond_tokens):
        # x_t: (B, 243, 9), flow_t: (B, 1), cond_tokens: (B, hidden_size)
        
        # 1. Extract and fuse spatiotemporal features
        t_feat = self.time_encoder(x_t[:, :, 8:]) # The 9th dimension is continuous time
        e_feat = self.event_embed(x_t[:, :, :8]) # The first 8 dimensions are events
        h = self.joint_proj(t_feat + e_feat)
        
        # 2. Prepare AdaLN modulation vector: fuse flow time step t and experimental condition cond
        c = self.flow_t_embed(flow_t.view(-1, 1)) + cond_tokens
        
        # 3. Transformer parallel evolution
        for block in self.blocks:
            h = block(h, c)
            
        return self.final_layer(h) # Predict vector field v