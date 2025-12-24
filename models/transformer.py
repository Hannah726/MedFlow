import torch
import torch.nn as nn
import math
from typing import Optional

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("WARNING: flash_attn not available, falling back to standard attention")


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for Flash Attention with automatic fallback to standard attention.
    Handles both packed QKV and separate Q, K, V inputs.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.causal = causal
        self.use_flash = FLASH_ATTN_AVAILABLE
        # Always initialize these
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward_flash(
        self,
        qkv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Flash attention forward with packed QKV.
        
        Args:
            qkv: (B, L, 3, n_heads, head_dim)
            key_padding_mask: (B, L) bool tensor, True for valid positions
        
        Returns:
            output: (B, L, d_model)
        """
        B, L, _, H, D = qkv.shape
        
        if key_padding_mask is not None:
            valid_mask = key_padding_mask.float()
            max_seqlen = valid_mask.sum(dim=1).max().int().item()
        else:
            max_seqlen = L
        
        output = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout if self.training else 0.0,
            causal=self.causal
        )
        
        output = output.reshape(B, L, self.d_model)
        
        if key_padding_mask is not None:
            output = output * key_padding_mask.unsqueeze(-1).float()
        
        return output
    
    def forward_standard(
        self,
        qkv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention.
        
        Args:
            qkv: (B, L, 3, n_heads, head_dim)
            key_padding_mask: (B, L) bool tensor, True for valid positions
        
        Returns:
            output: (B, L, d_model)
        """
        B, L, _, H, D = qkv.shape
        
        q, k, v = qkv.unbind(dim=2)
        
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        
        # Mask padding positions in keys
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask  # (B, L), True for padding
            attn_scores = attn_scores.masked_fill(
                attn_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, L)
                float('-inf')
            )
        
        # Causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, L, L)
        
        # Handle NaN from fully masked queries (all keys are padding)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)  # (B, H, L, D)
        output = output.transpose(1, 2).contiguous()  # (B, L, H, D)
        output = output.reshape(B, L, self.d_model)
        
        # Zero out padding positions in output
        if key_padding_mask is not None:
            output = output * key_padding_mask.unsqueeze(-1).float()
        
        return output
    
    def forward(
        self,
        qkv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            qkv: (B, L, 3, n_heads, head_dim)
            key_padding_mask: (B, L) bool tensor, True for valid positions
        """
        if self.use_flash and qkv.is_cuda:
            try:
                return self.forward_flash(qkv, key_padding_mask)
            except Exception as e:
                print(f"Flash attention failed: {e}, falling back to standard attention")
                self.use_flash = False
                return self.forward_standard(qkv, key_padding_mask)
        else:
            return self.forward_standard(qkv, key_padding_mask)


class FlowTransformerBlock(nn.Module):
    """
    Transformer block for flow matching with optional Flash Attention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True,
        causal: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        self.attn = FlashAttentionWrapper(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            causal=causal
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            mask: (B, L) bool tensor, True for valid positions
        
        Returns:
            x: (B, L, d_model)
        """
        residual = x
        x = self.norm1(x)
        
        B, L, D = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        
        attn_out = self.attn(qkv, key_padding_mask=mask)
        attn_out = self.out_proj(attn_out)
        
        x = residual + attn_out
        
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x


class FlowTransformer(nn.Module):
    """
    Full transformer encoder for flow matching.
    Includes flow time conditioning via adaptive modulation.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True,
        causal: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_flash=use_flash,
                causal=causal
            )
            for _ in range(n_layers)
        ])
        
        self.norm_out = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.time_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z: (B, L, d_model) joint state
            s: (B,) flow time in [0, 1]
            mask: (B, L) bool tensor, True for valid positions
        
        Returns:
            v: (B, L, d_model) velocity prediction
        """
        s_emb = self.time_mlp(s.unsqueeze(-1))
        z = z + s_emb.unsqueeze(1)
        
        for block in self.blocks:
            z = block(z, mask)
        
        v = self.norm_out(z)
        
        return v


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on flow time.
    Alternative conditioning mechanism to simple addition.
    """
    
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale_shift = nn.Linear(d_cond, 2 * d_model)
        
        # Initialize: scale part close to 0 (so 1+scale≈1), shift part is 0
        # Use small random initialization for weight, zero for bias
        # nn.init.normal_(self.scale_shift.weight, std=0.02)
        nn.init.zeros_(self.scale_shift.weight)
        # Initialize bias to zero for both scale and shift parts
        nn.init.zeros_(self.scale_shift.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            cond: (B, d_cond)
        
        Returns:
            x: (B, L, d_model) adaptively normalized
        """
        x_norm = self.norm(x)
        
        scale_shift = self.scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        x_cond = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x_cond


class AdaptiveFlowTransformerBlock(nn.Module):
    """
    Transformer block with AdaLN conditioning.
    More sophisticated alternative to simple time conditioning.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        self.attn = FlashAttentionWrapper(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            causal=False
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.ada_norm1 = AdaptiveLayerNorm(d_model, d_model)
        self.ada_norm2 = AdaptiveLayerNorm(d_model, d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            cond: (B, d_model) conditioning vector (e.g., time embedding)
            mask: (B, L) bool tensor
        """
        residual = x
        x = self.ada_norm1(x, cond)
        
        B, L, D = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        
        attn_out = self.attn(qkv, key_padding_mask=mask)
        attn_out = self.out_proj(attn_out)
        
        x = residual + attn_out
        
        residual = x
        x = self.ada_norm2(x, cond)
        x = residual + self.mlp(x)
        
        return x