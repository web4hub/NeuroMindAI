import torch
import torch.nn as nn

class NeuroMindRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) used in NeuroMindAI.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (Gamma) initialized to 1s
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Calculate the mean of squared activations along the last axis
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # 2. Divide by the root square (with an epsilon buffer to prevent division by zero)
        # 3. Multiply by the learnable weight parameter
        return x * torch.rsqrt(variance + self.eps) * self.weight


# --- Complete Block Integration (Interlocking All Built Components) ---

from typing import Any

class NeuroMindBlock(nn.Module):
    """
    A single residual Transformer layer combining RMSNorm, GQA, and SwiGLU.
    """
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, intermediate_dim: int):
        super().__init__()
        # Import the prior layers we wrote
        from __main__ import NeuroMindGQA, NeuroMindSwiGLU
        
        # Attention Sub-Layer Setup
        self.attn_norm = NeuroMindRMSNorm(embed_dim)
        self.attention = NeuroMindGQA(embed_dim, num_q_heads, num_kv_heads)
        
        # MLP Sub-Layer Setup
        self.mlp_norm = NeuroMindRMSNorm(embed_dim)
        self.feed_forward = NeuroMindSwiGLU(embed_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm configuration with Residual Connections
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.mlp_norm(x))
        return x

# --- Verification Run ---
if __name__ == "__main__":
    B, S, D = 2, 64, 512  
    Q_H, KV_H = 8, 2       
    hidden_dim = int(2 * (4 * D) / 3) 
    
    # Initialize the complete model layer block
    block = NeuroMindBlock(embed_dim=D, num_q_heads=Q_H, num_kv_heads=KV_H, intermediate_dim=hidden_dim)
    sample_input = torch.randn(B, S, D)
    
    output = block(sample_input)
    print(f"Full Block Input Shape:  {sample_input.shape}")
    print(f"Full Block Output Shape: {output.shape} (Successfully completed internal forward path)")
  class NeuroMindBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.attn_norm = RMSNorm(config.dim)
        # self.attn = NeuroMindGQA(config)
        # self.mlp_norm = RMSNorm(config.dim)
        # self.mlp = NeuroMindSwiGLU(config.dim, config.hidden_dim)
        pass

    def forward(self, x):
        # x = x + self.attn(self.attn_norm(x))
        # x = x + self.mlp(self.mlp_norm(x))
        return x
