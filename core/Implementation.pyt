import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuroMindRoPE(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE).
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        # dim should be the dimension per head
        self.dim = dim
        
        # Compute the frequency bands
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute frequencies for the maximum sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqss = torch.outer(t, self.inv_freq)
        
        # [max_seq_len, dim // 2] -> [max_seq_len, dim]
        emb = torch.cat((freqss, freqss), dim=-1)
        
        # Cache cos and sin embeddings
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dimensions."""
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch, num_heads, seq_len, head_dim]
        # Return cos and sin sliced to the current sequence length
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Broadcast shapes for [batch, num_heads, seq_len, head_dim]
        # cos/sin shape initially: [seq_len, head_dim] -> unsqueeze to match dimensions
        cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)


class NeuroMindGQA(nn.Module):
    """
    Grouped-Query Attention (GQA) layer matching NeuroMindAI specifications.
    """
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        
        # GQA validation: Q heads must be perfectly divisible by KV heads
        assert num_q_heads % num_kv_heads == 0, "Query heads must be divisible by KV heads."
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim, bias=False)
        
        # RoPE initialization
        self.rope = NeuroMindRoPE(dim=self.head_dim, max_seq_len=max_seq_len)

    def _repeat_kv(self, x: torch.Tensor, rep: int) -> torch.Tensor:
        """Repeats KV heads to match the number of Q heads for broadcasting."""
        if rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        # Expand and reshape to upsample the head dimension
        x = x.unsqueeze(2).expand(batch, num_kv_heads, rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * rep, seq_len, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 1. Project inputs to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 2. Compute and apply Rotary Position Embeddings (RoPE)
        cos, sin = self.rope(q, seq_len)
        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)
        
        # 3. Grouped-Query Expand: Broadcast K and V heads to match Q
        k = self._repeat_kv(k, self.num_queries_per_kv)
        v = self._repeat_kv(v, self.num_queries_per_kv)
        
        # 4. Scaled Dot-Product Attention with Causal Masking
        # Compute raw attention weights: [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create lower-triangular causal mask
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(1) # Broadcast over batch and heads
        
        # Softmax normalize over the keys dimension
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. Multiply by V context vector
        context = torch.matmul(attn_weights, v) # [B, H, S, D]
        
        # 6. Permute back, flatten heads, and project out
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(context)

# --- Verification Run ---
if __name__ == "__main__":
    # Mocking standard hyperparameters
    B, S, D = 2, 64, 512  # Batch=2, Seq_len=64, Embed_dim=512
    Q_H, KV_H = 8, 2       # 8 Query heads, 2 KV heads (GQA Ratio = 4)
    
    layer = NeuroMindGQA(embed_dim=D, num_q_heads=Q_H, num_kv_heads=KV_H)
    sample_input = torch.randn(B, S, D)
    
    output = layer(sample_input)
    print(f"Input Shape:  {sample_input.shape}")
    print(f"Output Shape: {output.shape} (Must match Input Shape)")
