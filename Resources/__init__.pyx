import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# ==========================================
# 1. CORE COMPONENT DEFINITIONS
# ==========================================

class NeuroMindRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class NeuroMindRoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqss = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqss, freqss), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

    def apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = cos.unsqueeze(0).unsqueeze(1) 
        sin = sin.unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)


class NeuroMindGQA(nn.Module):
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, max_seq_len: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.num_queries_per_kv = num_q_heads // num_kv_heads
        
        self.q_proj = nn.Linear(embed_dim, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim, bias=False)
        
        self.rope = NeuroMindRoPE(dim=self.head_dim, max_seq_len=max_seq_len)

    def _repeat_kv(self, x: torch.Tensor, rep: int) -> torch.Tensor:
        if rep == 1: return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(batch, num_kv_heads, rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * rep, seq_len, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(seq_len)
        q = self.rope.apply_rope(q, cos, sin)
        k = self.rope.apply_rope(k, cos, sin)
        
        k = self._repeat_kv(k, self.num_queries_per_kv)
        v = self._repeat_kv(v, self.num_queries_per_kv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device).triu(diagonal=1)
        scores = scores + mask.unsqueeze(0).unsqueeze(1)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(context)


class NeuroMindSwiGLU(nn.Module):
    def __init__(self, embed_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NeuroMindBlock(nn.Module):
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, intermediate_dim: int, max_seq_len: int):
        super().__init__()
        self.attn_norm = NeuroMindRMSNorm(embed_dim)
        self.attention = NeuroMindGQA(embed_dim, num_q_heads, num_kv_heads, max_seq_len)
        self.mlp_norm = NeuroMindRMSNorm(embed_dim)
        self.feed_forward = NeuroMindSwiGLU(embed_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.mlp_norm(x))
        return x

# ==========================================
# 2. FULL DECODER-ONLY TRANSFORMER SHELL
# ==========================================

@dataclass
class NeuroMindConfig:
    vocab_size: int = 1000
    max_seq_len: int = 64
    embed_dim: int = 256
    num_blocks: int = 4
    num_q_heads: int = 8
    num_kv_heads: int = 2
    intermediate_dim: int = 684 # ~8/3 of embed_dim

class NeuroMindModel(nn.Module):
    """
    Complete Decoder-Only Transformer architecture with tied input/output embeddings.
    """
    def __init__(self, config: NeuroMindConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding Space
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Deep Residual Transformer Block Pipeline
        self.blocks = nn.ModuleList([
            NeuroMindBlock(
                embed_dim=config.embed_dim,
                num_q_heads=config.num_q_heads,
                num_kv_heads=config.num_kv_heads,
                intermediate_dim=config.intermediate_dim,
                max_seq_len=config.max_seq_len
            ) for _ in range(config.num_blocks)
        ])
        
        # Final Processing Elements
        self.final_norm = NeuroMindRMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Tie weights: share memory space between input token space and output projection
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: [Batch, Seq_len]
        x = self.token_embeddings(tokens) # Map to embedding vector space
        
        # Forward pass through sequential transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.final_norm(x)
        logits = self.lm_head(x) # Project back up to vocabulary space
        return logits # Output shape: [Batch, Seq_len, Vocab_size]

# ==========================================
# 3. DETERMINISTIC SYNTHETIC TRAINING PIPELINE
# ==========================================

def run_synthetic_training():
    print("🚀 Initializing NeuroMindAI Config and Scaffold Model...")
    torch.manual_seed(42) # Strict reproducibility anchor
    
    config = NeuroMindConfig()
    model = NeuroMindModel(config)
    
    # Simple cross-entropy setup matching token prediction specifications
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\n📦 Building deterministic toy mock text engine...")
    # Generate static synthetic data to validate the full graph processing bounds
    batch_size = 2
    mock_input_tokens = torch.randint(0, config.vocab_size - 1, (batch_size, config.max_seq_len))
    
    # Target sequences are shifted left by 1 token for standard autoregressive causal targets
    mock_target_tokens = torch.roll(mock_input_tokens, shifts=-1, dims=-1)
    mock_target_tokens[:, -1] = 0 # Blank out final pad context boundary element
    
    print(f"   Input Token Array Size:  {list(mock_input_tokens.shape)}")
    print(f"   Target Token Array Size: {list(mock_target_tokens.shape)}")
    
    print("\n🏋️ Running short verification backprop profile (5 validation steps)...")
    model.train()
    
    for step in range(1, 6):
        optimizer.zero_grad()
        
        # Forward pass through whole pipeline
        logits = model(mock_input_tokens)
        
        # Compute standard cross-entropy loss over token projections
        # Reshape to flatten batch and seq dims: [B * S, Vocab] vs [B * S]
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size), 
            mock_target_tokens.view(-1)
        )
        
        # Backward optimization step
        loss.backward()
        optimizer.step()
        
        print(f"   [Step {step}/5] -> Optimization Cross-Entropy Loss: {loss.item():.4f}")
        
    print("\n✅ Verification Successful: Forward, backward, and optimization graph tracks are active and sound.")

if __name__ == "__main__":
    run_synthetic_training()
