import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# ============================================================================
# 1. CORE COMPONENT DEFINITIONS
# ============================================================================

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

# ============================================================================
# 2. FULL DECODER-ONLY TRANSFORMER SHELL
# ============================================================================

@dataclass
class NeuroMindConfig:
    vocab_size: int = 1000
    max_seq_len: int = 64
    embed_dim: int = 256
    num_blocks: int = 4
    num_q_heads: int = 8
    num_kv_heads: int = 2
    intermediate_dim: int = 684 

class NeuroMindModel(nn.Module):
    def __init__(self, config: NeuroMindConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = nn.ModuleList([
            NeuroMindBlock(
                embed_dim=config.embed_dim,
                num_q_heads=config.num_q_heads,
                num_kv_heads=config.num_kv_heads,
                intermediate_dim=config.intermediate_dim,
                max_seq_len=config.max_seq_len
            ) for _ in range(config.num_blocks)
        ])
        self.final_norm = NeuroMindRMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

# ============================================================================
# 3. BYTE-PAIR ENCODING (BPE) TOKENIZER ENGINE
# ============================================================================

class NeuroMindTokenizer:
    def __init__(self):
        self.encoder: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
        self.pad_token_id = 0
        self.bos_token_id = 256
        self.eos_token_id = 257
        
        self.encoder[b"<pad>"] = self.pad_token_id
        self.encoder[b"<bos>"] = self.bos_token_id
        self.encoder[b"<eos>"] = self.eos_token_id
        
        sample_merges = [b"th", b"he", b"in", b"an", b"er", b"the", b"and"]
        for idx, merge in enumerate(sample_merges, start=258):
            self.encoder[merge] = idx
            
        self.decoder: Dict[int, bytes] = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        raw_bytes = text.encode("utf-8")
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        idx = 0
        while idx < len(raw_bytes):
            matched = False
            for width in range(min(5, len(raw_bytes) - idx), 0, -1):
                slice_bytes = raw_bytes[idx : idx + width]
                if slice_bytes in self.encoder:
                    ids.append(self.encoder[slice_bytes])
                    idx += width
                    matched = True
                    break
            if not matched:
                ids.append(raw_bytes[idx])
                idx += 1
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        byte_fragments = []
        for token_id in ids:
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            if token_id in self.decoder:
                byte_fragments.append(self.decoder[token_id])
        return b"".join(byte_fragments).decode("utf-8", errors="replace")

# ============================================================================
# 4. TRANSFORMER CACHE STRUCT & OPTIMIZED INFERENCE HOOKS
# ============================================================================

@dataclass
class KVCacheTensor:
    key_states: torch.Tensor   
    value_states: torch.Tensor 


class NeuroMindInferenceModel(nn.Module):
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core = core_model

    def generate(
        self, 
        prompt_ids: List[int], 
        max_new_tokens: int = 10, 
        temperature: float = 0.7
    ) -> List[int]:
        self.core.eval()
        device = next(self.core.parameters()).device
        input_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated_sequence = list(prompt_ids)
        kv_cache: Dict[int, KVCacheTensor] = {}
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                seq_len = input_tokens.shape[1]
                x = self.core.token_embeddings(input_tokens)
                
                for idx, block in enumerate(self.core.blocks):
                    normed_x = block.attn_norm(x)
                    
                    q = block.attention.q_proj(normed_x).view(1, seq_len, block.attention.num_q_heads, block.attention.head_dim).transpose(1, 2)
                    k = block.attention.k_proj(normed_x).view(1, seq_len, block.attention.num_kv_heads, block.attention.head_dim).transpose(1, 2)
                    v = block.attention.v_proj(normed_x).view(1, seq_len, block.attention.num_kv_heads, block.attention.head_dim).transpose(1, 2)
                    
                    # Determine current total length for RoPE tracking
                    past_len = kv_cache[idx].key_states.shape[-2] if idx in kv_cache else 0
                    total_len = past_len + seq_len
                    
                    cos, sin = block.attention.rope(total_len)
                    # Apply RoPE slices matching current chunk positions
                    q = block.attention.rope.apply_rope(q, cos[past_len:total_len], sin[past_len:total_len])
                    k = block.attention.rope.apply_rope(k, cos[past_len:total_len], sin[past_len:total_len])
                    
                    if idx in kv_cache:
                        k = torch.cat([kv_cache[idx].key_states, k], dim=-2)
                        v = torch.cat([kv_cache[idx].value_states, v], dim=-2)
                    
                    kv_cache[idx] = KVCacheTensor(key_states=k, value_states=v)
                    
                    k_expanded = block.attention._repeat_kv(k, block.attention.num_queries_per_kv)
                    v_expanded = block.attention._repeat_kv(v, block.attention.num_queries_per_kv)
                    
                    scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(block.attention.head_dim)
                    
                    mask = torch.full((seq_len, k_expanded.shape[-2]), float("-inf"), device=device).triu(diagonal=1 + past_len)
                    scores = scores + mask.unsqueeze(0).unsqueeze(1)
                    
                    attn_weights = torch.softmax(scores, dim=-1)
                    context = torch.matmul(attn_weights, v_expanded).transpose(1, 2).contiguous().view(1, seq_len, -1)
                    
                    attn_out = block.attention.out_proj(context)
                    x = x + attn_out
                    x = x + block.feed_forward(block.mlp_norm(x))
                
                x = self.core.final_norm(x)
                logits = self.core.lm_head(x[:, -1, :]) / max(temperature, 1e-5)
                
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                generated_sequence.append(next_token_id)
                if next_token_id == 257: 
                    break
                    
                input_tokens = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                
        return generated_sequence

# ============================================================================
# EXECUTION ROUTINE
# ============================================================================

print("Executing validation pipeline profile...")
tokenizer = NeuroMindTokenizer()
config = NeuroMindConfig(vocab_size=tokenizer.vocab_size, max_seq_len=128)
model = NeuroMindModel(config)
inference_wrapper = NeuroMindInferenceModel(model)

# Training profile pass verification
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
mock_inputs = torch.randint(0, config.vocab_size - 1, (2, 16))
mock_targets = torch.randint(0, config.vocab_size - 1, (2, 16))

model.train()
optimizer.zero_grad()
loss = F.cross_entropy(model(mock_inputs).view(-1, config.vocab_size), mock_targets.view(-1))
loss.backward()
optimizer.step()
print(f"-> Train pass verified successfully! Initial validation step loss: {loss.item():.4f}")

# Generation pipeline verification
prompt = "the architecture validation"
encoded = tokenizer.encode(prompt)
output_ids = inference_wrapper.generate(encoded, max_new_tokens=5, temperature=0.7)
decoded = tokenizer.decode(output_ids)
print(f"-> Generation wrapper pass verified successfully! Generated output text: '{decoded}'")
