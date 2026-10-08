import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# ============================================================================
# EXTENSION 1: FROM-SCRATCH BYTE/CHARACTER TOKENIZER & REAL TEXT PIPELINE
# ============================================================================

class NeuroMindTokenizer:
    """
    A lightweight tokenizer that maps raw textual inputs down to discrete 
    vocabulary tokens based on UTF-8 bytes, ensuring a fixed-size vocabulary.
    """
    def __init__(self):
        # Base vocabulary is mapped to single bytes (0-255)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        
        # Shift character values up by 3 to accommodate special tokens
        self.vocab_offset = 3 
        self.vocab_size = 256 + self.vocab_offset

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Converts a string into an array of integer token IDs."""
        token_ids = [b + self.vocab_offset for b in text.encode("utf-8")]
        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        return token_ids

    def decode(self, ids: List[int]) -> str:
        """Converts an array of token IDs safely back into readable text."""
        byte_list = []
        for token_id in ids:
            if token_id >= self.vocab_offset and token_id < self.vocab_size:
                byte_list.append(token_id - self.vocab_offset)
        return bytes(byte_list).decode("utf-8", errors="replace")


class TextPretrainingDataset(Dataset):
    """
    Splits raw text strings into fixed-length causal contexts 
    where Target = Input shifted left by 1 token.
    """
    def __init__(self, raw_text: str, tokenizer: NeuroMindTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # Encode the entire raw corpus into an unrolled token list
        self.tokens = tokenizer.encode(raw_text, add_special_tokens=False)

    def __len__(self) -> int:
        # Number of sliding training sequences available
        return max(0, len(self.tokens) - self.max_seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract sliding chunk windows
        chunk = self.tokens[idx : idx + self.max_seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long) # Autoregressive shifted target
        return x, y


# ============================================================================
# EXTENSION 2: MEMORY-EFFICIENT INFERENCE ENGINE (KV-CACHING)
# ============================================================================

@dataclass
class KVCacheTensor:
    """Stores the historical context values to prevent O(N^2) computation lag."""
    key_states: torch.Tensor   # [Batch, KV_Heads, Seq_Len, Head_Dim]
    value_states: torch.Tensor # [Batch, KV_Heads, Seq_Len, Head_Dim]


class NeuroMindGenerator:
    """
    Wraps the core NeuroMind model to execute optimized, cache-aware 
    autoregressive token generation.
    """
    def __init__(self, model: nn.Module, tokenizer: NeuroMindTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 20, 
        temperature: float = 0.8
    ) -> str:
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 1. Prepare raw prompt tokens
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)[:-1] # Drop EOS for now
        generated_ids = list(prompt_ids)
        
        # Setup input processing sequence
        current_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        # Dictionary tracking KV historical caches across our sequential blocks
        # Keys map directly to the index position of the block depth layer
        kv_cache: Dict[int, KVCacheTensor] = {}

        for step in range(max_new_tokens):
            seq_len = current_input.shape[1]
            
            # --- CUSTOM CACHE-AWARE MODEL FORWARD OVERRIDE SIMULATION ---
            x = self.model.token_embeddings(current_input)
            
            for idx, block in enumerate(self.model.blocks):
                normed_x = block.attn_norm(x)
                
                # Project Q, K, V
                q = block.attention.q_proj(normed_x).view(1, seq_len, block.attention.num_q_heads, block.attention.head_dim).transpose(1, 2)
                k = block.attention.k_proj(normed_x).view(1, seq_len, block.attention.num_kv_heads, block.attention.head_dim).transpose(1, 2)
                v = block.attention.v_proj(normed_x).view(1, seq_len, block.attention.num_kv_heads, block.attention.head_dim).transpose(1, 2)
                
                # Apply RoPE Positional vectors
                cos, sin = block.attention.rope(seq_len)
                q = block.attention.rope.apply_rope(q, cos, sin)
                k = block.attention.rope.apply_rope(k, cos, sin)
                
                # KV Cache Read / Concatenate Update
                if idx in kv_cache:
                    k = torch.cat([kv_cache[idx].key_states, k], dim=-2)
                    v = torch.cat([kv_cache[idx].value_states, v], dim=-2)
                kv_cache[idx] = KVCacheTensor(key_states=k, value_states=v)
                
                # Expand GQA grouped heads
                k_exp = block.attention._repeat_kv(k, block.attention.num_queries_per_kv)
                v_exp = block.attention._repeat_kv(v, block.attention.num_queries_per_kv)
                
                # Standard causal self attention computation
                scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(block.attention.head_dim)
                mask = torch.full((seq_len, k_exp.shape[-2]), float("-inf"), device=device).triu(diagonal=1)
                scores = scores + mask.unsqueeze(0).unsqueeze(1)
                
                attn_weights = torch.softmax(scores, dim=-1)
                context = torch.matmul(attn_weights, v_exp).transpose(1, 2).contiguous().view(1, seq_len, -1)
                
                x = x + block.attention.out_proj(context)
                x = x + block.feed_forward(block.mlp_norm(x))
            
            x = self.model.final_norm(x)
            logits = self.model.lm_head(x[:, -1, :]) / max(temperature, 1e-5)
            
            # 2. Sample next token using temperature logic
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_token)
            if next_token == self.tokenizer.eos_id:
                break
                
            # 3. CRITICAL CACHE OPTIMIZATION: Update next step input tensor 
            # to contain ONLY the single new token, skipping historical reprocessing.
            current_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

        return self.tokenizer.decode(generated_ids)


# ============================================================================
# RUNNING PIPELINE SIMULATION
# ============================================================================

if __name__ == "__main__":
    # Import our base NeuroMindAI framework constructs
    from __main__ import NeuroMindModel, NeuroMindConfig
    print("🚀 Initializing NeuroMindAI Extensions Pipeline...")
    
    # Setup text processing objects
    tokenizer = NeuroMindTokenizer()
    config = NeuroMindConfig(vocab_size=tokenizer.vocab_size, max_seq_len=32)
    model = NeuroMindModel(config)
    
    # Simple real training string snippet
    corpus = "the neuromind network architecture processes sequences using attention mechanics."
    dataset = TextPretrainingDataset(corpus, tokenizer, max_seq_len=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Run a single pretraining optimization step using real text input
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()
    
    print("📋 Training model on real tokenized text streams...")
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        print(f"   Batch step computed successfully. Causal Loss: {loss.item():.4f}")
        break # Exit early for testing bounds validation
        
    # Execute cache-accelerated text production simulation
    print("\n🔮 Testing optimization engine via text prompt production...")
    generator = NeuroMindGenerator(model, tokenizer)
    generated_text = generator.generate(prompt="the neuromind network", max_new_tokens=10)
    print(f"   Prompt Input: 'the neuromind network'")
    print(f"   Model Output: '{generated_text}'")
