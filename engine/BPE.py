import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# ============================================================================
# PHASE 1: MINIMAL BYTE-PAIR ENCODING (BPE) TOKENIZER ENGINE
# ============================================================================

class NeuroMindTokenizer:
    """
    A lightweight, from-scratch Byte-Pair Encoding (BPE) Tokenizer 
    built to map raw textual inputs down to discrete vocabulary tokens.
    """
    def __init__(self):
        # Initialize base vocabulary with single bytes (0-255)
        self.encoder: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
        
        # Reserved special tokens
        self.pad_token_id = 0
        self.bos_token_id = 256
        self.eos_token_id = 257
        
        self.encoder[b"<pad>"] = self.pad_token_id
        self.encoder[b"<bos>"] = self.bos_token_id
        self.encoder[b"<eos>"] = self.eos_token_id
        
        # Simple sample structural merges for demonstration
        # In practice, these are learned over a corpus via statistical frequency
        sample_merges = [b"th", b"he", b"in", b"an", b"er", b"the", b"and"]
        for idx, merge in enumerate(sample_merges, start=258):
            self.encoder[merge] = idx
            
        self.decoder: Dict[int, bytes] = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Converts raw string text into an ordered list of integer IDs."""
        raw_bytes = text.encode("utf-8")
        ids: List[int] = []
        
        if add_special_tokens:
            ids.append(self.bos_token_id)
            
        # Greedy fallback tokenization matching against learned vocabulary merges
        idx = 0
        while idx < len(raw_bytes):
            matched = False
            # Check largest slice possibilities first (down to 1 byte)
            for width in range(min(5, len(raw_bytes) - idx), 0, -1):
                slice_bytes = raw_bytes[idx : idx + width]
                if slice_bytes in self.encoder:
                    ids.append(self.encoder[slice_bytes])
                    idx += width
                    matched = True
                    break
            if not matched:
                # Absolute single byte structural fallback security
                ids.append(raw_bytes[idx])
                idx += 1
                
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Converts token integer sequences cleanly back into user-readable text."""
        byte_fragments = []
        for token_id in ids:
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            if token_id in self.decoder:
                byte_fragments.append(self.decoder[token_id])
        
        # Re-stitch sequence structure safely, dropping corrupted tail segments
        return b"".join(byte_fragments).decode("utf-8", errors="replace")


# ============================================================================
# PHASE 2: TRANSFORMER CACHE STRUCT & OPTIMIZED INFERENCE HOOKS
# ============================================================================

@dataclass
class KVCacheTensor:
    """Stores the computational history tracking vectors across time."""
    key_states: torch.Tensor   # Shape: [Batch, KV_Heads, Prev_Seq_Len, Head_Dim]
    value_states: torch.Tensor # Shape: [Batch, KV_Heads, Prev_Seq_Len, Head_Dim]


class NeuroMindInferenceModel(nn.Module):
    """
    Wraps the core NeuroMindModel infrastructure to support dynamic, 
    memory-efficient text generation using KV-Caching.
    """
    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core = core_model

    def generate(
        self, 
        prompt_ids: List[int], 
        max_new_tokens: int = 10, 
        temperature: float = 0.7
    ) -> List[int]:
        """
        Generates text autoregressively using a rolling sequence memory cache
        to avoid recomputing key-value states for historical tokens.
        """
        self.core.eval()
        device = next(self.core.parameters()).device
        
        # Format running index context boundaries
        input_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated_sequence = list(prompt_ids)
        
        # Allocate storage bins to act as the KV-Cache system for each layer block
        # Indexed explicitly by block number depth mapping
        kv_cache: Dict[int, KVCacheTensor] = {}
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                seq_len = input_tokens.shape[1]
                
                # --- CACHE-AWARE ENGINE FORWARD PASS BINDING ---
                # Resolve underlying blocks embedding representations
                x = self.core.token_embeddings(input_tokens)
                
                for idx, block in enumerate(self.core.blocks):
                    # 1. Normalize sequence tokens 
                    normed_x = block.attn_norm(x)
                    
                    # 2. Compute projections for the prompt or current step token
                    q = block.attention.q_proj(normed_x).view(1, seq_len, block.attention.num_q_heads, block.attention.head_dim).transpose(1, 2)
                    k = block.attention.k_proj(normed_x).view(1, seq_len, block.attention.num_kv_heads, block.attention.head_dim).transpose(1, 2)
                    v = block.attention.v_proj(normed_x).view(1, seq_len, block.attention.num_kv_heads, block.attention.head_dim).transpose(1, 2)
                    
                    # 3. Dynamic RoPE calculation 
                    cos, sin = block.attention.rope(seq_len)
                    q = block.attention.rope.apply_rope(q, cos, sin)
                    k = block.attention.rope.apply_rope(k, cos, sin)
                    
                    # 4. KV-Cache Read/Write Update
                    if idx in kv_cache:
                        # Append new historical snapshots to avoid processing past contexts from scratch
                        k = torch.cat([kv_cache[idx].key_states, k], dim=-2)
                        v = torch.cat([kv_cache[idx].value_states, v], dim=-2)
                    
                    # Save current computed state back into the cache
                    kv_cache[idx] = KVCacheTensor(key_states=k, value_states=v)
                    
                    # 5. Grouped-Query Attention Repeat Operation
                    k_expanded = block.attention._repeat_kv(k, block.attention.num_queries_per_kv)
                    v_expanded = block.attention._repeat_kv(v, block.attention.num_queries_per_kv)
                    
                    # 6. Score Mapping Matrix calculation
                    scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(block.attention.head_dim)
                    
                    # Causal masking is only needed if we feed in multi-token prompt arrays
                    mask = torch.full((seq_len, k_expanded.shape[-2]), float("-inf"), device=device).triu(diagonal=1)
                    scores = scores + mask.unsqueeze(0).unsqueeze(1)
                    
                    attn_weights = torch.softmax(scores, dim=-1)
                    context = torch.matmul(attn_weights, v_expanded).transpose(1, 2).contiguous().view(1, seq_len, -1)
                    
                    # Residual Feedforward Stack processing
                    attn_out = block.attention.out_proj(context)
                    x = x + attn_out
                    x = x + block.feed_forward(block.mlp_norm(x))
                
                # Extract the output prediction projection from the final token step
                x = self.core.final_norm(x)
                logits = self.core.lm_head(x[:, -1, :]) / max(temperature, 1e-5)
                
                # Sample the next token from the vocabulary distribution
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # Append token to tracking arrays
                generated_sequence.append(next_token_id)
                
                if next_token_id == self.core.config.vocab_size or next_token_id == 257: # Stop if <eos> hit
                    break
                    
                # Update input context vector loop bound to process ONLY the freshly generated token ID next step
                input_tokens = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                
        return generated_sequence


# ============================================================================
# INTERPRETIVE VERIFICATION SIMULATION
# ============================================================================

if __name__ == "__main__":
    from __main__ import NeuroMindModel, NeuroMindConfig
    print("✨ Testing complete Pipeline integration...")
    
    # 1. Initialize custom BPE tokenizer space
    tokenizer = NeuroMindTokenizer()
    print(f"   [Tokenizer] Configured Vocabulary Size: {tokenizer.vocab_size} unique symbols")
    
    # 2. Build model base configuration matching tokenizer dimensions
    config = NeuroMindConfig(vocab_size=tokenizer.vocab_size, max_seq_len=128)
    raw_base_model = NeuroMindModel(config)
    inference_wrapper = NeuroMindInferenceModel(raw_base_model)
    
    # 3. Simulate operational prompt token flow parsing
    prompt = "the inner network"
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"   [Encode] User String Input: '{prompt}' -> Mapped Token sequence ID List: {encoded_prompt}")
    
