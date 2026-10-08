"""Byte-level text pretraining and KV-cached generation extensions."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from torch import nn
from torch.utils.data import Dataset

class NeuroMindTokenizer:
    pad_id, bos_id, eos_id, vocab_offset, vocab_size = 0, 1, 2, 3, 259
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [b + self.vocab_offset for b in text.encode("utf-8")]
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids
    def decode(self, ids: List[int]) -> str:
        data = bytearray(int(i) - self.vocab_offset for i in ids
                         if self.vocab_offset <= int(i) < self.vocab_size)
        return bytes(data).decode("utf-8", errors="replace")
    def __len__(self): return self.vocab_size

class TextPretrainingDataset(Dataset):
    def __init__(self, raw_text: str, tokenizer: NeuroMindTokenizer, max_seq_len: int):
        if max_seq_len < 1: raise ValueError("max_seq_len must be positive")
        self.tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        self.max_seq_len = max_seq_len
    def __len__(self): return max(0, len(self.tokens) - self.max_seq_len)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self): raise IndexError(idx)
        chunk = self.tokens[idx:idx + self.max_seq_len + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

@dataclass
class KVCacheTensor:
    key_states: torch.Tensor
    value_states: torch.Tensor

def _apply_rope_at(x, cos, sin, position: int):
    c = cos[position].view(1, 1, 1, -1).repeat_interleave(2, -1)
    s = sin[position].view(1, 1, 1, -1).repeat_interleave(2, -1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * c + rotated * s

class NeuroMindGenerator:
    """Autoregressive generator using the repository's real module names and GQA."""
    def __init__(self, model: nn.Module, tokenizer: NeuroMindTokenizer):
        self.model, self.tokenizer = model, tokenizer

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.8) -> str:
        if max_new_tokens < 0: raise ValueError("max_new_tokens must be non-negative")
        self.model.eval()
        device = next(self.model.parameters()).device
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)[:-1]
        if not ids: ids = [self.tokenizer.bos_id]
        if len(ids) > self.model.config.max_position_embeddings:
            ids = ids[-self.model.config.max_position_embeddings:]
        current = torch.tensor([ids], dtype=torch.long, device=device)
        caches: Dict[int, KVCacheTensor] = {}
        x = self.model.transformer.embed_tokens(current)

        for layer_idx, block in enumerate(self.model.transformer.layers):
            h = block.input_norm(x)
            attn = block.attention
            b, s, _ = h.shape
            q = attn.q_proj(h).view(b, s, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(h).view(b, s, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(h).view(b, s, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            q = attn._rope(q, 0) if hasattr(attn, "_rope") else _apply_rope_at(q, attn.rope_cos, attn.rope_sin, 0)
            k = attn._rope(k, 0) if hasattr(attn, "_rope") else _apply_rope_at(k, attn.rope_cos, attn.rope_sin, 0)
            # Repository RoPE is sequence-based; apply all prompt positions explicitly.
            q = torch.cat([_apply_rope_at(q[:, :, i:i+1], attn.rope_cos, attn.rope_sin, i) for i in range(s)], dim=2)
            k = torch.cat([_apply_rope_at(k[:, :, i:i+1], attn.rope_cos, attn.rope_sin, i) for i in range(s)], dim=2)
            caches[layer_idx] = KVCacheTensor(k, v)
            groups = attn.num_heads // attn.num_kv_heads
            ke = k.repeat_interleave(groups, dim=1) if groups != 1 else k
            ve = v.repeat_interleave(groups, dim=1) if groups != 1 else v
            scores = q @ ke.transpose(-2, -1) / math.sqrt(attn.head_dim)
            causal = torch.triu(torch.ones(s, s, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
            x = x + attn.o_proj(torch.softmax(scores.float(), -1).to(scores.dtype).matmul(ve)
                               .transpose(1, 2).contiguous().view(b, s, attn.hidden_size))
            x = x + block.mlp(block.post_attention_norm(x))
        x = self.model.transformer.norm(x)
        generated = list(ids)
        for _ in range(max_new_tokens):
            logits = self.model.lm_head(x[:, -1, :])
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1).float(), 1)
            token = int(nxt.item())
            generated.append(token)
            if token == self.tokenizer.eos_id: break

            position = len(generated) - 1
            if position >= self.model.config.max_position_embeddings: break
            current = nxt
            for layer_idx, block in enumerate(self.model.transformer.layers):
                h = block.input_norm(x[:, -1:, :]) if layer_idx == 0 else block.input_norm(x)
                # For subsequent layers x is the current token representation after the prior layer.
                if layer_idx == 0:
                    h = block.input_norm(self.model.transformer.embed_tokens(current))
                attn = block.attention
                q = attn.q_proj(h).view(1, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
                k = attn.k_proj(h).view(1, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                v = attn.v_proj(h).view(1, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
                q = _apply_rope_at(q, attn.rope_cos, attn.rope_sin, position)
                k = _apply_rope_at(k, attn.rope_cos, attn.rope_sin, position)
                old = caches[layer_idx]
                k_all, v_all = torch.cat((old.key_states, k), -2), torch.cat((old.value_states, v), -2)
                caches[layer_idx] = KVCacheTensor(k_all, v_all)
                groups = attn.num_heads // attn.num_kv_heads
                ke = k_all.repeat_interleave(groups, 1) if groups != 1 else k_all
                ve = v_all.repeat_interleave(groups, 1) if groups != 1 else v_all
                y = torch.softmax((q @ ke.transpose(-2, -1)) / math.sqrt(attn.head_dim), -1).matmul(ve)
                x = x[:, -1:, :] if layer_idx == 0 else x
                x = x + attn.o_proj(y.transpose(1, 2).contiguous().view(1, 1, attn.hidden_size))
                x = x + block.mlp(block.post_attention_norm(x))
            x = self.model.transformer.norm(x)
        return self.tokenizer.decode(generated)

__all__ = ["NeuroMindTokenizer", "TextPretrainingDataset", "KVCacheTensor", "NeuroMindGenerator"]
