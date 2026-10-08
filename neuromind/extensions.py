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
    def __init__(self, model: nn.Module, tokenizer: NeuroMindTokenizer):
        self.model, self.tokenizer = model, tokenizer

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.8) -> str:
        if max_new_tokens < 0: raise ValueError("max_new_tokens must be non-negative")
        self.model.eval()
        device = next(self.model.parameters()).device
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)[:-1] or [self.tokenizer.bos_id]
        max_pos = self.model.config.max_position_embeddings
        ids = ids[-max_pos:]
        x = self.model.transformer.embed_tokens(torch.tensor([ids], device=device))
        caches: Dict[int, KVCacheTensor] = {}

        for layer_idx, block in enumerate(self.model.transformer.layers):
            h = block.input_norm(x); a = block.attention
            b, s, _ = h.shape
            q = a.q_proj(h).view(b,s,a.num_heads,a.head_dim).transpose(1,2)
            k = a.k_proj(h).view(b,s,a.num_kv_heads,a.head_dim).transpose(1,2)
            v = a.v_proj(h).view(b,s,a.num_kv_heads,a.head_dim).transpose(1,2)
            q = torch.cat([_apply_rope_at(q[:,:,i:i+1],a.rope_cos,a.rope_sin,i) for i in range(s)],2)
            k = torch.cat([_apply_rope_at(k[:,:,i:i+1],a.rope_cos,a.rope_sin,i) for i in range(s)],2)
            caches[layer_idx] = KVCacheTensor(k,v)
            g = a.num_heads // a.num_kv_heads
            ke = k.repeat_interleave(g,1) if g != 1 else k
            ve = v.repeat_interleave(g,1) if g != 1 else v
            scores = (q @ ke.transpose(-2,-1)) / math.sqrt(a.head_dim)
            mask = torch.triu(torch.ones(s,s,device=device,dtype=torch.bool),1)
            y = torch.softmax(scores.masked_fill(mask,torch.finfo(scores.dtype).min).float(),-1).to(scores.dtype) @ ve
            x = x + a.o_proj(y.transpose(1,2).contiguous().view(b,s,a.hidden_size))
            x = x + block.mlp(block.post_attention_norm(x))
        x = self.model.transformer.norm(x)
        generated = list(ids)

        for _ in range(max_new_tokens):
            logits = self.model.lm_head(x[:,-1,:])
            nxt = logits.argmax(-1,keepdim=True) if temperature <= 0 else torch.multinomial(torch.softmax(logits/temperature,-1).float(),1)
            token = int(nxt.item()); generated.append(token)
            if token == self.tokenizer.eos_id: break
            position = len(generated)-1
            if position >= max_pos: break
            x = self.model.transformer.embed_tokens(nxt)
            for layer_idx, block in enumerate(self.model.transformer.layers):
                h = block.input_norm(x); a = block.attention
                q = a.q_proj(h).view(1,1,a.num_heads,a.head_dim).transpose(1,2)
                k = a.k_proj(h).view(1,1,a.num_kv_heads,a.head_dim).transpose(1,2)
                v = a.v_proj(h).view(1,1,a.num_kv_heads,a.head_dim).transpose(1,2)
                q = _apply_rope_at(q,a.rope_cos,a.rope_sin,position)
                k = _apply_rope_at(k,a.rope_cos,a.rope_sin,position)
                old_cache = caches[layer_idx]
                k_all = torch.cat((old_cache.key_states,k),-2)
                v_all = torch.cat((old_cache.value_states,v),-2)
                caches[layer_idx] = KVCacheTensor(k_all,v_all)
                g = a.num_heads // a.num_kv_heads
                ke = k_all.repeat_interleave(g,1) if g != 1 else k_all
                ve = v_all.repeat_interleave(g,1) if g != 1 else v_all
                y = torch.softmax((q @ ke.transpose(-2,-1) / math.sqrt(a.head_dim)).float(),-1) @ ve
                x = x + a.o_proj(y.transpose(1,2).contiguous().view(1,1,a.hidden_size))
                x = x + block.mlp(block.post_attention_norm(x))
            x = self.model.transformer.norm(x)
        return self.tokenizer.decode(generated)

__all__ = ["NeuroMindTokenizer","TextPretrainingDataset","KVCacheTensor","NeuroMindGenerator"]
