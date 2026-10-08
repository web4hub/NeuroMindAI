"""Contiguous token blocks for causal LM training."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
class TokenBlockDataset(Dataset):
    def __init__(self,tokens,seq_len):
        if seq_len<2: raise ValueError("seq_len must be >= 2")
        self.tokens=torch.as_tensor(tokens,dtype=torch.long); self.seq_len=seq_len; self.length=max(0,(len(self.tokens)-1)//seq_len)
    def __len__(self): return self.length
    def __getitem__(self,index):
        s=index*self.seq_len; x=self.tokens[s:s+self.seq_len+1]; return {"input_ids":x[:-1],"labels":x[1:]}
def encode_text(text,tokenizer): return tokenizer.encode(text,add_special_tokens=True)
