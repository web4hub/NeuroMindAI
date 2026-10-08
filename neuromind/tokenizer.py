"""Dependency-free baseline tokenizer interface."""
from __future__ import annotations
class SimpleTokenizer:
    def __init__(self,vocab,unk_token="<unk>",bos_token="<bos>",eos_token="<eos>"):
        self.vocab=dict(vocab); self.unk_token=unk_token; self.bos_token=bos_token; self.eos_token=eos_token
        self.unk_token_id=self.vocab.get(unk_token,0); self.bos_token_id=self.vocab.get(bos_token); self.eos_token_id=self.vocab.get(eos_token); self.id_to_token={i:t for t,i in self.vocab.items()}
    def encode(self,text,add_special_tokens=True):
        ids=[self.vocab.get(t,self.unk_token_id) for t in text.split()]
        if add_special_tokens and self.bos_token_id is not None: ids.insert(0,self.bos_token_id)
        if add_special_tokens and self.eos_token_id is not None: ids.append(self.eos_token_id)
        return ids
    def decode(self,ids,skip_special_tokens=True):
        special={self.unk_token,self.bos_token,self.eos_token}; return " ".join(t for i in ids if (t:=self.id_to_token.get(int(i),self.unk_token)) and not(skip_special_tokens and t in special))
    @classmethod
    def from_tokens(cls,tokens): return cls({t:i for i,t in enumerate(tokens)})
