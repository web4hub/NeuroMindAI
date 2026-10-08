"""Generation helpers."""
from __future__ import annotations
import torch

def sample_next_token(logits:torch.Tensor,temperature=1.0,top_k=0)->torch.Tensor:
    if temperature<=0: return logits.argmax(dim=-1,keepdim=True)
    logits=logits/temperature
    if top_k>0:
        k=min(int(top_k),logits.size(-1)); values,_=torch.topk(logits,k,dim=-1); logits=logits.masked_fill(logits<values[...,-1,None],float("-inf"))
    return torch.multinomial(torch.softmax(logits,dim=-1),1)
"""Safe bounded generation helpers."""
from __future__ import annotations
import torch

def sample_next_token(logits,temperature=1.0,top_k=0):
    if temperature<=0: return logits.argmax(-1,keepdim=True)
    logits=logits/temperature
    if top_k>0:
        k=min(int(top_k),logits.size(-1)); values,_=torch.topk(logits,k,dim=-1); logits=logits.masked_fill(logits<values[...,-1,None],float("-inf"))
    return torch.multinomial(torch.softmax(logits.float(),-1),1)

def generate(model,input_ids,max_new_tokens=32,temperature=1.0,top_k=0):
    if input_ids.ndim!=2: raise ValueError("input_ids must have shape [batch, sequence]")
    if max_new_tokens<0: raise ValueError("max_new_tokens must be non-negative")
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context=input_ids[:,-model.config.max_position_embeddings:]
            token=sample_next_token(model(context)["logits"][:,-1,:],temperature,top_k); input_ids=torch.cat((input_ids,token),1)
            if model.config.eos_token_id is not None and torch.all(token.eq(model.config.eos_token_id)): break
    return input_ids
