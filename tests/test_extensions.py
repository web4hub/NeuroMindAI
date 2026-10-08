import torch
from neuromind import NeuroMindConfig, NeuroMindForCausalLM, NeuroMindGenerator, NeuroMindTokenizer, TextPretrainingDataset

def test_byte_text_dataset():
    t=NeuroMindTokenizer()
    d=TextPretrainingDataset("hello NeuroMind 🚀",t,4)
    x,y=d[0]
    assert x.shape==y.shape==(4,)
    assert int(x.max()) < t.vocab_size

def test_kv_cached_generation():
    t=NeuroMindTokenizer()
    c=NeuroMindConfig(vocab_size=t.vocab_size,hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_kv_heads=2,intermediate_size=64,max_position_embeddings=32)
    m=NeuroMindForCausalLM(c)
    out=NeuroMindGenerator(m,t).generate("hello",max_new_tokens=2,temperature=0)
    assert isinstance(out,str)
