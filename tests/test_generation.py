import torch
from neuromind import NeuroMindConfig,NeuroMindForCausalLM
from neuromind.generation import generate
def test_generation():
 c=NeuroMindConfig(vocab_size=32,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,num_kv_heads=2,intermediate_size=64,max_position_embeddings=8); m=NeuroMindForCausalLM(c); x=torch.randint(0,32,(1,8)); assert generate(m,x,3,temperature=0).shape==(1,11)
