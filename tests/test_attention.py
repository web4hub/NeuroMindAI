import torch
from neuromind import NeuroMindConfig,NeuroMindForCausalLM
def test_gqa_shape():
 c=NeuroMindConfig(vocab_size=32,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,num_kv_heads=2,intermediate_size=64,max_position_embeddings=16); assert NeuroMindForCausalLM(c)(torch.randint(0,32,(2,7)))["logits"].shape==(2,7,32)
