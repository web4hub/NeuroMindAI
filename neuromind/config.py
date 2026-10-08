from dataclasses import dataclass

@dataclass
class NeuroMindConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_kv_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
