import torch

from neuromind import NeuroMindConfig
from neuromind.attention import NeuroMindAttention


def test_attention_output_shape_and_finite_values():
    config = NeuroMindConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
    )
    attention = NeuroMindAttention(config)
    x = torch.randn(2, 8, config.hidden_size)

    y = attention(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_attention_supports_grouped_query_attention():
    config = NeuroMindConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        max_position_embeddings=16,
    )
    attention = NeuroMindAttention(config)
    x = torch.randn(1, 6, config.hidden_size)

    y = attention(x)

    assert y.shape == (1, 6, config.hidden_size)
