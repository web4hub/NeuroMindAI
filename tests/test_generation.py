import torch

from neuromind import NeuroMindConfig, NeuroMindForCausalLM


def make_model():
    config = NeuroMindConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return NeuroMindForCausalLM(config)


def test_generate_greedy_shape():
    model = make_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))

    output = model.generate(input_ids, max_new_tokens=5, temperature=0)

    assert output.shape == (1, 9)
    assert torch.equal(output[:, :4], input_ids)


def test_generate_sampling_shape():
    model = make_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 3))

    output = model.generate(
        input_ids,
        max_new_tokens=4,
        temperature=1.0,
        top_k=8,
    )

    assert output.shape == (2, 7)
    assert output.dtype == torch.long
    assert int(output.min()) >= 0
    assert int(output.max()) < model.config.vocab_size
