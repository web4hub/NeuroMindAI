#pragma once
#include <cstddef>
#include <stdexcept>
namespace neuromind {
struct Config { std::size_t vocab_size=32000, hidden_size=512, num_hidden_layers=8, num_attention_heads=8, num_kv_heads=8, intermediate_size=2048, max_position_embeddings=2048; float rms_norm_eps=1e-5f, rope_theta=10000.f; int bos_token_id=1,eos_token_id=2,pad_token_id=0; bool tie_word_embeddings=true; void validate() const { if(!num_attention_heads||hidden_size%num_attention_heads||num_attention_heads%num_kv_heads) throw std::invalid_argument("invalid attention configuration"); }};
}