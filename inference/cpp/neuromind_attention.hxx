#pragma once
#include "neuromind_config.hxx"
namespace neuromind { struct AttentionShape { std::size_t batch,sequence,hidden; }; class Attention { Config config_; public: explicit Attention(Config c):config_(c){config_.validate();} AttentionShape shape(std::size_t b,std::size_t s) const {return {b,s,config_.hidden_size};} }; }