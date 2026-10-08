#pragma once
#include "neuromind_attention.hxx"
#include "neuromind_mlp.hxx"
namespace neuromind { class Block { Attention attention_; MLP mlp_; public: explicit Block(Config c):attention_(c),mlp_(c){} }; }
