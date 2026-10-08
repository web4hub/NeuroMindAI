#pragma once
#include "neuromind_config.hxx"
namespace neuromind { class Attention { Config c_; public: explicit Attention(Config c):c_(c){c_.validate();} std::size_t hidden_size()const{return c_.hidden_size;} }; }
