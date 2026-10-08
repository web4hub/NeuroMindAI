#pragma once
#include "neuromind_config.hxx"
namespace neuromind { class MLP { Config config_; public: explicit MLP(Config c):config_(c){} std::size_t input_size() const{return config_.hidden_size;} std::size_t intermediate_size() const{return config_.intermediate_size;} }; }
namespace neuromind { class MLP { Config c_; public: explicit MLP(Config c):c_(c){} std::size_t intermediate_size()const{return c_.intermediate_size;} }; }
