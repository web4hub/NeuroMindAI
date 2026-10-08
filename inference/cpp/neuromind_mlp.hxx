#pragma once
#include "neuromind_config.hxx"
namespace neuromind { class MLP { Config c_; public: explicit MLP(Config c):c_(c){} std::size_t intermediate_size()const{return c_.intermediate_size;} }; }
