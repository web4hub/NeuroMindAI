#pragma once
#include "neuromind_model.hxx"
namespace neuromind { class CausalLM { Model model_; public: explicit CausalLM(Config c):model_(c){} const Config& config() const{return model_.config();} }; }
namespace neuromind { class CausalLM { Model model_; public: explicit CausalLM(Config c):model_(c){} const Config& config()const{return model_.config();} }; }
