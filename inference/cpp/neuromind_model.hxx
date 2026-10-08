#pragma once
#include "neuromind_block.hxx"
#include <vector>
namespace neuromind { class Model { Config config_; std::vector<Block> layers_; public: explicit Model(Config c):config_(c),layers_(c.num_hidden_layers,Block(c)){} const Config& config() const{return config_;} std::size_t num_layers() const{return layers_.size();} }; }