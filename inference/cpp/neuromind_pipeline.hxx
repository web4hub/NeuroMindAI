#pragma once
#include "neuromind_causallm.hxx"
#include "neuromind_tokenizer.hxx"
#include <string>\n#include <utility>
namespace neuromind { class Pipeline { CausalLM model_; Tokenizer tokenizer_; public: Pipeline(CausalLM model,Tokenizer tokenizer):model_(std::move(model)),tokenizer_(std::move(tokenizer)){} std::vector<int> tokenize(const std::string& text) const{return tokenizer_.encode(text);} const CausalLM& model() const{return model_;} }; }
#include <string>
#include <utility>
namespace neuromind { class Pipeline { CausalLM model_; Tokenizer tokenizer_; public: Pipeline(CausalLM model,Tokenizer tokenizer):model_(std::move(model)),tokenizer_(std::move(tokenizer)){} std::vector<int> tokenize(const std::string& s)const{return tokenizer_.encode(s);} }; }
