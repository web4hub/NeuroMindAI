#pragma once
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <utility>
namespace neuromind { class Tokenizer { std::unordered_map<std::string,int> vocab_; int unk_; public: explicit Tokenizer(std::unordered_map<std::string,int> vocab,int unk_id=0):vocab_(std::move(vocab)),unk_(unk_id){} std::vector<int> encode(std::string_view text)const{std::vector<int> out;std::string w;for(char c:text){if(c==' '||c=='\n'||c=='\t'){if(!w.empty()){auto i=vocab_.find(w);out.push_back(i==vocab_.end()?unk_:i->second);w.clear();}}else w+=c;}if(!w.empty()){auto i=vocab_.find(w);out.push_back(i==vocab_.end()?unk_:i->second);}return out;}}; }
