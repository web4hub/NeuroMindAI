from neuromind.tokenizer import SimpleTokenizer

def test_roundtrip():
    t=SimpleTokenizer.from_tokens(["<unk>","<bos>","<eos>","hello","world"])
    assert t.decode(t.encode("hello world"))=="hello world"
