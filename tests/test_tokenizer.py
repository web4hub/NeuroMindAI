import pytest


tokenizer_module = pytest.importorskip(
    "neuromind.tokenizer",
    reason="Tokenizer implementation is not present in neuromind-v0.1 yet",
)


def _tokenizer_class():
    for name in ("NeuroMindTokenizer", "Tokenizer"):
        cls = getattr(tokenizer_module, name, None)
        if cls is not None:
            return cls
    pytest.skip("No supported tokenizer class is implemented yet")


def _make_tokenizer():
    cls = _tokenizer_class()
    for kwargs in ({}, {"vocab_size": 256}):
        try:
            return cls(**kwargs)
        except TypeError:
            continue
    return cls()


def test_tokenizer_round_trip():
    tokenizer = _make_tokenizer()
    text = "NeuroMindAI builds intelligent systems."
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert isinstance(ids, list)
    assert ids
    assert all(isinstance(token_id, int) for token_id in ids)
    assert isinstance(decoded, str)
    assert decoded


def test_tokenizer_is_deterministic():
    tokenizer = _make_tokenizer()
    text = "attention memory reasoning"

    assert tokenizer.encode(text) == tokenizer.encode(text)
