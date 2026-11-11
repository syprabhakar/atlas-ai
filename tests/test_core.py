from atlas.core import split_text, tokens

def test_split_empty():
    assert split_text("  ") == []

def test_split_overlap():
    chunks = split_text("word " * 200, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)

def test_tokenization():
    assert tokens("Payment FAILED, again!") == ["payment", "failed", "again"]
