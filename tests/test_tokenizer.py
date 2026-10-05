"""Tests for cad_code_gen.tokenizer (train/load a byte-level BPE tokenizer)."""

from cad_code_gen.tokenizer import load_tokenizer


def test_special_tokens_are_registered(tiny_tokenizer):
    assert tiny_tokenizer.pad_token == "<pad>"
    assert tiny_tokenizer.bos_token == "<bos>"
    assert tiny_tokenizer.eos_token == "<eos>"
    assert tiny_tokenizer.pad_token_id is not None
    assert tiny_tokenizer.bos_token_id != tiny_tokenizer.eos_token_id


def test_encode_decode_roundtrip(tiny_tokenizer):
    code = 'result = cq.Workplane("XY").box(10, 10, 10)'
    ids = tiny_tokenizer(code, add_special_tokens=False).input_ids
    decoded = tiny_tokenizer.decode(ids)
    assert decoded.strip() == code.strip()


def test_load_tokenizer_matches_trained_one(tmp_path, tiny_tokenizer):
    reloaded = load_tokenizer(tmp_path / "tokenizer")
    assert reloaded.vocab_size == tiny_tokenizer.vocab_size
    assert reloaded.pad_token_id == tiny_tokenizer.pad_token_id
