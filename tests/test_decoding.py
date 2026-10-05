"""Tests for cad_code_gen.decoding (greedy/beam decoding, code-cleanup regexes)."""

import torch

from cad_code_gen.decoding import (
    decode_beam,
    decode_beam_candidates,
    decode_greedy,
    decode_with_syntax_repair,
    fix_floats,
    select_first_parseable,
)
from cad_code_gen.models.baseline import Image2CADQuery

VOCAB_SIZE = 50
PAD_ID = 0


def test_decode_greedy_terminates_and_returns_one_string_per_image(tiny_tokenizer, tiny_model_config):
    model = Image2CADQuery(
        vocab_size=tiny_tokenizer.vocab_size,
        pad_token_id=tiny_tokenizer.pad_token_id,
        config=tiny_model_config,
        pretrained_backbone=False,
    )
    images = torch.randn(2, 3, 224, 224)

    outputs = decode_greedy(model, tiny_tokenizer, images, max_len=10)

    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)


def test_decode_beam_returns_a_string(tiny_tokenizer, tiny_model_config):
    model = Image2CADQuery(
        vocab_size=tiny_tokenizer.vocab_size,
        pad_token_id=tiny_tokenizer.pad_token_id,
        config=tiny_model_config,
        pretrained_backbone=False,
    )
    image = torch.randn(3, 224, 224)

    output = decode_beam(model, tiny_tokenizer, image, max_len=10, beam_size=2)

    assert isinstance(output, str)


def test_decode_beam_candidates_returns_beam_size_ranked_strings(tiny_tokenizer, tiny_model_config):
    model = Image2CADQuery(
        vocab_size=tiny_tokenizer.vocab_size,
        pad_token_id=tiny_tokenizer.pad_token_id,
        config=tiny_model_config,
        pretrained_backbone=False,
    )
    image = torch.randn(3, 224, 224)

    candidates = decode_beam_candidates(model, tiny_tokenizer, image, max_len=10, beam_size=4)

    assert len(candidates) == 4
    assert all(isinstance(c, str) for c in candidates)
    # decode_beam must agree with the top beam-search candidate
    assert decode_beam(model, tiny_tokenizer, image, max_len=10, beam_size=4) == candidates[0]


def test_select_first_parseable_prefers_valid_python():
    candidates = [
        "result = cq.Workplane(",  # invalid syntax
        "result = cq.Workplane('XY').box(10, 10, 10)",  # valid
        "result = cq.Workplane('XY').box(20, 20, 20)",  # also valid, but ranked lower
    ]
    assert select_first_parseable(candidates) == candidates[1]


def test_select_first_parseable_falls_back_to_first_when_all_invalid():
    candidates = ["result = cq.Workplane(", "def f(:"]
    assert select_first_parseable(candidates) == candidates[0]


def test_select_first_parseable_empty_list_returns_empty_string():
    assert select_first_parseable([]) == ""


def test_decode_with_syntax_repair_returns_a_string(tiny_tokenizer, tiny_model_config):
    model = Image2CADQuery(
        vocab_size=tiny_tokenizer.vocab_size,
        pad_token_id=tiny_tokenizer.pad_token_id,
        config=tiny_model_config,
        pretrained_backbone=False,
    )
    image = torch.randn(3, 224, 224)

    output = decode_with_syntax_repair(model, tiny_tokenizer, image, max_len=10, beam_size=3)

    assert isinstance(output, str)


def test_fix_floats_repairs_double_decimal():
    assert fix_floats("height = 3.5.6") == "height = 3.56"


def test_fix_floats_removes_trailing_dot():
    assert fix_floats("width = 12.\n") == "width = 12\n"


def test_fix_floats_is_a_noop_on_clean_code():
    code = "height = 60.0\nwidth = 80.0"
    assert fix_floats(code) == code
