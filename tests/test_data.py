"""Tests for cad_code_gen.data (dataset wrapper, image loading, batching)."""

import io

import torch
from PIL import Image

from cad_code_gen.data import CADCodeDataset, collate_batch, read_pil_image


def test_read_pil_image_from_pil():
    img = Image.new("RGB", (8, 8))
    assert read_pil_image(img).size == (8, 8)


def test_read_pil_image_from_bytes_dict():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    field = {"bytes": buf.getvalue()}
    out = read_pil_image(field)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"


def test_dataset_item_shapes_and_special_tokens(dummy_hf_dataset, tiny_tokenizer):
    dataset = CADCodeDataset(dummy_hf_dataset, tiny_tokenizer, max_len=32)
    item = dataset[0]

    assert item["image"].shape == (3, 224, 224)
    assert item["tokens"][0].item() == tiny_tokenizer.bos_token_id
    assert item["tokens"][-1].item() == tiny_tokenizer.eos_token_id
    assert item["code_string"] == dummy_hf_dataset[0]["cadquery"]
    assert item["id"] == "sample_0"


def test_dataset_truncates_to_max_len(dummy_hf_dataset, tiny_tokenizer):
    dataset = CADCodeDataset(dummy_hf_dataset, tiny_tokenizer, max_len=5)
    item = dataset[1]  # longer snippet than the truncation budget allows
    assert len(item["tokens"]) <= 5


def test_collate_batch_pads_to_longest_sequence(dummy_hf_dataset, tiny_tokenizer):
    dataset = CADCodeDataset(dummy_hf_dataset, tiny_tokenizer, max_len=32)
    samples = [dataset[0], dataset[1]]

    imgs, tokens, meta = collate_batch(samples, pad_token_id=tiny_tokenizer.pad_token_id)

    assert imgs.shape == (2, 3, 224, 224)
    max_len = max(len(s["tokens"]) for s in samples)
    assert tokens.shape == (2, max_len)
    assert meta["ids"] == ["sample_0", "sample_1"]

    shorter = samples[0]["tokens"]
    if len(shorter) < max_len:
        pad_region = tokens[0, len(shorter) :]
        assert torch.all(pad_region == tiny_tokenizer.pad_token_id)
