"""Shared pytest fixtures: a tiny tokenizer, a tiny fake dataset, and a tiny model.

Everything here is deliberately small (vocab size, embedding dims, image size) so the
whole test suite runs in a few seconds on CPU with no network access and no download
of the real 147K-example dataset or any pretrained-weight checkpoints.

`torch` is imported lazily inside `fake_batch_loader` (the only fixture that needs it),
not at module level: tests/test_metrics.py and the cadquery-only half of
tests/test_rl_finetune.py must stay collectible in an environment that has `cadquery`
but not a working `torch` (see README's "Platform note").
"""

import pytest
from PIL import Image

from cad_code_gen.config import BaselineModelConfig, SpatialModelConfig, TokenizerConfig
from cad_code_gen.tokenizer import train_tokenizer

TOY_CADQUERY_CODE = [
    'result = cq.Workplane("XY").box(10, 10, 10)',
    'result = cq.Workplane("XY").box(20, 15, 5).faces(">Z").workplane().hole(4)',
    'result = cq.Workplane("XY").circle(5).extrude(10)',
    'result = cq.Workplane("XY").rect(8, 8).extrude(3)',
]


@pytest.fixture
def tiny_tokenizer(tmp_path):
    """A byte-level BPE tokenizer trained on a handful of toy CadQuery snippets."""
    config = TokenizerConfig(vocab_size=300, min_frequency=1)
    return train_tokenizer(iter(TOY_CADQUERY_CODE), tmp_path / "tokenizer", config=config)


@pytest.fixture
def dummy_hf_dataset():
    """A list of dict rows shaped like the CADCODER/GenCAD-Code dataset."""
    rows = []
    for i, code in enumerate(TOY_CADQUERY_CODE):
        rows.append(
            {
                "image": Image.new("RGB", (64, 64), color=(i * 10 % 255, 0, 0)),
                "cadquery": code,
                "deepcad_id": f"sample_{i}",
            }
        )
    return rows


@pytest.fixture
def tiny_model_config():
    return BaselineModelConfig(embed_dim=32, n_layers=1, n_heads=2, ff_dim=64, max_position_embeddings=64)


@pytest.fixture
def tiny_spatial_model_config():
    # A 64x64 input downsampled 32x by ResNet-18 yields a 2x2=4-token grid;
    # max_image_tokens just needs to be >= that.
    return SpatialModelConfig(
        embed_dim=32, n_layers=1, n_heads=2, ff_dim=64, max_position_embeddings=64, max_image_tokens=8
    )


@pytest.fixture
def fake_batch_loader():
    """Factory for an in-memory stand-in for a `DataLoader` of (imgs, tgt_full, meta)
    batches, built directly from `torch.randn`/`torch.randint` rather than real images
    run through `CADCodeDataset` (which needs PIL -> tensor conversion). Anything that
    just needs *some* well-shaped batches to loop over (training/eval loops, RL
    fine-tuning) can use this instead of standing up a real dataset + tokenizer.
    """
    import torch

    def _make(
        n_batches: int = 2,
        batch_size: int = 2,
        seq_len: int = 6,
        vocab_size: int = 50,
        image_size: int = 224,
        code_strings: list[str] | None = None,
    ):
        batches = []
        for _ in range(n_batches):
            imgs = torch.randn(batch_size, 3, image_size, image_size)
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            meta = {
                "code_strings": (code_strings or ["result = cq.Workplane('XY').box(1, 1, 1)"] * batch_size)[
                    :batch_size
                ],
                "ids": [f"sample_{i}" for i in range(batch_size)],
            }
            batches.append((imgs, tokens, meta))
        return batches

    return _make
