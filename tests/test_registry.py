"""Tests for cad_code_gen.registry (shared model name -> builder lookup).

Only "baseline" and "spatial" are exercised with a real forward pass here: both build
from scratch with no network access when `pretrained_backbone=False`. "vision_prefix"
needs to download pretrained CLIP/CodeGen weights (see tests/test_models.py, which
covers its architecture directly with tiny from-scratch configs instead) -- here we
only check it's registered with the expected shape.
"""

import torch

from cad_code_gen.registry import MODEL_REGISTRY

VOCAB_SIZE = 50
PAD_ID = 0


def test_registry_has_all_three_architectures():
    assert set(MODEL_REGISTRY) == {"baseline", "spatial", "vision_prefix"}
    for spec in MODEL_REGISTRY.values():
        assert callable(spec.build)
        assert spec.description


def test_baseline_builder_forward_pass_shape():
    model = MODEL_REGISTRY["baseline"].build(
        VOCAB_SIZE, PAD_ID, pretrained_backbone=False, embed_dim=32, n_layers=1, n_heads=2, ff_dim=64
    )
    images = torch.randn(2, 3, 224, 224)
    tokens = torch.randint(0, VOCAB_SIZE, (2, 7))

    logits = model(images, tokens)

    assert logits.shape == (2, 7, VOCAB_SIZE)


def test_spatial_builder_forward_pass_shape():
    model = MODEL_REGISTRY["spatial"].build(
        VOCAB_SIZE,
        PAD_ID,
        pretrained_backbone=False,
        embed_dim=32,
        n_layers=1,
        n_heads=2,
        ff_dim=64,
        max_image_tokens=64,
    )
    images = torch.randn(2, 3, 224, 224)
    tokens = torch.randint(0, VOCAB_SIZE, (2, 7))

    logits = model(images, tokens)

    assert logits.shape == (2, 7, VOCAB_SIZE)
