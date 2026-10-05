"""Single source of truth for which model architectures exist and how to build one.

Every model built through this registry exposes the same interface --
`forward(images, tgt_tokens) -> logits` of shape `(B, L, vocab_size)` -- so
`cad_code_gen.training`, `cad_code_gen.decoding`, and `cad_code_gen.evaluation` work
identically across all of them, and `scripts/train.py`, `scripts/evaluate.py`, and the
notebooks all share one definition of "what models exist" instead of each keeping
their own copy of this lookup.
"""

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from cad_code_gen.config import BaselineModelConfig, SpatialModelConfig
from cad_code_gen.models import Image2CADQuery, Image2CADQuerySpatial, VisionPrefixLogitsAdapter, build_pretrained


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    build: Callable[..., nn.Module]  # (vocab_size, pad_token_id, **config_overrides) -> model


def _build_baseline(
    vocab_size: int, pad_token_id: int, pretrained_backbone: bool = True, **config_overrides
) -> nn.Module:
    return Image2CADQuery(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        config=BaselineModelConfig(**config_overrides),
        pretrained_backbone=pretrained_backbone,
    )


def _build_spatial(
    vocab_size: int, pad_token_id: int, pretrained_backbone: bool = True, **config_overrides
) -> nn.Module:
    return Image2CADQuerySpatial(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        config=SpatialModelConfig(**config_overrides),
        pretrained_backbone=pretrained_backbone,
    )


def _build_vision_prefix(vocab_size: int, pad_token_id: int, **config_overrides) -> nn.Module:
    # pad_token_id is unused, and config_overrides (if any) go straight to
    # build_pretrained's own kwargs (e.g. num_visual_tokens) -- this architecture's
    # hyperparameters aren't BaselineModelConfig/SpatialModelConfig fields.
    return VisionPrefixLogitsAdapter(build_pretrained(vocab_size=vocab_size, **config_overrides))


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "baseline": ModelSpec(
        name="baseline",
        description="ResNet-18 (pooled vector) + Transformer decoder, trained from scratch.",
        build=_build_baseline,
    ),
    "spatial": ModelSpec(
        name="spatial",
        description="ResNet-18 (spatial feature grid) + Transformer decoder cross-attention, trained from scratch.",
        build=_build_spatial,
    ),
    "vision_prefix": ModelSpec(
        name="vision_prefix",
        description="Frozen CLIP ViT + Perceiver resampler + CodeGen LM (visual-prefix prompting).",
        build=_build_vision_prefix,
    ),
}
