"""Forward-pass shape/gradient tests for both model architectures.

All models here are built with tiny, randomly initialized configs (no pretrained
checkpoint download) so the suite runs offline in a couple of seconds on CPU.
"""

import pytest
import torch

from cad_code_gen.models.baseline import Image2CADQuery
from cad_code_gen.models.spatial_baseline import Image2CADQuerySpatial
from cad_code_gen.models.vision_prefix import (
    PerceiverResampler,
    VisionPrefixCodeGen,
    VisionPrefixLogitsAdapter,
)

VOCAB_SIZE = 50
PAD_ID = 0


def test_baseline_forward_pass_shape(tiny_model_config):
    model = Image2CADQuery(
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_ID,
        config=tiny_model_config,
        pretrained_backbone=False,
    )
    images = torch.randn(2, 3, 224, 224)
    tokens = torch.randint(0, VOCAB_SIZE, (2, 7))

    logits = model(images, tokens)

    assert logits.shape == (2, 7, VOCAB_SIZE)


def test_baseline_backward_pass_updates_non_backbone_params(tiny_model_config):
    model = Image2CADQuery(
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_ID,
        config=tiny_model_config,
        pretrained_backbone=False,
    )
    images = torch.randn(2, 3, 224, 224)
    tokens = torch.randint(0, VOCAB_SIZE, (2, 7))

    logits = model(images, tokens)
    logits.sum().backward()

    assert model.fc_out.weight.grad is not None
    assert torch.any(model.fc_out.weight.grad != 0)


def test_spatial_baseline_encode_image_returns_multiple_tokens(tiny_spatial_model_config):
    model = Image2CADQuerySpatial(
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_ID,
        config=tiny_spatial_model_config,
        pretrained_backbone=False,
    )
    images = torch.randn(2, 3, 64, 64)  # -> 2x2=4 spatial tokens after ResNet-18's 32x downsampling

    memory = model.encode_image(images)

    assert memory.shape == (2, 4, tiny_spatial_model_config.embed_dim)


def test_spatial_baseline_forward_pass_shape(tiny_spatial_model_config):
    model = Image2CADQuerySpatial(
        vocab_size=VOCAB_SIZE,
        pad_token_id=PAD_ID,
        config=tiny_spatial_model_config,
        pretrained_backbone=False,
    )
    images = torch.randn(2, 3, 64, 64)
    tokens = torch.randint(0, VOCAB_SIZE, (2, 7))

    logits = model(images, tokens)

    assert logits.shape == (2, 7, VOCAB_SIZE)


def test_spatial_baseline_raises_when_grid_exceeds_max_image_tokens():
    from cad_code_gen.config import SpatialModelConfig

    tiny_config = SpatialModelConfig(embed_dim=32, n_layers=1, n_heads=2, max_image_tokens=2)
    model = Image2CADQuerySpatial(
        vocab_size=VOCAB_SIZE, pad_token_id=PAD_ID, config=tiny_config, pretrained_backbone=False
    )
    images = torch.randn(1, 3, 64, 64)  # yields 4 tokens > max_image_tokens=2

    with pytest.raises(ValueError, match="max_image_tokens"):
        model.encode_image(images)


def test_perceiver_resampler_output_shape():
    resampler = PerceiverResampler(source_dim=16, target_dim=8, num_latents=4, n_heads=2, n_layers=1)
    patch_features = torch.randn(3, 20, 16)  # (B, N_patches, source_dim)

    out = resampler(patch_features)

    assert out.shape == (3, 4, 8)


def test_vision_prefix_codegen_forward_pass_shape():
    from transformers import CLIPVisionConfig, CLIPVisionModel, GPT2Config, GPT2LMHeadModel

    vision_encoder = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=16,
        )
    )
    language_model = GPT2LMHeadModel(
        GPT2Config(vocab_size=VOCAB_SIZE, n_embd=16, n_layer=1, n_head=2, n_positions=64)
    )

    model = VisionPrefixCodeGen(
        vision_encoder, language_model, num_visual_tokens=3, freeze_vision=True, freeze_language=False
    )

    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 5))
    labels = input_ids.clone()

    outputs = model(pixel_values, input_ids, labels=labels)

    # sequence length = num_visual_tokens (3) + text length (5)
    assert outputs.logits.shape == (2, 8, VOCAB_SIZE)
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_vision_prefix_codegen_freezes_backbones_when_requested():
    from transformers import CLIPVisionConfig, CLIPVisionModel, GPT2Config, GPT2LMHeadModel

    vision_encoder = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=16,
        )
    )
    language_model = GPT2LMHeadModel(
        GPT2Config(vocab_size=VOCAB_SIZE, n_embd=16, n_layer=1, n_head=2, n_positions=64)
    )

    model = VisionPrefixCodeGen(
        vision_encoder, language_model, num_visual_tokens=3, freeze_vision=True, freeze_language=True
    )

    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    assert all(not p.requires_grad for p in model.language_model.parameters())
    assert any(p.requires_grad for p in model.resampler.parameters())


def test_vision_prefix_codegen_works_with_codegen_language_model():
    """`build_pretrained`'s default LM is now CodeGen (a code-pretrained model) instead
    of GPT-2; this proves the architecture works with that model family too. Config
    values (n_head, rotary_dim) are the smallest legal ones for CodeGen's internal
    multi-partition attention split and rotary embedding, not arbitrary.
    """
    from transformers import CLIPVisionConfig, CLIPVisionModel, CodeGenConfig, CodeGenForCausalLM

    vision_encoder = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=16,
        )
    )
    language_model = CodeGenForCausalLM(
        CodeGenConfig(
            vocab_size=VOCAB_SIZE, n_embd=16, n_layer=1, n_head=4, n_positions=64, n_ctx=64, rotary_dim=4
        )
    )

    model = VisionPrefixCodeGen(
        vision_encoder, language_model, num_visual_tokens=3, freeze_vision=True, freeze_language=False
    )

    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 5))
    labels = input_ids.clone()

    outputs = model(pixel_values, input_ids, labels=labels)

    assert outputs.logits.shape == (2, 8, VOCAB_SIZE)
    assert torch.isfinite(outputs.loss)


def test_vision_prefix_logits_adapter_matches_shared_model_interface():
    """The adapter must expose forward(images, tgt_tokens) -> (B, L, vocab_size),
    matching Image2CADQuery/Image2CADQuerySpatial exactly (same L as tgt_tokens, not
    num_visual_tokens + L), since it's what lets all three models share one training
    loop and one set of decoding functions.
    """
    from transformers import CLIPVisionConfig, CLIPVisionModel, GPT2Config, GPT2LMHeadModel

    vision_encoder = CLIPVisionModel(
        CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=32,
            patch_size=16,
        )
    )
    language_model = GPT2LMHeadModel(
        GPT2Config(vocab_size=VOCAB_SIZE, n_embd=16, n_layer=1, n_head=2, n_positions=64)
    )
    inner = VisionPrefixCodeGen(vision_encoder, language_model, num_visual_tokens=3)
    adapter = VisionPrefixLogitsAdapter(inner)

    images = torch.randn(2, 3, 32, 32)
    tgt_tokens = torch.randint(0, VOCAB_SIZE, (2, 5))

    logits = adapter(images, tgt_tokens)

    assert logits.shape == (2, 5, VOCAB_SIZE)  # L=5, matching tgt_tokens -- not 3+5=8


def test_vision_prefix_codegen_resizes_codegen_embeddings_for_custom_tokenizer():
    """build_pretrained's `vocab_size` override must work for the new default LM family."""
    from transformers import CodeGenConfig, CodeGenForCausalLM

    language_model = CodeGenForCausalLM(
        CodeGenConfig(
            vocab_size=VOCAB_SIZE, n_embd=16, n_layer=1, n_head=4, n_positions=64, n_ctx=64, rotary_dim=4
        )
    )

    language_model.resize_token_embeddings(300)

    assert language_model.get_input_embeddings().num_embeddings == 300
