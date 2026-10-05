"""State-of-the-art alternative: pretrained vision encoder + pretrained code LM,
bridged by a small trainable resampler (a "visual prefix", as in Frozen / Flamingo /
BLIP-2 / LLaVA), instead of training a ResNet + Transformer decoder from scratch.

Why this should beat the baseline (`Image2CADQuery`):

- The baseline encodes each image into a *single* pooled vector, discarding spatial
  detail before the decoder ever sees it. Here, a small set of learnable "latent"
  tokens cross-attend over *all* of the vision encoder's patch tokens (a Perceiver-style
  resampler), so the language model gets several image-grounded tokens instead of one.
- The baseline's Transformer decoder is trained from scratch on ~130K examples and has
  to learn Python/CadQuery syntax *and* geometry grounding simultaneously, which is
  almost certainly the main driver of its ~50% Valid Syntax Rate. Here, the decoder is
  a pretrained causal language model that already knows Python syntax; only the vision
  encoder->LM bridge (the resampler) is trained from scratch, and the LM can optionally
  stay frozen or be lightly fine-tuned (e.g. LoRA). This should raise VSR by construction,
  since syntactic well-formedness is inherited from pretraining rather than learned from
  the (comparatively small) CAD dataset.

Design simplification: the visual-prefix tokens and the text tokens are concatenated
into one sequence and passed through the language model's ordinary causal self-attention
(rather than a custom bidirectional-within-prefix mask). This means visual token i cannot
attend to visual token j > i, which is a minor inefficiency but keeps the implementation
correct and robust across `transformers` versions, since it relies only on the public
`inputs_embeds` / `attention_mask` / `labels` API common to every causal LM.
"""

from typing import Optional

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """Cross-attends a fixed number of learnable latent tokens over variable-length
    patch features, producing a fixed-size, information-dense summary of the image
    (Perceiver IO / Flamingo-style resampler).
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        num_latents: int = 32,
        n_heads: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, target_dim) * 0.02)
        self.input_proj = nn.Linear(source_dim, target_dim)
        layer = nn.TransformerDecoderLayer(d_model=target_dim, nhead=n_heads, batch_first=True)
        self.layers = nn.TransformerDecoder(layer, num_layers=n_layers)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """patch_features: (B, N_patches, source_dim) -> (B, num_latents, target_dim)."""
        batch_size = patch_features.shape[0]
        kv = self.input_proj(patch_features)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        return self.layers(latents, kv)


def _language_model_hidden_size(language_model: nn.Module) -> int:
    config = language_model.config
    return getattr(config, "hidden_size", None) or getattr(config, "n_embd")


class VisionPrefixCodeGen(nn.Module):
    """Bridges a frozen (or lightly tuned) pretrained vision encoder and a pretrained
    causal language model with a trainable `PerceiverResampler`, prepending the
    resampled visual tokens as a soft prompt in front of the code tokens.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        language_model: nn.Module,
        num_visual_tokens: int = 32,
        freeze_vision: bool = True,
        freeze_language: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model

        vision_dim = vision_encoder.config.hidden_size
        lm_dim = _language_model_hidden_size(language_model)
        self.resampler = PerceiverResampler(vision_dim, lm_dim, num_latents=num_visual_tokens)
        self.num_visual_tokens = num_visual_tokens

        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        if freeze_language:
            for p in self.language_model.parameters():
                p.requires_grad = False

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, H, W) -> visual prefix tokens (B, num_visual_tokens, lm_dim)."""
        patch_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        return self.resampler(patch_features)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            pixel_values: (B, 3, H, W)
            input_ids: (B, L) CadQuery code token IDs (teacher-forced input).
            attention_mask: (B, L) 1 for real tokens, 0 for padding.
            labels: (B, L) target token IDs (use -100 at padding positions to ignore).

        Returns:
            The language model's `CausalLMOutput` (has `.logits`, and `.loss` if
            `labels` was given), computed over the [visual prefix, text] sequence.
        """
        visual_tokens = self.encode_image(pixel_values)  # (B, V, D)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)  # (B, L, D)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        batch_size, num_visual = visual_tokens.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_mask = torch.ones(
            batch_size, num_visual, device=input_ids.device, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            ignore = torch.full(
                (batch_size, num_visual), -100, device=labels.device, dtype=labels.dtype
            )
            full_labels = torch.cat([ignore, labels], dim=1)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )


class VisionPrefixLogitsAdapter(nn.Module):
    """Wraps `VisionPrefixCodeGen` to expose `forward(images, tgt_tokens) -> logits`,
    the same calling convention as `Image2CADQuery`/`Image2CADQuerySpatial`.

    `VisionPrefixCodeGen.forward` is deliberately HF-style (`pixel_values`/`input_ids`,
    returns a `CausalLMOutput` with `.logits`/`.loss`) so it composes with the ordinary
    `transformers` ecosystem. This adapter is the glue that lets it *also* drop into
    `cad_code_gen.training`'s teacher-forced loop and `cad_code_gen.decoding`'s
    greedy/beam search unmodified, alongside the other two architectures -- so all
    three can be trained and decoded with identical code, differing only in which
    model object gets passed in.
    """

    def __init__(self, vision_prefix_model: VisionPrefixCodeGen):
        super().__init__()
        self.model = vision_prefix_model

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W), tgt_tokens: (B, L) -> logits: (B, L, vocab_size).

        The wrapped model's output covers [visual-prefix, text] positions
        (num_visual_tokens + L); callers expect one logit vector per *text* token
        (matching tgt_tokens' length), so the visual-prefix positions are sliced off.
        """
        outputs = self.model(pixel_values=images, input_ids=tgt_tokens)
        return outputs.logits[:, self.model.num_visual_tokens :, :]


def build_pretrained(
    vision_model_name: str = "openai/clip-vit-base-patch32",
    language_model_name: str = "Salesforce/codegen-350M-mono",
    num_visual_tokens: int = 32,
    vocab_size: Optional[int] = None,
    freeze_vision: bool = True,
    freeze_language: bool = True,
) -> VisionPrefixCodeGen:
    """Load pretrained weights from the Hugging Face Hub and assemble the model.

    Args:
        vision_model_name: any `transformers` CLIP-vision-compatible checkpoint.
        language_model_name: any `transformers` causal LM checkpoint. Defaults to
            CodeGen (pretrained on source code, including Python) rather than a
            natural-language model like GPT-2, for a much stronger built-in prior on
            valid Python syntax -- directly targeting the baseline's ~50% VSR ceiling.
            Swap in a larger code model (e.g. a StarCoder checkpoint) if compute allows.
        num_visual_tokens: number of resampled visual tokens fed to the LM.
        vocab_size: if the CadQuery byte-level BPE tokenizer (see `tokenizer.py`) is
            used instead of the LM's native tokenizer, pass its vocab size here to
            resize the LM's token embeddings/head accordingly.
        freeze_vision / freeze_language: keep the large pretrained backbones frozen
            and only train the resampler (fast, memory-light); unfreeze for full or
            LoRA fine-tuning once the resampler has converged.
    """
    from transformers import AutoModelForCausalLM, CLIPVisionModel

    vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
    language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
    if vocab_size is not None:
        language_model.resize_token_embeddings(vocab_size)

    return VisionPrefixCodeGen(
        vision_encoder,
        language_model,
        num_visual_tokens=num_visual_tokens,
        freeze_vision=freeze_vision,
        freeze_language=freeze_language,
    )
