"""Enhanced baseline: ResNet-18 *spatial* feature map + Transformer decoder cross-attention.

`Image2CADQuery` (the original baseline) collapses each image to a single pooled
vector before the decoder ever sees it, so the decoder can't attend to *different*
image regions while generating different tokens (e.g. one region for a hole's
diameter, another for the block's overall extent). This variant keeps the ResNet's
spatial feature grid (one token per receptive field, e.g. 7x7=49 tokens for a 224x224
input) as the decoder's cross-attention memory instead, using `nn.TransformerDecoder`'s
existing multi-token cross-attention -- no change to the decoder itself, only to what
the encoder hands it.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from cad_code_gen.config import SpatialModelConfig


class Image2CADQuerySpatial(nn.Module):
    """Same interface as `Image2CADQuery`, but with spatial cross-attention memory."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        config: SpatialModelConfig = SpatialModelConfig(),
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.config = config

        resnet = resnet18(weights="IMAGENET1K_V1" if pretrained_backbone else None)
        # Drop avgpool + fc (unlike the baseline, which only drops fc): keeps the
        # (B, 512, H', W') spatial feature map instead of a pooled (B, 512) vector.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feat_proj = nn.Linear(resnet.fc.in_features, config.embed_dim)
        self.spatial_pos_emb = nn.Embedding(config.max_image_tokens, config.embed_dim)

        self.tok_emb = nn.Embedding(vocab_size, config.embed_dim, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        self.fc_out = nn.Linear(config.embed_dim, vocab_size)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> memory: (B, H'*W', embed_dim), one token per grid cell.

        Raises:
            ValueError: if the backbone's feature grid has more cells than
                `config.max_image_tokens` has embeddings for.
        """
        feats = self.backbone(images)  # (B, C, H', W')
        b, c, h, w = feats.shape
        num_tokens = h * w
        if num_tokens > self.config.max_image_tokens:
            raise ValueError(
                f"Feature grid has {num_tokens} tokens (H'={h}, W'={w}) but "
                f"config.max_image_tokens={self.config.max_image_tokens}. Increase "
                "max_image_tokens or use a smaller input resolution."
            )

        feats = feats.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        memory = self.feat_proj(feats)  # (B, H'*W', embed_dim)

        positions = torch.arange(num_tokens, device=images.device).unsqueeze(0)
        return memory + self.spatial_pos_emb(positions)

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
            tgt_tokens: (B, L) token IDs, teacher-forced input (shifted right, with BOS).

        Returns:
            logits: (B, L, vocab_size)
        """
        _, seq_len = tgt_tokens.shape

        memory = self.encode_image(images)  # (B, num_image_tokens, embed_dim)

        positions = torch.arange(seq_len, device=images.device).unsqueeze(0)
        tgt = self.tok_emb(tgt_tokens) + self.pos_emb(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(images.device)
        decoded = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return self.fc_out(decoded)
