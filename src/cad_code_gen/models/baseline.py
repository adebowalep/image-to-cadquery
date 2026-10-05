"""Baseline model: ResNet-18 image encoder + Transformer decoder over code tokens."""

import torch
import torch.nn as nn
from torchvision.models import resnet18

from cad_code_gen.config import BaselineModelConfig


class Image2CADQuery(nn.Module):
    """Generates CadQuery code tokens autoregressively, conditioned on one input image.

    The image is reduced to a single embedding (global-average-pooled ResNet-18
    features, linearly projected), which is fed to the Transformer decoder as its
    one-token "memory" sequence; the decoder then attends to it via cross-attention
    while generating code tokens under a causal mask (teacher forcing at train time).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        config: BaselineModelConfig = BaselineModelConfig(),
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.config = config

        resnet = resnet18(weights="IMAGENET1K_V1" if pretrained_backbone else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop fc, keep (B,512,1,1)
        self.img_proj = nn.Linear(resnet.fc.in_features, config.embed_dim)

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
        """images: (B, 3, H, W) -> memory: (B, 1, embed_dim)."""
        b = images.shape[0]
        feats = self.backbone(images).view(b, -1)  # (B, 512)
        return self.img_proj(feats).unsqueeze(1)  # (B, 1, embed_dim)

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
            tgt_tokens: (B, L) token IDs, teacher-forced input (shifted right, with BOS).

        Returns:
            logits: (B, L, vocab_size)
        """
        _, seq_len = tgt_tokens.shape

        memory = self.encode_image(images)

        positions = torch.arange(seq_len, device=images.device).unsqueeze(0)
        tgt = self.tok_emb(tgt_tokens) + self.pos_emb(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(images.device)
        decoded = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return self.fc_out(decoded)
