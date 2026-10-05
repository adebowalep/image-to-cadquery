"""Hyperparameter configs for the baseline model and training loop."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineModelConfig:
    """Architecture hyperparameters for `Image2CADQuery`."""

    embed_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_position_embeddings: int = 1024


@dataclass(frozen=True)
class SpatialModelConfig(BaselineModelConfig):
    """Architecture hyperparameters for `Image2CADQuerySpatial`.

    Adds `max_image_tokens`, the size of the learned positional-embedding table for
    the ResNet's spatial feature grid (e.g. a 224x224 input downsampled 32x by
    ResNet-18 yields a 7x7=49-token grid) -- must be >= the actual grid size used.
    """

    max_image_tokens: int = 64


@dataclass(frozen=True)
class TrainConfig:
    """Optimization/data hyperparameters for training the baseline model."""

    seed: int = 42
    batch_size_train: int = 32
    batch_size_eval: int = 64
    num_workers: int = 2
    max_seq_len: int = 256
    n_epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3


@dataclass(frozen=True)
class TokenizerConfig:
    """Byte-level BPE tokenizer training hyperparameters."""

    vocab_size: int = 16_000
    min_frequency: int = 2
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
