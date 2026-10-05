"""Train/load a byte-level BPE tokenizer over CadQuery source code."""

from pathlib import Path
from typing import Iterable

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from cad_code_gen.config import TokenizerConfig


def train_tokenizer(
    code_iterator: Iterable[str],
    save_dir: Path,
    config: TokenizerConfig = TokenizerConfig(),
) -> PreTrainedTokenizerFast:
    """Train a byte-level BPE tokenizer over an iterable of CadQuery source strings.

    Args:
        code_iterator: yields raw CadQuery code strings (e.g. one dataset column).
        save_dir: directory the tokenizer files are written to.
        config: vocab size / special tokens.

    Returns:
        The trained tokenizer, wrapped for use with Hugging Face APIs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_tokenizer = ByteLevelBPETokenizer()
    raw_tokenizer.train_from_iterator(
        code_iterator,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=[config.pad_token, config.bos_token, config.eos_token],
    )
    raw_tokenizer.save_model(str(save_dir))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token=config.bos_token,
        eos_token=config.eos_token,
        pad_token=config.pad_token,
    )
    tokenizer.save_pretrained(str(save_dir))
    return tokenizer


def load_tokenizer(save_dir: Path) -> PreTrainedTokenizerFast:
    """Load a tokenizer previously written by `train_tokenizer`."""
    return PreTrainedTokenizerFast.from_pretrained(str(save_dir))
