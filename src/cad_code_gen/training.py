"""Training/validation loops for `Image2CADQuery` (teacher-forced cross-entropy)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _shift_for_teacher_forcing(tgt_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a padded token batch into decoder input / prediction target.

    tgt_in is everything but the last token (starts with BOS); tgt_out is everything
    but the first token, i.e. the token the model should predict at each position.
    """
    return tgt_full[:, :-1], tgt_full[:, 1:]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    vocab_size: int,
) -> float:
    """Run one training pass over `loader`. Returns the average loss."""
    model.train()
    running_loss = 0.0

    for imgs, tgt_full, _meta in loader:
        imgs, tgt_full = imgs.to(device), tgt_full.to(device)
        tgt_in, tgt_out = _shift_for_teacher_forcing(tgt_full)

        logits = model(imgs, tgt_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    vocab_size: int,
) -> float:
    """Compute average teacher-forced loss over `loader` without updating weights."""
    model.eval()
    running_loss = 0.0

    for imgs, tgt_full, _meta in loader:
        imgs, tgt_full = imgs.to(device), tgt_full.to(device)
        tgt_in, tgt_out = _shift_for_teacher_forcing(tgt_full)

        logits = model(imgs, tgt_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        running_loss += loss.item()

    return running_loss / len(loader)
