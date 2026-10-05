"""Resume-from-checkpoint support, so a disconnected session (Colab, preempted VM)
restarts training where it left off instead of from epoch 1.

The per-epoch `ckpt_eNN.pt` files stay plain model `state_dict`s (that's what evaluation
loads). Resume needs more -- optimizer state and the epoch counter -- so it lives in a
separate, overwritten-each-epoch `last_state.pt` next to them.
"""

from pathlib import Path

import torch
import torch.nn as nn

RESUME_FILENAME = "last_state.pt"


def save_resume_state(checkpoint_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    """Write model + optimizer + completed-epoch count, atomically (temp file then rename),
    so a crash mid-write can't leave a corrupt resume file behind.
    """
    checkpoint_dir = Path(checkpoint_dir)
    final_path = checkpoint_dir / RESUME_FILENAME
    tmp_path = checkpoint_dir / (RESUME_FILENAME + ".tmp")
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, tmp_path)
    tmp_path.replace(final_path)


def load_resume_state(
    checkpoint_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: str
) -> int:
    """Restore model + optimizer from `last_state.pt` if present.

    Returns:
        The number of epochs already completed (0 if there is nothing to resume).
    """
    path = Path(checkpoint_dir) / RESUME_FILENAME
    if not path.exists():
        return 0

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"])
