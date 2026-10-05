"""RL fine-tuning stage: REINFORCE with a rendering-based reward.

Teacher-forced cross-entropy training (`training.py`) only ever conditions on
*ground-truth* prefixes and optimizes per-token likelihood, which doesn't line up with
what the project's own metrics actually measure: whether the *whole* generated script
executes (Valid Syntax Rate) and reconstructs the right 3D geometry (Best-IoU). Both of
those require executing arbitrary generated Python and voxelizing a rendered mesh
(`metrics/best_iou.py`), so they're not differentiable and can't be backpropagated
through directly.

REINFORCE sidesteps that: it samples a full code string from the model's own
distribution (rather than teacher-forcing), scores it with the real reward (VSR +
rendering-based Best-IoU), and uses the reward as a scalar signal that weights the
log-probability of having sampled that particular sequence -- so the (non-differentiable)
reward still produces a valid gradient w.r.t. the model's parameters.

Meant to run *after* supervised warm-start training (`training.train_one_epoch`), the
same way RLHF fine-tunes a language model after supervised fine-tuning.
"""

from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader

from cad_code_gen.decoding import strip_bos_eos

RewardFn = Callable[[str, str], float]


def compute_reward(pred_code: str, gt_code: str, iou_weight: float = 0.8) -> float:
    """Reward = a fixed bonus for syntactically valid code, plus `iou_weight` times the
    rendering-based Best-IoU against the ground-truth shape (0 if the code doesn't run).

    Args:
        pred_code: the model's sampled CadQuery script.
        gt_code: the ground-truth CadQuery script for the same image.
        iou_weight: weight on the IoU term; `(1 - iou_weight)` is the weight on the
            validity bonus, so a syntactically valid-but-wrong prediction still scores
            above an invalid one -- giving the model a gradient signal even before it
            starts getting the geometry right.
    """
    # Deferred: metrics/best_iou.py imports cadquery, a heavy native dependency that
    # (on some platforms) can't coexist in the same environment as a working torch
    # build (see README's "Platform note"). Keeping this import inside the function
    # means the rest of rl_finetune.py -- the sampling/policy-gradient mechanics, which
    # need torch but not cadquery -- stays importable and testable without it.
    from metrics.best_iou import get_iou_best
    from metrics.valid_syntax_rate import evaluate_syntax_rate_simple

    is_valid = evaluate_syntax_rate_simple({"sample": pred_code}) == 1.0
    if not is_valid:
        return 0.0

    try:
        iou = get_iou_best(gt_code, pred_code)
    except Exception:
        iou = 0.0  # executes as Python but isn't a renderable/voxelizable solid

    return (1.0 - iou_weight) * 1.0 + iou_weight * iou


def sample_with_log_probs(
    model, images: torch.Tensor, bos_id: int, eos_id: int, max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively *sample* (not greedy-decode) a token sequence per image, and
    accumulate the log-probability of each sampled token.

    Unlike `decoding.decode_greedy`, this must run with gradients enabled: the returned
    `log_probs_sum` is what `reinforce_step` backpropagates the reward through.

    Args:
        images: (B, 3, H, W)

    Returns:
        tgt: (B, L) sampled token IDs, including the leading BOS.
        log_probs_sum: (B,) sum of log P(token) over each sequence's sampled tokens
            (padding after EOS excluded).
    """
    device = images.device
    batch_size = images.shape[0]

    tgt = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    log_probs_sum = torch.zeros(batch_size, device=device)

    for _ in range(max_len):
        logits = model(images, tgt)
        step_logits = logits[:, -1, :]

        dist = torch.distributions.Categorical(logits=step_logits)
        next_token = dist.sample()
        log_probs_sum = log_probs_sum + dist.log_prob(next_token) * (~finished).float()

        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break

    return tgt, log_probs_sum


def decode_sampled_batch(tokenizer, tgt: torch.Tensor, bos_id: int, eos_id: int) -> list[str]:
    """Decode each row of a sampled token batch (from `sample_with_log_probs`) to a string."""
    return [tokenizer.decode(strip_bos_eos(seq, bos_id, eos_id)) for seq in tgt.tolist()]


def reinforce_step(
    model,
    tokenizer,
    images: torch.Tensor,
    gt_codes: Sequence[str],
    optimizer: torch.optim.Optimizer,
    max_len: int = 256,
    reward_fn: RewardFn = compute_reward,
) -> tuple[float, float]:
    """One REINFORCE update over a batch.

    Samples code from the model, scores each sample against its ground truth with
    `reward_fn`, and takes a policy-gradient step using "reward minus batch-mean" as a
    variance-reducing baseline (standard REINFORCE-with-baseline).

    Args:
        images: (B, 3, H, W)
        gt_codes: B ground-truth CadQuery strings, one per image, same order as `images`.
        reward_fn: `(pred_code, gt_code) -> float`. Injectable so this function is
            testable without executing real CadQuery code (see tests/test_rl_finetune.py).

    Returns:
        (mean_reward, loss) for logging.
    """
    model.train()
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id

    tgt, log_probs_sum = sample_with_log_probs(model, images, bos_id, eos_id, max_len)
    pred_codes = decode_sampled_batch(tokenizer, tgt, bos_id, eos_id)

    rewards = torch.tensor(
        [reward_fn(pred, gt) for pred, gt in zip(pred_codes, gt_codes)],
        device=images.device,
        dtype=torch.float32,
    )
    baseline = rewards.mean()
    advantage = (rewards - baseline).detach()  # no gradient through the reward itself

    loss = -(advantage * log_probs_sum).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return rewards.mean().item(), loss.item()


def reinforce_finetune_epoch(
    model,
    tokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_len: int = 256,
    reward_fn: RewardFn = compute_reward,
) -> float:
    """Run one epoch of REINFORCE fine-tuning over `loader`. Returns the average reward."""
    total_reward = 0.0
    n_batches = 0

    for imgs, _tgt_full, meta in loader:
        imgs = imgs.to(device)
        mean_reward, _loss = reinforce_step(
            model, tokenizer, imgs, meta["code_strings"], optimizer, max_len=max_len, reward_fn=reward_fn
        )
        total_reward += mean_reward
        n_batches += 1

    return total_reward / n_batches if n_batches else 0.0
