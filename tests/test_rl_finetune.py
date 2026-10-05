"""Tests for cad_code_gen.rl_finetune (REINFORCE with a rendering-based reward).

`compute_reward` is tested against the real metrics (needs `cadquery`, not `torch`
numpy-interop, so it runs even in a torch-broken environment). The sampling/policy-
gradient mechanics are tested with a tiny random model and an injected dummy
`reward_fn`, since a randomly initialized model's sampled "code" is just noise text
that wouldn't exercise the real reward function meaningfully anyway.
"""

import torch

from cad_code_gen.models.baseline import Image2CADQuery
from cad_code_gen.rl_finetune import (
    compute_reward,
    reinforce_finetune_epoch,
    reinforce_step,
    sample_with_log_probs,
)

VOCAB_SIZE = 50
PAD_ID = 0

VALID_BOX = 'result = cq.Workplane("XY").box(10, 10, 10)'
VALID_BOX_TALL = 'result = cq.Workplane("XY").box(10, 10, 40)'
SYNTAX_ERROR = 'result = cq.Workplane("XY").box(10, 10, 10'


def test_compute_reward_is_zero_for_invalid_syntax():
    assert compute_reward(SYNTAX_ERROR, VALID_BOX) == 0.0


def test_compute_reward_is_positive_for_valid_code():
    assert compute_reward(VALID_BOX, VALID_BOX) > 0.0


def test_compute_reward_rewards_better_geometry_match_more():
    identical = compute_reward(VALID_BOX, VALID_BOX)
    different = compute_reward(VALID_BOX_TALL, VALID_BOX)
    assert identical > different


def test_compute_reward_iou_weight_controls_validity_vs_geometry_tradeoff():
    # With iou_weight=0, only the fixed validity bonus matters: any valid code scores
    # the same regardless of how well it matches the ground-truth geometry.
    assert compute_reward(VALID_BOX, VALID_BOX, iou_weight=0.0) == compute_reward(
        VALID_BOX_TALL, VALID_BOX, iou_weight=0.0
    )


def _make_model(tiny_model_config):
    return Image2CADQuery(
        vocab_size=VOCAB_SIZE, pad_token_id=PAD_ID, config=tiny_model_config, pretrained_backbone=False
    )


def test_sample_with_log_probs_shapes_and_gradient_flow(tiny_model_config):
    model = _make_model(tiny_model_config)
    images = torch.randn(3, 3, 224, 224)

    tgt, log_probs_sum = sample_with_log_probs(model, images, bos_id=1, eos_id=2, max_len=8)

    assert tgt.shape[0] == 3
    assert log_probs_sum.shape == (3,)
    assert log_probs_sum.requires_grad  # backprop target for the REINFORCE loss


def test_reinforce_step_updates_weights_and_returns_finite_values(tiny_model_config):
    model = _make_model(tiny_model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    images = torch.randn(4, 3, 224, 224)
    gt_codes = [VALID_BOX] * 4

    def dummy_reward_fn(pred_code: str, gt_code: str) -> float:
        # Deterministic stand-in reward, independent of cadquery, so the test only
        # exercises the sampling/policy-gradient mechanics.
        return float(len(pred_code) % 5)

    before = model.fc_out.weight.clone()
    mean_reward, loss = reinforce_step(
        model, _FakeTokenizer(), images, gt_codes, optimizer, max_len=8, reward_fn=dummy_reward_fn
    )

    assert isinstance(mean_reward, float)
    assert isinstance(loss, float)
    assert not torch.equal(before, model.fc_out.weight)


def test_reinforce_finetune_epoch_returns_average_reward(tiny_model_config, fake_batch_loader):
    model = _make_model(tiny_model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loader = fake_batch_loader(n_batches=2, batch_size=2, vocab_size=VOCAB_SIZE, code_strings=[VALID_BOX] * 2)

    def dummy_reward_fn(pred_code: str, gt_code: str) -> float:
        return 1.0

    mean_reward = reinforce_finetune_epoch(
        model, _FakeTokenizer(), loader, optimizer, "cpu", max_len=8, reward_fn=dummy_reward_fn
    )

    assert mean_reward == 1.0


class _FakeTokenizer:
    """Minimal stand-in exposing exactly what rl_finetune.py needs, to avoid paying
    for a real BPE tokenizer in tests that don't care about actual decoded text.
    """

    bos_token_id = 1
    eos_token_id = 2

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)
