"""Tests for cad_code_gen.training (teacher-forced train/eval loops)."""

import torch
import torch.nn as nn

from cad_code_gen.models.baseline import Image2CADQuery
from cad_code_gen.training import evaluate_loss, train_one_epoch

VOCAB_SIZE = 50
PAD_ID = 0


def _make_model(tiny_model_config):
    return Image2CADQuery(
        vocab_size=VOCAB_SIZE, pad_token_id=PAD_ID, config=tiny_model_config, pretrained_backbone=False
    )


def test_train_one_epoch_returns_finite_average_loss(tiny_model_config, fake_batch_loader):
    model = _make_model(tiny_model_config)
    loader = fake_batch_loader(n_batches=3, vocab_size=VOCAB_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    avg_loss = train_one_epoch(model, loader, optimizer, criterion, "cpu", VOCAB_SIZE)

    assert isinstance(avg_loss, float)
    assert torch.isfinite(torch.tensor(avg_loss))


def test_train_one_epoch_actually_updates_weights(tiny_model_config, fake_batch_loader):
    model = _make_model(tiny_model_config)
    loader = fake_batch_loader(n_batches=3, vocab_size=VOCAB_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    before = model.fc_out.weight.clone()
    train_one_epoch(model, loader, optimizer, criterion, "cpu", VOCAB_SIZE)

    assert not torch.equal(before, model.fc_out.weight)


def test_evaluate_loss_does_not_update_weights(tiny_model_config, fake_batch_loader):
    model = _make_model(tiny_model_config)
    loader = fake_batch_loader(n_batches=2, vocab_size=VOCAB_SIZE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    before = model.fc_out.weight.clone()
    avg_loss = evaluate_loss(model, loader, criterion, "cpu", VOCAB_SIZE)

    assert torch.equal(before, model.fc_out.weight)
    assert torch.isfinite(torch.tensor(avg_loss))
