"""Tests for cad_code_gen.checkpointing (resume-from-checkpoint)."""

import torch
import torch.nn as nn

from cad_code_gen.checkpointing import RESUME_FILENAME, load_resume_state, save_resume_state


def _model_and_optimizer():
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, optimizer


def _train_step(model, optimizer):
    loss = model(torch.randn(8, 4)).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_load_returns_zero_when_nothing_to_resume(tmp_path):
    model, optimizer = _model_and_optimizer()
    assert load_resume_state(tmp_path, model, optimizer, "cpu") == 0


def test_roundtrip_restores_weights_optimizer_and_epoch(tmp_path):
    model, optimizer = _model_and_optimizer()
    _train_step(model, optimizer)
    save_resume_state(tmp_path, model, optimizer, epoch=3)

    fresh_model, fresh_optimizer = _model_and_optimizer()
    completed = load_resume_state(tmp_path, fresh_model, fresh_optimizer, "cpu")

    assert completed == 3
    for saved, restored in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(saved, restored)
    # AdamW's step counter only exists in the optimizer state once it has taken a step
    assert len(fresh_optimizer.state_dict()["state"]) > 0


def test_save_overwrites_previous_state_and_leaves_no_temp_file(tmp_path):
    model, optimizer = _model_and_optimizer()
    save_resume_state(tmp_path, model, optimizer, epoch=1)
    save_resume_state(tmp_path, model, optimizer, epoch=2)

    assert load_resume_state(tmp_path, *_model_and_optimizer(), "cpu") == 2
    assert [p.name for p in tmp_path.iterdir()] == [RESUME_FILENAME]
