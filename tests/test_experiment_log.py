"""Tests for cad_code_gen.experiment_log."""

from cad_code_gen.experiment_log import best_epoch_by_val_loss, load_runs, log_run


def test_load_runs_returns_empty_dataframe_when_file_missing(tmp_path):
    df = load_runs(tmp_path / "does_not_exist.jsonl")
    assert df.empty


def test_log_run_then_load_runs_roundtrips(tmp_path):
    path = tmp_path / "runs.jsonl"
    log_run({"model": "baseline", "epoch": 1, "val_loss": 0.5}, path=path)
    log_run({"model": "baseline", "epoch": 2, "val_loss": 0.4}, path=path)

    df = load_runs(path)

    assert len(df) == 2
    assert list(df["epoch"]) == [1, 2]
    assert list(df["val_loss"]) == [0.5, 0.4]


def test_log_run_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "runs.jsonl"
    log_run({"a": 1}, path=path)
    assert path.exists()


def test_best_epoch_by_val_loss_picks_lowest_val_loss_not_last():
    history = [
        {"epoch": 1, "val_loss": 0.30},
        {"epoch": 2, "val_loss": 0.20},  # the actual minimum
        {"epoch": 3, "val_loss": 0.25},  # worse again -- overfitting after epoch 2
    ]
    assert best_epoch_by_val_loss(history) == {"epoch": 2, "val_loss": 0.20}
