"""Lightweight run-tracking shared between training and evaluation notebooks/scripts.

Training appends one JSON record per epoch to a `.jsonl` file; evaluation (a separate
notebook, run independently, possibly much later) reads it back to know what was
trained and pick a checkpoint -- without ever needing to re-run training itself.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DEFAULT_LOG_PATH = Path("results/runs.jsonl")


def log_run(record: Dict[str, Any], path: Path = DEFAULT_LOG_PATH) -> None:
    """Append one record (e.g. `{"model": "baseline", "epoch": 6, "train_loss": ...,
    "val_loss": ...}`) as a line of JSON. Creates the parent directory if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_runs(path: Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    """Load every logged record as a DataFrame (empty if the log doesn't exist yet)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(records)


def best_epoch_by_val_loss(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the record with the lowest `val_loss` from one model's per-epoch history.

    Training loss keeps improving as long as you keep training; validation loss is
    what tells you when the model started overfitting instead. Picking the
    lowest-val-loss checkpoint (rather than always the final epoch) is a cheap,
    effective form of early stopping applied after the fact.
    """
    return min(history, key=lambda record: record["val_loss"])
