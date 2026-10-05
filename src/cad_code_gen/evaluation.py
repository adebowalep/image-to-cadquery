"""Evaluate a trained model's predictions with the provided VSR / Best-IoU metrics."""

from typing import Dict

import torch
from torch.utils.data import DataLoader

from cad_code_gen.decoding import decode_greedy
from metrics.best_iou import get_iou_best
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple


@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    loader: DataLoader,
    device: str,
    max_len: int = 256,
) -> Dict[str, str]:
    """Run greedy decoding over every batch in `loader`, keyed by sample id."""
    model.eval()
    predictions: Dict[str, str] = {}

    for imgs, _tgt_full, meta in loader:
        imgs = imgs.to(device)
        pred_codes = decode_greedy(model, tokenizer, imgs, max_len=max_len)
        for sample_id, code in zip(meta["ids"], pred_codes):
            predictions[str(sample_id)] = code

    return predictions


def evaluate_predictions(
    predictions: Dict[str, str],
    ground_truth: Dict[str, str],
) -> Dict[str, float]:
    """Score predicted CadQuery code against ground truth with VSR and Best-IoU.

    Args:
        predictions: {sample_id: predicted_code}
        ground_truth: {sample_id: ground_truth_code}, superset of predictions' keys.

    Returns:
        {"vsr": ..., "mean_iou_best": ...}
    """
    vsr = evaluate_syntax_rate_simple(predictions)

    ious = []
    for sample_id, pred_code in predictions.items():
        if sample_id not in ground_truth:
            continue
        try:
            ious.append(get_iou_best(ground_truth[sample_id], pred_code))
        except Exception:
            continue  # invalid/non-executable code contributes 0 to VSR, not to mean IoU

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return {"vsr": vsr, "mean_iou_best": mean_iou}
