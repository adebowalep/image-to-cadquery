#!/usr/bin/env python
"""CLI entry point for scoring a trained checkpoint with VSR and Best-IoU.

Usage:
    python scripts/evaluate.py --model baseline --checkpoint checkpoints/baseline/ckpt_e06.pt
"""

import argparse
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))  # for `metrics/`, imported by cad_code_gen.evaluation

from cad_code_gen.data import CADCodeDataset, build_image_transform, collate_batch
from cad_code_gen.evaluation import evaluate_predictions, generate_predictions
from cad_code_gen.registry import MODEL_REGISTRY
from cad_code_gen.tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="baseline")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="CADCODER/GenCAD-Code")
    parser.add_argument("--cache-dir", default="huggingface_cache")
    parser.add_argument("--tokenizer-dir", default="cadquery_tokenizer")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset  # deferred: heavy import, only needed here

    ds_test = load_dataset(args.dataset, split="test", cache_dir=args.cache_dir)
    tokenizer = load_tokenizer(pathlib.Path(args.tokenizer_dir))

    img_transform = build_image_transform()
    test_ds = CADCodeDataset(ds_test, tokenizer, max_len=args.max_len, img_transform=img_transform)

    def _collate(samples):
        return collate_batch(samples, pad_token_id=tokenizer.pad_token_id)

    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    # For "baseline"/"spatial", skip downloading ImageNet weights before immediately
    # overwriting them from the checkpoint below; "vision_prefix" has no such flag.
    build_kwargs = {"pretrained_backbone": False} if args.model in ("baseline", "spatial") else {}
    model = MODEL_REGISTRY[args.model].build(tokenizer.vocab_size, tokenizer.pad_token_id, **build_kwargs).to(
        args.device
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    predictions = generate_predictions(model, tokenizer, test_dl, args.device, max_len=args.max_len)
    ground_truth = {str(row["deepcad_id"]): row["cadquery"] for row in ds_test}

    metrics = evaluate_predictions(predictions, ground_truth)
    print(f"Model             : {args.model}")
    print(f"Valid Syntax Rate : {metrics['vsr']:.3f}")
    print(f"Mean IoU (best)   : {metrics['mean_iou_best']:.3f}")


if __name__ == "__main__":
    main()
