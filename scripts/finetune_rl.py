#!/usr/bin/env python
"""CLI entry point for RL fine-tuning (REINFORCE + rendering-based reward).

Run this *after* `scripts/train.py` has produced a supervised checkpoint -- this script
loads it as a warm start and continues training with a policy-gradient objective driven
by the actual VSR + Best-IoU metrics, rather than teacher-forced cross-entropy.

Usage:
    python scripts/finetune_rl.py --init-checkpoint checkpoints/ckpt_e15.pt --epochs 3
"""

import argparse
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))  # for `metrics/`, imported lazily by rl_finetune.compute_reward

from cad_code_gen.config import BaselineModelConfig, SpatialModelConfig, TrainConfig
from cad_code_gen.data import CADCodeDataset, build_image_transform, collate_batch
from cad_code_gen.models import Image2CADQuery, Image2CADQuerySpatial
from cad_code_gen.rl_finetune import reinforce_finetune_epoch
from cad_code_gen.tokenizer import load_tokenizer

MODEL_REGISTRY = {"baseline": Image2CADQuery, "spatial": Image2CADQuerySpatial}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CADCODER/GenCAD-Code")
    parser.add_argument("--cache-dir", default="huggingface_cache")
    parser.add_argument("--tokenizer-dir", default="cadquery_tokenizer")
    parser.add_argument("--init-checkpoint", required=True, help="supervised warm-start checkpoint")
    parser.add_argument("--checkpoint-dir", default="checkpoints_rl")

    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)  # smaller: sampling is O(max_len) forwards
    parser.add_argument("--lr", type=float, default=1e-5)  # small: fine-tuning, not training from scratch
    parser.add_argument("--max-seq-len", type=int, default=TrainConfig.max_seq_len)
    parser.add_argument("--iou-weight", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="cad-code-generation")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    wandb_run = None
    if args.wandb:
        import wandb  # optional dependency: `uv sync --extra tracking`

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    from datasets import load_dataset  # deferred: heavy import, only needed here

    ds_train = load_dataset(args.dataset, split="train", cache_dir=args.cache_dir)
    tokenizer = load_tokenizer(pathlib.Path(args.tokenizer_dir))

    img_transform = build_image_transform()
    train_ds = CADCodeDataset(ds_train, tokenizer, max_len=args.max_seq_len, img_transform=img_transform)

    def _collate(samples):
        return collate_batch(samples, pad_token_id=tokenizer.pad_token_id)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    model_cls = MODEL_REGISTRY[args.model]
    config_cls = SpatialModelConfig if args.model == "spatial" else BaselineModelConfig
    model = model_cls(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        config=config_cls(),
        pretrained_backbone=False,  # overwritten by the loaded checkpoint below
    ).to(args.device)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location=args.device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def reward_fn(pred_code: str, gt_code: str) -> float:
        from cad_code_gen.rl_finetune import compute_reward

        return compute_reward(pred_code, gt_code, iou_weight=args.iou_weight)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        mean_reward = reinforce_finetune_epoch(
            model, tokenizer, train_dl, optimizer, args.device, max_len=args.max_seq_len, reward_fn=reward_fn
        )
        print(f"[RL {epoch:02d}/{args.epochs}] mean_reward={mean_reward:.4f}")

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "mean_reward": mean_reward})

        torch.save(model.state_dict(), checkpoint_dir / f"rl_ckpt_e{epoch:02d}.pt")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
