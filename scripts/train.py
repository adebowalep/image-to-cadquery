#!/usr/bin/env python
"""CLI entry point for training the image -> CadQuery-code model.

Usage:
    python scripts/train.py --epochs 15 --batch-size 32 --tokenizer-dir cadquery_tokenizer
    python scripts/train.py --model spatial --embed-dim 256 --n-layers 6 --wandb
    python scripts/train.py --model vision_prefix --epochs 5
"""

import argparse
import pathlib
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cad_code_gen.checkpointing import load_resume_state, save_resume_state
from cad_code_gen.config import BaselineModelConfig, TrainConfig
from cad_code_gen.data import CADCodeDataset, build_image_transform, collate_batch
from cad_code_gen.experiment_log import log_run
from cad_code_gen.registry import MODEL_REGISTRY
from cad_code_gen.tokenizer import load_tokenizer, train_tokenizer
from cad_code_gen.training import evaluate_loss, train_one_epoch

# Architecture hyperparameters only apply to "baseline"/"spatial" (both trained from
# scratch with a BaselineModelConfig/SpatialModelConfig); "vision_prefix" is built
# entirely from pretrained checkpoints via MODEL_REGISTRY and ignores these.
CONFIGURABLE_MODELS = {"baseline", "spatial"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CADCODER/GenCAD-Code")
    parser.add_argument("--cache-dir", default="huggingface_cache")
    parser.add_argument("--tokenizer-dir", default="cadquery_tokenizer")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-log", default="results/runs.jsonl")
    parser.add_argument("--no-resume", action="store_true", help="ignore any last_state.pt and start from epoch 1")

    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="baseline")
    parser.add_argument("--embed-dim", type=int, default=BaselineModelConfig.embed_dim)
    parser.add_argument("--n-layers", type=int, default=BaselineModelConfig.n_layers)
    parser.add_argument("--n-heads", type=int, default=BaselineModelConfig.n_heads)
    parser.add_argument("--ff-dim", type=int, default=BaselineModelConfig.ff_dim)
    parser.add_argument("--dropout", type=float, default=BaselineModelConfig.dropout)

    parser.add_argument("--epochs", type=int, default=TrainConfig.n_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size_train)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--max-seq-len", type=int, default=TrainConfig.max_seq_len)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="cad-code-generation")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def build_model(args: argparse.Namespace, vocab_size: int, pad_token_id: int) -> nn.Module:
    overrides = {}
    if args.model in CONFIGURABLE_MODELS:
        overrides = dict(
            embed_dim=args.embed_dim, n_layers=args.n_layers, n_heads=args.n_heads,
            ff_dim=args.ff_dim, dropout=args.dropout,
        )
    return MODEL_REGISTRY[args.model].build(vocab_size, pad_token_id, **overrides)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    wandb_run = None
    if args.wandb:
        import wandb  # optional dependency: `uv sync --extra tracking`

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    from datasets import load_dataset  # deferred: heavy import, only needed here

    ds_train_full, ds_test = load_dataset(
        args.dataset, num_proc=16, split=["train", "test"], cache_dir=args.cache_dir
    )
    split = ds_train_full.shuffle(seed=args.seed).train_test_split(test_size=0.10, seed=args.seed)
    hf_train, hf_val = split["train"], split["test"]

    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    if tokenizer_dir.exists():
        tokenizer = load_tokenizer(tokenizer_dir)
    else:
        tokenizer = train_tokenizer((text for text in hf_train["cadquery"]), tokenizer_dir)  # text column only: no image decoding

    img_transform = build_image_transform()
    train_ds = CADCodeDataset(hf_train, tokenizer, max_len=args.max_seq_len, img_transform=img_transform)
    val_ds = CADCodeDataset(hf_val, tokenizer, max_len=args.max_seq_len, img_transform=img_transform)

    def _collate(samples):
        return collate_batch(samples, pad_token_id=tokenizer.pad_token_id)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=_collate)

    model = build_model(args, tokenizer.vocab_size, tokenizer.pad_token_id).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    checkpoint_dir = pathlib.Path(args.checkpoint_dir) / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed_epochs = 0 if args.no_resume else load_resume_state(checkpoint_dir, model, optimizer, args.device)
    if completed_epochs:
        print(f"Resuming {args.model} after epoch {completed_epochs}")

    for epoch in range(completed_epochs + 1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_dl, optimizer, criterion, args.device, tokenizer.vocab_size
        )
        val_loss = evaluate_loss(model, val_dl, criterion, args.device, tokenizer.vocab_size)
        print(f"[{args.model} {epoch:02d}/{args.epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        checkpoint_path = checkpoint_dir / f"ckpt_e{epoch:02d}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        log_run(
            {
                "model": args.model,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "checkpoint": str(checkpoint_path),
            },
            path=args.results_log,
        )
        save_resume_state(checkpoint_dir, model, optimizer, epoch)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
