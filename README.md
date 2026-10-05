# CAD Code Generation from Images

A vision-to-code model that generates [CadQuery](https://cadquery.readthedocs.io/) Python scripts from 2D images of CAD parts — translating a picture of a mechanical part into parametric code that reconstructs it.

- **Dataset**: 147K paired examples of images and CadQuery scripts from [CADCODER/GenCAD-Code](https://huggingface.co/datasets/CADCODER/GenCAD-Code).
- **Baseline model**: a ResNet-18 image encoder feeding a Transformer decoder that autoregressively generates CadQuery tokens.
- **Evaluation**:
  - **Valid Syntax Rate (VSR)** — fraction of generated scripts that execute without error.
  - **Best IoU** — geometric similarity (after best-fit rotation) between the mesh produced by the predicted code and the ground-truth mesh.

The original baseline notebook (`notebooks/archive/`) reported ~50% valid syntax rate and ~0.78 mean IoU — but that number was computed on **2 test samples**, a sanity check rather than a benchmark (see "Status" below). Everything past the baseline (spatial-attention and vision-prefix models, syntax-repair decoding, RL fine-tuning, experiment tracking) is new code built on top of it, unit-tested but not yet trained on the full dataset.

## Repository structure

```
.
├── notebooks/
│   ├── 00_data_exploration.ipynb          # thin: dataset stats, sample visualizations, code-length/operation distributions
│   ├── 01_train_models.ipynb              # thin: trains every model in the registry, logs checkpoints + loss history
│   ├── 02_evaluate_and_compare.ipynb      # thin: reads 01's output, scores + compares models, qualitative inference
│   └── archive/
│       └── 01_baseline_training_and_evaluation_original_run.ipynb  # the original, single-model Colab/Udacity run, WITH its real outputs
├── src/cad_code_gen/          # reusable package the notebook's logic was refactored into
│   ├── config.py                # dataclass hyperparameter configs
│   ├── tokenizer.py             # train/load the byte-level BPE code tokenizer
│   ├── data.py                  # dataset wrapper, image preprocessing, batching
│   ├── training.py              # teacher-forced supervised train/eval loops
│   ├── rl_finetune.py           # REINFORCE fine-tuning with a rendering-based reward
│   ├── decoding.py              # greedy / beam-search / syntax-repair decoding
│   ├── evaluation.py            # scores predictions with metrics/ (VSR, Best-IoU)
│   ├── experiment_log.py        # shared run-tracking: 01 writes results/runs.jsonl, 02 reads it
│   ├── registry.py              # MODEL_REGISTRY: single source of truth for "what models exist"
│   └── models/
│       ├── baseline.py            # Image2CADQuery: pooled-vector ResNet-18 + Transformer decoder
│       ├── spatial_baseline.py    # Image2CADQuerySpatial: spatial-feature-map cross-attention variant
│       └── vision_prefix.py       # VisionPrefixCodeGen + VisionPrefixLogitsAdapter: pretrained-backbones alternative
├── scripts/
│   ├── train.py                 # CLI: supervised training (either model, exposed hyperparameters, optional W&B)
│   ├── finetune_rl.py           # CLI: RL fine-tuning from a supervised checkpoint
│   └── evaluate.py              # CLI: score a checkpoint with VSR / Best-IoU
├── sweep.yaml                  # W&B Bayesian hyperparameter sweep over scripts/train.py
├── tests/                      # pytest unit tests for the modules above
├── metrics/                    # provided evaluation metrics (best_iou.py, valid_syntax_rate.py)
├── pyproject.toml / uv.lock     # dependencies
└── README.md
```

## Environment setup

```bash
pip install uv
uv sync --extra dev
source .venv/bin/activate
```

Then either work through the notebooks in order:

```bash
# notebooks/00_data_exploration.ipynb    -- dataset stats & visualizations (no modeling)
# notebooks/01_train_models.ipynb        -- trains baseline + spatial (+ vision_prefix if enabled)
# notebooks/02_evaluate_and_compare.ipynb -- reads 01's output, scores + compares, no retraining
```

or run the equivalent scripts, one model at a time:

```bash
python scripts/train.py --model baseline --epochs 15
python scripts/train.py --model spatial --epochs 15
python scripts/finetune_rl.py --model baseline --init-checkpoint checkpoints/baseline/ckpt_e06.pt --epochs 3  # optional RL stage
python scripts/evaluate.py --model baseline --checkpoint checkpoints/baseline/ckpt_e06.pt
```

Add `--wandb` to `train.py`/`finetune_rl.py` to log to Weights & Biases (needs `uv sync --extra tracking`). `--model {baseline,spatial,vision_prefix}` selects the architecture (via `cad_code_gen.registry.MODEL_REGISTRY`, the single source of truth both the scripts and the notebooks read from); run `python scripts/train.py --help` for all exposed hyperparameters (embed dim, layers, heads, dropout, lr, weight decay, ...), which are also what `sweep.yaml` searches over (`wandb sweep sweep.yaml && wandb agent <sweep_id>`).

> `notebooks/archive/01_baseline_training_and_evaluation_original_run.ipynb` is the original, single-model notebook that produced the ~50%/0.78 numbers — kept as-is (with its real outputs) for historical reference. It was originally developed on Google Colab (GPU), so its first setup cell installs packages inline rather than relying purely on `uv sync`. `01_train_models.ipynb`/`02_evaluate_and_compare.ipynb` and `scripts/*.py` are the actively-maintained pipeline, all built on the same `src/cad_code_gen/` package.

### Running on a hosted GPU workspace (e.g. Udacity)

`requires-python` is `>=3.10,<3.14` and `cadquery`/`cadquery-ocp` are pinned to exact versions (`2.6.0`/`7.8.1.1`) specifically so this installs cleanly on hosted Linux workspaces with an older glibc (`manylinux_2_31`, e.g. Ubuntu 20.04-based images) and only Python 3.10 available — not just on a local Mac/Colab. If `uv sync` still can't find a matching interpreter:

```bash
rm -rf .venv uv.lock          # any lockfile/venv from a previous, broader requires-python must go
uv sync --python $(which python3.10)   # or wherever the workspace's Python 3.10 lives
source .venv/bin/activate
```

Two things that bit a previous attempt at this and are now fixed by the pins above:
- **`No solution found ... python_full_version == '3.9.*'`** — happened with an open-ended bound like `requires-python = "<=3.11"` (no lower bound), which makes `uv` try to solve for every version down to 3.9 too. `>=3.10,<3.14` avoids that.
- **`cadquery-ocp==7.7.2 ... doesn't have a wheel for manylinux_2_31_x86_64`** — the default (unpinned) `cadquery-ocp` resolved to a release that only ships `manylinux_2_35` wheels. `cadquery-ocp==7.8.1.1` is pinned explicitly because it's the newest release that still ships `manylinux_2_31` wheels for `cp310`–`cp313` (and also has `macosx_11_0_{x86_64,arm64}`/`win_amd64` wheels, so it doesn't break other platforms).

If the notebook still can't `import cadquery` after `uv sync` succeeds, the venv likely isn't registered as a Jupyter kernel yet:

```bash
python -m ipykernel install --user --name cad-code-gen --display-name "CAD Code Gen (uv)"
```

Then select that kernel in the notebook UI. Finally, confirm the GPU is actually visible to this `torch` install before training: `python -c "import torch; print(torch.cuda.is_available())"`; if that's `False` despite the workspace having a GPU, the default PyPI `torch` wheel likely doesn't match the workspace's CUDA driver version, and you'll need `torch`'s CUDA-specific index URL instead (see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the exact command for that CUDA version).

### Running on Google Colab (or Kaggle)

The notebooks are built to run unchanged on Colab: the first code cell of each one detects Colab, mounts Drive, clones the repo, and installs only what Colab lacks (`cadquery`, `cadquery-ocp`, `trimesh`, `datasets` — it already ships `torch`/`transformers`/`pandas`). It then puts `src/` and the repo root on `sys.path`, so `from cad_code_gen...` and `import metrics` work with no `pip install -e`.

1. Push this repo to GitHub (private is fine) and set `REPO_URL` in the first cell of `00`/`01`/`02` (private repo: `https://<token>@github.com/<you>/<repo>.git`).
2. Runtime → Change runtime type → a GPU.
3. Run `01_train_models.ipynb` top to bottom, then `02_evaluate_and_compare.ipynb`.

Everything that must outlive the session — dataset cache, tokenizer, checkpoints, `results/runs.jsonl` — is stored under `STORAGE_ROOT` (`/content/drive/MyDrive/cad_code_gen` on Colab, the repo itself locally). **Training resumes automatically**: every epoch writes `last_state.pt` (model + optimizer + epoch) next to the `ckpt_eNN.pt` files, and re-running `01` after a disconnect continues from there; a model that already finished all `N_EPOCHS` is skipped. `scripts/train.py` does the same (`--no-resume` to start over). The per-epoch `ckpt_eNN.pt` files stay plain `state_dict`s, so evaluation is unaffected.

## Running the tests

```bash
uv run pytest
```

The suite (`tests/`) covers the provided metrics, the tokenizer, dataset/batching, the supervised training loop, all three model architectures' forward passes, greedy/beam/syntax-repair decoding, and REINFORCE fine-tuning mechanics — all with tiny, randomly initialized configs and injected dummy dependencies where needed, so it runs in seconds, offline, with no GPU and no dataset download.

> **Platform note (Intel Macs only):** `cadquery` requires `numpy>=2`, but PyTorch dropped Intel-Mac (`x86_64`) wheels after `2.2.2`, and `2.2.2` predates NumPy 2 ABI support — so on an Intel Mac specifically, `cadquery` and a working `torch` cannot be installed in the *same* environment. This isn't a real project constraint: on Linux/Colab/Apple Silicon (the actual training platform), the latest `torch` and `numpy>=2` install together without issue, and `uv sync` here reflects that. If you're on an Intel Mac and want to run the full test suite locally, use two environments — the main `uv`-managed one (has `cadquery`; covers `tests/test_metrics.py` and most of `tests/test_rl_finetune.py`), and a second scratch environment with `numpy<2`, `torch==2.2.2`, `transformers<5` for whatever touches real image tensors or `transformers` model classes (`tests/test_data.py`, the CLIP/CodeGen cases in `tests/test_models.py`). Every module that needs `cadquery` (`rl_finetune.compute_reward`, `evaluation.py`) imports it lazily inside the function that needs it, precisely so the rest of that module stays importable without it.

## What's in the pipeline

1. **Data loading & exploration** — loads the GenCAD-Code dataset, visualizes sample image/code pairs, and profiles code length and operation frequency (`00_data_exploration.ipynb`).
2. **Tokenizer** — a byte-level BPE tokenizer trained on the CadQuery code corpus (`cad_code_gen.tokenizer`).
3. **Models** — three architectures, selectable via `--model` in `scripts/train.py` (see below for the two enhanced ones).
4. **Supervised training** — teacher-forced cross-entropy training with AdamW (`cad_code_gen.training`).
5. **RL fine-tuning (optional second stage)** — REINFORCE with a rendering-based reward (`cad_code_gen.rl_finetune`).
6. **Decoding** — greedy, beam-search, and a syntax-repair decoder that filters beam candidates by `ast.parse` validity (`cad_code_gen.decoding`).
7. **Evaluation** — VSR and Best-IoU computed over the test split, using the metrics in `metrics/` (`cad_code_gen.evaluation`).

## Design choices & known bottlenecks (original baseline)

- **Single global image embedding**: `Image2CADQuery` collapses each image to one vector (global average pooling) before decoding, rather than exposing a spatial feature map to a cross-attention decoder — this is what `Image2CADQuerySpatial` (below) addresses.
- **BPE tokenization of code** was chosen over a grammar/AST-aware representation for simplicity; it does not guarantee syntactic validity, which is the main driver of the ~50% VSR ceiling — this is what syntax-repair decoding and `VisionPrefixCodeGen`'s code-pretrained LM (below) address.
- **Training budget**: due to Colab's compute limits, the baseline was trained for a relatively small number of epochs on a ResNet-18 backbone rather than a larger vision transformer — VSR/IoU are the *relative* signal here (baseline vs. enhanced), not an absolute ceiling for the architecture.

## Enhancement 1: spatial cross-attention — `Image2CADQuerySpatial`

`src/cad_code_gen/models/spatial_baseline.py` keeps the ResNet-18's spatial feature grid (e.g. 7x7=49 tokens for a 224x224 input, one per receptive field) as the Transformer decoder's cross-attention memory, instead of collapsing it to a single pooled vector first. No change to the decoder itself is needed — `nn.TransformerDecoder` already supports multi-token cross-attention memory; only what the encoder hands it changes. This lets the decoder attend to *different* image regions while generating different tokens (e.g. one region for a hole's diameter, another for the block's overall extent), which the pooled-vector baseline structurally cannot do.

## Enhancement 2: pretrained backbones — `VisionPrefixCodeGen`

`src/cad_code_gen/models/vision_prefix.py` implements a second architecture that follows the "frozen pretrained backbones + small trainable bridge" recipe used by Frozen, Flamingo, BLIP-2, and LLaVA, rather than training everything from scratch:

- A **pretrained vision encoder** (e.g. CLIP ViT) produces patch-level features instead of one pooled vector.
- A small trainable **`PerceiverResampler`** cross-attends a fixed number of learnable latent tokens over those patches, producing a handful of image-grounded "visual tokens" (this is the only large module trained from scratch).
- A **pretrained causal language model** — defaults to [CodeGen](https://huggingface.co/Salesforce/codegen-350M-mono) (pretrained on source code, including Python) rather than a natural-language model like GPT-2 — consumes those visual tokens as a prefix in front of the code tokens and generates CadQuery code, reusing knowledge of Python syntax it already has instead of relearning it from ~130K examples.

In practice, only the resampler (and optionally a LoRA adapter on the language model) needs training, which is far cheaper than the baseline's full ResNet+decoder training run. `build_pretrained(...)` assembles the model from Hugging Face checkpoints; unit tests exercise the architecture with tiny from-scratch configs (including CodeGen specifically) so they don't require downloading weights or a GPU.

## Enhancement 3: syntax-repair decoding

`cad_code_gen.decoding.decode_with_syntax_repair` is a cheap, training-free way to push VSR up regardless of which model architecture is used: it beam-searches several candidate scripts, repairs common numeric-literal glitches (`fix_floats`), then uses `select_first_parseable` to return the first candidate that is valid Python (`ast.parse` succeeds) instead of always taking the single top-scoring beam. This isn't full grammar-constrained decoding (it doesn't mask the vocabulary to only grammatically-valid tokens at every generation step, which would need an incremental Python-aware parser wired into the decode loop) — it's a "generate a few, keep the syntactically valid one" approximation, but it directly targets what VSR actually checks (`exec()` not raising), at the cost of a few extra forward passes rather than any retraining.

## Enhancement 4: RL fine-tuning with a rendering-based reward

`cad_code_gen.rl_finetune` adds an optional second training stage after supervised warm-start. Teacher-forced cross-entropy only ever conditions on ground-truth prefixes and optimizes per-token likelihood, which doesn't line up with what the project's metrics actually measure — whether the *whole* script executes (VSR) and reconstructs the right geometry (Best-IoU). Both require executing arbitrary generated Python and voxelizing a rendered mesh (`metrics/best_iou.py`), so they're not differentiable. REINFORCE sidesteps that: `sample_with_log_probs` samples a full code string from the model's own distribution (instead of teacher-forcing), `compute_reward` scores it with the real VSR + rendering-based Best-IoU (a validity bonus plus `iou_weight` times Best-IoU, so a syntactically-valid-but-wrong prediction still scores above an invalid one), and `reinforce_step` uses that scalar reward — with a batch-mean baseline for variance reduction — to weight the log-probability of having sampled that sequence, producing a valid policy-gradient update despite the reward itself being non-differentiable. Run via `scripts/finetune_rl.py` after a supervised checkpoint exists.

## Enhancement 5: experiment tracking & hyperparameter sweeps

`scripts/train.py` and `scripts/finetune_rl.py` take a `--wandb` flag (optional dependency: `uv sync --extra tracking`) that logs per-epoch train/val loss (or mean reward, for RL) to Weights & Biases. Architecture and optimization hyperparameters (`--model`, `--embed-dim`, `--n-layers`, `--n-heads`, `--ff-dim`, `--dropout`, `--lr`, `--weight-decay`, `--batch-size`) are all exposed as CLI flags rather than hard-coded, and `sweep.yaml` defines a Bayesian search over them (minimizing `val_loss`) that a `wandb agent` can run across as many machines as needed.

## Status: what has and hasn't been run

The original baseline was really trained on GPU (A100, via Colab then re-run on a Udacity GPU workspace) — see `notebooks/archive/01_baseline_training_and_evaluation_original_run.ipynb` for the actual training curve (15 epochs; validation loss bottoms out around **epoch 6** and gets worse after, i.e. the model overfits past that point) and per-sample evaluation output. Its widely-quoted "~50% VSR, ~0.78 mean IoU" came from evaluating **2 test samples** — a quick sanity check, not a benchmark. `02_evaluate_and_compare.ipynb` fixes this by evaluating a much larger random subset (`EVAL_SAMPLE_SIZE`, default 200) and by picking each model's *best-val-loss* checkpoint automatically (`cad_code_gen.experiment_log.best_epoch_by_val_loss`) instead of always using the final epoch.

Everything else — `Image2CADQuerySpatial`, `VisionPrefixCodeGen` (+ its `VisionPrefixLogitsAdapter`, which lets it share the exact same training loop and decoding functions as the other two architectures), syntax-repair decoding, RL fine-tuning, the shared model registry, experiment logging, and the W&B/sweep integration — is implemented and unit-tested (tiny configs, dummy rewards/tokenizers, no real training), but **has not yet been trained or evaluated on the full dataset** in this environment. Run `01_train_models.ipynb` on real GPU compute, then `02_evaluate_and_compare.ipynb` for the actual comparison.

## What I'd do with more time

- Run `01_train_models.ipynb` with `MODELS_TO_TRAIN` including `"vision_prefix"`, and `02_evaluate_and_compare.ipynb` at a larger `EVAL_SAMPLE_SIZE` (ideally the full test set), for a real three-way comparison.
- Give `vision_prefix` its own image preprocessing (CLIP's actual normalization stats) instead of reusing the other two models' ImageNet-normalized transform, which `01_train_models.ipynb` currently does purely for loop simplicity.
- Replace `select_first_parseable`'s after-the-fact filtering with true grammar-constrained decoding (masking invalid next-tokens during generation, not just rejecting whole invalid sequences after the fact).
- Run the `sweep.yaml` hyperparameter sweep for real, rather than the hand-picked settings currently in `config.py`.
- Extend `02_evaluate_and_compare.ipynb`'s qualitative section to render the predicted mesh next to the ground-truth mesh (not just the code), for a more direct visual comparison than reading CadQuery source.
