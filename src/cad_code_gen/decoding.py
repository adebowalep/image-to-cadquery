"""Autoregressive decoding strategies for turning model logits into CadQuery code strings."""

import ast
import re

import torch


@torch.no_grad()
def decode_greedy(model, tokenizer, imgs: torch.Tensor, max_len: int = 256) -> list[str]:
    """Greedy decoding: always pick the argmax token. Fast, single hypothesis per image.

    Args:
        model: an Image2CADQuery-like model with `forward(images, tgt_tokens) -> logits`.
        tokenizer: exposes `bos_token_id`, `eos_token_id`, and `decode`.
        imgs: (B, C, H, W)
        max_len: maximum number of tokens to generate per sequence.

    Returns:
        One decoded code string per image in the batch.
    """
    model.eval()
    device = imgs.device
    batch_size = imgs.shape[0]
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id

    tgt = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model(imgs, tgt)
        next_token = logits[:, -1, :].argmax(dim=-1)
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == eos_id
        if finished.all():
            break

    pred_codes = []
    for seq in tgt:
        seq = seq[1:]  # drop BOS
        eos_positions = (seq == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            seq = seq[: eos_positions[0]]
        pred_codes.append(tokenizer.decode(seq.tolist()))
    return pred_codes


@torch.no_grad()
def _beam_search(model, img_tensor: torch.Tensor, bos_id: int, eos_id: int, max_len: int, beam_size: int):
    """Shared beam-search loop. Returns [(token_id_list, score), ...], best score first."""
    model.eval()
    device = img_tensor.device

    sequences = [(torch.tensor([[bos_id]], device=device), 0.0)]

    for _ in range(max_len):
        candidates = []
        for seq, score in sequences:
            logits = model(img_tensor.unsqueeze(0), seq)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
            top_probs, top_ids = log_probs.topk(beam_size)

            for i in range(beam_size):
                next_id = top_ids[i].item()
                new_seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
                candidates.append((new_seq, score + top_probs[i].item()))

        sequences = sorted(candidates, key=lambda c: c[1], reverse=True)[:beam_size]

        if all(seq[0, -1].item() == eos_id for seq, _ in sequences):
            break

    return [(seq[0].tolist(), score) for seq, score in sequences]


def strip_bos_eos(token_ids: list[int], bos_id: int, eos_id: int) -> list[int]:
    if token_ids and token_ids[0] == bos_id:
        token_ids = token_ids[1:]
    if eos_id in token_ids:
        token_ids = token_ids[: token_ids.index(eos_id)]
    return token_ids


def decode_beam(model, tokenizer, img_tensor: torch.Tensor, max_len: int = 256, beam_size: int = 3) -> str:
    """Beam search decoding for a single image (higher quality, higher cost than greedy).

    Args:
        model: an Image2CADQuery-like model.
        tokenizer: exposes `bos_token_id`, `eos_token_id`, and `decode`.
        img_tensor: (C, H, W) -- a single image, no batch dimension.
        max_len: maximum sequence length.
        beam_size: number of hypotheses kept at each step.

    Returns:
        The decoded code string for the single best beam.
    """
    return decode_beam_candidates(model, tokenizer, img_tensor, max_len=max_len, beam_size=beam_size)[0]


def decode_beam_candidates(
    model, tokenizer, img_tensor: torch.Tensor, max_len: int = 256, beam_size: int = 3
) -> list[str]:
    """Like `decode_beam`, but returns every final beam's decoded string, best score first."""
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    beams = _beam_search(model, img_tensor, bos_id, eos_id, max_len, beam_size)
    return [tokenizer.decode(strip_bos_eos(ids, bos_id, eos_id)) for ids, _score in beams]


def select_first_parseable(candidates: list[str]) -> str:
    """Return the first candidate that is valid Python syntax, else the first candidate.

    This is a cheap approximation of grammar-constrained decoding: rather than masking
    the vocabulary to only grammatically-valid next-tokens at every generation step
    (which needs an incremental Python-aware parser wired into the decoding loop),
    this generates a handful of complete candidates and filters by whole-sequence
    parseability after the fact. `ast.parse` checks Python syntax only -- it doesn't
    guarantee the code *executes* (e.g. it won't catch an undefined CadQuery method),
    but syntax errors are exactly what the Valid Syntax Rate metric's `exec()` call
    raises first, so this directly targets that failure mode at negligible extra cost.
    """
    for candidate in candidates:
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return candidates[0] if candidates else ""


def decode_with_syntax_repair(
    model, tokenizer, img_tensor: torch.Tensor, max_len: int = 256, beam_size: int = 5
) -> str:
    """Beam search, then prefer whichever beam is syntactically valid Python.

    Combines `decode_beam_candidates` with `fix_floats` (numeric-literal cleanup) and
    `select_first_parseable` (syntax filter) to bias the final output toward code that
    will actually pass the Valid Syntax Rate check, without any extra training.
    """
    candidates = decode_beam_candidates(model, tokenizer, img_tensor, max_len=max_len, beam_size=beam_size)
    candidates = [fix_floats(c) for c in candidates]
    return select_first_parseable(candidates)


def fix_floats(code_str: str) -> str:
    """Repair common decimal-literal glitches seen in generated code (e.g. '3.5.6' -> '3.56')."""
    code_str = re.sub(r"(\d+\.\d+)\.(\d+)", r"\1\2", code_str)
    code_str = re.sub(r"\.(?=\.)", "", code_str)
    code_str = re.sub(r"(\d+)\.(\s|$)", r"\1\2", code_str)
    return code_str
