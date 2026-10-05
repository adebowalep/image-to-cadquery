"""Dataset, image preprocessing, and batching for image -> CadQuery-code training."""

import io
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform() -> transforms.Compose:
    """Standard ImageNet preprocessing used by the ResNet encoder."""
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=Image.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def read_pil_image(img_field: Any) -> Image.Image:
    """Normalize a Hugging Face image field (raw bytes dict or PIL.Image) to RGB PIL.Image."""
    if isinstance(img_field, dict):  # streaming mode: {"bytes": ...}
        return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    if isinstance(img_field, Image.Image):  # materialized dataset
        return img_field.convert("RGB")
    raise TypeError(f"Unknown image type: {type(img_field)}")


class CADCodeDataset(Dataset):
    """Wraps a Hugging Face image/code dataset for image -> CadQuery-code training.

    Converts each row's image to a tensor and tokenizes its CadQuery code with the
    supplied tokenizer, adding BOS/EOS and truncating to `max_len`.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        code_col: str = "cadquery",
        id_col: str = "deepcad_id",
        max_len: int = 256,
        img_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.code_col = code_col
        self.id_col = id_col
        self.max_len = max_len
        self.img_transform = img_transform or build_image_transform()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]

        pil_img = read_pil_image(row["image"])
        img = self.img_transform(pil_img)

        bos_id, eos_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        ids = self.tokenizer(row[self.code_col], add_special_tokens=False).input_ids[
            : self.max_len - 2
        ]
        ids = [bos_id] + ids + [eos_id]

        return {
            "image": img,
            "tokens": torch.tensor(ids, dtype=torch.long),
            "code_string": row[self.code_col],
            "id": row[self.id_col],
        }


def collate_batch(samples: list[dict], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Batch a list of `CADCodeDataset` items into padded tensors.

    Returns:
        imgs: FloatTensor (B, 3, 224, 224)
        tgt_padded: LongTensor (B, L_max) of token IDs, right-padded with `pad_token_id`
        meta: {"code_strings": [...], "ids": [...]}
    """
    imgs = torch.stack([s["image"] for s in samples])
    seqs = [s["tokens"] for s in samples]
    tgt_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
    meta = {
        "code_strings": [s["code_string"] for s in samples],
        "ids": [s["id"] for s in samples],
    }
    return imgs, tgt_padded, meta
