"""CIFAR-10 loader that pulls from HuggingFace's parquet mirror.

torchvision's default CIFAR-10 host (www.cs.toronto.edu) is throttled to a few
hundred bytes/sec in this environment; huggingface.co is not, so we fetch the
dataset's parquet shards directly instead of using torchvision.datasets.CIFAR10.
"""
import io
from pathlib import Path
from urllib.request import urlretrieve

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

_BASE_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
_SHARDS = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


class HFCifar10(Dataset):
    def __init__(self, root, split="train", transform=None):
        assert split in _SHARDS
        self.transform = transform
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        local_path = root / _SHARDS[split]
        if not local_path.exists():
            url = f"{_BASE_URL}/{_SHARDS[split]}"
            print(f"downloading {url} -> {local_path}")
            urlretrieve(url, local_path)
        table = pq.read_table(local_path)
        self._img_bytes = table.column("img").combine_chunks().field("bytes").to_pylist()
        self._labels = table.column("label").to_pylist()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self._img_bytes[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self._labels[idx]
