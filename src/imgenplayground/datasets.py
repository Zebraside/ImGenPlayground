import io

import torch
import polars as pl
from PIL import Image
from torchvision import transforms as tr
from transformers import CLIPTokenizer


class NarutoDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_preprocessed, pretrained_model_name_or_path):
        super().__init__()
        self.data = pl.read_parquet(path_to_preprocessed)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

        self.transforms = tr.Compose([
            tr.Resize(512),
            tr.ToTensor(),
            tr.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        task = self.data.row(idx, named=True)
        image = Image.open(io.BytesIO(task["image"]["bytes"]))
        return {
            "pixel_values": self.transforms(image),
            "input_ids": self.tokenizer(task["text"], max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids,
        }

    def __len__(self):
        return len(self.data)
