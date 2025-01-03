import json
import torch
from transformers import AutoTokenizer
from typing import Mapping

class QGDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str) -> None:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        item = self.data[index]
        text_representation = item["labeled_text"]
        text_description = item["description"]

        return text_representation, text_description


class QGDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 0) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch):
        text_representations, text_descriptions = zip(*batch)

        representations = self.tokenizer(
            text_representations,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        descriptions = self.tokenizer(
            text_descriptions,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        labels = torch.arange(len(batch), dtype=torch.long)

        return representations, descriptions, labels
    