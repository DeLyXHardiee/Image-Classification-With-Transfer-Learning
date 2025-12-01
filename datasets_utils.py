"""Utility helpers for dataset preparation and data augmentation tricks."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class FilteredFlowersDataset(Dataset):

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Iterable[int],
        label_map: Dict[int, int],
        transform: Callable | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        image, label = self.base_dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        remapped_label = self.label_map[label]
        return image, remapped_label


def mixup_data(inputs: Tensor, targets: Tensor, alpha: float = 0.4):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, targets_a, targets_b, lam


def mixup_criterion(criterion, preds: Tensor, targets_a: Tensor, targets_b: Tensor, lam: float):
    return lam * criterion(preds, targets_a) + (1 - lam) * criterion(preds, targets_b)
