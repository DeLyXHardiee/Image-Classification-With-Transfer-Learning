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
        classes: Iterable[int],
        label_map: Dict[int, int] | None = None,
        indices: Iterable[int] | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.classes = sorted(list(classes))
        self.label_map = label_map if label_map is not None else {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        if indices is None:
            self.indices = self._find_indices()
        else:
            self.indices = list(indices)

    def _find_indices(self) -> list[int]:
        def get_labels(ds):
            # Handle ConcatDataset first to ensure recursion
            if isinstance(ds, torch.utils.data.ConcatDataset):
                labels = []
                for sub_ds in ds.datasets:
                    labels.extend(get_labels(sub_ds))
                return labels

            if isinstance(ds, torch.utils.data.Subset):
                 full_labels = get_labels(ds.dataset)
                 return [full_labels[i] for i in ds.indices]

            # Handle standard datasets with labels/targets attributes
            if hasattr(ds, '_labels'):
                return [int(x) for x in ds._labels]
            if hasattr(ds, 'labels'):
                return [int(x) for x in ds.labels]
            if hasattr(ds, 'targets'):
                return [int(x) for x in ds.targets]

            raise ValueError(f"Could not extract labels from dataset type: {type(ds)}")

        all_labels = get_labels(self.base_dataset)
        print(f"FilteredFlowersDataset: Found {len(all_labels)} total labels in base_dataset")
        
        indices = [i for i, label in enumerate(all_labels) if label in self.label_map]
        print(f"FilteredFlowersDataset: Selected {len(indices)} images for {len(self.classes)} classes")
        
        return indices

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
