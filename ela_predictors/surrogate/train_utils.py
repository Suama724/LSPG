from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Subset


def normalize_batch_y(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample normalization for y with shape [batch, n_eval].
    """
    if y.ndim != 2:
        raise ValueError(f"Expected [batch, n_eval], got shape={tuple(y.shape)}")
    mean = y.mean(dim=1, keepdim=True)
    std = y.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (y - mean) / std


def stratified_train_val_split(dataset, val_ratio: float, seed: int = 42):
    """
    Stratified split by func_kind to keep train/val class ratio aligned.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")
    if not hasattr(dataset, "df") or "func_kind" not in dataset.df.columns:
        raise ValueError("dataset must expose df with 'func_kind' for stratified split")

    labels = dataset.df["func_kind"].to_numpy()
    rng = np.random.RandomState(seed)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_ratio))
        n_val = max(1, min(len(cls_idx) - 1, n_val))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)

