from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch



def compare_coord(
    coords: Union[np.ndarray, torch.Tensor],
    coords_pred: Union[np.ndarray, torch.Tensor],
    dim: int,
    description: str,
    save_path: Optional[str] = None,
) -> float:
    """
    Compare 2D coordinates and return mean L2 distance.
    """
    gt = _to_numpy(coords)
    pred = _to_numpy(coords_pred)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 2:
        raise ValueError(f"coords must be [batch, 2], got {gt.shape}")

    dists = np.linalg.norm(gt - pred, axis=1)
    mean_l2 = float(np.mean(dists))
    _plot(gt, pred, dim=dim, description=description, mean_l2=mean_l2, save_path=save_path)
    return mean_l2


def _to_numpy(coords: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(coords, torch.Tensor):
        return coords.detach().cpu().numpy()
    return np.asarray(coords)


def _plot(
    gt: np.ndarray,
    pred: np.ndarray,
    dim: int,
    description: str,
    mean_l2: float,
    save_path: Optional[str] = None,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(gt[:, 0], gt[:, 1], s=20, alpha=0.7, label="compute & encode")
    plt.scatter(pred[:, 0], pred[:, 1], s=20, alpha=0.7, label="net predict")
    plt.title(f"{description} | dim={dim} | mean_l2={mean_l2:.4f}")
    plt.xlabel("axis0")
    plt.ylabel("axis1")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
        return
    plt.show()