
import os
from typing import List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# 全局三张图文件名
PLOT_01_ORIGINAL_AND_SLICED = "01_original_and_sliced.png"
PLOT_02_GENERATED_AND_OUTPUT = "02_generated_and_output.png"
PLOT_03_FULL_COMPARISON = "03_full_comparison.png"
# 每个切片目录内的轨迹图文件名
TRAJECTORY_PLOT_FILENAME = "trajectory.png"


def latent_2d(latent: np.ndarray) -> np.ndarray:
    """保证坐标至少 2"""
    latent = np.atleast_2d(np.asarray(latent))
    if latent.shape[1] < 2:
        latent = np.hstack([latent, np.zeros((latent.shape[0], 2 - latent.shape[1]))])
    return latent


# Pic 1: Origin and Slices
def save_original_and_sliced_plot(
    original_latent: np.ndarray,
    sliced_latents: np.ndarray,
    save_path: str,
    title: str = "Original & Sliced",
) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    orig = latent_2d(original_latent)
    sliced = latent_2d(sliced_latents)
    plt.figure(figsize=(6, 5))
    plt.scatter(orig[:, 0], orig[:, 1], c="green", s=18, marker="o", edgecolors="black", label="Original function", zorder=5)
    plt.scatter(sliced[:, 0], sliced[:, 1], c="blue", s=9, alpha=0.8, edgecolors="black", label="Sliced functions", zorder=3)
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# Pic 2: Single Slice and its trajectory
def save_slice_trajectory_plot(
    trajectory_latents: List[np.ndarray],
    final_latent: np.ndarray,
    target_latent: np.ndarray,
    save_dir: str,
    slice_index: int,
    title: Optional[str] = None,
) -> str:
    """每个 generation 结束时可选调用，更新同一张图，便于过程中观察."""
    os.makedirs(save_dir, exist_ok=True)
    if title is None:
        title = f"Slice {slice_index} trajectory (best per gen) & target"
    traj = np.array([np.asarray(p).ravel() for p in trajectory_latents])
    if len(traj) == 0:
        traj = np.empty((0, 2))
    else:
        traj = latent_2d(traj)
    final = latent_2d(final_latent)
    target = latent_2d(target_latent)

    plt.figure(figsize=(6, 5))
    if len(traj):
        plt.scatter(traj[:, 0], traj[:, 1], c="steelblue", s=6, alpha=0.6, label="Trajectory (best per gen)", zorder=2)
    plt.scatter(final[:, 0], final[:, 1], c="orange", s=12, alpha=0.9, edgecolors="black", label="Final best", zorder=4)
    plt.scatter(target[:, 0], target[:, 1], c="red", s=20, marker="*", edgecolors="black", label="Target", zorder=5)
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    out_path = os.path.join(save_dir, TRAJECTORY_PLOT_FILENAME)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path



# Pic 3: Output and generated low dim funcs
def save_generated_and_output_plot(
    generated_latents: np.ndarray,
    output_latent: np.ndarray,
    save_path: str,
    title: str = "Generated slices & Summed Output",
) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    gen = latent_2d(generated_latents)
    out = latent_2d(output_latent)
    plt.figure(figsize=(6, 5))
    plt.scatter(gen[:, 0], gen[:, 1], c="orange", s=9, alpha=0.8, edgecolors="black", label="Generated slices", zorder=3)
    plt.scatter(out[:, 0], out[:, 1], c="red", s=18, marker="s", edgecolors="black", label="Output summed function", zorder=5)
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# Pic 4: all points for comparison
def save_full_comparison_plot(
    original_latent: Optional[np.ndarray],
    sliced_latents: Optional[np.ndarray],
    generated_latents: Optional[np.ndarray],
    output_latent: Optional[np.ndarray],
    save_path: str,
    title: str = "Comparison: Original / Sliced / Generated / Output",
) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 7))
    if original_latent is not None:
        p = latent_2d(original_latent)
        plt.scatter(p[:, 0], p[:, 1], c="green", s=18, marker="o", edgecolors="black", label="Original function", zorder=5)
    if sliced_latents is not None and len(sliced_latents):
        s = latent_2d(sliced_latents)
        plt.scatter(s[:, 0], s[:, 1], c="blue", s=9, alpha=0.7, label="Sliced functions", zorder=3)
    if generated_latents is not None and len(generated_latents):
        g = latent_2d(generated_latents)
        plt.scatter(g[:, 0], g[:, 1], c="orange", s=9, alpha=0.7, label="Generated slice functions", zorder=4)
    if output_latent is not None:
        o = latent_2d(output_latent)
        plt.scatter(o[:, 0], o[:, 1], c="red", s=18, marker="s", edgecolors="black", label="Output summed function", zorder=5)
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
