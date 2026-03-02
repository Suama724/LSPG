import os
from typing import Optional

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

matplotlib.use("Agg")

PLOT_01_ORIGINAL = "01_original_function_latent.png"
PLOT_02_SLICED = "02_sliced_functions_latent.png"
PLOT_03_GENERATED = "03_generated_slice_functions_latent.png"
PLOT_04_OUTPUT = "04_output_summed_function_latent.png"
PLOT_05_COMPARISON = "05_comparison_all_latent.png"


def latent_2d(latent: np.ndarray) -> np.ndarray:
    # 保证坐标至少2D
    latent = np.atleast_2d(np.asarray(latent))
    if latent.shape[1] < 2:
        latent = np.hstack([latent, np.zeros((latent.shape[0], 2 - latent.shape[1]))])
    return latent


def save_slice_latent_plot(
    latent: np.ndarray,
    slice_index: int,
    save_dir: str,
    title: Optional[str] = None,
) -> str:
    """画单个切片的latent"""
    os.makedirs(save_dir, exist_ok=True)
    latent = latent_2d(latent)
    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c="red", s=80, edgecolors="black")
    if title is None:
        title = f"Slice {slice_index} latent"
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    out_path = os.path.join(save_dir, f"slice_{slice_index}_latent.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def save_single_point_plot(
    latent: np.ndarray,
    save_path: str,
    title: str,
    color: str = "red",
) -> str:
    """处理单点 用于 01 04 """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    latent = latent_2d(latent)
    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=color, s=120, edgecolors="black", label=title)
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_points_plot(
    points: np.ndarray,
    save_path: str,
    title: str,
    color: str = "blue",
) -> str:
    """处理多点 用于 02 与 03 """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    points = np.atleast_2d(points)
    points = points.reshape(-1, points.shape[-1])
    points = latent_2d(points)
    plt.figure(figsize=(6, 5))
    plt.scatter(points[:, 0], points[:, 1], c=color, s=60, edgecolors="black", alpha=0.8)
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_comparison_plot(
    original_latent: Optional[np.ndarray],
    sliced_latents: Optional[np.ndarray],
    generated_latents: Optional[np.ndarray],
    output_latent: Optional[np.ndarray],
    save_path: str,
) -> str:
    """05"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 7))
    if original_latent is not None:
        p = latent_2d(original_latent)
        plt.scatter(p[:, 0], p[:, 1], c="green", s=150, marker="o", edgecolors="black", label="Original function", zorder=5)
    if sliced_latents is not None and len(sliced_latents):
        s = latent_2d(sliced_latents)
        plt.scatter(s[:, 0], s[:, 1], c="blue", s=50, alpha=0.7, label="Sliced functions", zorder=3)
    if generated_latents is not None and len(generated_latents):
        g = latent_2d(generated_latents)
        plt.scatter(g[:, 0], g[:, 1], c="orange", s=50, alpha=0.7, label="Generated slice functions", zorder=4)
    if output_latent is not None:
        o = latent_2d(output_latent)
        plt.scatter(o[:, 0], o[:, 1], c="red", s=150, marker="s", edgecolors="black", label="Output summed function", zorder=5)
    plt.title("Comparison: Original / Sliced / Generated / Output")
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="best")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
