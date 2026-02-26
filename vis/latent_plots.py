"""
潜空间可视化：数据集分布图与采样点图。
可被 encode 流程与 sample 流程直接调用，图与数据同目录存放。
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from net.AE import load_model, encode_ela_feats


# 默认图名
DATASET_PLOT_NAME = "dataset.png"
SAMPLE_PLOT_NAME = "sampled_points.png"


def plot_dataset_latent_space(
    dataset_path,
    model_path,
    scaler_path,
    save_dir=None,
    latent_points=None,
    labels=None,
    dim=None,
    time_str=None,
    with_clouds=True,
):
    os.makedirs(save_dir, exist_ok=True)

    if latent_points is not None and labels is not None:
        # 已编码，直接绘图
        pass
    else:
        df_raw = pd.read_pickle(dataset_path)
        features = np.array(df_raw["ela_feats"].to_list())
        labels = df_raw["meta_func_id"].values
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        features_scaled = scaler.transform(features)
        device = "cpu"
        ela_feats_num = features.shape[1]
        model = load_model(model_path, ela_feats_num, device)
        latent_points = encode_ela_feats(model, features_scaled, device)

    # 推断 dim / time 用于标题
    if dim is None or time_str is None:
        basename = os.path.basename(dataset_path)
        if basename.startswith("results_") and "_" in basename:
            parts = basename.replace("results_", "").replace(".pkl", "").split("_")
            if len(parts) >= 2:
                dim = dim or int(parts[0].replace("D", ""))
                time_str = time_str or "_".join(parts[1:])
    title_dim = dim if dim is not None else "?"
    title_time = time_str if time_str is not None else ""

    plt.figure(figsize=(10, 8))
    unique_fids = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_fids))

    for i, fid in enumerate(unique_fids):
        mask = labels == fid
        points_of_fid = latent_points[mask]
        if len(points_of_fid) == 0:
            continue
        if with_clouds:
            plt.scatter(
                points_of_fid[:, 0],
                points_of_fid[:, 1],
                color=colors[i],
                s=15,
                alpha=0.15,
                edgecolors="none",
                zorder=1,
            )
        centroid = points_of_fid.mean(axis=0)
        plt.scatter(
            centroid[0],
            centroid[1],
            color=colors[i],
            marker="o",
            s=180,
            edgecolors="white",
            linewidth=1.5,
            label=f"F{int(fid)} Center" if i < 5 else None,
            zorder=10,
        )
        rng = np.random.default_rng(abs(int(fid)) % (2 ** 31))
        text_pos = centroid + rng.normal(0, 0.08, size=centroid.shape)
        plt.text(
            text_pos[0],
            text_pos[1],
            f"F{int(fid)}",
            fontsize=7,
            weight="bold",
            color="black",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor=colors[i],
                boxstyle="round,pad=0.2",
            ),
            zorder=11,
        )

    title = f"Latent Space Distribution ({title_dim}D"
    if title_time:
        title += f", Time: {title_time}"
    title += ")"
    plt.title(title, fontsize=14)
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.4, zorder=0)

    out_path = os.path.join(save_dir, DATASET_PLOT_NAME)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dataset latent plot saved: {out_path}")
    return out_path


def plot_sample_latent_space(data, npy_path=None, save_dir=None, title=None):
    save_dir = save_dir 
    os.makedirs(save_dir, exist_ok=True)
    if data is not None:
        sampled_points = data
    else:
        sampled_points = np.load(npy_path)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        sampled_points[:, 0],
        sampled_points[:, 1],
        c="red",
        marker="o",
        s=40,
        linewidth=1.5,
        edgecolors="black",
        label="Sampled Points",
        zorder=10,
    )
    if title is None:
        title = "Latent Space Sampling (No Background)"
    plt.title(title, fontsize=15)
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")

    out_path = os.path.join(save_dir, SAMPLE_PLOT_NAME)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Sample latent plot saved: {out_path}")
    return out_path
