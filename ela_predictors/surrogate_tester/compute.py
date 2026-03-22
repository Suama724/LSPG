from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pickle
import torch
import torch.nn as nn

from dataset_generation.config import config_ela_data_pipeline
from dataset_generation.utils.create_initial_sample import create_initial_sample
from dataset_generation.utils.ela_feature import get_ela_feature
from dataset_generation.net.AE import AutoEncoder, encode_ela_feats, load_model
from surrogate.train_utils import normalize_batch_y

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_GEN_ROOT = PROJECT_ROOT / "dataset_generation"

def predict_ela_with_net(func_list: Iterable[Any], net: nn.Module, x_path: str | Path) -> np.ndarray:
    """
    Predict 2D coords directly from function evaluations by surrogate net.
    y: (batch, n_eval), return coords: (batch, 2)
    """
    cfg = config_ela_data_pipeline
    X = np.asarray(np.load(Path(x_path)), dtype=np.float64)

    ys = []
    for problem in func_list:
        y = np.asarray(problem.eval(X), dtype=np.float32).reshape(-1)
        ys.append(y)
    batch_y = np.stack(ys, axis=0)

    net.eval()
    try:
        net_device = next(net.parameters()).device
    except StopIteration:
        net_device = torch.device("cpu")
    with torch.no_grad():
        y_tensor = torch.tensor(batch_y, dtype=torch.float32, device=net_device)
        y_tensor = normalize_batch_y(y_tensor)
        coords = net(y_tensor)
    return np.asarray(coords.detach().cpu(), dtype=np.float32)


def compute_ela(func_list, x_path) -> np.ndarray:
    """
    Compute classical ELA features on shared fixed X for each function.
    return: ela_feats with shape (batch, 21)
    """
    cfg = config_ela_data_pipeline
    X = np.asarray(np.load(Path(x_path)), dtype=np.float64)

    all_ela = []
    for idx, problem in enumerate(func_list):
        y = np.asarray(problem.eval(X), dtype=np.float64).reshape(-1)
        ela, _, _ = get_ela_feature(
            problem=problem,
            Xs=X,
            Ys=y,
            random_state=int(cfg["seed"]) + idx,
            ela_conv_nsample=min(cfg["ela_conv_nsample"], cfg["n_eval"]),
        )
        all_ela.append(np.asarray(ela, dtype=np.float32).reshape(-1))
    return np.stack(all_ela, axis=0)


def encode_ela(ela_feats: np.ndarray, encoder: AutoEncoder | None = None) -> np.ndarray:
    """
    Encode ELA features to 2D.
    If encoder is None, auto-load trained encoder + scaler from dataset_generation/ela_encoder.
    """
    cfg = config_ela_data_pipeline
    ela_arr = np.asarray(ela_feats, dtype=np.float32)

    if ela_arr.ndim == 1:
        ela_arr = ela_arr.reshape(1, -1)

    model_path = Path(cfg["model_path"])
    scaler_path = Path(cfg["scaler_path"])
    if not model_path.is_absolute():
        model_path = DATASET_GEN_ROOT / model_path
    if not scaler_path.is_absolute():
        scaler_path = DATASET_GEN_ROOT / scaler_path

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    ela_arr = np.asarray(scaler.transform(ela_arr), dtype=np.float32)

    if encoder is None:
        encoder = load_model(
            str(model_path),
            ela_feats_num=21,
            device="cpu",
        )


    encoded = encode_ela_feats(
        encoder,
        torch.tensor(ela_arr, dtype=torch.float32),
        device=cfg["device"],
    )
    return np.asarray(encoded, dtype=np.float32)


def compute_and_encode_ela(func_list: Iterable[Any], x_path: str | Path, encoder: AutoEncoder | None = None) -> np.ndarray:

    ela_feats = compute_ela(func_list=func_list, x_path=x_path)
    return encode_ela(ela_feats=ela_feats, encoder=encoder)



