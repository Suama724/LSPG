from __future__ import annotations

import time
from typing import Any

import numpy as np
import pickle
import torch

from .types import FuncMeta, SamplePayload

from net.AE import encode_ela_feats, load_model
from utils.create_initial_sample import create_initial_sample
from utils.ela_feature import get_ela_feature


class EncoderBundle:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model = load_model(
            cfg["model_path"],
            ela_feats_num=21,
            device=cfg["device"],
        )
        with open(cfg["scaler_path"], "rb") as f:
            self.scaler = pickle.load(f)

    def encode(self, ela_features: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(ela_features.reshape(1, -1))
        z = encode_ela_feats(
            self.model,
            torch.tensor(x, dtype=torch.float32),
            device=self.cfg["device"],
        )
        return np.asarray(z, dtype=np.float32).reshape(-1)


def evaluate_and_encode(
    problem: Any,
    sample_seed: int,
    cfg: dict[str, Any],
    encoder: EncoderBundle,
    func_kind: int,
    meta: FuncMeta,
    fixed_X: np.ndarray | None = None,
) -> SamplePayload | None:
    t0 = time.perf_counter()
    if fixed_X is None:
        X = np.asarray(
            create_initial_sample(
                dim=cfg["dim"],
                n=cfg["n_eval"],
                sample_type=cfg["sample_type"],
                lower_bound=cfg["lower_bound"],
                upper_bound=cfg["upper_bound"],
                seed=sample_seed,
            ),
            dtype=np.float64,
        )
    else:
        X = np.asarray(fixed_X, dtype=np.float64)
    y = np.asarray(problem.eval(X), dtype=np.float64).reshape(-1)
    if y.shape[0] != cfg["n_eval"]:
        return None
    if (not np.isfinite(y).all()) or np.std(y) < cfg["std_epsilon"]:
        return None

    ela, cost_fes, cost_time = get_ela_feature(
        problem=problem,
        Xs=X,
        Ys=y,
        random_state=sample_seed,
        ela_conv_nsample=min(cfg["ela_conv_nsample"], cfg["n_eval"]),
    )
    ela = np.asarray(ela, dtype=np.float64).reshape(-1)
    if ela.shape[0] != 21:
        return None
    if not np.isfinite(ela).all():
        return None

    z2d = encoder.encode(ela)
    if z2d.shape[0] != 2 or (not np.isfinite(z2d).all()):
        return None

    rec = meta.to_record()
    rec["ela_cost_fes"] = int(cost_fes)
    rec["ela_cost_time"] = float(cost_time)
    rec["pipeline_time_sec"] = float(time.perf_counter() - t0)
    return SamplePayload(
        y=y.astype(np.float32),
        z2d=z2d.astype(np.float32),
        seed=int(sample_seed),
        func_kind=int(func_kind),
        meta=rec,
    )

