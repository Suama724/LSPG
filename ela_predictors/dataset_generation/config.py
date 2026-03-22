from __future__ import annotations

import json
import os
from typing import Any

MODEL_PATH = "./ela_encoder/autoencoder_best.pth"
SCALER_PATH = "./ela_encoder/scaler.pkl"
OUTPUT_PATH = "./dataset"

pipeline_defaults: dict[str, Any] = {
    "output_root": OUTPUT_PATH,
    "run_name": "ela_predictor_100d",
    "target_size": 50000,
    "dim": 100,
    "n_eval": 230,
    "chunk_size": 2000,
    "ratio_bbob": 0.3,
    "seed": 42,
    "sample_type": "lhs",
    "x_seed": 42062,
    "x_filename": "X.npy",
    "lower_bound": -5.0,
    "upper_bound": 5.0,
    "do_shift": True,
    "do_rotation": True,
    "do_bias": False,
    "bbob_ids": [i for i in range(1, 25)],
    "compose_k_min": 2,
    "compose_k_max": 5,
    "model_path": MODEL_PATH,
    "scaler_path": SCALER_PATH,
    "device": "cpu",
    "ela_conv_nsample": 200,
    "std_epsilon": 1e-10,
    "max_retries": 3,
    "use_ray": True,
    "num_workers": 3,
    "ray_wait_timeout_s": 1.0,
    "progress_log_filename": "process.log",
    "log_every_n": 200,
}


def normalize_pipeline_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(pipeline_defaults)
    if cfg:
        out.update(cfg)
    if out["target_size"] <= 0:
        raise ValueError("target_size must be positive")
    if out["dim"] <= 0:
        raise ValueError("dim must be positive")
    if out["n_eval"] <= 0:
        raise ValueError("n_eval must be positive")
    if out["chunk_size"] <= 0:
        raise ValueError("chunk_size must be positive")
    if not isinstance(out["x_seed"], int):
        raise ValueError("x_seed must be int")
    if not (0.0 <= out["ratio_bbob"] <= 1.0):
        raise ValueError("ratio_bbob must be in [0, 1]")
    if out["compose_k_min"] <= 0 or out["compose_k_max"] < out["compose_k_min"]:
        raise ValueError("compose_k range is invalid")
    if not out["bbob_ids"]:
        raise ValueError("bbob_ids cannot be empty")
    if out["num_workers"] <= 0:
        raise ValueError("num_workers must be positive")
    if out["ray_wait_timeout_s"] <= 0:
        raise ValueError("ray_wait_timeout_s must be positive")
    if out["log_every_n"] <= 0:
        raise ValueError("log_every_n must be positive")
    return out


def dump_config_json(cfg: dict[str, Any], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=True, indent=2)


config_ela_data_pipeline = normalize_pipeline_config({})


config_ela_data_validate = {
    "run_dir": os.path.join("dataset", "raw", "ela_predictor_smoke"),
    "check_n": 8,
}
