from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from total_files.config import config_ela_data_validate, normalize_pipeline_config
from .function_factory import problem_from_meta_record
from utils.create_initial_sample import create_initial_sample


def validate_run(run_dir: str, check_n: int = 8) -> None:
    run_path = Path(run_dir)
    manifest_path = run_path / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with (run_path / "config.snapshot.json").open("r", encoding="utf-8") as f:
        cfg = normalize_pipeline_config(json.load(f))
    df = pd.read_parquet(manifest_path).sort_values("global_idx").reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("manifest is empty")

    ratio = float((df["func_kind"] == 0).mean())
    if abs(ratio - cfg["ratio_bbob"]) > 0.05:
        raise RuntimeError(f"func_kind ratio mismatch: got {ratio:.3f}, expect {cfg['ratio_bbob']:.3f}")

    chunk_dir = run_path / "chunks"
    fixed_x_path = run_path / cfg.get("x_filename", "X.npy")
    fixed_X = None
    if fixed_x_path.exists():
        fixed_X = np.asarray(np.load(fixed_x_path), dtype=np.float64)
        if fixed_X.shape != (cfg["n_eval"], cfg["dim"]):
            raise RuntimeError(f"fixed X shape mismatch: {fixed_X.shape}")

    sample_indices = np.linspace(0, len(df) - 1, num=min(check_n, len(df)), dtype=int)
    for i in sample_indices:
        row = df.iloc[int(i)]
        chunk = np.load(chunk_dir / row["chunk_file"])
        local_idx = int(row["local_idx"])
        y_saved = chunk["y"][local_idx]
        z_saved = chunk["z2d"][local_idx]
        if y_saved.shape[0] != cfg["n_eval"]:
            raise RuntimeError("y shape mismatch")
        if z_saved.shape[0] != 2:
            raise RuntimeError("z2d shape mismatch")
        if not np.isfinite(y_saved).all() or not np.isfinite(z_saved).all():
            raise RuntimeError("NaN/Inf detected in saved arrays")

        problem = problem_from_meta_record(row.to_dict(), cfg)
        if fixed_X is not None:
            X = fixed_X
        else:
            X = np.asarray(
                create_initial_sample(
                    dim=cfg["dim"],
                    n=cfg["n_eval"],
                    sample_type=cfg["sample_type"],
                    lower_bound=cfg["lower_bound"],
                    upper_bound=cfg["upper_bound"],
                    seed=int(row["seed"]),
                ),
                dtype=np.float64,
            )
        y_rebuild = np.asarray(problem.eval(X), dtype=np.float64).reshape(-1).astype(np.float32)
        if not np.allclose(y_saved, y_rebuild, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"rebuild y mismatch at global_idx={int(row['global_idx'])}")

    summary = {
        "n_samples": int(len(df)),
        "ratio_bbob_single": ratio,
        "check_count": int(len(sample_indices)),
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def run_validate(cfg: dict[str, Any] | None = None) -> None:
    final_cfg = cfg or config_ela_data_validate
    validate_run(run_dir=final_cfg["run_dir"], check_n=int(final_cfg.get("check_n", 8)))


if __name__ == "__main__":
    run_validate()

