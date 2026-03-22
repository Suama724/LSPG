from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LatentCoordDataset(Dataset):
    def __init__(self, run_dir: str, return_meta: bool = False):
        self.run_dir = Path(run_dir)
        self.return_meta = return_meta
        manifest_path = self.run_dir / "manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.df = pd.read_parquet(manifest_path).sort_values("global_idx").reset_index(drop=True)
        self.chunk_dir = self.run_dir / "chunks"
        self._chunk_cache_name: str | None = None
        self._chunk_cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        chunk_file = row["chunk_file"]
        local_idx = int(row["local_idx"])
        chunk = self._load_chunk(chunk_file)

        y = torch.from_numpy(chunk["y"][local_idx].astype(np.float32))
        z2d = torch.from_numpy(chunk["z2d"][local_idx].astype(np.float32))
        if not self.return_meta:
            return y, z2d

        meta = {
            "global_idx": int(row["global_idx"]),
            "sample_id": int(row["sample_id"]),
            "func_kind": int(row["func_kind"]),
            "seed": int(row["seed"]),
            "task_seed": int(row["task_seed"]),
            "base_meta_func_id": None if pd.isna(row["base_meta_func_id"]) else int(row["base_meta_func_id"]),
            "sub_meta_func_ids": self._maybe_json_list(row.get("sub_meta_func_ids")),
            "sub_seeds": self._maybe_json_list(row.get("sub_seeds")),
            "weights": self._maybe_json_list(row.get("weights")),
            "agg": row.get("agg"),
        }
        return y, z2d, meta

    def _load_chunk(self, chunk_file: str) -> dict[str, np.ndarray]:
        if self._chunk_cache_name == chunk_file and self._chunk_cache is not None:
            return self._chunk_cache
        path = self.chunk_dir / chunk_file
        arr = np.load(path)
        self._chunk_cache_name = chunk_file
        self._chunk_cache = {k: arr[k] for k in arr.files}
        return self._chunk_cache

    @staticmethod
    def _maybe_json_list(value: Any) -> list[Any]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, str):
            return list(json.loads(value))
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

