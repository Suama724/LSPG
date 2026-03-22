from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from total_files.config import dump_config_json
from .types import SamplePayload


@dataclass
class RunContext:
    run_dir: Path
    chunk_dir: Path
    manifest_path: Path
    schema_path: Path
    progress_path: Path
    config_path: Path


@dataclass
class WriteResult:
    chunk_id: int
    n_samples: int
    chunk_file: str


class ChunkStorageManager:
    def __init__(self, cfg: dict[str, Any], resume: bool = False):
        self.cfg = cfg
        run_name = cfg.get("run_name") or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path(cfg["output_root"]) / run_name
        self.ctx = RunContext(
            run_dir=run_dir,
            chunk_dir=run_dir / "chunks",
            manifest_path=run_dir / "manifest.parquet",
            schema_path=run_dir / "schema.json",
            progress_path=run_dir / "progress.json",
            config_path=run_dir / "config.snapshot.json",
        )
        self.ctx.chunk_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_df = self._load_manifest_if_any(resume=resume)
        self._next_global_idx = int(self._manifest_df["global_idx"].max() + 1) if len(self._manifest_df) else 0
        self._next_chunk_id = int(self._manifest_df["chunk_id"].max() + 1) if len(self._manifest_df) else 0
        self._write_schema_if_needed()
        dump_config_json(cfg, str(self.ctx.config_path))

    @property
    def run_dir(self) -> str:
        return str(self.ctx.run_dir)

    @property
    def completed_size(self) -> int:
        return int(self._next_global_idx)

    def append_batch(self, payloads: list[SamplePayload]) -> WriteResult:
        if not payloads:
            raise ValueError("payloads cannot be empty")
        chunk_id = self._next_chunk_id
        chunk_file = f"chunk_{chunk_id:06d}.npz"
        chunk_path = self.ctx.chunk_dir / chunk_file

        y = np.stack([p.y for p in payloads]).astype(np.float32)
        z2d = np.stack([p.z2d for p in payloads]).astype(np.float32)
        seed = np.asarray([p.seed for p in payloads], dtype=np.int64)
        func_kind = np.asarray([p.func_kind for p in payloads], dtype=np.uint8)
        np.savez_compressed(chunk_path, y=y, z2d=z2d, seed=seed, func_kind=func_kind)

        records: list[dict[str, Any]] = []
        for i, p in enumerate(payloads):
            rec = dict(p.meta)
            rec.update(
                {
                    "global_idx": int(self._next_global_idx + i),
                    "chunk_id": int(chunk_id),
                    "local_idx": int(i),
                    "chunk_file": chunk_file,
                    "seed": int(p.seed),
                    "func_kind": int(p.func_kind),
                    "sub_meta_func_ids": json.dumps(rec.get("sub_meta_func_ids", []), ensure_ascii=True),
                    "sub_seeds": json.dumps(rec.get("sub_seeds", []), ensure_ascii=True),
                    "weights": json.dumps(rec.get("weights", []), ensure_ascii=True),
                }
            )
            records.append(rec)
        append_df = pd.DataFrame(records)
        self._manifest_df = pd.concat([self._manifest_df, append_df], ignore_index=True)
        self._flush_manifest()

        self._next_global_idx += len(payloads)
        self._next_chunk_id += 1
        self._flush_progress()
        return WriteResult(chunk_id=chunk_id, n_samples=len(payloads), chunk_file=chunk_file)

    def finalize_manifest(self) -> None:
        self._flush_manifest()
        self._flush_progress()

    def _load_manifest_if_any(self, resume: bool) -> pd.DataFrame:
        if resume and self.ctx.manifest_path.exists():
            return pd.read_parquet(self.ctx.manifest_path)
        return pd.DataFrame()

    def _flush_manifest(self) -> None:
        self._manifest_df.to_parquet(self.ctx.manifest_path, index=False)

    def _flush_progress(self) -> None:
        info = {
            "completed_size": int(self._next_global_idx),
            "next_chunk_id": int(self._next_chunk_id),
            "run_dir": str(self.ctx.run_dir),
        }
        with self.ctx.progress_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=True, indent=2)

    def _write_schema_if_needed(self) -> None:
        if self.ctx.schema_path.exists():
            return
        schema = {
            "version": 1,
            "description": "ELA predictor dataset schema (ela not persisted)",
            "fixed_x_file": self.cfg.get("x_filename", "X.npy"),
            "chunk_fields": {
                "y": "float32[B, n_eval]",
                "z2d": "float32[B, 2]",
                "seed": "int64[B]",
                "func_kind": "uint8[B]",
            },
            "manifest_fields": [
                "global_idx",
                "chunk_id",
                "local_idx",
                "chunk_file",
                "seed",
                "func_kind",
                "sample_id",
                "task_seed",
                "base_meta_func_id",
                "sub_meta_func_ids",
                "sub_seeds",
                "weights",
                "agg",
                "ela_cost_fes",
                "ela_cost_time",
                "pipeline_time_sec",
            ],
        }
        with self.ctx.schema_path.open("w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=True, indent=2)

