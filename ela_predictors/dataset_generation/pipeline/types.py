from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    sample_id: int
    func_kind: int  # 0=bbob_single, 1=bbob_composed
    seed: int


@dataclass
class FuncMeta:
    sample_id: int
    func_kind: int
    task_seed: int
    base_meta_func_id: int | None = None
    sub_meta_func_ids: list[int] = field(default_factory=list)
    sub_seeds: list[int] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    agg: str = "weighted_sum"

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "func_kind": self.func_kind,
            "task_seed": self.task_seed,
            "base_meta_func_id": self.base_meta_func_id,
            "sub_meta_func_ids": self.sub_meta_func_ids,
            "sub_seeds": self.sub_seeds,
            "weights": self.weights,
            "agg": self.agg,
        }


@dataclass
class SamplePayload:
    y: np.ndarray
    z2d: np.ndarray
    seed: int
    func_kind: int
    meta: dict[str, Any]

