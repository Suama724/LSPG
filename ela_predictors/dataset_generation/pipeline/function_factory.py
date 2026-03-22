import json
from typing import Any

import numpy as np

from .types import FuncMeta, TaskSpec

from ..problem_form.bbob_problem import build_instance


def build_task_plan(target_size: int, ratio_bbob: float, seed: int) -> list[TaskSpec]:
    rng = np.random.RandomState(seed)
    n_bbob = int(round(target_size * ratio_bbob))
    n_bbob = max(0, min(target_size, n_bbob))
    tasks: list[TaskSpec] = []
    for i in range(target_size):
        kind = 0 if i < n_bbob else 1
        task_seed = int(rng.randint(1, 2**31 - 1))
        tasks.append(TaskSpec(sample_id=i, func_kind=kind, seed=task_seed))
    rng.shuffle(tasks)
    return tasks


def _build_single_bbob_instance(
    meta_func_id: int,
    dim: int,
    upperbound: float,
    shifted: bool,
    rotated: bool,
    biased: bool,
    seed: int,
):
    return build_instance(
        meta_func_id=meta_func_id,
        dim=dim,
        upperbound=upperbound,
        shifted=shifted,
        rotated=rotated,
        biased=biased,
        seed=seed,
    )


class WeightedSumProblem:
    def __init__(self, sub_problems: list[Any], weights: list[float]):
        self.sub_problems = sub_problems
        self.weights = np.asarray(weights, dtype=np.float64)

    def eval(self, x):
        out = None
        for w, p in zip(self.weights, self.sub_problems):
            y = np.asarray(p.eval(x), dtype=np.float64)
            out = y * w if out is None else out + y * w
        return np.asarray(out, dtype=np.float64)


def instantiate_function(task: TaskSpec, cfg: dict[str, Any]) -> tuple[Any, FuncMeta]:
    rng = np.random.RandomState(task.seed)
    if task.func_kind == 0:
        fid = int(rng.choice(cfg["bbob_ids"]))
        problem = _build_single_bbob_instance(
            meta_func_id=fid,
            dim=cfg["dim"],
            upperbound=cfg["upper_bound"],
            shifted=cfg["do_shift"],
            rotated=cfg["do_rotation"],
            biased=cfg["do_bias"],
            seed=task.seed,
        )
        meta = FuncMeta(
            sample_id=task.sample_id,
            func_kind=task.func_kind,
            task_seed=task.seed,
            base_meta_func_id=fid,
            agg="identity",
        )
        return problem, meta

    k = int(rng.randint(cfg["compose_k_min"], cfg["compose_k_max"] + 1))
    sub_ids = [int(rng.choice(cfg["bbob_ids"])) for _ in range(k)]
    sub_seeds = [int(rng.randint(1, 2**31 - 1)) for _ in range(k)]
    raw_weights = rng.dirichlet(alpha=np.ones(k)).astype(np.float64)
    weights = (raw_weights / np.sum(raw_weights)).tolist()
    sub_problems = [
        _build_single_bbob_instance(
            meta_func_id=fid,
            dim=cfg["dim"],
            upperbound=cfg["upper_bound"],
            shifted=cfg["do_shift"],
            rotated=cfg["do_rotation"],
            biased=cfg["do_bias"],
            seed=s,
        )
        for fid, s in zip(sub_ids, sub_seeds)
    ]
    problem = WeightedSumProblem(sub_problems=sub_problems, weights=weights)
    meta = FuncMeta(
        sample_id=task.sample_id,
        func_kind=task.func_kind,
        task_seed=task.seed,
        sub_meta_func_ids=sub_ids,
        sub_seeds=sub_seeds,
        weights=weights,
        agg="weighted_sum",
    )
    return problem, meta


def problem_from_meta_record(record: dict[str, Any], cfg: dict[str, Any]):
    func_kind = int(record["func_kind"])
    if func_kind == 0:
        return _build_single_bbob_instance(
            meta_func_id=int(record["base_meta_func_id"]),
            dim=cfg["dim"],
            upperbound=cfg["upper_bound"],
            shifted=cfg["do_shift"],
            rotated=cfg["do_rotation"],
            biased=cfg["do_bias"],
            seed=int(record["task_seed"]),
        )

    sub_ids = _to_int_list(record.get("sub_meta_func_ids", []))
    sub_seeds = _to_int_list(record.get("sub_seeds", []))
    weights = _to_float_list(record.get("weights", []))
    sub_problems = [
        _build_single_bbob_instance(
            meta_func_id=fid,
            dim=cfg["dim"],
            upperbound=cfg["upper_bound"],
            shifted=cfg["do_shift"],
            rotated=cfg["do_rotation"],
            biased=cfg["do_bias"],
            seed=s,
        )
        for fid, s in zip(sub_ids, sub_seeds)
    ]
    return WeightedSumProblem(sub_problems=sub_problems, weights=weights)


def _to_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    return [int(x) for x in value]


def _to_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    return [float(x) for x in value]

