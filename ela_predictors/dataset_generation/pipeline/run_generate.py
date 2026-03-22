from __future__ import annotations

from dataclasses import asdict
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
from total_files.config import normalize_pipeline_config
from .feature_pipeline import EncoderBundle, evaluate_and_encode
from .function_factory import build_task_plan, instantiate_function
from .storage import ChunkStorageManager
from .types import SamplePayload, TaskSpec

import ray
from utils.create_initial_sample import create_initial_sample

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def run_generate(cfg: dict[str, Any], resume: bool = False) -> str:
    config = normalize_pipeline_config(cfg)
    storage = ChunkStorageManager(cfg=config, resume=resume)
    log_path = Path(storage.run_dir) / config["progress_log_filename"]
    _log_progress(log_path, f"[Init] run_dir={storage.run_dir}")
    _log_progress(log_path, f"[Init] resume={resume}, use_ray={config['use_ray']}, workers={config['num_workers']}")

    fixed_X, x_source = _load_or_create_fixed_x(config=config, run_dir=storage.run_dir)
    _log_progress(log_path, f"[Init] fixed_X={x_source}, shape={tuple(fixed_X.shape)}")

    tasks = build_task_plan(config["target_size"], config["ratio_bbob"], config["seed"])
    done = storage.completed_size
    if done > 0:
        tasks = [t for t in tasks if t.sample_id >= done]
    tasks = sorted(tasks, key=lambda x: x.sample_id)
    _log_progress(log_path, f"[Init] total_tasks={len(tasks)}, completed_before_resume={done}")

    if config["use_ray"]:
        if ray is None:
            raise RuntimeError("Ray is not available. Please install ray or set use_ray=False.")
        return _run_generate_parallel(config=config, storage=storage, tasks=tasks, fixed_X=fixed_X, log_path=log_path)
    return _run_generate_sequential(config=config, storage=storage, tasks=tasks, fixed_X=fixed_X, log_path=log_path)


def _run_generate_sequential(
    config: dict[str, Any],
    storage: ChunkStorageManager,
    tasks: list[TaskSpec],
    fixed_X: np.ndarray,
    log_path: Path,
) -> str:
    encoder = EncoderBundle(config)
    batch: list[SamplePayload] = []
    total = len(tasks)
    succeeded = 0
    failed = 0
    started_at = dt.datetime.now()
    pbar = _new_pbar(total, "Generating(sequential)")

    for task in tasks:
        payload = None
        for retry in range(config["max_retries"]):
            try:
                problem, meta = instantiate_function(task, config)
                payload = evaluate_and_encode(
                    problem=problem,
                    sample_seed=task.seed + retry,
                    cfg=config,
                    encoder=encoder,
                    func_kind=task.func_kind,
                    meta=meta,
                    fixed_X=fixed_X,
                )
                if payload is not None:
                    break
            except Exception:
                payload = None
        if payload is None:
            failed += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"ok": succeeded, "fail": failed, "buf": len(batch)})
            _maybe_log_checkpoint(log_path, config, started_at, succeeded + failed, total, succeeded, failed)
            continue

        succeeded += 1
        batch.append(payload)
        if len(batch) >= config["chunk_size"]:
            wr = storage.append_batch(batch)
            _log_progress(log_path, f"[Chunk] wrote={wr.chunk_file}, size={wr.n_samples}, chunk_id={wr.chunk_id}")
            batch = []
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"ok": succeeded, "fail": failed, "buf": len(batch)})
        _maybe_log_checkpoint(log_path, config, started_at, succeeded + failed, total, succeeded, failed)

    if batch:
        wr = storage.append_batch(batch)
        _log_progress(log_path, f"[Chunk] wrote={wr.chunk_file}, size={wr.n_samples}, chunk_id={wr.chunk_id}")
    storage.finalize_manifest()
    if pbar is not None:
        pbar.close()
    elapsed = (dt.datetime.now() - started_at).total_seconds()
    _log_progress(log_path, f"[Done] mode=sequential, total={total}, ok={succeeded}, fail={failed}, elapsed_s={elapsed:.2f}")
    return storage.run_dir


def _run_generate_parallel(
    config: dict[str, Any],
    storage: ChunkStorageManager,
    tasks: list[TaskSpec],
    fixed_X: np.ndarray,
    log_path: Path,
) -> str:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    worker_num = min(config["num_workers"], max(1, len(tasks)))
    workers = [RayGenerateWorker.remote(config, fixed_X) for _ in range(worker_num)]
    in_flight = {}
    next_worker = 0
    cursor = 0
    batch: list[SamplePayload] = []
    total = len(tasks)
    succeeded = 0
    failed = 0
    started_at = dt.datetime.now()
    pbar = _new_pbar(total, "Generating(ray)")
    _log_progress(log_path, f"[Ray] workers={worker_num}, inflight_cap={worker_num * 2}")

    while cursor < len(tasks) or in_flight:
        while cursor < len(tasks) and len(in_flight) < worker_num * 2:
            task = tasks[cursor]
            w = workers[next_worker]
            ref = w.process_task.remote(asdict(task))
            in_flight[ref] = task.sample_id
            cursor += 1
            next_worker = (next_worker + 1) % worker_num

        done_refs, _ = ray.wait(
            list(in_flight.keys()),
            num_returns=min(worker_num, len(in_flight)),
            timeout=config["ray_wait_timeout_s"],
        )
        if not done_refs:
            continue

        for ref in done_refs:
            in_flight.pop(ref, None)
            res = ray.get(ref)
            if res is None:
                failed += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"ok": succeeded, "fail": failed, "in_flight": len(in_flight), "buf": len(batch)})
                _maybe_log_checkpoint(log_path, config, started_at, succeeded + failed, total, succeeded, failed)
                continue
            payload = SamplePayload(
                y=res["y"],
                z2d=res["z2d"],
                seed=int(res["seed"]),
                func_kind=int(res["func_kind"]),
                meta=res["meta"],
            )
            succeeded += 1
            batch.append(payload)
            if len(batch) >= config["chunk_size"]:
                wr = storage.append_batch(batch)
                _log_progress(log_path, f"[Chunk] wrote={wr.chunk_file}, size={wr.n_samples}, chunk_id={wr.chunk_id}")
                batch = []
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"ok": succeeded, "fail": failed, "in_flight": len(in_flight), "buf": len(batch)})
            _maybe_log_checkpoint(log_path, config, started_at, succeeded + failed, total, succeeded, failed)

    if batch:
        wr = storage.append_batch(batch)
        _log_progress(log_path, f"[Chunk] wrote={wr.chunk_file}, size={wr.n_samples}, chunk_id={wr.chunk_id}")
    storage.finalize_manifest()
    if pbar is not None:
        pbar.close()
    elapsed = (dt.datetime.now() - started_at).total_seconds()
    _log_progress(log_path, f"[Done] mode=ray, total={total}, ok={succeeded}, fail={failed}, elapsed_s={elapsed:.2f}")
    return storage.run_dir


def _load_or_create_fixed_x(config: dict[str, Any], run_dir: str) -> tuple[np.ndarray, str]:
    x_path = Path(run_dir) / config["x_filename"]
    if x_path.exists():
        X = np.load(x_path)
        source = "loaded"
    else:
        X = np.asarray(
            create_initial_sample(
                dim=config["dim"],
                n=config["n_eval"],
                sample_type=config["sample_type"],
                lower_bound=config["lower_bound"],
                upper_bound=config["upper_bound"],
                seed=config["x_seed"],
            ),
            dtype=np.float64,
        )
        np.save(x_path, X)
        source = "created"
    if X.shape != (config["n_eval"], config["dim"]):
        raise ValueError(f"Fixed X shape mismatch: got {X.shape}, expect ({config['n_eval']}, {config['dim']})")
    return np.asarray(X, dtype=np.float64), source


def _new_pbar(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit="task", dynamic_ncols=True)


def _log_progress(log_path: Path, message: str) -> None:
    line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if tqdm is not None:
        tqdm.write(line)
    else:
        print(line)


def _maybe_log_checkpoint(
    log_path: Path,
    config: dict[str, Any],
    started_at: dt.datetime,
    completed: int,
    total: int,
    succeeded: int,
    failed: int,
) -> None:
    if completed <= 0:
        return
    if completed % config["log_every_n"] != 0 and completed != total:
        return
    elapsed = (dt.datetime.now() - started_at).total_seconds()
    rate = completed / max(elapsed, 1e-9)
    eta = (total - completed) / max(rate, 1e-9)
    _log_progress(
        log_path,
        f"[Progress] done={completed}/{total}, ok={succeeded}, fail={failed}, speed={rate:.2f} task/s, eta_s={eta:.1f}",
    )


if ray is not None:
    @ray.remote
    class RayGenerateWorker:
        def __init__(self, cfg_dict: dict[str, Any], fixed_X: np.ndarray):
            self.config = normalize_pipeline_config(cfg_dict)
            self.encoder = EncoderBundle(self.config)
            self.fixed_X = np.asarray(fixed_X, dtype=np.float64)

        def process_task(self, task_dict: dict[str, Any]) -> dict[str, Any] | None:
            task = TaskSpec(**task_dict)
            payload = None
            for retry in range(self.config["max_retries"]):
                try:
                    problem, meta = instantiate_function(task, self.config)
                    payload = evaluate_and_encode(
                        problem=problem,
                        sample_seed=task.seed + retry,
                        cfg=self.config,
                        encoder=self.encoder,
                        func_kind=task.func_kind,
                        meta=meta,
                        fixed_X=self.fixed_X,
                    )
                    if payload is not None:
                        break
                except Exception:
                    payload = None
            if payload is None:
                return None
            return {
                "y": payload.y,
                "z2d": payload.z2d,
                "seed": int(payload.seed),
                "func_kind": int(payload.func_kind),
                "meta": payload.meta,
            }

