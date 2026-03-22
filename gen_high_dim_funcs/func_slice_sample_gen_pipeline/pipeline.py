
import os
import pickle
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import ray

from utils.create_initial_sample import create_initial_sample
from utils.ela_feature import get_ela_feature
from net.AE import load_model, encode_ela_feats
from gp.engine import EvolutionEngine

from .sliced_problem import HighDimSlicedWrapper, SliceInfo, ProblemLike
from . import plots




class SummedSliceProgramProblem:

    def __init__(
        self,
        slice_results: List[Tuple[int, int, Any]],  # (start, end, program(have .eval()))
        full_dim: int,
    ):
        self.slice_results = slice_results
        self.full_dim = full_dim

    def eval(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        out = np.zeros(n, dtype=np.float64)
        for start, end, program in self.slice_results:
            x_slice = x[:, start:end]
            out += np.asarray(program.eval(x_slice), dtype=np.float64).ravel()
        return out


class SliceELAGenPipeline:

    def __init__(
        self,
        problem: ProblemLike,
        full_dim: int,
        model_path: str,
        scaler_path: str,
        slice_len: int = 100,
        fill_value: float = 0.0,
        num_ela_feats: int = 21,
        bound: float = 5.0,
        seed: int = 42,
        X_sampling_num_ela: int = 230,
        population_size: int = 500,
        generation: int = 20,
        n_jobs: int = 4,
        save_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.wrapper = HighDimSlicedWrapper(
            problem=problem,
            full_dim=full_dim,
            slice_len=slice_len,
            fill_value=fill_value,
        )
        self.full_dim = full_dim
        self.problem = problem
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.num_ela_feats = num_ela_feats
        self.bound = bound
        self.seed = seed
        self.X_sampling_num_ela = X_sampling_num_ela
        self.population_size = population_size
        self.generation = generation
        self.n_jobs = n_jobs
        self.device = device

        self.save_path = os.path.join(
            save_path,
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"),
        )
        os.makedirs(self.save_path, exist_ok=True)

        self.model = load_model(
            self.model_path,
            ela_feats_num=self.num_ela_feats,
            device=self.device,
        )
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.slice_outputs: List[Dict[str, Any]] = []
        self.summed_problem: Optional[SummedSliceProgramProblem] = None
        self.original_ela_feats: Optional[np.ndarray] = None
        self.original_latent: Optional[np.ndarray] = None
        self.output_ela_feats: Optional[np.ndarray] = None
        self.output_latent: Optional[np.ndarray] = None

    def _ela_encode_full_problem(
        self,
        problem: ProblemLike,
        full_dim: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        对完整高维问题做一次 ELA 并encode 返回 (ela_feats, latent) 
        但这里因为用的是100D模型评估更高维模型, 不能保证有效
        对于更高维的评估应该找学习器看直接效果, 
        毕竟目前即使仅看ela也无法做到直接对1k维问题进行ela数据集生成
        """
        X = np.array(
            create_initial_sample(
                dim=full_dim,
                n=self.X_sampling_num_ela,
                sample_type="lhs",
                lower_bound=-self.bound,
                upper_bound=self.bound,
                seed=self.seed,
            ),
            dtype=np.float64,
        )
        Y = problem.eval(X)
        if np.isnan(Y).any() or np.isinf(Y).any() or np.std(Y) < 1e-10:
            return None, None
        ela_feats, _, _ = get_ela_feature(
            problem, X, Y, random_state=self.seed, ela_conv_nsample=200
        )
        ela_feats = np.asarray(ela_feats).ravel()
        if np.isnan(ela_feats).any() or np.isinf(ela_feats).any():
            return None, None
        ela_scaled = self.scaler.transform(ela_feats.reshape(1, -1))
        latent = encode_ela_feats(
            self.model,
            torch.tensor(ela_scaled, dtype=torch.float32),
            device=self.device,
        )
        return ela_feats, np.asarray(latent).ravel()

    def _encode_program_to_latent(
        self,
        program: Any,
        dim: int,
        seed_offset: int = 0,
    ) -> Optional[np.ndarray]:
        """对给定 program encode 用于后面轨迹图"""
        X = np.array(
            create_initial_sample(
                dim=dim,
                n=self.X_sampling_num_ela,
                sample_type="lhs",
                lower_bound=-self.bound,
                upper_bound=self.bound,
                seed=self.seed + seed_offset,
            ),
            dtype=np.float64,
        )
        Y = program.eval(X)
        if np.isnan(Y).any() or np.isinf(Y).any() or np.std(Y) < 1e-10:
            return None
        ela_feats, _, _ = get_ela_feature(
            program, X, Y, random_state=self.seed, ela_conv_nsample=self.X_sampling_num_ela
        )
        ela_feats = np.asarray(ela_feats).ravel()
        if np.isnan(ela_feats).any() or np.isinf(ela_feats).any():
            return None
        ela_scaled = self.scaler.transform(ela_feats.reshape(1, -1))
        latent = encode_ela_feats(
            self.model,
            torch.tensor(ela_scaled, dtype=torch.float32),
            device=self.device,
        )
        return np.asarray(latent).ravel()

    def _run_slice_ela_encode(
        self,
        info: SliceInfo,
        slice_dir: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        dim = info.slice_dim
        X = np.array(
            create_initial_sample(
                dim=dim,
                n=self.X_sampling_num_ela,
                sample_type="lhs",
                lower_bound=-self.bound,
                upper_bound=self.bound,
                seed=self.seed + info.slice_index,
            ),
            dtype=np.float64,
        )
        Y = info.problem.eval(X)
        if np.isnan(Y).any() or np.isinf(Y).any() or np.std(Y) < 1e-10:
            return None, None
        ela_feats, _, _ = get_ela_feature(
            info.problem, X, Y, random_state=self.seed, ela_conv_nsample=200
        )
        ela_feats = np.asarray(ela_feats).ravel()
        if np.isnan(ela_feats).any() or np.isinf(ela_feats).any():
            return None, None
        ela_scaled = self.scaler.transform(ela_feats.reshape(1, -1))
        latent = encode_ela_feats(
            self.model,
            torch.tensor(ela_scaled, dtype=torch.float32),
            device=self.device,
        )
        latent = np.asarray(latent).ravel()

        np.save(os.path.join(slice_dir, "ela_feats.npy"), ela_feats)
        np.save(os.path.join(slice_dir, "latent.npy"), latent)
        return ela_feats, latent

    def _run_slice_gen_fun(
        self,
        info: SliceInfo,
        target_latent: np.ndarray,
        slice_dir: str,
    ) -> Optional[Any]:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        target_dim = info.slice_dim
        X = np.array(
            create_initial_sample(
                dim=target_dim,
                n=self.X_sampling_num_ela,
                sample_type="lhs",
                lower_bound=-self.bound,
                upper_bound=self.bound,
                seed=self.seed + info.slice_index,
            ),
            dtype=np.float64,
        )
        dataset_dict = {target_dim: (X, None)}

        engine = EvolutionEngine(
            population_size=self.population_size,
            generations=self.generation,
            n_jobs=self.n_jobs,
            random_state=self.seed + 1000 + info.slice_index,
            verbose=1,
        )
        log_path = os.path.join(slice_dir, "evolution_log.txt")
        best_program, history = engine.fit(
            dataset_dict=dataset_dict,
            model=self.model,
            scaler=self.scaler,
            target_coords=target_latent,
            n_feats=target_dim,
            log_path=log_path,
        )
        if best_program is not None:
            with open(os.path.join(slice_dir, f"slice_{info.slice_index}_program.pickle"), "wb") as f:
                pickle.dump({"program": best_program, "history": history}, f)

        trajectory_latents = self._trajectory_from_history(
            history, best_program, target_dim, seed_offset=3000 + info.slice_index
        )
        if trajectory_latents:
            plots.save_slice_trajectory_plot(
                trajectory_latents,
                trajectory_latents[-1],
                target_latent,
                slice_dir,
                info.slice_index,
                title=f"Slice {info.slice_index} [{info.start}:{info.end}]",
            )
        return best_program

    def _trajectory_from_history(
        self,
        history: Any,
        final_best_program: Any,
        dim: int,
        seed_offset: int = 0,
    ) -> List[np.ndarray]:
        """
        从 engine.fit 返回的 history 还原每代最优的 latent 序列，用于画轨迹图。
        gp.engine.EvolutionEngine 的 history 为 list[dict]，每项含 gen/best_fitness 等，不含 program，
        故当前只能得到 [encode(final_best)]；若 engine 日后在 history 中提供每代 best program，此处可自动利用。
        """
        out: List[np.ndarray] = []
        if final_best_program is None:
            return out
        # EvolutionEngine 的 history：list of dict.
        if isinstance(history, (list, tuple)) and len(history) > 0:
            first = history[0]
            if isinstance(first, dict):
                # Prefer per-generation best program if provided by engine.
                programs_to_encode: List[Any] = []
                for item in history:
                    if not isinstance(item, dict):
                        continue
                    p = item.get("best_program", None)
                    if p is not None and hasattr(p, "eval"):
                        programs_to_encode.append(p)
                if programs_to_encode:
                    for i, prog in enumerate(programs_to_encode):
                        lat = self._encode_program_to_latent(
                            prog, dim, seed_offset=seed_offset + i
                        )
                        if lat is not None:
                            out.append(lat)
                    if out:
                        return out
            # 否则按「list of programs」或「list of (gen, program)」尝试解析
            programs_to_encode: List[Any] = []
            for x in history:
                if hasattr(x, "eval"):
                    programs_to_encode.append(x)
                elif isinstance(x, (list, tuple)) and len(x) >= 2 and hasattr(x[1], "eval"):
                    programs_to_encode.append(x[1])
            if programs_to_encode:
                for i, prog in enumerate(programs_to_encode):
                    lat = self._encode_program_to_latent(
                        prog, dim, seed_offset=seed_offset + i
                    )
                    if lat is not None:
                        out.append(lat)
                return out
        elif isinstance(history, dict):
            for key in ("best_per_generation", "best_individuals", "hall_of_fame"):
                if key in history and isinstance(history[key], (list, tuple)):
                    for i, p in enumerate(history[key]):
                        if hasattr(p, "eval"):
                            lat = self._encode_program_to_latent(
                                p, dim, seed_offset=seed_offset + i
                            )
                            if lat is not None:
                                out.append(lat)
                    if out:
                        return out
                    break  # 未解析到 program，fallback 到 final_best
        # history 为空或无法解析：仅用最终最优
        lat = self._encode_program_to_latent(
            final_best_program, dim, seed_offset=seed_offset
        )
        if lat is not None:
            out.append(lat)
        return out

    def run(self) -> SummedSliceProgramProblem:

        slice_results: List[Tuple[int, int, Any]] = []
        valid_slice_infos: List[SliceInfo] = []
        task_dataset_dict: Dict[int, Any] = {}
        task_target_coords: Dict[int, np.ndarray] = {}

        # 1. 原函数整体 ELA + encode --> 保存 npy 与图 01 
        self.original_ela_feats, self.original_latent = self._ela_encode_full_problem(
            self.problem, self.full_dim
        )
        if self.original_latent is not None:
            np.save(os.path.join(self.save_path, "original_ela_feats.npy"), self.original_ela_feats)
            np.save(os.path.join(self.save_path, "original_latent.npy"), self.original_latent)

        # 2. 切片循环：先做每片 ELA+encode，并收集多任务数据
        for info in self.wrapper.iter_slices():
            slice_dir = os.path.join(self.save_path, f"slice_{info.slice_index}")
            os.makedirs(slice_dir, exist_ok=True)

            ela_feats, latent = self._run_slice_ela_encode(info, slice_dir)
            if latent is None:
                self.slice_outputs.append({
                    "slice_index": info.slice_index,
                    "start": info.start,
                    "end": info.end,
                    "ela_feats": None,
                    "latent": None,
                    "program": None,
                })
                continue

            X_ga = np.array(
                create_initial_sample(
                    dim=info.slice_dim,
                    n=self.X_sampling_num_ela,
                    sample_type="lhs",
                    lower_bound=-self.bound,
                    upper_bound=self.bound,
                    seed=self.seed + info.slice_index,
                ),
                dtype=np.float64,
            )
            task_dataset_dict[info.slice_index] = (X_ga,)
            task_target_coords[info.slice_index] = np.asarray(latent, dtype=np.float64).ravel()
            valid_slice_infos.append(info)

            self.slice_outputs.append({
                "slice_index": info.slice_index,
                "start": info.start,
                "end": info.end,
                "ela_feats": ela_feats,
                "latent": latent,
                "program": None,
            })

        # 3. 多任务一次性进化（任务=所有有效切片）
        history = []
        if task_dataset_dict:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            task_key_order = list(task_dataset_dict.keys())
            sid_to_task_idx = {sid: i for i, sid in enumerate(task_key_order)}
            task_idx_to_sid = {i: sid for sid, i in sid_to_task_idx.items()}
            first_info = valid_slice_infos[0]
            dim = first_info.slice_dim
            task_log_paths: Dict[int, str] = {}
            for sid, task_idx in sid_to_task_idx.items():
                slice_dir = os.path.join(self.save_path, f"slice_{sid}")
                os.makedirs(slice_dir, exist_ok=True)
                log_fp = os.path.join(slice_dir, "evolution_task_log.txt")
                task_log_paths[task_idx] = log_fp
                with open(log_fp, "w", encoding="utf-8") as f:
                    f.write("")

            def _append_task_logs_per_gen(gen: int, rec: dict, elapsed_s: float):
                snapshots = rec.get("task_snapshots", [])
                for task_idx, log_fp in task_log_paths.items():
                    if task_idx >= len(snapshots):
                        continue
                    snap = snapshots[task_idx]
                    sid = task_idx_to_sid.get(task_idx, -1)
                    best_program_text = str(snap.get("best_program", "None"))
                    sep = "=" * 88
                    lines = [
                        sep,
                        f"[Gen] {int(gen)}",
                        f"[SliceId] {int(sid)}",
                        f"[TaskId] {int(task_idx)}",
                        f"[TimeCostSec] {float(elapsed_s):.4f}",
                        f"[BestTotal] {float(snap.get('best_total', np.nan)):.12g}",
                        f"[BestRawDist] {float(snap.get('best_raw_dist', np.nan)):.12g}",
                        f"[BestPenalty] {float(snap.get('best_penalty', np.nan)):.12g}",
                        f"[GlobalBestFitness] {float(rec.get('best_fitness', np.nan)):.12g}",
                        f"[GlobalMeanFitness] {float(rec.get('mean_fitness', np.nan)):.12g}",
                        f"[BestProgram] {best_program_text}",
                        sep,
                        "",
                    ]
                    with open(log_fp, "a", encoding="utf-8") as f:
                        f.write("\n".join(lines))

            engine = EvolutionEngine(
                population_size=self.population_size,
                generations=self.generation,
                n_jobs=self.n_jobs,
                random_state=self.seed + 1000,
                verbose=1,
            )
            log_path = os.path.join(self.save_path, "evolution_log.txt")
            best_programs, history = engine.fit(
                dataset_dict=task_dataset_dict,
                model=self.model,
                scaler=self.scaler,
                target_coords=task_target_coords,
                n_feats=dim,
                log_path=log_path,
                on_generation_end=_append_task_logs_per_gen,
            )

            if isinstance(best_programs, dict):
                for i, out in enumerate(self.slice_outputs):
                    sid = out["slice_index"]
                    if sid in best_programs:
                        out["program"] = best_programs[sid]
            else:
                # backward compatibility for single-task return shape
                for out in self.slice_outputs:
                    if out["slice_index"] in task_dataset_dict:
                        out["program"] = best_programs

            for out in self.slice_outputs:
                if out.get("program") is not None:
                    slice_results.append((out["start"], out["end"], out["program"]))
                    sid = out["slice_index"]
                    slice_dir = os.path.join(self.save_path, f"slice_{sid}")
                    with open(os.path.join(slice_dir, f"slice_{sid}_program.pickle"), "wb") as f:
                        pickle.dump({"program": out["program"], "history": history}, f)
                    info = self.wrapper.get_slice(sid)
                    trajectory_latents = self._trajectory_from_history(
                        history, out["program"], info.slice_dim, seed_offset=3000 + sid
                    )
                    if trajectory_latents:
                        plots.save_slice_trajectory_plot(
                            trajectory_latents,
                            trajectory_latents[-1],
                            out["latent"],
                            slice_dir,
                            sid,
                            title=f"Slice {sid} [{out['start']}:{out['end']}]",
                        )

        self.summed_problem = SummedSliceProgramProblem(
            slice_results=slice_results,
            full_dim=self.full_dim,
        )
        with open(os.path.join(self.save_path, "slice_outputs.pickle"), "wb") as f:
            pickle.dump(self.slice_outputs, f)

        sliced_latents = np.array([o["latent"] for o in self.slice_outputs if o["latent"] is not None])
        if len(sliced_latents):
            np.save(os.path.join(self.save_path, "sliced_latents.npy"), sliced_latents)

        # 4. 图1: 原函数点 + 切片点（两色）
        if self.original_latent is not None and len(sliced_latents):
            plots.save_original_and_sliced_plot(
                self.original_latent,
                sliced_latents,
                os.path.join(self.save_path, plots.PLOT_01_ORIGINAL_AND_SLICED),
            )

        # 5. 对每个有 program 的切片 encode 
        generated_latents_list: List[np.ndarray] = []
        for out in self.slice_outputs:
            if out.get("program") is None:
                continue
            info = self.wrapper.get_slice(out["slice_index"])
            prog_problem = out["program"]
            X_ga = np.array(
                create_initial_sample(
                    dim=info.slice_dim,
                    n=self.X_sampling_num_ela,
                    sample_type="lhs",
                    lower_bound=-self.bound,
                    upper_bound=self.bound,
                    seed=self.seed + 2000 + out["slice_index"],
                ),
                dtype=np.float64,
            )
            Y_ga = prog_problem.eval(X_ga)
            if np.isnan(Y_ga).any() or np.isinf(Y_ga).any() or np.std(Y_ga) < 1e-10:
                continue
            ela_p, _, _ = get_ela_feature(
                prog_problem, X_ga, Y_ga, random_state=self.seed, ela_conv_nsample=200
            )
            ela_p = np.asarray(ela_p).ravel()
            if np.isnan(ela_p).any() or np.isinf(ela_p).any():
                continue
            ela_scaled_p = self.scaler.transform(ela_p.reshape(1, -1))
            lat_p = encode_ela_feats(
                self.model,
                torch.tensor(ela_scaled_p, dtype=torch.float32),
                device=self.device,
            )
            generated_latents_list.append(np.asarray(lat_p).ravel())
        if generated_latents_list:
            generated_latents = np.array(generated_latents_list)
            np.save(os.path.join(self.save_path, "generated_slice_latents.npy"), generated_latents)
        else:
            generated_latents = np.empty((0, 2))

        # 6. 输出相加后高维问题 ELA + encode --> 保存 npy / pickle
        self.output_ela_feats, self.output_latent = self._ela_encode_full_problem(
            self.summed_problem, self.full_dim
        )
        if self.output_latent is not None:
            np.save(os.path.join(self.save_path, "output_ela_feats.npy"), self.output_ela_feats)
            np.save(os.path.join(self.save_path, "output_latent.npy"), self.output_latent)
        with open(os.path.join(self.save_path, "summed_problem.pickle"), "wb") as f:
            pickle.dump(
                {
                    "summed_problem": self.summed_problem,
                    "slice_results": slice_results,
                    "full_dim": self.full_dim,
                },
                f,
            )

        # 7. 图2: 生成切片点 + 高维输出点（两色）
        if len(generated_latents) and self.output_latent is not None:
            plots.save_generated_and_output_plot(
                generated_latents,
                self.output_latent,
                os.path.join(self.save_path, plots.PLOT_02_GENERATED_AND_OUTPUT),
            )

        # 8. 图3: 图1+图2 合并（四类点）
        plots.save_full_comparison_plot(
            self.original_latent,
            sliced_latents if len(sliced_latents) else None,
            generated_latents if generated_latents_list else None,
            self.output_latent,
            os.path.join(self.save_path, plots.PLOT_03_FULL_COMPARISON),
        )

        return self.summed_problem


def run_slice_ela_gen_pipeline(
    problem: ProblemLike,
    full_dim: int,
    model_path: str,
    scaler_path: str,
    *,
    slice_len: int = 50,
    fill_value: float = 0.0,
    X_sampling_num_ela: Optional[int] = None,
    X_sampling_num_ga: Optional[int] = None,
    **kwargs: Any,
) -> SummedSliceProgramProblem:
    pipeline = SliceELAGenPipeline(
        problem=problem,
        full_dim=full_dim,
        slice_len=slice_len,
        fill_value=fill_value,
        model_path=model_path,
        scaler_path=scaler_path,
        X_sampling_num_ela=X_sampling_num_ela,
        X_sampling_num_ga=X_sampling_num_ga,
        **kwargs,
    )
    return pipeline.run()
