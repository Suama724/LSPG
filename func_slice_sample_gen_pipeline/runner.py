
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
from renew_gp.engine import EvolutionEngine

from .sliced_problem import HighDimSlicedWrapper, SliceInfo, ProblemLike
from . import plots


class _ProgramAsProblem:
    """将 program.execute(X) 封装成带 .eval(x) 的 ProblemLike 供 ELA 使用。"""

    def __init__(self, program: Any):
        self.program = program

    def eval(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.asarray(self.program.execute(x), dtype=np.float64).ravel()

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
            out += np.asarray(program.execute(x_slice), dtype=np.float64).ravel()
        return out


class SliceELAGenPipeline:

    def __init__(
        self,
        problem: ProblemLike,
        full_dim: int,
        *,
        slice_len: int = 50,
        fill_value: float = 0.0,
        model_path: str,
        scaler_path: str,
        num_ela_feats: int = 21,
        bound: float = 5.0,
        seed: int = 42,
        X_sampling_num_ela: int = 500,
        X_sampling_num_ga: int = 500,
        population_size: int = 100,
        generation: int = 20,
        n_jobs: int = 2,
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
        self.X_sampling_num_ga = X_sampling_num_ga
        self.population_size = population_size
        self.generation = generation
        self.n_jobs = n_jobs
        self.device = device

        if save_path is None:
            "save_path": os.path.join('./artifacts', "slice_ela_gen_pipeline")
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
        n_samples: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """对完整高维问题做一次 ELA 并encode 返回 (ela_feats, latent) """
        n = n_samples if n_samples is not None else self.X_sampling_num_ela
        X = np.array(
            create_initial_sample(
                dim=full_dim,
                n=n,
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
        plots.save_slice_latent_plot(
            latent,
            info.slice_index,
            slice_dir,
            title=f"Slice {info.slice_index} [{info.start}:{info.end}] latent",
        )
        return ela_feats, latent

    def _run_slice_gen_fun(
        self,
        info: SliceInfo,
        target_latent: np.ndarray,
        slice_dir: str,
    ) -> Optional[Any]:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        dim = info.slice_dim
        X = np.array(
            create_initial_sample(
                dim=dim,
                n=self.X_sampling_num_ga,
                sample_type="lhs",
                lower_bound=-self.bound,
                upper_bound=self.bound,
                seed=self.seed + info.slice_index,
            ),
            dtype=np.float64,
        )
        dataset_dict = {dim: (X, None)}

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
            n_feats=dim,
            log_path=log_path,
        )
        if best_program is not None:
            with open(os.path.join(slice_dir, f"slice_{info.slice_index}_program.pickle"), "wb") as f:
                pickle.dump({"program": best_program, "history": history}, f)
        return best_program

    def run(self) -> SummedSliceProgramProblem:

        slice_results: List[Tuple[int, int, Any]] = []

        # 1. 原函数整体 ELA + encode --> 保存 npy 与图 01 
        self.original_ela_feats, self.original_latent = self._ela_encode_full_problem(
            self.problem, self.full_dim
        )
        if self.original_latent is not None:
            np.save(os.path.join(self.save_path, "original_ela_feats.npy"), self.original_ela_feats)
            np.save(os.path.join(self.save_path, "original_latent.npy"), self.original_latent)
            plots.save_single_point_plot(
                self.original_latent,
                os.path.join(self.save_path, plots.PLOT_01_ORIGINAL),
                title="Original function (full-dim ELA → latent)",
                color="green",
            )

        # 2. 切片循环：每片 ELA+encode --> 保存 Gen Fun
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

            program = self._run_slice_gen_fun(info, latent, slice_dir)
            self.slice_outputs.append({
                "slice_index": info.slice_index,
                "start": info.start,
                "end": info.end,
                "ela_feats": ela_feats,
                "latent": latent,
                "program": program,
            })
            if program is not None:
                slice_results.append((info.start, info.end, program))

        self.summed_problem = SummedSliceProgramProblem(
            slice_results=slice_results,
            full_dim=self.full_dim,
        )
        with open(os.path.join(self.save_path, "slice_outputs.pickle"), "wb") as f:
            pickle.dump(self.slice_outputs, f)

        # 3. 切片坐标
        sliced_latents = np.array([o["latent"] for o in self.slice_outputs if o["latent"] is not None])
        if len(sliced_latents):
            np.save(os.path.join(self.save_path, "sliced_latents.npy"), sliced_latents)
            plots.save_points_plot(
                sliced_latents,
                os.path.join(self.save_path, plots.PLOT_02_SLICED),
                title="Sliced functions (each slice ELA → latent)",
                color="blue",
            )

        # 4. 对每个有 program 的切片 encode 
        generated_latents_list: List[np.ndarray] = []
        for out in self.slice_outputs:
            if out.get("program") is None:
                continue
            info = self.wrapper.get_slice(out["slice_index"])
            prog_problem = _ProgramAsProblem(out["program"])
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
            plots.save_points_plot(
                generated_latents,
                os.path.join(self.save_path, plots.PLOT_03_GENERATED),
                title="Generated slice functions (each fitted program ELA → latent)",
                color="orange",
            )
        else:
            generated_latents = np.empty((0, 2))

        # 5. 输出相加后高维问题整体 ELA + encode --> 保存 npy pickle 与图 04 
        self.output_ela_feats, self.output_latent = self._ela_encode_full_problem(
            self.summed_problem, self.full_dim
        )
        if self.output_latent is not None:
            np.save(os.path.join(self.save_path, "output_ela_feats.npy"), self.output_ela_feats)
            np.save(os.path.join(self.save_path, "output_latent.npy"), self.output_latent)
            plots.save_single_point_plot(
                self.output_latent,
                os.path.join(self.save_path, plots.PLOT_04_OUTPUT),
                title="Output summed function (full-dim ELA → latent)",
                color="red",
            )
        with open(os.path.join(self.save_path, "summed_problem.pickle"), "wb") as f:
            pickle.dump(
                {
                    "summed_problem": self.summed_problem,
                    "slice_results": slice_results,
                    "full_dim": self.full_dim,
                },
                f,
            )

        # 6. 总图: 原函数 / 切片 / 生成切片 / 输出 
        plots.save_comparison_plot(
            self.original_latent,
            sliced_latents if len(sliced_latents) else None,
            generated_latents if generated_latents_list else None,
            self.output_latent,
            os.path.join(self.save_path, plots.PLOT_05_COMPARISON),
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
