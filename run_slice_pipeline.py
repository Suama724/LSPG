import numpy as np

from config import config_slice_ela_gen
from func_slice_sample_gen_pipeline import SliceELAGenPipeline


class HighDimEllipsoidRastrigin:
    """300D 椭球项 + Rastrigin 振荡项"""

    def __init__(self, dim=300):
        self.dim = dim
        # 条件数约 1e6 的权重
        self.weights = 10.0 ** (6 * np.linspace(0, 1, dim))

    def eval(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n, d = x.shape
        if d != self.dim:
            raise ValueError(f"{self.dim}D expected, but get {d}D")
        ellipsoid = np.sum(self.weights * (x ** 2), axis=1)
        rastrigin_term = -10 * np.sum(np.cos(2 * np.pi * x), axis=1) + 10 * self.dim
        return (ellipsoid + rastrigin_term).ravel()

'''
def get_problem():
    """
    返回 (problem, full_dim)
    problem 需实现 .eval(x), x 为 (n, full_dim) 的数组 
    """
'''

def get_problem():
    full_dim = 300
    problem = HighDimEllipsoidRastrigin(dim=full_dim)
    return problem, full_dim


def main():
    cfg = config_slice_ela_gen
    problem, full_dim = get_problem()

    pipeline = SliceELAGenPipeline(
        problem=problem,
        full_dim=full_dim,
        slice_len=cfg.get("slice_len", 50),
        fill_value=cfg.get("fill_value", 0.0),
        model_path=cfg["model_path"],
        scaler_path=cfg["scaler_path"],
        num_ela_feats=cfg.get("num_ela_feats", 21),
        bound=cfg.get("bound", 5.0),
        seed=cfg.get("seed", 42),
        X_sampling_num_ela=cfg.get("X_sampling_num_ela", 500),
        X_sampling_num_ga=cfg.get("X_sampling_num_ga", 500),
        population_size=cfg.get("population_size", 100),
        generation=cfg.get("generation", 20),
        n_jobs=cfg.get("n_jobs", 2),
        save_path=cfg.get("save_path"),
    )
    summed = pipeline.run()
    print(f"Pipeline finished. Output directory: {pipeline.save_path}")
    return summed


if __name__ == "__main__":
    main()
