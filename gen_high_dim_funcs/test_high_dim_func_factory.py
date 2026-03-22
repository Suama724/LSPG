import numpy as np

"""
idx: 
1: 椭球加 Rastrigin 振荡
2:
"""



def get_problem(idx: int):
    if idx == 1:
        return HighDimEllipsoidRastrigin(dim=1000), 1000
    elif idx == 2:
        pass


    else:
        raise ValueError("Not Implemented yet")

class HighDimEllipsoidRastrigin:
    """椭球项 + Rastrigin 振荡项"""

    def __init__(self, dim):
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