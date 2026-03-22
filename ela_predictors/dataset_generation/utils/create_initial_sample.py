import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, Sobol
from typing import List, Optional, Union


def create_initial_sample(
    dim: int,
    n: Optional[int] = None,
    sample_coefficient: int = 50,
    lower_bound: Union[List[float], float] = 0,
    upper_bound: Union[List[float], float] = 1,
    sample_type: str = "lhs",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if sample_type not in ["lhs", "random", "sobol"]:
        raise ValueError('Unknown sample type selected. Valid options are "lhs", "sobol", and "random"')

    if not isinstance(lower_bound, list) and type(lower_bound) is not np.ndarray:
        lower_bound = np.array([lower_bound] * dim)
    if isinstance(lower_bound, list):
        lower_bound = np.array(lower_bound)

    if not isinstance(upper_bound, list) and type(upper_bound) is not np.ndarray:
        upper_bound = np.array([upper_bound] * dim)
    if isinstance(upper_bound, list):
        upper_bound = np.array(upper_bound)

    if len(lower_bound) != dim or len(upper_bound) != dim:
        raise ValueError("Length of lower-/upper bound is not the same as the problem dimension")
    if not (lower_bound < upper_bound).all():
        raise ValueError("Not all elements of lower bound are smaller than upper bound")
    if n is None:
        n = dim * sample_coefficient

    if seed is not None:
        np.random.seed(seed)

    if sample_type == "lhs":
        sampler = LatinHypercube(d=dim, seed=seed)
        X = sampler.random(n=n)
    elif sample_type == "sobol":
        sampler = Sobol(d=dim, seed=50)
        X = sampler.random(n)
    else:
        X = np.random.rand(n, dim)

    X = X * (upper_bound - lower_bound) + lower_bound
    colnames = ["x" + str(x) for x in range(dim)]
    return pd.DataFrame(X, columns=colnames)

