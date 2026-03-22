from typing import Any

import numpy as np

from dataset_generation.problem_form.bbob_problem import build_instance


def sample_random_funcs(sampling_num: int, dim: int) -> list[Any]:
    """
    Return random BBOB function instances with `.eval(X)` interface.
    """
    if sampling_num <= 0:
        raise ValueError("sampling_num must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    rng = np.random.RandomState(42)
    func_ids = rng.randint(1, 25, size=sampling_num)
    seeds = rng.randint(1, 2**31 - 1, size=sampling_num)

    funcs = []
    for fid, seed in zip(func_ids, seeds):
        # Keep construction aligned with dataset_generation defaults.
        funcs.append(
            build_instance(
                meta_func_id=int(fid),
                dim=dim,
                upperbound=5.0,
                shifted=True,
                rotated=True,
                biased=False,
                seed=int(seed),
            )
        )
    return funcs

