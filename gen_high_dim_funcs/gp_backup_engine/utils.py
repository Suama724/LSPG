import numbers
from joblib import cpu_count

import numpy as np

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(int(seed))
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(f"{seed!r} cannot be used to seed a numpy Generator")


def _get_n_jobs(n_jobs):
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    if n_jobs == 0:
        raise ValueError("n_jobs == 0 has no meaning")
    return n_jobs


def partition_estimators(n_estimators, n_jobs):
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)
    per_job = (n_estimators // n_jobs) * np.ones(n_jobs, dtype=int)
    per_job[: n_estimators % n_jobs] += 1
    return n_jobs, per_job.tolist()
