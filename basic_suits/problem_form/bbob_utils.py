import numpy as np

# func for shift and rotate
def  shift_rotate(x: np.array, Os: np.array, Mr: np.array):
    y = x[:, :Os.shape[-1]] - Os
    return np.matmul(Mr, y.transpose()).transpose()

def gen_rotate_matrix_householder(dim, random_state=None):  # T(f): O(n^4)
    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
        x = rng.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    H = (D * H.T).T
    return H

def gen_rotate_matrix_qr(dim, random_state=None): # T(f): O(n^3) 
    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state    
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    ph = d / np.absolute(d)
    q *= ph
    return q

def bbob_osc_transform(x):
    y = np.copy(x)
    idx_p = x > 0
    if np.any(idx_p):
        log_x = np.log(x[idx_p]) / 0.1
        y[idx_p] = np.exp(0.1 * (log_x + 0.49 * (np.sin(log_x) + np.sin(0.79 * log_x))))
    idx_n = x < 0
    if np.any(idx_n):
        log_x = np.log(-x[idx_n]) / 0.1
        y[idx_n] = -np.exp(0.1 * (log_x + 0.49 * (np.sin(0.55 * log_x) + np.sin(0.31 * log_x))))
    return y

def bbob_asy_transform(x, beta):
    NP, dim = x.shape
    y = np.copy(x)
    idx = x > 0
    lin_space = np.linspace(0, 1, dim)
    safe_sqrt = np.zeros_like(x)
    np.sqrt(x, where=idx, out=safe_sqrt)
    exponent = 1.0 + beta * lin_space * safe_sqrt
    y[idx] = x[idx] ** exponent[idx]
    return y

def bbob_pen_func(x, ub):
    return np.sum(np.maximum(0., np.abs(x) - ub) ** 2, axis=-1)
