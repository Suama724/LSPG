import numpy as np
import time

from problem_form.bbob_utils import *

class BasicProblem:
    def __init__(self):
        self.time_cost = 0

    def reset(self):
        self.time_cost = 0

    def eval(self, x):
        time_start = time.perf_counter()
        if not isinstance(x, np.ndarray):
            x = np.array(x)
       
        # ---> (nums, ndim)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x.reshape(-1, x.shape[-1])
        
        y = self.func(x)
        y = np.atleast_1d(np.asarray(y))
        time_end = time.perf_counter()
        self.time_cost += (time_end - time_start) * 1000
        if y.size == 0:
            return np.nan
        return y.flat[0] if y.size == 1 else y

    def func(self, x):
        raise NotImplementedError        
    
class BasicProblemBBOB(BasicProblem):
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        super().__init__()
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.lb = lb
        self.ub = ub
        self.FES = 0 # 记录函数分析的开销
        self.opt = self.shift
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        return self.opt

    def func(self, x):
        raise NotImplementedError
    
class NoisyProblem(BasicProblemBBOB):
    def noisy(self, ftrue):
        raise NotImplementedError

    def eval(self, x):
        ftrue = super().eval(x)
        return self.noisy(ftrue)

    def boundaryHandling(self, x):
        return 100. * bbob_pen_func(x, self.ub)

class GaussNoisyProblem(NoisyProblem):
    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased * np.exp(self.gauss_beta * np.random.randn(*ftrue_unbiased.shape))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)


class UniformNoisyProblem(NoisyProblem):
    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased * (np.random.rand(*ftrue_unbiased.shape) ** self.uniform_beta) * \
                          np.maximum(1., (1e9 / (ftrue_unbiased + 1e-99)) ** (self.uniform_alpha * (0.49 + 1. / self.dim) * np.random.rand(*ftrue_unbiased.shape)))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)


class CauchyNoisyProblem(NoisyProblem):
    def noisy(self, ftrue):
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased + self.cauchy_alpha * np.maximum(0.,
                          1e3 + (np.random.rand(*ftrue_unbiased.shape) < self.cauchy_p) * np.random.randn(*ftrue_unbiased.shape) / (np.abs(np.random.randn(*ftrue_unbiased.shape)) + 1e-199))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)


class Sphere(BasicProblemBBOB):
    """
    Abstract Sphere
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        return np.sum(z ** 2, axis=-1) + self.bias + self.boundaryHandling(x)

class Step_Ellipsoidal(BasicProblemBBOB):
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.diag(scales), rotate)
        self.Q_rotate_T = gen_rotate_matrix_qr(dim).T
        self.weights = 100 ** np.linspace(0, 1, dim)
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
            self.FES += x.shape[0]
            z_hat = shift_rotate(x, self.shift, self.rotate)
            rounded_z = np.where(np.abs(z_hat) > 0.5, 
                                np.floor(0.5 + z_hat), 
                                np.floor(0.5 + 10. * z_hat) / 10.)
            z = np.matmul(rounded_z, self.Q_rotate_T)
            main_term = np.sum(self.weights * (z ** 2), axis=-1)
            first_dim_term = np.abs(z_hat[:, 0]) / 1e4
            return 0.1 * np.maximum(first_dim_term, main_term) + \
                self.boundaryHandling(x) + self.bias

class Rosenbrock(BasicProblemBBOB):
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        super().__init__(dim, shift, rotate, bias, lb, ub)
        shift *= 0.75  # range_of_shift=0.8*0.75*ub=0.6*ub
        rotate = np.eye(dim)

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = max(1., self.dim ** 0.5 / 8.) * shift_rotate(x, self.shift, self.rotate) + 1
        return np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=-1) + self.bias + self.boundaryHandling(x)

class Ellipsoidal(BasicProblemBBOB):

    condition = None

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        nx = self.dim
        z = shift_rotate(x, self.shift, self.rotate)
        z = bbob_osc_transform(z)
        i = np.arange(nx)
        return np.sum((self.condition ** (i / (nx - 1))) * (z ** 2), -1) + self.bias + self.boundaryHandling(x)

class Dif_powers(BasicProblemBBOB):

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        i = np.arange(self.dim)
        return np.power(np.sum(np.power(np.fabs(z), 2 + 4 * i / max(1, self.dim - 1)), -1), 0.5) + self.bias + self.boundaryHandling(x)


class Scaffer(BasicProblemBBOB):
    condition = None  # need to be defined in subclass
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (self.condition ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF_T = np.matmul(np.diag(scales), gen_rotate_matrix_qr(dim)).T
        self.inv_dim_minus_1 = 1.0 / max(1, dim - 1)
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
            self.FES += x.shape[0]
            z = shift_rotate(x, self.shift, self.rotate)
            z_asy = bbob_asy_transform(z, beta=0.5) 
            z_final = np.matmul(z_asy, self.linearTF_T)
            z_sq = z_final ** 2
            s = np.sqrt(z_sq[:, :-1] + z_sq[:, 1:])
            inner_term = np.sqrt(s) * (np.sin(50 * (s ** 0.2))**2 + 1)
            return (self.inv_dim_minus_1 * np.sum(inner_term, axis=-1))**2 + \
                self.boundaryHandling(x) + self.bias
    
class Composite_Grie_rosen(BasicProblemBBOB):
    factor = None
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scale = max(1., dim ** 0.5 / 8.)
        self.linearTF = scale * rotate
        shift = np.matmul(0.5 * np.ones(dim) / (scale ** 2.), self.linearTF)
        super().__init__(dim, shift, rotate, bias, lb, ub)
        
    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
    
    def func(self, x):
        self.FES += x.shape[0]
        z = np.matmul(x, self.linearTF.T) + 0.5
        s = 100. * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (1. - z[:, :-1]) ** 2
        return self.factor + self.factor * np.sum(s / 4000. - np.cos(s), axis=-1) / (self.dim - 1.) + self.bias + self.boundaryHandling(x)

class Gallagher(BasicProblemBBOB):
    n_peaks = None

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        if self.n_peaks == 101:   # F21
            opt_shrink = 1.       # shrink of global & local optima
            global_opt_alpha = 1e3
        elif self.n_peaks == 21:  # F22
            opt_shrink = 0.98     # shrink of global & local optima
            global_opt_alpha = 1e6
        else:
            raise ValueError(f'{self.n_peaks} peaks Gallagher is not supported yet.')

        # generate global & local optima y[i]
        self.y = opt_shrink * (np.random.rand(self.n_peaks, dim) * (ub - lb) + lb)  # [n_peaks, dim]
        self.y[0] = shift * opt_shrink  # the global optimum
        shift = self.y[0]

        # generate the matrix C[i]
        sqrt_alpha = 1000 ** np.random.permutation(np.linspace(0, 1, self.n_peaks - 1))
        sqrt_alpha = np.insert(sqrt_alpha, obj=0, values=np.sqrt(global_opt_alpha))
        self.C = [np.random.permutation(sqrt_alpha[i] ** np.linspace(-0.5, 0.5, dim)) for i in range(self.n_peaks)]
        self.C = np.vstack(self.C)  # [n_peaks, dim]
        self.rotate_T = rotate.T
        # generate the weight w[i]
        self.w = np.insert(np.linspace(1.1, 9.1, self.n_peaks - 1), 0, 10.)  # [n_peaks]
        self.const_factor = -0.5 / dim
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        # x: (NP, D), self.y: (n_peaks, D), self.rotate_T: (D, D)
        z = np.matmul(x[:, np.newaxis, :] - self.y, self.rotate_T)
        exponent = self.const_factor * np.sum(self.C * (z ** 2), axis=-1)
        z_max = np.max(self.w * np.exp(exponent), axis=-1)
        
        return bbob_osc_transform(10 - z_max) ** 2 + self.bias + self.boundaryHandling(x)
    
    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
