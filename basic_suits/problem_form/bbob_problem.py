from problem_form.abc_problem import *

class F1(Sphere):
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Sphere'


class F101(GaussNoisyProblem, Sphere):
    gauss_beta = 0.01
    def __str__(self):
        return 'Sphere_moderate_gauss'


class F102(UniformNoisyProblem, Sphere):
    uniform_alpha = 0.01
    uniform_beta = 0.01
    def __str__(self):
        return 'Sphere_moderate_uniform'


class F103(CauchyNoisyProblem, Sphere):
    cauchy_alpha = 0.01
    cauchy_p = 0.05
    def __str__(self):
        return 'Sphere_moderate_cauchy'


class F107(GaussNoisyProblem, Sphere):
    gauss_beta = 1.
    def __str__(self):
        return 'Sphere_gauss'


class F108(UniformNoisyProblem, Sphere):
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Sphere_uniform'


class F109(CauchyNoisyProblem, Sphere):
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Sphere_cauchy'


class F2(BasicProblemBBOB):
    """
    Ellipsoidal
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Ellipsoidal'

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
        return np.sum(np.power(10, 6 * i / (nx - 1)) * (z ** 2), -1) + self.bias


class F3(BasicProblemBBOB):
    """
    Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        self.scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rastrigin'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = self.scales * bbob_asy_transform(bbob_osc_transform(shift_rotate(x, self.shift, self.rotate)), beta=0.2)
        return 10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + self.bias


class F4(BasicProblemBBOB):
    """
    Buche_Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift[::2] = np.abs(shift[::2])
        self.scales = ((10. ** 0.5) ** np.linspace(0, 1, dim))
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Buche_Rastrigin'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        z = bbob_osc_transform(z)
        even = z[:, ::2]
        even[even > 0.] *= 10.
        z *= self.scales
        return 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + 100 * bbob_pen_func(x, self.ub) + self.bias


class F5(BasicProblemBBOB):
    """
    Linear_Slope
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift = np.sign(shift)
        shift[shift == 0.] = np.random.choice([-1., 1.], size=(shift == 0.).sum())
        shift = shift * ub
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Linear_Slope'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = x.copy()
        exceed_bound = (x * self.shift) > (self.ub ** 2)
        z[exceed_bound] = np.sign(z[exceed_bound]) * self.ub  # clamp back into the domain
        s = np.sign(self.shift) * (10 ** np.linspace(0, 1, self.dim))
        return np.sum(self.ub * np.abs(s) - z * s, axis=-1) + self.bias


class F6(BasicProblemBBOB):
    """
    Attractive_Sector
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(gen_rotate_matrix_qr(dim), np.diag(scales)), rotate)
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Attractive_Sector'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
    
    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        idx = (z * self.get_optimal()) > 0.
        z[idx] *= 100.
        return bbob_osc_transform(np.sum(z ** 2, -1)) ** 0.9 + self.bias

class F7(Step_Ellipsoidal):
    def boundaryHandling(self, x):
        return bbob_pen_func(x, self.ub)

    def __str__(self):
        return 'Step_Ellipsoidal'


class F113(GaussNoisyProblem, Step_Ellipsoidal):
    gauss_beta = 1.
    def __str__(self):
        return 'Step_Ellipsoidal_gauss'


class F114(UniformNoisyProblem, Step_Ellipsoidal):
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Step_Ellipsoidal_uniform'


class F115(CauchyNoisyProblem, Step_Ellipsoidal):
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Step_Ellipsoidal_cauchy'


class F8(Rosenbrock):
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Rosenbrock_original'


class F104(GaussNoisyProblem, Rosenbrock):
    gauss_beta = 0.01
    def __str__(self):
        return 'Rosenbrock_moderate_gauss'


class F105(UniformNoisyProblem, Rosenbrock):
    uniform_alpha = 0.01
    uniform_beta = 0.01
    def __str__(self):
        return 'Rosenbrock_moderate_uniform'


class F106(CauchyNoisyProblem, Rosenbrock):
    cauchy_alpha = 0.01
    cauchy_p = 0.05
    def __str__(self):
        return 'Rosenbrock_moderate_cauchy'


class F110(GaussNoisyProblem, Rosenbrock):
    gauss_beta = 1.
    def __str__(self):
        return 'Rosenbrock_gauss'


class F111(UniformNoisyProblem, Rosenbrock):
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Rosenbrock_uniform'


class F112(CauchyNoisyProblem, Rosenbrock):
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Rosenbrock_cauchy'


class F9(BasicProblemBBOB):
    """
    Rosenbrock_rotated
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scale = max(1., dim ** 0.5 / 8.)
        self.linearTF = scale * rotate
        shift = np.matmul(0.5 * np.ones(dim), self.linearTF) / (scale ** 2)
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rosenbrock_rotated'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = np.matmul(x, self.linearTF.T) + 0.5
        return np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=-1) + self.bias


class F10(Ellipsoidal):
    condition = 1e6
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Ellipsoidal_high_cond'


class F116(GaussNoisyProblem, Ellipsoidal):
    condition = 1e4
    gauss_beta = 1.
    def __str__(self):
        return 'Ellipsoidal_gauss'


class F117(UniformNoisyProblem, Ellipsoidal):
    condition = 1e4
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Ellipsoidal_uniform'


class F118(CauchyNoisyProblem, Ellipsoidal):
    condition = 1e4
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Ellipsoidal_cauchy'


class F11(BasicProblemBBOB):
    """
    Discus
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Discus'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
        
    
    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        z = bbob_osc_transform(z)
        return np.power(10, 6) * (z[:, 0] ** 2) + np.sum(z[:, 1:] ** 2, -1) + self.bias


class F12(BasicProblemBBOB):
    """
    Bent_Cigar
    """
    beta = 0.5

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Bent_Cigar'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
    
    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        z = bbob_asy_transform(z, beta=self.beta)
        z = np.matmul(z, self.rotate.T)
        return z[:, 0] ** 2 + np.sum(np.power(10, 6) * (z[:, 1:] ** 2), -1) + self.bias


class F13(BasicProblemBBOB):
    """
    Sharp_Ridge
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10 ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(gen_rotate_matrix_qr(dim), np.diag(scales)), rotate)
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Sharp_Ridge'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        return z[:, 0] ** 2. + 100. * np.sqrt(np.sum(z[:, 1:] ** 2., axis=-1)) + self.bias

class F14(Dif_powers):
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Different_Powers'


class F119(GaussNoisyProblem, Dif_powers):
    gauss_beta = 1.
    def __str__(self):
        return 'Different_Powers_gauss'


class F120(UniformNoisyProblem, Dif_powers):
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Different_Powers_uniform'


class F121(CauchyNoisyProblem, Dif_powers):
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Different_Powers_cauchy'


class F15(BasicProblemBBOB):
    """
    Rastrigin_F15
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF = np.matmul(np.matmul(rotate, np.diag(scales)), gen_rotate_matrix_qr(dim))
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rastrigin_F15'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)

    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        z = bbob_asy_transform(bbob_osc_transform(z), beta=0.2)
        z = np.matmul(z, self.linearTF.T)
        return 10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + self.bias


class F16(BasicProblemBBOB):
    """
    Weierstrass
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (0.01 ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF = np.matmul(np.matmul(rotate, np.diag(scales)), gen_rotate_matrix_qr(dim))
        self.aK = 0.5 ** np.arange(12)
        self.bK = 3.0 ** np.arange(12)
        self.f0 = np.sum(self.aK * np.cos(np.pi * self.bK))
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Weierstrass'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
               
    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        z = np.matmul(bbob_osc_transform(z), self.linearTF.T)
        return 10 * np.power(np.mean(np.sum(self.aK * np.cos(np.matmul(2 * np.pi * (z[:, :, None] + 0.5), self.bK[None, :])), axis=-1), axis=-1) - self.f0, 3) + \
               10 / self.dim * bbob_pen_func(x, self.ub) + self.bias

class F17(Scaffer):
    condition = 10.
    def boundaryHandling(self, x):
        return 10 * bbob_pen_func(x, self.ub)

    def __str__(self):
        return 'Schaffers'


class F18(Scaffer):
    condition = 1000.
    def boundaryHandling(self, x):
        return 10 * bbob_pen_func(x, self.ub)

    def __str__(self):
        return 'Schaffers_high_cond'


class F122(GaussNoisyProblem, Scaffer):
    condition = 10.
    gauss_beta = 1.
    def __str__(self):
        return 'Schaffers_gauss'


class F123(UniformNoisyProblem, Scaffer):
    condition = 10.
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Schaffers_uniform'


class F124(CauchyNoisyProblem, Scaffer):
    condition = 10.
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Schaffers_cauchy'

class F19(Composite_Grie_rosen):
    factor = 10.
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Composite_Grie_rosen'


class F125(GaussNoisyProblem, Composite_Grie_rosen):
    factor = 1.
    gauss_beta = 1.
    def __str__(self):
        return 'Composite_Grie_rosen_gauss'


class F126(UniformNoisyProblem, Composite_Grie_rosen):
    factor = 1.
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Composite_Grie_rosen_uniform'


class F127(CauchyNoisyProblem, Composite_Grie_rosen):
    factor = 1.
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Composite_Grie_rosen_cauchy'


class F20(BasicProblemBBOB):
    """
    Schwefel
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift = 0.5 * 4.2096874633 * np.random.choice([-1., 1.], size=dim)
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Schwefel'
    
    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
            
    def func(self, x):
        self.FES += x.shape[0]
        tmp = 2 * np.abs(self.shift)
        scales = (10 ** 0.5) ** np.linspace(0, 1, self.dim)
        z = 2 * np.sign(self.shift) * x
        z[:, 1:] += 0.25 * (z[:, :-1] - tmp[:-1])
        z = 100. * (scales * (z - tmp) + tmp)
        b = 4.189828872724339
        return b - 0.01 * np.mean(z * np.sin(np.sqrt(np.abs(z))), axis=-1) + 100 * bbob_pen_func(z / 100, self.ub) + self.bias

class F21(Gallagher):
    n_peaks = 101
    def boundaryHandling(self, x):
        return bbob_pen_func(x, self.ub)

    def __str__(self):
        return 'Gallagher_101Peaks'


class F22(Gallagher):
    n_peaks = 21
    def boundaryHandling(self, x):
        return bbob_pen_func(x, self.ub)

    def __str__(self):
        return 'Gallagher_21Peaks'


class F128(GaussNoisyProblem, Gallagher):
    n_peaks = 101
    gauss_beta = 1.
    def __str__(self):
        return 'Gallagher_101Peaks_gauss'


class F129(UniformNoisyProblem, Gallagher):
    n_peaks = 101
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Gallagher_101Peaks_uniform'


class F130(CauchyNoisyProblem, Gallagher):
    n_peaks = 101
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Gallagher_101Peaks_cauchy'


class F23(BasicProblemBBOB):
    """
    Katsuura
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (100. ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(gen_rotate_matrix_qr(dim), np.diag(scales)), rotate)
        BasicProblemBBOB.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Katsuura'
    
    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
    
    def func(self, x):
        self.FES += x.shape[0]
        z = shift_rotate(x, self.shift, self.rotate)
        tmp3 = np.power(self.dim, 1.2)
        tmp1 = np.repeat(np.power(np.ones((1, 32)) * 2, np.arange(1, 33)), x.shape[0], 0)
        res = np.ones(x.shape[0])
        for i in range(self.dim):
            tmp2 = tmp1 * np.repeat(z[:, i, None], 32, 1)
            temp = np.sum(np.fabs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1, -1)
            res *= np.power(1 + (i + 1) * temp, 10 / tmp3)
        tmp = 10 / self.dim / self.dim
        return res * tmp - tmp + bbob_pen_func(x, self.ub) + self.bias


class F24(BasicProblemBBOB):
    """
    Lunacek_bi_Rastrigin
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        self.mu0 = 2.5 / 5 * ub
        shift = np.random.choice([-1., 1.], size=dim) * self.mu0 / 2
        scales = (100 ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(gen_rotate_matrix_qr(dim), np.diag(scales)), rotate)
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Lunacek_bi_Rastrigin'

    def __call__(self,x):
        if len(x.shape) == 1 and x.shape[-1] == self.dim:
            x = x.reshape(1,-1)
            return self.func(x).reshape(-1)[0]
        else:
            return self.func(x).reshape(-1)
    
    def func(self, x):
        self.FES += x.shape[0]
        x_hat = 2. * np.sign(self.shift) * x
        z = np.matmul(x_hat - self.mu0, self.rotate.T)
        s = 1. - 1. / (2. * np.sqrt(self.dim + 20.) - 8.2)
        mu1 = -np.sqrt((self.mu0 ** 2 - 1) / s)
        return np.minimum(np.sum((x_hat - self.mu0) ** 2., axis=-1), self.dim + s * np.sum((x_hat - mu1) ** 2., axis=-1)) + \
               10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + 1e4 * bbob_pen_func(x, self.ub) + self.bias
