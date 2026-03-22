import numpy as np
from joblib import wrap_non_picklable_objects

__all__ = [
    'Operator',
    'make_operator',
    'ALL_OPS',
    'AGGREGATE_OPS',
    'BINARY_OPS',
    'UNARY_OPS'
]

class Operator:
    def __init__(self, function, name, arity: int,
                is_aggregate=False,
                is_commutative=False,
                is_protected=False,
                is_trigonometric=False):
        self.function = function
        self.name = name
        self.arity = arity

        self.is_aggregate = is_aggregate
        self.is_commutative = is_commutative
        self.is_protected = is_protected
        self.is_trigonometric = is_trigonometric

    def __call__(self, *args):
        return self.function(*args)
    
    def __repr__(self):
        return f"Operator({self.name}, arity={self.arity})"
    
    def __str__(self):
        return self.name

def make_operator(function, name, arity, **kwargs):
    
    wrapped_func = wrap_non_picklable_objects(function)

    return Operator(function=wrapped_func, name=name, arity=arity, **kwargs)


def _protected_division(x1, x2):
    '''分母接近0时返回1'''
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

def _protected_sqrt(x1):
    """对绝对值开方"""
    return np.sqrt(np.abs(x1))

def _protected_log(x1):
    """对绝对值取对数，接近 0 时返回 0"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def _protected_inverse(x1):
    """倒数接近 0 时返回 0"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)

def _protected_exp(x1):
    """指数防止溢出导致 Inf, 这里选择截断输入"""
    with np.errstate(over='ignore', invalid='ignore'):
        return np.exp(np.clip(x1, -100, 100))

def _protected_power(x1, x2):
    """
    幂运算(x1 ^ x2)
    如果底数绝对值接近0且指数为负, 或者底数为负且指数不是整数，则返回 1.
    """
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        '''转换为 float64 避免整型溢出'''
        base = np.array(x1, dtype=np.float64)
        exp = np.array(x2)
        # 1. 底数过小 且 指数 < 0 -> 导致除零 -> 返回 1.0
        # 2. 底数 < 0 且 指数非整数 -> 复数域 -> 返回 1.0 
        # 原代码逻辑：(abs(X) > 0.0001) | (c >= 0) 才计算
        return np.where((np.abs(base) > 0.001) | (exp >= 0), np.power(base, exp), 1.)

def _protected_sub(x1, x2):
    """减法处理溢出"""
    with np.errstate(over='ignore', invalid='ignore'):
        return np.subtract(x1, x2)

def _sigmoid(x1):
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def _tanh(x1):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        return np.tanh(x1)


# [B, dim]
def _sum_func(x): return np.sum(x, axis=1, keepdims=True)
def _prod_func(x): return np.prod(x, axis=1, keepdims=True)
def _mean_func(x): return np.mean(x, axis=1, keepdims=True)

# Arity 2 (Binary)
add  = Operator(np.add,              'add',  2, is_commutative=True)
sub  = Operator(_protected_sub,      'sub',  2, is_commutative=False)
mul  = Operator(np.multiply,         'mul',  2, is_commutative=True)
div  = Operator(_protected_division, 'div',  2, is_commutative=False, is_protected=True)
max_ = Operator(np.maximum,          'max',  2, is_commutative=True)
min_ = Operator(np.minimum,          'min',  2, is_commutative=True)
pow_ = Operator(_protected_power,    'pow',  2, is_commutative=False, is_protected=True)

# Arity 1 (Unary)
sqrt = Operator(_protected_sqrt,     'sqrt', 1, is_protected=True)
log  = Operator(_protected_log,      'log',  1, is_protected=True)
abs_ = Operator(np.abs,              'abs',  1)
neg  = Operator(np.negative,         'neg',  1)
inv  = Operator(_protected_inverse,  'inv',  1, is_protected=True)
sin  = Operator(np.sin,              'sin',  1, is_trigonometric=True)
cos  = Operator(np.cos,              'cos',  1, is_trigonometric=True)
tan  = Operator(np.tan,              'tan',  1, is_trigonometric=True)
tanh = Operator(_tanh,               'tanh', 1)
exp  = Operator(_protected_exp,      'exp',  1)
sig  = Operator(_sigmoid,            'sig',  1)

# Arity 0 -> Aggregate Functions
# 操作数数量可能在变异中改变，或者作用于整个向量
# arity=0 作为标记
sum_op  = Operator(_sum_func,  'sum',  1, is_aggregate=True)
prod_op = Operator(_prod_func, 'prod', 1, is_aggregate=True)
mean_op = Operator(_mean_func, 'mean', 1, is_aggregate=True)

ALL_OPS = [
    add, sub, mul, div, max_, min_, pow_,  # two
    sqrt, log, abs_, neg, inv, sin, cos, tan, tanh, exp, sig, # one
    sum_op, prod_op, mean_op # Aggregate
]

OP_MAP = {op.name: op for op in ALL_OPS}

AGGREGATE_OPS = [op for op in ALL_OPS if op.is_aggregate]

BINARY_OPS = [op for op in ALL_OPS if op.arity == 2]

UNARY_OPS = [op for op in ALL_OPS if op.arity == 1]

TRIG_OPS = [op for op in ALL_OPS if op.is_trigonometric]

ELEMENTARY_OPS = [sin, cos, tan, log, tanh]


if __name__ == "__main__":
    print("All Operators:", [op.name for op in ALL_OPS])
    print("Aggregate Ops:", [op.name for op in AGGREGATE_OPS])
    print("Is 'add' commutative?", add.is_commutative)
    print("Testing Protected Div(1, 0):", div(np.array([1.]), np.array([0.])))

