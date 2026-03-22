import numpy as np
from .structure import ExpressionTree, FunctionNode

def length_penalty(tree: ExpressionTree) -> float:
    return float(len(tree))

def depth_penalty(tree: ExpressionTree) -> float:
    return float(tree.depth)

def dimension_penalty(tree: ExpressionTree, target_dim):
    used_idx = set()
    for node in tree.program_list:
        if not getattr(node, "is_var_node", False):
            continue
        start = int(getattr(node, "start", 0))
        step = int(getattr(node, "step", 1))
        width = int(getattr(node, "output_dim", 0))
        if width <= 0:
            continue
        for i in range(width):
            used_idx.add(start + i * step)
    return float(abs(target_dim - len(used_idx)))

def op_cost_penalty(tree: ExpressionTree, heavy_ops=None, **kwargs) -> float:
    if heavy_ops is None:
        heavy_ops = {'exp', 'log', 'pow', 'sin', 'cos', 'tan', 'tanh'}
    count = 0
    for node in tree.program_list:
        if isinstance(node, FunctionNode) and node.name in heavy_ops:
            count += 1
    return float(count)

def symbol_cost_penalty(tree: ExpressionTree, budget=None, **kwargs) -> float:
    """
    Total "symbol cost" in the same units as the old remaining budget.
    Cost per op: sum/prod/mean=2, min/max=1, pow=1, sin/cos/tan/log/tanh=1, exp=1.
    If budget=(agg,pow,elem,exp) is given, returns max(0, total_agg-agg) + ... (over-budget part).
    Else returns raw total cost (then you tune weight to soft-limit).
    """
    agg_cost, pow_c, elem_c, exp_c = 0, 0, 0, 0
    for node in tree.program_list:
        if not isinstance(node, FunctionNode):
            continue
        name = node.name
        if name in ('sum', 'prod', 'mean'):
            agg_cost += 2
        elif name in ('min', 'max'):
            agg_cost += 1
        elif name == 'pow':
            pow_c += 1
        elif name == 'exp':
            exp_c += 1
        elif name in ('sin', 'cos', 'tan', 'log', 'tanh'):
            elem_c += 1
    if budget is not None:
        agg_b, pow_b, elem_b, exp_b = budget
        return (max(0, agg_cost - agg_b) + max(0, pow_c - pow_b) +
                max(0, elem_c - elem_b) + max(0, exp_c - exp_b))
    return float(agg_cost + pow_c + elem_c + exp_c)



def compute_penalty(tree: ExpressionTree,
                    w_len: float,
                    w_depth: float,
                    w_dim: float,
                    w_symbol_cost: float,
                    target_dim=None) -> float:
    total = 0.
    dim_term = 0.0
    if target_dim is not None:
        dim_term = w_dim * dimension_penalty(tree, target_dim)
    total += (
        w_len * length_penalty(tree)
        + w_depth * depth_penalty(tree)
        + dim_term
        + w_symbol_cost * symbol_cost_penalty(tree)
    )
    return total
    