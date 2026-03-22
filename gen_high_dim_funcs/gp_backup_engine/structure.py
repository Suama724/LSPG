import numpy as np
from typing import List
from copy import deepcopy
import random
import math

from . import ops

class Node:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

        self.depth = 0
        '''
        如add(x, y), add 在idx 0, x 在1, y 在2, 
        则add...._list=[1, 2]
        '''
        self.child_distance_list = []
        self.output_dim = 0 
        self.input_dim = 0 # 树创建时填入具体数字

        self.is_func_node = False
        self.is_con_node = False
        self.is_var_node = False
    def __repr__(self):
        return self.name

class FunctionNode(Node):
    # 封装一个 Operator 对象
    #__slots__ = ('operator', 'input_dimension')

    def __init__(self, operator: ops.Operator, output_dim=0):
        super().__init__(name=operator.name, 
                        arity=operator.arity,
                        output_dim=output_dim)
        self.operator = operator
        self.is_func_node = True

    def __call__(self, *args):
        return self.operator(*args)

    def __repr__(self):
        return f"{self.name}[{self.output_dim}]"

class TerminalNode(Node):
    def __init__(self, name, output_dim):
        super().__init__(name, arity=0, output_dim=output_dim)

class VariableNode(TerminalNode):
    #__slots__ = ('start', 'end', 'step', 'tuple')

    def __init__(self, start: int, end: int, step: int,
                output_dim, total_dim):
        # 这里的 end 指的是到末尾的距离
        name = f"X[{start+1}:{total_dim - end}:{step}]"
        super().__init__(name, output_dim)
        self.start = start
        self.end = end
        self.step = step
        self.tuple = (self.start, self.end, self.step)

        self.is_var_node = True

class ConstantNode(TerminalNode):
    __slots__ = ("value",)

    def __init__(self, value):
        value = np.asarray(value, dtype=np.float64)
        name = f"{value[0]:.3f}" 
        if value.size > 1:
            name += "..."
        super().__init__(name, output_dim=len(value))
        self.value = value

        self.is_con_node = True
    


class ExpressionTree:
    #__slots__ = ('program_list', 'raw_dist',
    #            'fitness', 'raw_fitness_tasks_list',
    #            'best_task_id', 'scaler_fitness',
    #            'skill_factor')
    def __init__(self, program_list: List[Node]=[]):
        self.program_list = program_list
        self.raw_dist = None
        self.fitness = None
        self.raw_per_task = None
        self.best_task_id = None
        self.scaler_fitness = None
        self.skill_factor = None

    def __len__(self):
        return len(self.program_list)
    
    def __str__(self):
        return ",".join(str(n) for n in self.program_list)
    
    @property
    def depth(self):
        if not self.program_list:
            return 0
        return max(n.depth for n in self.program_list)
    
    @property
    def output_dimension(self):
        if not self.program_list:
            return 0
        return self.program_list[0].output_dimension # 嵌套结构的体现, 层层包裹


    def eval(self, X: np.array):
        # X: [B, dim]
        if not self.program_list:
            return np.zeros(X.shape[0])
        batch = X.shape[0]
        apply_stack = []

        for node in self.program_list:
            if node.is_func_node:
                apply_stack.append([node, []])
            else: 
                val = self._get_terminal_val(node, X)
                if not apply_stack:
                    return self._finalize_result(val, batch)
                apply_stack[-1][1].append(val)

            while apply_stack and len(apply_stack[-1][1]) == apply_stack[-1][0].arity:
                func_node, args = apply_stack.pop()
                res = self._compute_func(func_node, args)
                if not apply_stack:
                    return self._finalize_result(res, batch)
                apply_stack[-1][1].append(res)
            
        return np.zeros(batch)

    def _get_terminal_val(self, node: Node, X: np.array):
        if node.is_var_node:
            real_end = X.shape[1] - node.end
            return X[:, node.start:real_end:node.step]

        if node.is_con_node:
            return np.array([[node.value]])

        return np.array([[0.0]])

    def _compute_func(self, node: FunctionNode, args):
        """
        强制对齐
        会损失可表达性
        目前来说, 忍着
        """
        max_dim = max((a.shape[1] if a.ndim > 1 else 1 for a in args), default=0) or 1
        aligned = []

        for a in args:
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            batch, dim = a.shape[0], a.shape[1]

            if dim == max_dim:
                aligned.append(a)
            elif dim == 1:
                aligned.append(np.tile(a, (1, max_dim)) if max_dim > 1 else a)
            else:
                if dim > max_dim:
                    aligned.append(a[:, :max_dim])
                else:
                    reps = int(np.ceil(max_dim / dim))
                    aligned.append(np.tile(a, (1, reps))[:, :max_dim])
        res = node(*aligned)
        if res.ndim == 1:
            res = res.reshape(-1, 1)
        return res
        
    def _finalize_result(self, res: np.array, batch):
        
        # [batch, dim] -> [batch,] as output
        out = res[:, 0] if res.ndim == 2 else res
        if out.shape[0] == 1 and batch > 1:
            return np.full(batch, out[0])
        return out

    @classmethod
    def create_random_tree(cls, random_state, total_dim,
                        init_depth=(3, 6),
                        method='half and half'):
        method = 'full' if random_state.integers(2) else 'grow' if method == 'half and half' else method
        max_depth = random_state.integers(*init_depth)
        program_list = []
        stack = [(0, None, 1)] # (depth, parent_idx, out_dim)

        while stack:
            depth, parent_idx, out_dim = stack.pop()
            is_terminal = depth >= max_depth or (method == 'grow' and depth > 0 and random_state.random() < 0.15)
            if not is_terminal:
                valid = cls._get_valid_ops()
                if not valid:
                    is_terminal = True
            if is_terminal:
                node = cls._create_terminal_node(random_state,
                                                total_dim,
                                                out_dim)
            
            else: 
                    op = valid[random_state.integers(len(valid))]
                    node = cls._create_func_node(op, total_dim, out_dim)
            
            node.depth = depth
            if parent_idx is not None:
                curr = len(program_list)
                program_list[parent_idx].child_distance_list.append(curr - parent_idx)
                node.parent_distance = parent_idx - curr 
            program_list.append(node)

            if node.is_func_node:
                child_dim = node.input_dim
                for _ in range(node.arity):
                    stack.append((depth + 1, len(program_list) - 1, child_dim))

        return cls(program_list)

    @staticmethod
    def _get_valid_ops():
        out = []
        out.extend(ops.BINARY_OPS)
        out.extend(ops.UNARY_OPS)
        out.extend(ops.AGGREGATE_OPS)
        return out

    @staticmethod
    def _create_func_node(op: ops.Operator, input_dim, output_dim):
        node = FunctionNode(op)
        node.output_dim = output_dim
        if op.is_aggregate:
            node.arity = 1
            node.input_dim = input_dim 
        else:
            node.arity = op.arity
            node.input_dim = output_dim
        return node

    @staticmethod
    def _create_terminal_node(random_state, total_dim, output_dim):
        if random_state.uniform() < 0.8 and output_dim <= total_dim:
            max_start = total_dim - output_dim
            start = random_state.integers(0, max_start + 1)
            end = total_dim - start - output_dim
            return VariableNode(start, end, 1, output_dim, total_dim)
        val = random_state.uniform(-1, 1, size=output_dim)
        return ConstantNode(val)
