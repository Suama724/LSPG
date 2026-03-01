import numpy as np
from copy import deepcopy
import random
import math

from . import ops

class Node:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

        # 用于树结点
        self.depth = 0
        self.parent_distance = 0
        '''
        如add(x, y), add 在idx 0, x 在1, y 在2, 
        则add...._list=[1, 2]
        '''
        self.child_distance_list = []

        # 用于维度检查
        self.input_dimension = 0
        self.output_dimension = 0

        # 约束条件
        #[aggregate权重. pow权重, 初等函数权重, exp次数]
        self.remaining = [4, 1, 1, 1]
        self.total = [0, 0, 0, 0]
        self.constant_num = 0

        # 值域, 用于后续剪枝
        self.value_range = np.array([])

    def __repr__(self):
        return self.name
    
class FunctionNode(Node):
    '''封装node'''
    def __init__(self, operator: ops.Operator):
        super().__init__(name=operator.name, arity=operator.arity)
        self.operator = operator
    
    def __call__(self, *args):
        return self.operator(*args)
    
    def __repr__(self):
        return f'{self.name}[{self.output_dimension}]'
    
class TerminalNode(Node):
    def __init__(self, name, output_dimension):
        super().__init__(name, arity=0)
        self.output_dimension = output_dimension
        self.input_dimension = 0
    
class VariableNode(TerminalNode):
    # 对输入数据X的一个切片
    def __init__(self, start, end, step, output_dimension, feature_names=None):
        name = f"X[{start}:{end}:{step}]"

        if feature_names is not None and output_dimension == 1:
            if 0 <= start < len(feature_names):
                name = feature_names[start]

        super().__init__(name, output_dimension)
        self.start = start
        self.end = end
        self.step = step
        self.tuple = (start, end, step)

    def get_data(self, X):
        cols = X.shape[1]
        real_end = cols - self.end
        return X[:, self.start : real_end : self.step]

class ConstantNode(TerminalNode):
    # 用来表示具体数值

    def __init__(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        
        '''[1.23124]-->1.231, [1,2]-->1...'''
        name = f'{value[0]:.3f}' if value.size > 0 else "Const"
        if value.size > 1: 
            name += '...'
        
        super().__init__(name, output_dimension=len(value))
        self.value = value

    def get_data(self, n_samples):
        '''(D,)->(D,1)->(batch, D, 1), batch内每个数据相同'''
        return np.tile(self.value.reshape(-1, 1), (1, n_samples))

class ExpressionTree:
    def __init__(self, program_list=None):
        # program_list 对树的扁平化结构表示
        self.program_list: list[Node] = program_list if program_list is not None else []

        self.raw_fitness_ = None
        self.fitness_ = None

        self.ela_feature = None
        self.coordi_2D = None
        self.best_dim = None

    def __len__(self):
        return len(self.program_list)
    
    def __str__(self):
        return ",".join([str(node) for node in self.program_list])
    
    @property
    def depth(self):
        if not self.program_list: 
            return 0
        return max(node.depth for node in self.program_list)
    
    def execute(self, X):
        '''
        在这里计算结果
        Args:
            X: 输入数据 [n_samples, n_features]
        Returns:
            y_pred: 计算结果 [n_samples]
        '''
        if not self.program_list:
            return np.zeros(X.shape[0])
        X_T = X.T
        n_samples = X.shape[0]

        # 存储为[FunctionNode, args_list]
        # args_list 是 [dim, n_samples](或dim, 1)的np.array
        apply_stack: list[list[FunctionNode, np.array]] = []
        
        for node in self.program_list:
            if isinstance(node, FunctionNode):
                # 开辟一个新的待办
                apply_stack.append([node, []]) 
            else: 
                val = self._get_terminal_val(node, X_T)

                if not apply_stack: # 单节点树直接算结果不进栈处理
                    return self._finalize_result(val, n_samples)

                apply_stack[-1][1].append(val) # 放到列表里

            while apply_stack and len(apply_stack[-1][1]) == apply_stack[-1][0].arity:
                func_node, args = apply_stack.pop()

                res = self._compute_function(func_node, args)
                if not apply_stack:
                    return self._finalize_result(res, n_samples)
                
                apply_stack[-1][1].append(res)
            
        return np.zeros(n_samples) # 不该到这一步

    def _get_terminal_val(self, node, X_T):
        if isinstance(node, VariableNode):
            real_end = X_T.shape[0] - node.end
            return X_T[node.start:real_end:node.step]
        elif isinstance(node, ConstantNode):
            return node.value.reshape(-1, 1)
        
        return np.array([[0.]]) # 兜底用

    def _compute_function(self, func_node: FunctionNode, args: list[np.array]):
        max_dim = 0
        for arg in args:
            if arg.shape[0] > max_dim:
                max_dim = arg.shape[0]

        if max_dim == 0 : max_dim = 1
        aligned_args = []
        for arg in args:
            curr_dim = arg.shape[0]
            if curr_dim == max_dim:
                aligned_args.append(arg)
            elif curr_dim == 1:
                #[1, n_samples] -> [max_dim, n_samples] 利用tile
                if max_dim > 1:
                    aligned_args.append(np.tile(arg, (max_dim, 1)))
                else: 
                    aligned_args.append(arg)
            else:
                if curr_dim > max_dim:
                    #截断
                    aligned_args.append(arg[:max_dim, :])
                else:
                    # 循环填充
                    repeats = int(np.ceil(max_dim / curr_dim))
                    tiled = np.tile(arg, (repeats, 1))
                    aligned_args.append(tiled[:max_dim, :])
        op = func_node.operator
        if op.is_aggregate:
            if func_node.arity == 1:
                res = op(aligned_args[0])
                if res.ndim == 1:
                    res = res.reshape(1, -1)
            else:
                res = op(np.array(aligned_args))
        else:
            res = op(*aligned_args)
        return res

    def _finalize_result(self, val, n_samples):
        if val.ndim == 1:
            return val
        if val.shape[1] == 1 and n_samples > 1:
            return np.full(n_samples, val[0, 0])
        return val[0, :]

    @classmethod
    def create_random_tree(cls,
                           random_state: np.random.Generator,
                           n_features,
                           init_depth=(2, 6),
                           method='half and half',
                           feature_names=None):
        if method == 'half and half':
            method = 'full' if random_state.integers(2) else 'grow'
        max_depth = random_state.integers(*init_depth)

        program = []
        initial_budget = [4, 1, 1, 1]
        stack = [(0, None, 1, initial_budget)]
        while stack:
            depth, parent_idx, out_dim, budget = stack.pop()

            is_terminal = False
            
            if depth >= max_depth:
                is_terminal = True
            elif method == 'full':
                is_terminal = False
            else:
                if random_state.random() < 0.1 and depth > 0: 
                    is_terminal = True

            if not is_terminal:
                valid_ops = cls._get_valid_ops(budget, out_dim)
                if not valid_ops:
                    is_terminal = True
                else:
                    op = valid_ops[random_state.integers(len(valid_ops))]
                    node = cls._create_function_node(op, random_state, n_features, out_dim, budget)
                
            if is_terminal:
                node = cls._create_terminal_node(random_state, n_features, out_dim, feature_names)
            
            node.depth = depth
            if parent_idx is not None:
                curr_idx = len(program)
                dist = curr_idx - parent_idx
                program[parent_idx].child_distance_list.append(dist)
                node.parent_distance = -dist
            
            program.append(node)

            if isinstance(node, FunctionNode):
                new_budget = node.remaining
                child_dim = node.input_dimension
                if node.operator.name == 'sum' and node.output_dimension == 1:
                    child_dim = n_features if n_features > 1 else 1

                for _ in range(node.arity):
                    stack.append((depth + 1, len(program) - 1, child_dim, new_budget))
        return cls(program)
    
    @staticmethod
    def _get_valid_ops(remaining, output_dim):
        candidates = []
        if remaining[0] >= 2:
            candidates.extend(ops.AGGREGATE_OPS)

        candidates.extend(ops.BINARY_OPS)

        for op in ops.UNARY_OPS:
            if op.name == 'exp' and remaining[3] < 1: continue
            if op.name in ['sin', 'cos', 'tan', 'log', 'tanh'] and remaining[2] < 1 : continue
            if op.name == 'pow' and remaining[1] < 1 : continue
            candidates.append(op)

        return candidates
    
    @staticmethod
    def _create_function_node(operator: ops.Operator, random_state: np.random.Generator, n_features, output_dim, parent_remaining):
        node = FunctionNode(operator)
        node.remaining = ExpressionTree._update_remaining_logic(parent_remaining, operator)
        node.output_dimension = output_dim

        if operator.is_aggregate:
            if output_dim == 1:
                node.arity = 1
                node.input_dimension = n_features if n_features > 1 else 1 # 内部视为整体
            else: 
                node.input_dimension = output_dim
                if n_features <= 1 or operator.name == 'sum':
                    node.arity = 2
                else:
                    node.arity = random_state.integers(2, 5)
        else:
            node.arity = operator.arity
            node.input_dimension = output_dim
            
        return node       

    @staticmethod
    def _create_terminal_node(random_state: np.random.Generator, n_features, output_dim, feature_names):
        if random_state.uniform() < 0.8:
            if output_dim > n_features:
                return ExpressionTree._create_constant_node(random_state, output_dim)
            max_start = n_features - output_dim
            start = random_state.integers(0, max_start + 1)
            end = n_features - start - output_dim
            
            return VariableNode(start, end, 1, output_dim, feature_names)
        else:
            return ExpressionTree._create_constant_node(random_state, output_dim)

    @staticmethod
    def _create_constant_node(random_state, output_dim):
        # 简单的随机常数生成 (-1, 1)
        val = random_state.uniform(-1, 1, size=output_dim)
        return ConstantNode(val)

    @staticmethod
    def _update_remaining_logic(remaining, operator: ops.Operator):
        new_remaining = deepcopy(remaining)
        name = operator.name

        if operator.is_aggregate:
            new_remaining[0] -= 2 
        elif name in ['min', 'max']:
            new_remaining[0] -= 1
        elif name == 'pow':
            new_remaining[1] -= 1
        elif operator.is_trigonometric or name in ['log', 'tanh']:
            new_remaining[2] -= 1
        elif name == 'exp':
            new_remaining[3] -= 1
            
        return new_remaining
#=======================================================================

    @staticmethod
    def create_function_node(operator: ops.Operator, random_state: np.random.Generator, n_features, output_dim, parent_remaining):
        node = FunctionNode(operator)
        node.remaining = ExpressionTree._update_remaining_logic(parent_remaining, operator)
        node.output_dimension = output_dim

        if operator.is_aggregate:
            if output_dim == 1:
                # 此时输出单位标量, 则对单个子树的结果进行聚合
                node.arity = 1
                node.input_dimension = 1 # 接受任意维输入, 但被内部视为整体
            else: # 此时多维向量
                if n_features <= 1:
                    node.arity = 2
                else:
                    if operator.name == 'sum': # 防溢出
                        node.arity = 2
                    else:
                        # 对于mean, prod等进行更多样的组合
                        node.arity = random_state.integers(2, 5)

                node.input_dimension = output_dim
        
        else:
            node.arity = operator.arity
            node.input_dimension = output_dim
        return node
    
    @staticmethod
    def _update_remaining_logic(remaining, operator: ops.Operator):
        new_remaining = deepcopy(remaining)
        name = operator.name

        if operator.is_aggregate:
            new_remaining[0] -= 2 # 增强限制
        elif name in ['min', 'max']:
            new_remaining[0] -= 1
        elif name == 'pow':
            new_remaining[1] -= 1
        elif operator.is_trigonometric or name in ['log', 'tanh']:
            new_remaining[2] -= 1
        elif name == 'exp':
            new_remaining[3] -= 1
            
        return new_remaining            
