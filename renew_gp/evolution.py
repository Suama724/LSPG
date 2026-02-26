import random
import numpy as np
from copy import deepcopy

from .structure import ExpressionTree, FunctionNode, TerminalNode, ConstantNode
from .structure import Node

from . import ops

class EvolutionOps:
    @staticmethod
    def _get_candidates_by_dim(program_list: list[Node]):
        '''
        查看每个结点的output_dim, 然后将所有dim相同的program的索引记录在一起
        如dim_1: [2, 6]
        dim_10: [3, 5]
        '''
        candidates: dict[int, list[int]] = {}
        for i, node in enumerate(program_list):
            dim = node.output_dimension
            if dim not in candidates:
                candidates[dim] = []
            candidates[dim].append(i)
        return candidates

    @staticmethod
    def _get_subtree_span(program_list: list[Node], start_idx: int):
        '''找子树范围'''
        stack = 1
        end = start_idx
        while stack > 0:
            if end >= len(program_list):
                '''bug'''
                raise IndexError("Program incomplete or logic error in subtree span.")
            
            node = program_list[end]
            stack += node.arity - 1
            end += 1
        return start_idx, end

    @staticmethod
    def _generate_random_subtree_by_dim(random_state: np.random.Generator, 
                                        n_features: int, 
                                        output_dim: int,
                                        init_depth):
        max_depth = random_state.integers(*init_depth)
        stack = [(0, output_dim, [4, 1, 1, 1])]
        program = []

        while stack:
            depth, output_dim, budget = stack.pop()

            is_terminal = (depth >= max_depth) or (random_state.random() < 0.1 and depth > 0)
            node = None
            if not is_terminal:
                valid_ops = ExpressionTree._get_valid_ops(budget, output_dim)
                if valid_ops:
                    op = valid_ops[random_state.integers(len(valid_ops))]
                    node = ExpressionTree._create_function_node(op, random_state, n_features, output_dim, budget)
                else: 
                    is_terminal = True
            
            if is_terminal:
                node = ExpressionTree._create_terminal_node(random_state, n_features, output_dim, None)
            
            node.depth = depth
            program.append(node)

            if isinstance(node, FunctionNode):
                child_dim = node.input_dimension
                if node.operator.name == 'sum' and node.output_dimension == 1:
                    child_dim = n_features if n_features > 1 else 1
                
                for _ in range(node.arity):
                    stack.append((depth + 1, child_dim, node.remaining))
            
        return program 

#=====================================================

    @staticmethod
    def crossover(donor: ExpressionTree,
                  receiver: ExpressionTree,
                  random_state: np.random.Generator):
        receiver_candidates = EvolutionOps._get_candidates_by_dim(receiver.program_list)
        donor_candidates = EvolutionOps._get_candidates_by_dim(donor.program_list)

        common_dims = set(receiver_candidates.keys()) & set(donor_candidates.keys())

        '''无共同维度时直接返回副本不交叉'''
        if not common_dims:
            return deepcopy(receiver)

        chosen_dim = random_state.choice(list(common_dims))

        receiver_idx = random_state.choice(receiver_candidates[chosen_dim])
        donor_idx = random_state.choice(donor_candidates[chosen_dim])

        r_start, r_end = EvolutionOps._get_subtree_span(receiver.program_list, receiver_idx)
        d_start, d_end = EvolutionOps._get_subtree_span(donor.program_list, donor_idx)

        new_program_list = (
            receiver.program_list[ : r_start] + 
            donor.program_list[d_start : d_end] + 
            receiver.program_list[r_end : ]
        )

        return ExpressionTree(new_program_list)
    
    @staticmethod
    def subtree_mutation(individual: ExpressionTree,
                         random_state: np.random.Generator,
                         n_features: int,
                         init_depth = (0, 2),
                         feature_names = None):
        program_list = individual.program_list
        idx = random_state.integers(0, len(program_list))

        target_node = program_list[idx]
        target_dim = target_node.output_dimension

        new_subtree = EvolutionOps._generate_random_subtree_by_dim(
            random_state, n_features, target_dim, init_depth
        )

        start, end = EvolutionOps._get_subtree_span(program_list, idx)
        new_program_list = (
            program_list[ : start] +
            new_subtree +
            program_list[end : ]
        )

        return ExpressionTree(new_program_list)
    
    @staticmethod
    def point_mutation(individual: ExpressionTree,
                       random_state: np.random.Generator,
                       p_point_replace: float,
                       n_features: int,
                       feature_names=None):
        '''点突变: 遍历, 概率p替换每个结点, 保持arity, output_dim不变'''
        new_program = deepcopy(individual.program_list)

        for i, node in enumerate(new_program):
            if random_state.uniform() < p_point_replace:
                if isinstance(node, FunctionNode):
                    if node.operator.is_aggregate:
                        candidates = [
                            op for op in ops.AGGREGATE_OPS
                            if op.name != node.name
                        ]
                    else:
                        candidates = [
                            op for op in ops.ALL_OPS 
                            if op.arity == node.arity
                            and not op.is_aggregate
                            and op.name != node.name 
                        ]
                    if candidates:
                        new_op: ops.Operator = random_state.choice(candidates)
                        new_node = deepcopy(node)
                        new_node.operator = new_op
                        new_node.name = new_op.name
                        new_program[i] = new_node                
                
                elif isinstance(node, TerminalNode):
                    new_node = ExpressionTree._create_terminal_node(
                        random_state, n_features, node.output_dimension, feature_names
                    )
                    new_program[i] = new_node
        return ExpressionTree(new_program)
    
    @staticmethod
    def hoist_mutation(individual: ExpressionTree, 
                       random_state: np.random.Generator):
        program = individual.program_list
        candidates = [i for i, node in enumerate(program)
                      if isinstance(node, FunctionNode)]
        if not candidates:
            return deepcopy(individual)
        
        target_idx = random_state.choice(candidates)
        target_start, target_end = EvolutionOps._get_subtree_span(program, target_idx)
        target_subtree = program[target_start:target_end]
        target_dim = program[target_start].output_dimension

        sub_candidates = EvolutionOps._get_candidates_by_dim(target_subtree[1:])
        
        if target_dim not in sub_candidates:
            return deepcopy(individual) 
        
        hoist_relative_idx = random_state.choice(sub_candidates[target_dim]) + 1 
        
        hoist_start, hoist_end = EvolutionOps._get_subtree_span(target_subtree, hoist_relative_idx)
        hoist_subtree = target_subtree[hoist_start:hoist_end]
        
        new_program_list = program[:target_start] + hoist_subtree + program[target_end:]
        
        return ExpressionTree(new_program_list)