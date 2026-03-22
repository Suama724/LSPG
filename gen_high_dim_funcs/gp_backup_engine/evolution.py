import numpy as np
from copy import deepcopy
from typing import List

from .structure import ExpressionTree, FunctionNode, TerminalNode, Node
from . import ops

def _recalculate_topology(program_list: List[Node]):
    if not program_list:
        return []
    
    for node in program_list:
        node.child_distance_list = []
        node.parent_distance = None
        node.depth
    
    stack = []

    for i, node in enumerate(program_list):
        if stack:
            parent_idx, rem = stack.pop()
            program_list[parent_idx].child_distance_list.append(i - parent_idx)
            node.parent_distance = parent_idx - i
            node.depth = program_list[parent_idx].depth + 1

            if rem > 1:
                stack.append((parent_idx, rem - 1))

        if node.is_func_node and node.arity > 0:
            stack.append((i, node.arity))
            
    return program_list

def _get_candidates_by_dim(program_list):
    """Map output_dim -> list of node indices (for subtree selection)."""
    cand = {}
    for i, node in enumerate(program_list):
        d = node.output_dim  
        cand.setdefault(d, []).append(i)
    return cand

def _get_subtree_span(program_list, start_idx):
    """Return (start, end) indices of subtree rooted at start_idx."""
    stack = 1
    end = start_idx
    while stack > 0 and end < len(program_list):
        stack += program_list[end].arity - 1
        end += 1
    return start_idx, end

def _generate_random_subtree(random_state, total_dim, output_dim, init_depth):
    max_d = random_state.integers(*init_depth)
    program = []
    stack = [(0, output_dim)]
    
    while stack:
        depth, out_dim = stack.pop()
        is_terminal = depth >= max_d or (depth > 0 and random_state.random() < 0.1)
        
        if not is_terminal:
            valid = ExpressionTree._get_valid_ops()  
            if not valid:
                is_terminal = True
                
        if is_terminal:
            node = ExpressionTree._create_terminal_node(random_state, total_dim, out_dim)
        else:
            op = valid[random_state.integers(len(valid))]
            node = ExpressionTree._create_func_node(op, total_dim, out_dim)
            
        node.depth = depth
        program.append(node)
        
        if node.is_func_node:
            cd = node.input_dim
            for _ in range(node.arity):
                stack.append((depth + 1, cd))

    return _recalculate_topology(program)


def crossover(donor: ExpressionTree, 
            receiver: ExpressionTree, 
            random_state: np.random.Generator) -> ExpressionTree:
    rc = _get_candidates_by_dim(receiver.program_list)
    dc = _get_candidates_by_dim(donor.program_list)
    common = set(rc) & set(dc)
    if not common:
        return deepcopy(receiver)
        
    dim = random_state.choice(list(common))
    ri = random_state.choice(rc[dim])
    di = random_state.choice(dc[dim])
    
    rs, re = _get_subtree_span(receiver.program_list, ri)
    ds, de = _get_subtree_span(donor.program_list, di)
    
    # 使用 deepcopy 防止列表切片拼接时产生对象引用污染
    new_list = deepcopy(receiver.program_list[:rs]) + deepcopy(donor.program_list[ds:de]) + deepcopy(receiver.program_list[re:])
    
    # 核心修复：重装拼接后的相对偏移量
    return ExpressionTree(_recalculate_topology(new_list))


def subtree_mutation(individual: ExpressionTree, 
                    random_state, 
                    total_dim, 
                    init_depth=(0, 2)) -> ExpressionTree:
    pl = individual.program_list
    idx = random_state.integers(0, len(pl))
    target_dim = pl[idx].output_dim  # 修正
    
    new_sub = _generate_random_subtree(random_state, total_dim, target_dim, init_depth)
    s, e = _get_subtree_span(pl, idx)
    
    new_list = deepcopy(pl[:s]) + new_sub + deepcopy(pl[e:])
    return ExpressionTree(_recalculate_topology(new_list))


def point_mutation(individual: ExpressionTree, random_state, p_point_replace, total_dim) -> ExpressionTree:
    new_list = deepcopy(individual.program_list)
    for i, node in enumerate(new_list):
        if random_state.uniform() >= p_point_replace:
            continue
            
        if node.is_func_node:
            if node.operator.is_aggregate:
                cand = [o for o in ops.AGGREGATE_OPS if o.name != node.name]
            else:
                cand = [o for o in ops.ALL_OPS if o.arity == node.arity and not o.is_aggregate and o.name != node.name]
            if cand:
                new_op = random_state.choice(cand)
                new_node = deepcopy(node)
                new_node.operator = new_op
                new_node.name = new_op.name
                new_list[i] = new_node
                
        elif node.is_con_node or node.is_var_node:
            new_list[i] = ExpressionTree._create_terminal_node(random_state, total_dim, node.output_dim)
            
    # 点变异通常不改变 arity 导致结构变化，但终端节点的重构可能会刷新元数据，重新算一次防患于未然
    return ExpressionTree(_recalculate_topology(new_list))


def hoist_mutation(individual: ExpressionTree, random_state) -> ExpressionTree:
    pl = individual.program_list
    func_indices = [i for i, n in enumerate(pl) if n.is_func_node]
    if not func_indices:
        return deepcopy(individual)
        
    idx = random_state.choice(func_indices)
    s, e = _get_subtree_span(pl, idx)
    sub = pl[s:e]
    
    target_dim = sub[0].output_dim
    cand = _get_candidates_by_dim(sub[1:])
    
    if target_dim not in cand:
        return deepcopy(individual)
        
    rel = random_state.choice(cand[target_dim]) + 1
    hs, he = _get_subtree_span(sub, rel)
    hoist = sub[hs:he]
    
    new_list = deepcopy(pl[:s]) + hoist + deepcopy(pl[e:])
    return ExpressionTree(_recalculate_topology(new_list))