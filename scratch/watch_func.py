import os
import numpy as np
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from renew_gp.structure import ExpressionTree
from renew_gp.structure import FunctionNode, TerminalNode


def to_latex(program, dim=10):

    cursor = 0 

    def parse():
        nonlocal cursor
        node = program[cursor]
        cursor += 1

        if isinstance(node, TerminalNode):
            if 'X' in node.name:
                try:
                    parts = node.name.split('[')[1].split(']')[0].split(':')
                    return f"x_{{{parts[0]},{parts[1]}}}"
                except:
                    return f"x_{{{node.name}}}"
            return str(node.name)

        children = [parse() for _ in range(node.arity)]

        op_name = node.name

        if op_name == 'add': return f"({children[0]} + {children[1]})"
        if op_name == 'sub': return f"({children[0]} - {children[1]})"
        if op_name == 'mul': return f"({children[0]} \\cdot {children[1]})"
        if op_name == 'div': return f"\\frac{{{children[0]}}}{{{children[1]}}}"
        if op_name == 'pow': return f"{{{children[0]}}}^{{{children[1]}}}"
        if op_name == 'max': return f"\\max({children[0]}, {children[1]})"
        if op_name == 'min': return f"\\min({children[0]}, {children[1]})"

        if op_name == 'sqrt': return f"\\sqrt{{{children[0]}}}"
        if op_name == 'exp':  return f"e^{{{children[0]}}}"
        if op_name == 'log':  return f"\\ln|{children[0]}|"
        if op_name == 'abs':  return f"|{children[0]}|"
        if op_name == 'sin':  return f"\\sin({children[0]})"
        if op_name == 'cos':  return f"\\cos({children[0]})"
        if op_name == 'tan':  return f"\\tan({children[0]})"
        if op_name == 'tanh': return f"\\tanh({children[0]})"
        if op_name == 'sig':  return f"\\sigma({children[0]})"
        if op_name == 'inv':  return f"\\frac{{1}}{{{children[0]}}}"
        if op_name == 'neg':  return f"-({children[0]})"

        if op_name == 'sum':  return f"\\sum({children[0]})"
        if op_name == 'mean': return f"\\text{{mean}}({children[0]})"
        
        return f"\\text{{{op_name}}}({', '.join(children)})"

    return parse()

if __name__ == '__main__':
    file_path = os.path.join(config.ARTIFACTS_DIR, 'generated_functions', 'batch_01', 'func0_10D_best.pickle')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    tree = data['program']
    target_dim = 10
    formula = to_latex(tree.program_list, target_dim)
    print(f'{formula}')