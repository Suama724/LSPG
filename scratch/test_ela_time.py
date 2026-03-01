import numpy as np
import time

from utils.ela_feature import get_ela_feature
from problem_form.abc_problem import BasicProblem

class TestFuncSquare(BasicProblem):
    def func(self, x: np.array):
        x = np.atleast_2d(x)
        val = np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2.0)**2.0 + (1.0 - x[:, :-1])**2.0, axis=1)
        return val[0] if val.size == 1 else val       
    
if __name__ == '__main__':
    problem = TestFuncSquare()
    dim = 50
    time_eval_total = 0
    time_ela_total = 0
    '''
    for i in range(10):
        Xs = np.random.rand(100, dim)
        t_0 = time.perf_counter()
        Ys = problem.eval(Xs)
        t_1 = time.perf_counter()

        ela_feats, _, _ = get_ela_feature(problem, Xs, Ys, random_state=42)
        time_2 = time.perf_counter()
        time_eval_total += t_1 - t_0
        time_ela_total += time_2 - t_1
    '''
    Xs = np.random.rand(1000, dim)
    t_0 = time.perf_counter()
    Ys = problem.eval(Xs)
    t_1 = time.perf_counter()

    ela_feats, _, _ = get_ela_feature(problem, Xs, Ys, random_state=42)
    time_2 = time.perf_counter()
    time_eval_total += t_1 - t_0
    time_ela_total += time_2 - t_1

    print(f"Time eval: {time_eval_total:.6f}")
    print(f"Time ela: {time_ela_total:.6f}")

    with open('./scratch/record_ela_time.txt', 'a') as f:
        f.write(f"""time: {time.strftime('%Y-%m-%d %H:%M:%S')}, 
dim: {dim}, 
Time eval: {time_eval_total:.6f}, 
Time ela: {time_ela_total:.6f}, 

""")