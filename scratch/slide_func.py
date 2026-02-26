import numpy as np

from problem_form.abc_problem import BasicProblem

class TestFuncSquare(BasicProblem):
    def func(self, x: np.array):
        return x ** 2
    

if __name__ == '__main__':
    x = np.array([i for i in range(1, 10)])
    func = TestFuncSquare()
    y = func.eval(x)
    print(y)
    print(func.time_cost)

