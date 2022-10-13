from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
from math import pow
import random
from bisect import bisect_left
from multiprocessing import cpu_count
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import math as math


class ObjectiveFunction(ObjectiveFunctionInterface):

    def get_value(self, i, j=None):
        if self.is_discrete(i):
            if j:
                return self._discrete_values[i][j]
            return self._discrete_values[i][random.randint(0, len(self._discrete_values[i]) - 1)]
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def get_num_discrete_values(self, i):
        if self.is_discrete(i):
            return len(self._discrete_values[i])
        return float('+inf')

    def get_index(self, i, v):

        return ObjectiveFunction.binary_search(self._discrete_values[i], v)

    @staticmethod
    def binary_search(a, x):
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        return self._discrete_values[i] is not None

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return hasattr(self, '_random_seed') and self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize

    def __init__(self):
        self._lower_bounds = [None, -1000]
        self._upper_bounds = [None, 1000]
        self._variable = [True, True]
        self._discrete_values = [[x for x in range(-100, 101)], None]

        self._maximize = True
        self._max_imp = 50000  # maximum number of improvisations
        self._hms = 100  # harmony memory size
        self._hmcr = 0.75  # harmony memory considering rate
        self._par = 0.5  # pitch adjusting rate
        self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 10  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only

    def get_fitness(self, vector):
        x = vector[0]
        y = vector[1]
        a = 1 - x
        b = y - x * x
        d = (1 - x)
        e = math.pow(d, 2)
        f = y - math.pow(d, 2)
        g = math.pow(b, 2)
        c = math.log(1 + e + 100 * g)
        return -c  # Rosenbrock Test Function

if __name__ == '__main__':
    obj_fun = ObjectiveFunction()
    num_processes = cpu_count()  # number of logical CPUs
    num_iterations = num_processes * 5  # each process does 5 iterations
    results = harmony_search(obj_fun, num_processes, num_iterations)
    print('Elapsed time: {}\nBest harmony: {}\nBest fitness: {}\nHarmony memories:'.format(results.elapsed_time, results.best_harmony, results.best_fitness))
    pprint(results.harmony_memories)

    #fig = plt.figure()
    #x=results.best_harmony
    #y=results.elapsed_time
    #x = np.linspace(x, y, 1000)
    #plt.plot(x, np.sin(x), '-g')
    #plt.plot(x, np.cos(y), ':b')
    #plt.axis('equal')

    #plt.legend() ;
    #plt.show()




