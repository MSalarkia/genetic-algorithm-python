from selection_functions import roulette_wheel_selection
import numpy as np
import random


def crossover(x1,x2):
    p_single_point = 0.1
    p_double_point = 0.2
    p_uniform = 1 - p_single_point - p_double_point
    p = np.array([p_single_point, p_double_point, p_uniform])
    method = roulette_wheel_selection(p)
    options = {0: single_point_crossover
              , 1: double_point_crossover
              , 2: uniform_crossover}
    y1, y2 = options[method](x1, x2)
    return y1, y2


def uniform_crossover(x1, x2):
    alpha = np.array([random.randint(0, 1) for i in range(len(x1))])
    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1
    return y1, y2


def double_point_crossover(x1, x2):
    n_var = len(x1)

    cc = random.sample(range(n_var - 1), 2)
    c1 = min(cc)
    c2 = max(cc)
    y1 = np.concatenate((x1[0:c1 + 1], x2[c1 + 1:c2 + 1], x1[c2 + 1:]), axis=0)
    y2 = np.concatenate((x2[0:c1 + 1], x1[c1 + 1:c2 + 1], x2[c2 + 1:]), axis=0)

    return y1, y2


def single_point_crossover(x1,x2):
    n_var = len(x1)
    c = random.randint(1, n_var-1)
    y1 = np.concatenate((x1[0:c+1], x2[c+1:]), axis=0)
    y2 = np.concatenate((x2[0:c+1], x1[c+1:]), axis=0)
    return y1, y2
