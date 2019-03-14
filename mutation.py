import math
import random
import numpy as np


def mutate(x, mu):
    n_var = len(x)
    nmu = math.ceil(mu * n_var)
    j = np.array(random.sample(range(n_var), nmu))
    y = x
    y[j] = 1 - x[j]
    return y

