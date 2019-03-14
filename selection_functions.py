import random


def roulette_wheel_selection(p):
    r = random.random()
    c = p.cumsum()
    i = (r < c).argmax()
    return i

