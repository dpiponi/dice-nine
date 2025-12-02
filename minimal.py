# fmt: off
# type: ignore

# import time
# from tabulate import tabulate
import logging
import numpy as np
# import matplotlib.pyplot as plt
from rich.logging import RichHandler, Console
# import tabulate
import dice9 as d9
import dice9.factor
# import math

if 0:
    console = Console(force_terminal=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[RichHandler(markup=True, show_time=False, console=console)])

if 0:
    from rich.traceback import install
    install(show_locals=True)   # show_locals dumps variable values

p = np.array([0.1, 0.2, 0.3])
i = np.array([10, 10, 1])
j = np.array([0, 0, 2])
l = np.array([5, 7, 9])

print("i = ", i)
print("j = ", j)
print("l = ", l)

idx, p_sum = d9.factor.dedupe_and_aggregate([i, j], p, d9.factor.hash_tensors, d9.Real64())

print(f"idx = {idx}, p_sum = {p_sum})")

e = d9.factor.expectation(l, [i, j], p, d9.factor.hash_tensors, d9.Real64())

print(f"result = {e}")

d = None
E = None
P = None
# log = None

def transition(x):
    if x == 0:
        if d(2) == 1:
            x = 1
        else:
            x = 0
    else:
        if d(4) == 1:
            x = 1
        else:
            x = 0
    return x
            
@d9.dist
def main():
    x = d(2) - 1
    x0 = x

    x1 = transition(x0)
    x2 = transition(x1)
    x3 = transition(x2)

    kl1 = E[E[log(P(x1, x2) / P(x2)) : x2]]
    kl2 = E[E[E[log(P(x1, x2, x) / P(x2, x)) : x2, x] : x]]
    return kl1, kl2


pmf = main()
for i, j in pmf.items():
    print(i, j)
