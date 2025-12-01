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

@d9.dist
def main():
    x = d(2)
    y = d(2)
    return -E(log(P(x, y)))

pmf = main()
for i, j in pmf.items():
    print(i, j)
