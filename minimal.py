# import time
# from tabulate import tabulate
import logging
# import matplotlib.pyplot as plt
from rich.logging import RichHandler, Console
# import tabulate
import dice9 as d9
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


@d9.dist
def f():
    hp1 = lazy_sum(36 @ d(8))
    hp2 = lazy_sum(18 @ d(8))

    for i in range(14):
        print("round", i)
        if hp1 > 0 and d(20) > 1:
            for x in 5 @ d(4):
                hp2 = max(0, hp2 - x)

        if hp2 > 0: 
            if d(20) > 1:
                hp1 -= d(6)
            if d(20) > 1:
                hp1 -= d(6)
            if d(20) > 1:
                for x in 5 @ d(8):
                    hp1 = max(0, hp1 - x)

    win1 = hp2 == 0
    win2 = hp1 == 0

    return win1, win2

print(f())
