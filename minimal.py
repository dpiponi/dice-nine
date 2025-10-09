# import time
# from tabulate import tabulate
import logging
# import matplotlib.pyplot as plt
from rich.logging import RichHandler, Console
# import tabulate
import dice9 as d9
# import math

if 1:
    console = Console(force_terminal=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[RichHandler(markup=True, show_time=False, console=console)])

if 0:
    from rich.traceback import install
    install(show_locals=True)   # show_locals dumps variable values


@d9.dist
def main():
    return 2 * d(2) - 3

print(main())
