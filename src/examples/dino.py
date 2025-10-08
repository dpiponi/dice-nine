import logging
from rich.logging import RichHandler, Console

if 0:
    console = Console(force_terminal=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[RichHandler(markup=True, show_time=False, console=console)])

#console = Console(force_terminal=True)

#logging.basicConfig(
#    level=logging.DEBUG,  # Root logger level
#    format="%(message)s",
#    handlers=[RichHandler(console=console, markup=True, show_time=False, level=logging.DEBUG)]
#)

import time
from dice9.factor import Real64, LogReal64, BigFraction

import dice9 as d9

# Brachiosaurus vs Tyrannosaurus, rounds=14, p=0.362511

# s = BigFraction(512)
s = Real64()

@d9.dist(semiring=s)
def f():
    #hp1 = 36 * 9 / 2 # lazy_sum(36 @ d(8))
    #hp2 = 18 * 9 / 2 # lazy_sum(18 @ d(8))
    hp1 = lazy_sum(36 @ d(8))
    hp2 = lazy_sum(18 @ d(8))

    for i in range(14):
        print("round", i)
        if hp1 > 0 and d(20) > 1:
            for x in 5 @ d(4):
                hp2 -= x
            hp2 = max(0, hp2)

        if hp2 > 0: 
            if d(20) > 1:
                hp1 -= d(6)
            if d(20) > 1:
                hp1 -= d(6)
            if d(20) > 1:
                for x in 5 @ d(8):
                    hp1 -= x
            hp1 = max(hp1, 0)

        __dumpvars__()

    win1 = hp2 == 0
    win2 = hp1 == 0

    return win1, win2


start = time.perf_counter()
result = f()
end = time.perf_counter()
print(result)
print(f"Time taken: {end - start:.6f} seconds")
for k, v in result.items():
    #print(k, v)
    print(k, v)
