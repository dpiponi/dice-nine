import time
import logging
import sys
import matplotlib.pyplot as plt

# sys.tracebacklimit = 8  # Show only the last 2 levels of traceback
# logging.basicConfig(level=logging.DEBUG)

import dice9 as d9

def f():
    x = [0, 0]
    for i in range(20):
        x[d(2) - 1] += 1
    return x

pmf = d9.run(f, squeeze=True, show_analysis=True)
print(pmf)
