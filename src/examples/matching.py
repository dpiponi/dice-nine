# This is the "all matching sets" example demonstrated by icepool
# https://github.com/HighDiceRoller/icepool

import dice9 as d9
d9.use('tf', profile=False)

from dice9.problib import *

def f():
    m = 10
    x = zeros(11)
    for i in m @ d(10):
        x = sort(move(x) + one_hot(i, 11), -1)
    return x


import time

start = time.perf_counter()
result = d9.run(f, debug=False)
print(result)
end = time.perf_counter()
print(f"Time taken: {end - start:.6f} seconds")
