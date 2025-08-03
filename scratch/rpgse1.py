# https://rpg.stackexchange.com/questions/208990/anydice-number-of-successes-when-dice-can-be-boosted

import dice9 as d9

d9.use("np", profile=False)
import termplotlib as tpl

from dice9.problib import *


def f():
    points = 3
    x = reduce_sum(
        cumsum(sort(6 - top_k(list(4 @ d(6)), 3), -1), -1) <= points, -1)
    return x

import time

start = time.perf_counter()
result = d9.run(f, debug=False, squeeze=True)
end = time.perf_counter()
# print(result)

print(f"Time taken: {end - start:.6f} seconds")

for k, v in sorted(result.items(), key=lambda item: item[1]):
    print(f"{k}: {v}")
