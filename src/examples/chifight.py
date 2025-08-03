# https://rpg.stackexchange.com/questions/212909/help-with-counting-successes-with-multiple-target-numbers

import dice9 as d9

d9.use("np", profile=False)
import termplotlib as tpl

from dice9.problib import *


def f():
    pchi = 20
    nchi = 20

    for rounds in range(41):
        p = sum(5 @ (d(6) >= 3)) >= sum(5 @ (d(6) >= 3))
        if nchi >= 0 and pchi >= 0:
            if p:
                pchi = max(move(pchi) - 1, -1)

        if nchi >= 0 and pchi >= 0:
            if not p:
                nchi = max(move(nchi) - 1, -1)

    return (nchi > -1, pchi > -1)



import time

start = time.perf_counter()
result = d9.run(f, debug=False)
end = time.perf_counter()
print(result)

print(f"Time taken: {end - start:.6f} seconds")
