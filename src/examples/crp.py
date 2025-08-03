import sys
import dice9 as bbP

bbP.use(sys.argv[1])

from dice9.problib import *


@bbP.dist
def f():
    r = constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for k in range(20):
        print("k=", k)
        add1 = r + one_hot(argmin(r, -1), 16)
        padded = concat([[0], move(add1)], -1)
        s = cumsum(move(padded), -1)
        i = argmin(dd(s[16]) >= move(s), -1) - 1
        r += one_hot(move(i), 16)

    return reduce_any(r == 4, -1)

result = f()
print("r=",result)
import math
print(1-math.exp(-1/4))
