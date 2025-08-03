# See https://anydice.com/articles/legend-of-the-five-rings/

import dice9 as d9
d9.use('np', profile=False)
from dice9.problib import *
import termplotlib as tpl

def append(zs, w):
    y1 = reshape(move(w), [1])
    return concat([move(zs), move(y1)], -1)

def my_top_k(k, s):
    xs = zeros(reshape(k, [1]))

    for y in s:
        c = append(move(xs), move(y))
        xs = top_k(move(c), k)

    return xs

def e():
    x = d(10)
    x = x + d(10) if x == 10 else move(x)
    x = x + d(10) if x == 20 else move(x)
    x = x + d(10) if x == 30 else move(x)

    return x

def test():
    return reduce_sum(my_top_k(3, (e() for i in range(6))), -1)

if 1:
    d = d9.run(test)

    labels, data = zip(*sorted(d.items()))

    fig = tpl.figure()
    fig.barh(
        data,
        labels=labels,
        force_ascii=False
    )
    fig.show()
