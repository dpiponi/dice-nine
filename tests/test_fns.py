# test_fns.py

import pytest
import math
import functools
from fractions import Fraction

import dice9 as d9

#from dice9.problib import *

from pytest import approx

def test_abs1():
    @d9.dist
    def f():
        x = d(3)
        return abs(x - 2)

    pmf = f()
    assert pmf[1] == approx(2 / 3, rel=1e-5)

def test_reshape1():
    @d9.dist
    def reshape1():
        return reshape(1, [1])

    pmf = reshape1()
    assert pmf[(1,)] == approx(1.0, rel=1e-5)
    assert len(pmf) == 1

def test_reshape2():
    @d9.dist
    def reshape2():
        return reshape(1, [1, 1])

    pmf = reshape2()
    assert pmf[((1,),)] == approx(1.0, rel=1e-5)
    assert len(pmf) == 1

import pytest

def test_argmin1():
    @d9.dist
    def argmin1():
        x = [[1, 2], [8, 4]]
        return argmin(x, axis=-2)

    pmf = argmin1()
    assert pmf[(0, 0)] == pytest.approx(1.0)

    def argmin2():
        x = [[1, 2], [8, 4]]
        return argmin(x, axis=-1)

    @d9.dist
    def argmin3():
        x = [[1, 2], [8, 4]]
        return argmin(x, axis=0)

    pmf = argmin3()
    assert pmf[(0, 0)] == pytest.approx(1.0)

    @d9.dist
    def argmin4():
        x = [[1, 2], [8, 4]]
        return argmin(x, axis=1)

    pmf = argmin4()
    assert pmf[(0, 1)] == pytest.approx(1.0)

    @d9.dist
    def argmax4():
        x = [[1, 2], [8, 4]]
        return argmax(x, axis=1)

    pmf = argmax4()
    assert pmf[(1, 0)] == pytest.approx(1.0)

def test_argsort1():
    @d9.dist
    def f():
        return argsort([d(2), d(2)], -1)

    pmf = f()
    assert pmf[(0, 1)] == pytest.approx(3 / 4)
    assert pmf[(1, 0)] == pytest.approx(1 / 4)

def test_argsort2():
    @d9.dist
    def f():
        return argsort([d(2), d(2)], 0)

    pmf = f()
    assert pmf[(0, 1)] == pytest.approx(3 / 4)
    assert pmf[(1, 0)] == pytest.approx(1 / 4)

def test_argsort3():
    @d9.dist
    def f():
        return argsort([d(2), d(2)])

    pmf = f()
    assert pmf[(0, 1)] == pytest.approx(3 / 4)
    assert pmf[(1, 0)] == pytest.approx(1 / 4)

def test_cumsum1():
    @d9.dist
    def f():
        z = [d(2), d(2)]
        return cumsum(z)

    pmf = f()
    assert pmf[(1, 2)] == pytest.approx(1 / 4)
    assert pmf[(2, 4)] == pytest.approx(1 / 4)

def test_cumsum2():
    @d9.dist
    def f():
        z = [d(2), d(2)]
        return cumsum(z, -1)

    pmf = f()
    assert pmf[(1, 2)] == pytest.approx(1 / 4)
    assert pmf[(2, 4)] == pytest.approx(1 / 4)

def test_zeros():
    @d9.dist
    def f():
        return zeros([2, 2])

    pmf = f()
    assert pmf[((0, 0), (0, 0))] == pytest.approx(1)

def test_ones():
    @d9.dist
    def f():
        return ones([2, 2])

    pmf = f()
    assert pmf[((1, 1), (1, 1))] == pytest.approx(1)

def test_dd1():
    @d9.dist
    def f():
        x = d(2)
        y = dd(x)
        return y

    pmf = f()
    assert pmf[0] == pytest.approx(0.75)
    assert pmf[1] == pytest.approx(0.25)

def test_dd2():
    @d9.dist
    def f():
        x = d(2)
        y = d(2)
        return dd(x) + dd(y)

    pmf = f()
    assert pmf[0] == pytest.approx(0.75**2)
    assert pmf[1] == pytest.approx(2 * 0.75 * 0.25)
    assert pmf[2] == pytest.approx(0.25**2)

def test_dd3():
    @d9.dist
    def f():
        x = d(2)
        return dd(x) + dd(x)

    pmf = f()
    assert pmf[0] == pytest.approx(0.5 + 0.5 * 0.5**2)
    assert pmf[2] == pytest.approx(0.5 * 0.5**2)

def test_dd4():
    @d9.dist
    def f():
        return dd(d(2)) + dd(d(2))

    pmf = f()
    assert pmf[0] == pytest.approx(0.75**2)
    assert pmf[1] == pytest.approx(2 * 0.75 * 0.25)
    assert pmf[2] == pytest.approx(0.25**2)

def test_ifexp1():
    @d9.dist
    def f():
        return 10 if d(2) == 1 else 20

    pmf = f()
    assert pmf[10] == pytest.approx(0.5)
    assert pmf[20] == pytest.approx(0.5)

def test_ifexp2():
    @d9.dist
    def f():
        x = d(6)
        x = x + d(6) if x == 6 else x
        return x

    pmf = f()
    assert pmf[1] == pytest.approx(1 / 6)
    assert pmf[12] == pytest.approx(1 / 36)

def test_ifexp3():
    @d9.dist
    def f():
        x = 0
        for i in range(3):
            x = x + d(6) if x == 6 * i else x
        return x

    pmf = f()
    assert pmf[1] == pytest.approx(1 / 6)
    assert pmf[18] == pytest.approx(1 / 216)

def test_ifexp4():
    @d9.dist
    def f(x):
        return 2 * x if x % 2 == 0 else x

    @d9.dist
    def g():
        return f(d(10))

    pmf = g()
    for i in [1, 4, 3, 8, 5, 12, 7, 16, 9, 20]:
        assert pmf[i] == pytest.approx(1 / 10)

def test_int():
    @d9.dist
    def f():
        x = d(6)
        return int(x == 1)

    pmf = f()
    assert pmf[1] == pytest.approx(1 / 6)

def test_minimal_gen():
    def f():
        yield 1

    @d9.dist
    def g():
        x = 0
        for y in f():
            x += y
        return x

    pmf = g()
    assert pmf[1] == pytest.approx(1)

def test_small_gen():
    def f():
        yield d(6)

    @d9.dist
    def g():
        x = 0
        for y in f():
            x += y
        return x

    pmf = g()
    assert pmf[1] == pytest.approx(1 / 6)

def test_small_gen():
    def f():
        for i in range(3):
            yield d(6)

    @d9.dist
    def g():
        x = 0
        for y in f():
            x += y
        return x

    pmf = g()
    assert pmf[3] == pytest.approx(1 / 6**3)
    assert pmf[18] == pytest.approx(1 / 6**3)

def test_gen1():
    def f(n):
        for i in range(n):
            yield d(6)

    @d9.dist
    def g():
        x = 0
        for y in f(3):
            x += y
        return x

    pmf = g()
    assert pmf[3] == pytest.approx(1 / 6**3)
    assert pmf[18] == pytest.approx(1 / 6**3)

def test_gen2():
    def h():
        yield d(2)
        yield d(2)
        yield d(2)

    def g():
        for i in h():
            yield i

    @d9.dist
    def f():
        t = 0
        for i in g():
            t += i
        return t

    pmf = f()
    assert pmf[3] == pytest.approx(0.125)
    assert pmf[6] == pytest.approx(0.125)

def test_sum1():
    def g():
        yield d(6)
        yield d(6)
        yield d(6)

    @d9.dist
    def f():
        x = sum(*g())
        return x

    pmf = f()
    assert pmf[3] == pytest.approx(1 / 6**3)
    assert pmf[18] == pytest.approx(1 / 6**3)

def test_sum2():
    def g():
        for i in range(3):
            yield d(6)

    @d9.dist
    def f():
        x = sum(*g())
        return x

    pmf = f()
    assert pmf[3] == pytest.approx(1 / 6**3)
    assert pmf[18] == pytest.approx(1 / 6**3)

def test_genexp1():
    def g():
        yield 1
        yield 2

    @d9.dist
    def f():
        z = sum(*(x for x in g()))
        return z

    pmf = f()
    assert pmf[3] == pytest.approx(1)

def test_genexp2():
    def g():
        yield d(6)
        yield d(6)

    @d9.dist
    def f():
        z = sum(*(x for x in g()))
        return z

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 36)
    assert pmf[12] == pytest.approx(1 / 36)

def test_genexp3():
    def g(n):
        for i in range(n):
            yield d(6)

    @d9.dist
    def f():
        z = sum(*(x - 1 for x in g(3)))
        return z

    pmf = f()
    assert pmf[0] == pytest.approx(1 / 216)
    assert pmf[15] == pytest.approx(1 / 216)

def test_genexp4():
    @d9.dist
    def f():
        return sum(*(int(d(20) == 1) for i in range(20)))

    pmf = f()
    for i in range(20):
        assert pmf[i] == pytest.approx(math.comb(20, i) * (1 / 20) ** i * (19 / 20) ** (20 - i))

def test_genexp5():
    @d9.dist
    def f():
        return sum(*(20 @ int(d(20) == 1)))

    pmf = f()
    for i in range(20):
        assert pmf[i] == pytest.approx(math.comb(20, i) * (1 / 20) ** i * (19 / 20) ** (20 - i))

def test_subscript1():
    @d9.dist
    def f():
        x = [10, 20, 30, 40, 50, 60]
        return x[d(6) - 1]

    pmf = f()
    assert pmf[10] == pytest.approx(1 / 6)
    assert pmf[60] == pytest.approx(1 / 6)

def test_topk1():
    def my_top_2(s):
        xs = [0, 0]

        for y in s:
            c = [*xs, y]
            xs = top_k(c, 2)

        return xs

    def es():
        for i in range(6):
            x = d(6)
            x = x + d(6) if x == 6 else x

            yield x

    @d9.dist
    def g():
        top2 = my_top_2(es())
        return reduce_sum(top2, -1)

    pmf = g()
    assert pmf[2] == pytest.approx(2.1433470507544594e-05)
    assert pmf[24] == pytest.approx(0.010743132472754508)

def test_topk2():
    @d9.dist
    def f():
        a = [d(4), d(6), d(6), d(10)]
        b = top_k(a, 2)
        return reduce_sum(b, -1)

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 1440)
    assert pmf[16] == pytest.approx(11 / 360)

def test_isin1():
    @d9.dist
    def f():
        x = d(6)
        return d(6) if x in [1, 2] else x

    pmf = f()
    assert pmf[1] == pytest.approx(1 / 18)
    assert pmf[6] == pytest.approx(2 / 9)

def test_isin2():
    @d9.dist
    def f():
        x = d(6)
        return d(6) if x not in [1, 2] else x

    pmf = f()
    assert pmf[1] == pytest.approx(5 / 18)
    assert pmf[6] == pytest.approx(1 / 9)

def test_birthday():
    @d9.dist
    def f():
        number_unique_birthdays = 0

        for i in range(23):
            number_unique_birthdays += int(
                d(365) > number_unique_birthdays)

        return number_unique_birthdays == 23

    pmf = f()
    prob = functools.reduce(lambda x, y: x * y, (1 - i / 365 for i in range(23)))
    assert pmf[True] == pytest.approx(prob)

def test_list1():
    @d9.dist
    def f():
        return [1]

    pmf = f()
    assert pmf[(1,)] == pytest.approx(1)

def test_list2():
    @d9.dist
    def f():
        return [1, 2]

    pmf = f()
    assert pmf[(1, 2)] == pytest.approx(1)

def test_list3():
    @d9.dist
    def f():
        return [d(2), d(6)]

    pmf = f()
    assert pmf[(1, 1)] == pytest.approx(1 / 12)
    assert pmf[(2, 6)] == pytest.approx(1 / 12)

def test_list4():
    @d9.dist
    def f():
        return [d(2), min(d(3), 2)]

    pmf = f()
    assert pmf[(1, 1)] == pytest.approx(1 / 6)
    assert pmf[(2, 2)] == pytest.approx(1 / 3)

def test_list5():
    def step(n):
        for i in range(n):
            yield [2 * d(2) - 3, 2 * d(2) - 3]

    @d9.dist
    def f():
        return sum(*step(4))

    pmf = f()
    assert pmf[(0, 0)] == pytest.approx(9 / 64)

def test_list6():
    @d9.dist
    def f():
        x = [d(6), d(6)]
        y = d(6)
        return reduce_sum([*x, y])

    pmf = f()
    assert pmf[3] == pytest.approx(1 / 216)
    assert pmf[18] == pytest.approx(1 / 216)

def test_bincount1():
    @d9.dist
    def f():
        a = [1, 3, 2, 5, 1, 3]
        return bincount(a, 8)

    pmf = f()
    assert pmf[(0, 2, 1, 2, 0, 1, 0, 0)] == pytest.approx(1.0)

def test_reduce_sum1():
    @d9.dist
    def f():
        return reduce_sum([d(4), d(4)], axis=-1)

    pmf = f()
    assert pmf[2] == pytest.approx(0.0625)
    assert pmf[8] == pytest.approx(0.0625)

def test_reduce_sum2():
    @d9.dist
    def f():
        return reduce_sum([d(4), d(4)])

    pmf = f()
    assert pmf[2] == pytest.approx(0.0625)
    assert pmf[8] == pytest.approx(0.0625)

def test_reduce_min1():
    @d9.dist
    def f():
        return reduce_min([d(4), d(4)], axis=-1)

    pmf = f()
    assert pmf[1] == pytest.approx(0.4375)
    assert pmf[4] == pytest.approx(0.0625)

def test_reduce_min2():
    @d9.dist
    def f():
        return reduce_min([d(4), d(4)])

    pmf = f()
    assert pmf[1] == pytest.approx(0.4375)
    assert pmf[4] == pytest.approx(0.0625)

def test_reduce_max():
    @d9.dist
    def f():
        return reduce_max([d(4), d(4)], axis=-1)

    pmf = f()
    assert pmf[1] == pytest.approx(0.0625)
    assert pmf[4] == pytest.approx(0.4375)

def test_reduce_any1():
    @d9.dist
    def f():
        return reduce_any([d(6), d(6), d(6)] == 1, -1)

    pmf = f()
    assert pmf[True] == pytest.approx(1 - (5 / 6)**3)

def test_reduce_any2():
    @d9.dist
    def f():
        return reduce_any([d(6), d(6), d(6)] == 1)

    pmf = f()
    assert pmf[True] == pytest.approx(1 - (5 / 6)**3)

def test_reduce_all():
    @d9.dist
    def f():
        return reduce_all([d(6), d(6), d(6)] == 1)

    pmf = f()
    assert pmf[True] == pytest.approx(1 / 216)

def test_reduce_all2():
    @d9.dist
    def f():
        return reduce_all([d(6), d(6), d(6)] == 1, -1)

    pmf = f()
    assert pmf[True] == pytest.approx(1 / 216)

def test_matmult():
    @d9.dist
    def f():
        t = 0
        for x in 3 @ d(6):
            t += x
        return t

    pmf = f()
    assert pmf[18] == pytest.approx(1 / 216)

def test_multiroll2():
    @d9.dist
    def f():
        x = d(6)
        return d[1 : x : 1]

    pmf = f()
    for i in range(1, 7):
        p = sum(1/(6 * n) for n in range(i, 7))
        assert pmf[i] == pytest.approx(p)

def test_multiroll3():
    @d9.dist
    def f():
        return d[1 : 2, 5 : 6]

    pmf = f()
    assert pmf[1] == pytest.approx(1 / 4)
    assert pmf[6] == pytest.approx(1 / 4)

def test_multiroll4():
    @d9.dist
    def f():
        return d[1, 2:3, 4:6:1]

    pmf = f()
    assert pmf[1] == pytest.approx(1 / 6)
    assert pmf[6] == pytest.approx(1 / 6)

def test_multiroll5():
    @d9.dist
    def f():
        return d[0:0:99, 1]

    pmf = f()
    assert pmf[0] == pytest.approx(0.99)
    assert pmf[1] == pytest.approx(0.01)

def test_multiroll6():
    @d9.dist
    def f():
        return d[1::2, 2::1]

    pmf = f()
    assert pmf[1] == pytest.approx(2 / 3)
    assert pmf[2] == pytest.approx(1 / 3)

def test_flip1():
    @d9.dist
    def f():
        x = [d(2), d(5)]
        x = flip(x, -1)
        return x[1] > x[0]

    pmf = f()
    assert pmf[True] == pytest.approx(0.1)

def test_flip2():
    @d9.dist
    def f():
        x = [d(2), d(5)]
        x = flip(x)
        return x[1] > x[0]

    pmf = f()
    assert pmf[True] == pytest.approx(0.1)

def test_len1():
    @d9.dist
    def f():
        return len([d(2), d(3)])

    pmf = f()
    assert pmf[2] == pytest.approx(1)

def test_len2():
    @d9.dist
    def f():
      t = []
      for r in {d(2), d(2), d(2)}:
        t = [*t, r]
        if len(t) > 2:
            t = top_k(t, 2)

      return t

    pmf = f()
    assert pmf[(1, 1)] == pytest.approx(0.125)
    assert pmf[(2, 2)] == pytest.approx(0.5)

def test_bitwise1():
    @d9.dist
    def f():
        return 1 << d(6)

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 6)
    assert pmf[64] == pytest.approx(1 / 6)

def test_bitwise2():
    @d9.dist
    def f():
        return 128 >> d(6)

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 6)
    assert pmf[64] == pytest.approx(1 / 6)

def test_bitwise3():
    @d9.dist
    def f():
        return (1 << d(6)) | (128 >> d(6))

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 36)
    assert pmf[96] == pytest.approx(1 / 18)

def test_bitwise4():
    @d9.dist
    def f():
        return (1 << d(6)) & (128 >> d(6))

    pmf = f()
    assert pmf[0] == pytest.approx(5 / 6)
    assert pmf[64] == pytest.approx(1 / 36)

def test_bitwise5():
    @d9.dist
    def f():
        return 8 + ~d(6)

    pmf = f()
    assert pmf[1] == pytest.approx(1 / 6)
    assert pmf[6] == pytest.approx(1 / 6)

def test_bitwise6():
    @d9.dist
    def f():
        return (1 << d(6)) ^ (128 >> d(6))

    pmf = f()
    assert pmf[0] == pytest.approx(1 / 6)
    assert pmf[96] == pytest.approx(1 / 18)

def test_boolean1():
    @d9.dist
    def f():
        a = d(2)
        b = d(2)
        c = d(2)
        return a==2 and b==2 and c==2

    pmf = f()
    assert pmf[True] == pytest.approx(1 / 8)

def test_boolean2():
    @d9.dist
    def f():
        a = d(2)
        b = d(2)
        c = d(2)
        return a==2 or b==2 or c==2

    pmf = f()
    assert pmf[False] == pytest.approx(1 / 8)

def test_compare1():
    @d9.dist
    def f():
        a = d(6)
        assert(1 < a < 6)
        return a

    pmf = f()

def test_semiring1():
    @d9.dist(semiring=d9.BigFraction(64))
    def f():
        t = 0
        for i in range(10):
            t += d(2)
        return t

    pmf = f()
    total = sum(pmf.values())
    assert total == 1

def test_semiring1a():
    @d9.dist(semiring=d9.BigInteger(64))
    def f():
        t = 0
        for i in range(10):
            t += d(2)
        return t

    pmf = f()
    total = sum(pmf.values())
    assert total == 1024

def test_semiring2():
    @d9.dist(semiring=d9.LogReal64())
    def f():
        t = 0
        for i in range(10):
            t += d(2)
        return t

    pmf = f()
    total = functools.reduce(d9.LogReal64().add, pmf.values())
    assert total == pytest.approx(0, abs=5e-9)

def test_semiring2a():
    @d9.dist(semiring=d9.LogReal64())
    def f():
        t = 0
        for i in range(10):
            t += d[1:2]
        return t

    pmf = f()
    total = functools.reduce(d9.LogReal64().add, pmf.values())
    assert total == pytest.approx(0, abs=5e-9)

def test_semiring2a():
    @d9.dist(semiring=d9.LogReal64())
    def test():
        t = 0
        for i in range(5):
            t += d[1:1:1048575, 2:2:1]
        return t

    pmf = test()
    assert pmf[10] == pytest.approx(-69.3147180559939)

def test_semring3():
    @d9.dist(semiring=d9.BigInteger(128))
    def f():
        x = 0
        y = 0
        for i in range(35):
            d = d(4)
            if d == 1:
                x += 1
            elif d == 2:
                x -= 1
            elif d == 3:
                y += 1
            else:
                y -= 1
            assert not(x <= 0 and y == 0)
        assert x == 1 and y == 0
        return x

        pmf = f()
        assert pmf[1] == 3116285494907301262

def test_semring4():
    @d9.dist(semiring=d9.SemiringProduct(d9.Real64(), d9.BigFraction(256)))
    def f():
        total = 0
        total += d(4)
        total += d(5)
        total += d(8)
        total += d(10)
        total += d(12)
        total += d(20)
        return total

    pmf = f()
    for k, v in pmf.items():
        assert v[0] == pytest.approx(v[1])

def test_indep1():
    @d9.dist
    def f():
        x = d(2)
        y = d(2)
        return x

    pmf = f()
    assert pmf[1] == 1 / 2
    assert pmf[2] == 1 / 2

def test_lazy_bincount1():
    @d9.dist
    def f():
      return lazy_bincount(3 @ d(6), 7)

    pmf = f()
    assert pmf[(0, 3, 0, 0, 0, 0, 0)] == pytest.approx(1 / 216)
    assert pmf[(0, 2, 1, 0, 0, 0, 0)] == pytest.approx(3 / 216)
    assert pmf[(0, 1, 1, 1, 0, 0, 0)] == pytest.approx(6 / 216)

def test_lazy_topk1():
    @d9.dist
    def f():
      return lazy_topk(3 @ d(6), 2)

    pmf = f()
    assert pmf[(1, 1)] == pytest.approx(1 / 216)
    assert pmf[(6, 6)] == pytest.approx((5 * 3 + 1) / 216)

def test_lazy_sort():
    @d9.dist
    def f():
      return lazy_sort(2 @ d(6))

    pmf = f()
    assert pmf[(1, 1)] == pytest.approx(1 / 36)
    assert pmf[(1, 6)] == pytest.approx(2 / 36)
    assert pmf[(6, 6)] == pytest.approx(1 / 36)

def test_lazy_sum():
    @d9.dist
    def f():
      return lazy_sum(2 @ d(6))

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 36)
    assert pmf[7] == pytest.approx(6 / 36)
    assert pmf[12] == pytest.approx(1 / 36)

def test_kth():
    @d9.dist
    def f():
      return lazy_kth(2 @ d(6), 2)

    pmf = f()
    assert pmf[1] == pytest.approx((2 * 5 + 1) / 36)
    assert pmf[6] == pytest.approx(1 / 36)

def test_first():
    @d9.dist
    def f():
        roll = d(6)
        comp = roll >= [6, 5, 4, 3, 2, 1]
        return first(comp)

    pmf = f()
    assert pmf[0] == pytest.approx(1 / 6)
    assert pmf[5] == pytest.approx(1 / 6)

def test_last():
    @d9.dist
    def f():
        roll = d(6)
        comp = roll >= [6, 5, 4, 3, 2, 1]
        return first(comp)

    pmf = f()
    assert pmf[0] == pytest.approx(1 / 6)
    assert pmf[5] == pytest.approx(1 / 6)

def test_split1():
    def g():
        for i in range(3):
            yield i + d(6)

    @d9.dist
    def f():
        total = 0
        for x in g():
            if x >= 8:
                total += x
        return total

    pmf = f()
    assert pmf[8] == pytest.approx(1 / 6)

def test_split2():
    def g():
        x = 0
        for i in range(3):
            if d(2) == 1:
                x += 1
            yield i + d(6) + 0 * x

    @d9.dist
    def f():
        total = 0
        for x in g():
            if x >= 8:
                total += x
        return total

    pmf = f()
    assert pmf[8] == pytest.approx(1 / 6)

def test__max__():
    @d9.dist
    def f():
        return __max__(d(100))

    pmf = f()
    assert pmf[100] == pytest.approx(1)

def test_weighted():
    @d9.dist(semiring=d9.BigFraction(64))
    def f():
        return weighted([10, 20, 30], [1, 2, 3])

    pmf = f()
    assert pmf[10] == pytest.approx(1 / 6)
    assert pmf[20] == pytest.approx(1 / 3)
    assert pmf[30] == pytest.approx(1 / 2)

def test_float1():
    @d9.dist
    def f():
        return 1 / d(2) + 1 / d(2)

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 4)
    assert pmf[1.5] == pytest.approx(1 / 2)
    assert pmf[1] == pytest.approx(1 / 4)

def test_float2():
    @d9.dist
    def f():
        return reduce_sum(list(3 @ (1 / d(2))))

    pmf = f()
    assert pmf[1.5] == pytest.approx(1 / 8)
    assert pmf[3] == pytest.approx(1 / 8)

def test_float3():
    @d9.dist
    def f():
        a = [0.25, 0.5, 0.25]
        b = [d(2), d(2), d(2)]
        return reduce_sum(a * b)

    pmf = f()
    assert pmf[1] == pytest.approx(1 / 8)
    assert pmf[2] == pytest.approx(1 / 8)

def test_indep1():
    @d9.dist
    def f():
        a = [d(6), d(6)]
        t = 0
        for i in a:
            t += i
        return t

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 36)
    assert pmf[12] == pytest.approx(1 / 36)

def test_indep2():
    @d9.dist
    def f():
        x = d(6)
        a = [x, x]
        t = 0
        for i in a:
            t += i
        return t

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 6)
    assert pmf[12] == pytest.approx(1 / 6)

def test_indep3():
    def gen():
        yield d(6)
        yield d(6)

    @d9.dist
    def f():
        t = 0
        for i in gen():
            t += i
        return t

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 36)
    assert pmf[12] == pytest.approx(1 / 36)

def test_indep4():
    def gen():
        a = d(6)
        yield a
        yield a

    @d9.dist
    def f():
        t = 0
        for i in gen():
            t += i
        return t

    pmf = f()
    assert pmf[2] == pytest.approx(1 / 6)
    assert pmf[12] == pytest.approx(1 / 6)

def test_importance1():
    @d9.dist
    def f():
        importance(0.5)
        return 1

    pmf = f()
    assert pmf[1] == pytest.approx(0.5)

def test_importance2():
    @d9.dist(normalize=True)
    def main():
        r = d(2)
        importance(r)
        return r

    pmf = main()
    assert pmf[1] == pytest.approx(1 / 3)
    assert pmf[2] == pytest.approx(2 / 3)

def test_quantum1():
    def sqrtnot(i):
        r = d(2) == 2
        if i == r:
            importance(1 + 1j)
        else:
            importance(1 - 1j)
        return r

    @d9.dist(semiring=d9.Complex128())
    def main1():
        return sqrtnot(sqrtnot(True))

    pmf = main1()
    assert pmf[False] == pytest.approx(1)
    assert pmf[True] == pytest.approx(0)

    @d9.dist(semiring=d9.Complex128())
    def main2():
        return sqrtnot(sqrtnot(sqrtnot(sqrtnot(True))))

    pmf = main2()
    assert pmf[False] == pytest.approx(0)
    assert pmf[True] == pytest.approx(1)

def test_aug1():
    @d9.dist
    def main():
        total = 0
        for i in range(6):
            total += d(2)
        return total

    pmf = main()
    assert pmf[6] == pytest.approx(1 / 2**6)
    assert pmf[12] == pytest.approx(1 / 2**6)

def test_aug2():
    @d9.dist
    def main():
        total = 0
        for i in range(6):
            total |= (d(2) - 1) * (1 << i)
        return total

    pmf = main()
    print(pmf)
    assert pmf[0] == pytest.approx(1 / 2**6)
    assert pmf[63] == pytest.approx(1 / 2**6)

def test_int1():
    @d9.dist
    def main():
        return int(d(6) / 2)

    pmf = main()

    assert pmf[0] == pytest.approx(1 / 6)
    assert pmf[2] == pytest.approx(1 / 3)

def test_int2():
    @d9.dist
    def main():
        return -int(d(2) == 2)

    pmf = main()
    assert pmf[0] == pytest.approx(1 / 2)
    assert pmf[-1] == pytest.approx(1 / 2)

def test_options1():

    @d9.dist
    def f(n=1):
        return d(n)

    pmf = f(6)
    assert pmf[1] == pytest.approx(1 / 6)
    assert pmf[6] == pytest.approx(1 / 6)

    pmf = f(n=4, _options={'semiring': d9.BigFraction(64)})
    assert pmf[1] == Fraction(1, 4)
    assert pmf[4] == pytest.approx(1 / 4)
