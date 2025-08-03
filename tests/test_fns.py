# test_fns.py

import unittest
import math
import functools
import dice9 as d9

#from dice9.problib import *

class TestProbLangFns(unittest.TestCase):
    def test_abs1(self):
        def f():
            x = d(3)
            return abs(x - 2)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 2 / 3, places=5)
        self.assertAlmostEqual(pmf[(1,)], 2 / 3, places=5)

    def test_reshape1(self):
        def reshape1():
            return reshape(1, [1])

        pmf = d9.run(reshape1)

        self.assertAlmostEqual(pmf[((1,),)], 1.0, places=5)
        self.assertEqual(len(pmf), 1)

    def test_reshape2(self):
        def reshape2():
            return reshape(1, [1, 1])

        pmf = d9.run(reshape2)

        self.assertAlmostEqual(pmf[(((1,),),)], 1.0, places=5)
        self.assertEqual(len(pmf), 1)

#    def test_det(self):
#        def det2x2():
#            x = list(4 @ d(2))
#            m = reshape(x, [2, 2])
#            return det(m) % 2
#
#        singular_dist = d9.run(det2x2)
#        self.assertAlmostEqual(singular_dist[(True,)], 3 / 8)

    def test_argmin1(self):
        def argmin1():
            x = [[1, 2], [8, 4]]
            return argmin(x, axis=-2)

        pmf = d9.run(argmin1)
        self.assertAlmostEqual(pmf[((0, 0),)], 1.0)

        def argmin2():
            x = [[1, 2], [8, 4]]
            return argmin(x, axis=-1)

        def argmin3():
            x = [[1, 2], [8, 4]]
            return argmin(x, axis=0)

        pmf = d9.run(argmin3)
        self.assertAlmostEqual(pmf[((0, 0),)], 1.0)

        def argmin4():
            x = [[1, 2], [8, 4]]
            return argmin(x, axis=1)

        pmf = d9.run(argmin4)
        self.assertAlmostEqual(pmf[((0, 1),)], 1.0)

        def argmax4():
            x = [[1, 2], [8, 4]]
            return argmax(x, axis=1)

        pmf = d9.run(argmax4)
        self.assertAlmostEqual(pmf[((1, 0),)], 1.0)

    def test_argsort1(self):
        def f():
            return argsort([d(2), d(2)], -1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((0, 1),)], 3 / 4)
        self.assertAlmostEqual(pmf[((1, 0),)], 1 / 4)

    def test_argsort2(self):
        def f():
            return argsort([d(2), d(2)], 0)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((0, 1),)], 3 / 4)
        self.assertAlmostEqual(pmf[((1, 0),)], 1 / 4)

    def test_argsort3(self):
        def f():
            return argsort([d(2), d(2)])

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((0, 1),)], 3 / 4)
        self.assertAlmostEqual(pmf[((1, 0),)], 1 / 4)

    def test_cumsum1(self):
        def f():
            z = [d(2), d(2)]
            return cumsum(z)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((1, 2),)], 1 / 4)
        self.assertAlmostEqual(pmf[((2, 4),)], 1 / 4)

    def test_cumsum2(self):
        def f():
            z = [d(2), d(2)]
            return cumsum(z, -1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((1, 2),)], 1 / 4)
        self.assertAlmostEqual(pmf[((2, 4),)], 1 / 4)

    def test_zeros(self):
        def f():
            return zeros([2, 2])

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(((0, 0), (0, 0)),)], 1)

    def test_ones(self):
        def f():
            return ones([2, 2])

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(((1, 1), (1, 1)),)], 1)

    def test_dd1(self):
        def f():
            x = d(2)
            y = dd(x)
            return y

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(0,)], 0.75)
        self.assertAlmostEqual(pmf[(1,)], 0.25)

    def test_dd2(self):
        def f():
            x = d(2)
            y = d(2)
            return dd(x) + dd(y)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(0,)], 0.75**2)
        self.assertAlmostEqual(pmf[(1,)], 2 * 0.75 * 0.25)
        self.assertAlmostEqual(pmf[(2,)], 0.25**2)

    def test_dd3(self):
        def f():
            x = d(2)
            return dd(x) + dd(x)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(0,)], 0.5 + 0.5 * 0.5**2)
        self.assertAlmostEqual(pmf[(2,)], 0.5 * 0.5**2)

    def test_dd4(self):
        def f():
            return dd(d(2)) + dd(d(2))

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(0,)], 0.75**2)
        self.assertAlmostEqual(pmf[(1,)], 2 * 0.75 * 0.25)
        self.assertAlmostEqual(pmf[(2,)], 0.25**2)

    def test_ifexp1(self):
        def f():
            return 10 if d(2) == 1 else 20

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(10,)], 0.5)
        self.assertAlmostEqual(pmf[(20,)], 0.5)

    def test_ifexp2(self):
        def f():
            x = d(6)
            x = x + d(6) if x == 6 else x
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 1 / 6)
        self.assertAlmostEqual(pmf[(12,)], 1 / 36)

    def test_ifexp3(self):
        def f():
            x = 0
            for i in range(3):
                x = x + d(6) if x == 6 * i else x
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 1 / 6)
        self.assertAlmostEqual(pmf[(18,)], 1 / 216)

    def test_ifexp4(self):
        def f(x):
            return 2 * x if x % 2 == 0 else x

        def g():
            return f(d(10))

        pmf = d9.run(g)
        for i in [1, 4, 3, 8, 5, 12, 7, 16, 9, 20]:
            self.assertAlmostEqual(pmf[(i,)], 1 / 10)

    def test_int(self):
        def f():
            x = d(6)
            return int(x == 1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 1 / 6)

    def test_minimal_gen(self):
        def f():
            yield 1

        def g():
            x = 0
            for y in f():
                x += y
            return x

        pmf = d9.run(g)
        self.assertAlmostEqual(pmf[(1,)], 1)

    def test_small_gen(self):
        def f():
            yield d(6)

        def g():
            x = 0
            for y in f():
                x += y
            return x

        pmf = d9.run(g)
        self.assertAlmostEqual(pmf[(1,)], 1 / 6)

    def test_small_gen(self):
        def f():
            for i in range(3):
                yield d(6)

        def g():
            x = 0
            for y in f():
                x += y
            return x

        pmf = d9.run(g)
        self.assertAlmostEqual(pmf[(3,)], 1 / 6**3)
        self.assertAlmostEqual(pmf[(18,)], 1 / 6**3)

    def test_gen1(self):
        def f(n):
            for i in range(n):
                yield d(6)

        def g():
            x = 0
            for y in f(3):
                x += y
            return x

        pmf = d9.run(g)
        self.assertAlmostEqual(pmf[(3,)], 1 / 6**3)
        self.assertAlmostEqual(pmf[(18,)], 1 / 6**3)

    def test_gen2(self):
        def h():
            yield d(2)
            yield d(2)
            yield d(2)

        def g():
            for i in h():
                yield i

        def f():
            t = 0
            for i in g():
                t += i
            return t

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[3], 0.125)
        self.assertAlmostEqual(pmf[6], 0.125)

    def test_sum1(self):
        def g():
            yield d(6)
            yield d(6)
            yield d(6)

        def f():
            x = sum(*g())
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(3,)], 1 / 6**3)
        self.assertAlmostEqual(pmf[(18,)], 1 / 6**3)

    def test_sum2(self):
        def g():
            for i in range(3):
                yield d(6)

        def f():
            x = sum(*g())
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(3,)], 1 / 6**3)
        self.assertAlmostEqual(pmf[(18,)], 1 / 6**3)

    def test_genexp1(self):
        def g():
            yield 1
            yield 2

        def f():
            z = sum(*(x for x in g()))
            return z

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(3,)], 1)

    def test_genexp2(self):
        def g():
            yield d(6)
            yield d(6)

        def f():
            z = sum(*(x for x in g()))
            return z

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(2,)], 1 / 36)
        self.assertAlmostEqual(pmf[(12,)], 1 / 36)

    def test_genexp3(self):
        def g(n):
            for i in range(n):
                yield d(6)

        def f():
            z = sum(*(x - 1 for x in g(3)))
            return z

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(0,)], 1 / 216)
        self.assertAlmostEqual(pmf[(15,)], 1 / 216)

    def test_genexp4(self):
        def f():
            return sum(*(int(d(20) == 1) for i in range(20)))

        pmf = d9.run(f)
        for i in range(20):
            self.assertAlmostEqual(
                pmf[(i,)], math.comb(20, i) * (1 / 20) ** i * (19 / 20) ** (20 - i)
            )

    def test_genexp5(self):
        def f():
            return sum(*(20 @ int(d(20) == 1)))

        pmf = d9.run(f)
        for i in range(20):
            self.assertAlmostEqual(
                pmf[(i,)], math.comb(20, i) * (1 / 20) ** i * (19 / 20) ** (20 - i)
            )

    def test_subscript1(self):
        def f():
            x = [10, 20, 30, 40, 50, 60]
            return x[d(6) - 1]

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(10,)], 1 / 6)
        self.assertAlmostEqual(pmf[(60,)], 1 / 6)

    def test_topk1(self):
        def my_top_2(s):
            xs = [0, 0]

            for y in s:
                y1 = reshape(y, [1])
                c = concat([xs, y1], -1)
                xs = top_k(c, 2)

            return xs

        def es():
            for i in range(6):
                x = d(6)
                x = x + d(6) if x == 6 else x

                yield x

        def g():
            top2 = my_top_2(es())
            return reduce_sum(top2, -1)

        pmf = d9.run(g)
        self.assertAlmostEqual(pmf[(2,)], 2.1433470507544594e-05)
        self.assertAlmostEqual(pmf[(24,)], 0.010743132472754508)

    def test_topk2(self):
        def f():
            a = [d(4), d(6), d(6), d(10)]
            b = top_k(a, 2)
            return reduce_sum(b, -1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(2,)], 1 / 1440)
        self.assertAlmostEqual(pmf[(16,)], 11 / 360)

    def test_isin1(self):
        def f():
            x = d(6)
            return d(6) if isin(x, [1, 2]) else x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 5 / 90)
        self.assertAlmostEqual(pmf[(6,)], 2 / 9)

    def test_isin2(self):
        def f():
            x = d(6)
            return d(6) if x in [1, 2] else x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 5 / 90)
        self.assertAlmostEqual(pmf[(6,)], 2 / 9)

    def test_birthday(self):
        def f():
            number_unique_birthdays = 0

            for i in range(23):
                number_unique_birthdays += int(
                    d(365) > number_unique_birthdays)

            return number_unique_birthdays == 23

        pmf = d9.run(f)
        prob = functools.reduce(lambda x, y: x * y, (1 - i / 365 for i in range(23)))
        self.assertAlmostEqual(pmf[(True,)], prob)

    def test_list1(self):
        def f():
            return [1]

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((1,),)], 1)

    def test_list2(self):
        def f():
            return [1, 2]

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((1, 2),)], 1)

    def test_list3(self):
        def f():
            return [d(2), d(6)]

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((1, 1),)], 1 / 12)
        self.assertAlmostEqual(pmf[((2, 6),)], 1 / 12)

    def test_list4(self):
        def f():
            return [d(2), min(d(3), 2)]

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((1, 1),)], 1 / 6)
        self.assertAlmostEqual(pmf[((2, 2),)], 1 / 3)

    def test_list5(self):
        def step(n):
            for i in range(n):
                yield [2 * d(2) - 3, 2 * d(2) - 3]

        def f():
            return sum(*step(4))

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((0, 0),)], 9 / 64)

    def test_bincount1(self):
        def f():
            a = [1, 3, 2, 5, 1, 3]
            return bincount(a, 8)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((0, 2, 1, 2, 0, 1, 0, 0),)], 1.0)

    def test_reduce_sum1(self):
        def f():
            return reduce_sum([d(4), d(4)], axis=-1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(2,)], 0.0625)
        self.assertAlmostEqual(pmf[(8,)], 0.0625)

    def test_reduce_sum2(self):
        def f():
            return reduce_sum([d(4), d(4)])

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(2,)], 0.0625)
        self.assertAlmostEqual(pmf[(8,)], 0.0625)

    def test_reduce_min1(self):
        def f():
            return reduce_min([d(4), d(4)], axis=-1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 0.4375)
        self.assertAlmostEqual(pmf[(4,)], 0.0625)

    def test_reduce_min2(self):
        def f():
            return reduce_min([d(4), d(4)])

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 0.4375)
        self.assertAlmostEqual(pmf[(4,)], 0.0625)

    def test_reduce_max(self):
        def f():
            return reduce_max([d(4), d(4)], axis=-1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 0.0625)
        self.assertAlmostEqual(pmf[(4,)], 0.4375)

    def test_reduce_any1(self):
        def f():
            return reduce_any([d(6), d(6), d(6)] == 1, -1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(True,)], 1 - (5 / 6)**3)

    def test_reduce_any2(self):
        def f():
            return reduce_any([d(6), d(6), d(6)] == 1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(True,)], 1 - (5 / 6)**3)

    def test_reduce_all(self):
        def f():
            return reduce_all([d(6), d(6), d(6)] == 1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(True,)], 1 / 216)

    def test_reduce_all2(self):
        def f():
            return reduce_all([d(6), d(6), d(6)] == 1, -1)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(True,)], 1 / 216)

    def test_matmult(self):
        def f():
            t = 0
            for x in 3 @ d(6):
                t += x
            return t

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(18,)], 1 / 216)

    def test_cast1(self):
        def f():
            return cast(d(2) == 1, "int64")

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[0], 1 / 2)
        self.assertAlmostEqual(pmf[1], 1 / 2)

    def test_cast2(self):
        def f():
            return cast(d(2) - 1, "bool")

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[True], 1 / 2)
        self.assertAlmostEqual(pmf[False], 1 / 2)

    def test_multiroll2(self):
        def f():
            x = d(6)
            return d[1 : x + 1 : 1]

        pmf = d9.run(f, squeeze=True)
        for i in range(1, 7):
            p = sum(1/(6 * n) for n in range(i, 7))
            self.assertAlmostEqual(pmf[i], p)

    def test_flip1(self):
        def f():
            x = [d(2), d(5)]
            x = flip(x, -1)
            return x[1] > x[0]

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[True], 0.1)

    def test_flip2(self):
        def f():
            x = [d(2), d(5)]
            x = flip(x)
            return x[1] > x[0]

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[True], 0.1)

    def test_len1(self):
        def f():
            return len([d(2), d(3)])

        pmf = d9.run(f, squeeze=True)
        self.assertEqual(pmf, [2])

    def test_len2(self):
        def f():
          t = []
          for r in {d(2), d(2), d(2)}:
            t = concat([t, [r]], -1)
            if len(t) > 2:
                t = top_k(t, 2)

          return t

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[(1, 1)], 0.125)
        self.assertAlmostEqual(pmf[(2, 2)], 0.5)
