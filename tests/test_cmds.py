# test_cmds.py

import unittest
import math

import dice9 as d9

#from dice9.problib import *

class TestProbLangCmds(unittest.TestCase):
    def test_return(self):
        def two_coins():
            x = d(2)
            y = d(2)
            return x, y

        pmf = d9.run(two_coins)

        for x in range(1, 3):
            for y in range(1, 3):
                self.assertAlmostEqual(pmf[(x, y)], 0.25, places=5)

        self.assertEqual(len(pmf), 4)

    def test_return2(self):
        def g():
            return x

        with self.assertRaisesRegex(ValueError, r".*move.*"):
            pmf = d9.run(g)

    def test_for1(self):
        def sum_ten_coins():
            x = 0
            for i in range(10):
                x = x + d(2)
            return x

        pmf = d9.run(sum_ten_coins)

        for x in range(10, 21):
            self.assertAlmostEqual(pmf[(x,)], math.comb(10, x - 10) * 2**-10, places=5)

        self.assertEqual(len(pmf), 11)

    def test_for2(self):
        def sum_six_coins():
            x = 0
            for i in range(2):
                for j in range(3):
                    x = x + d(2)
            return x

        pmf = d9.run(sum_six_coins)

        for x in range(6, 13):
            self.assertAlmostEqual(pmf[(x,)], math.comb(6, x - 6) * 2**-6, places=5)

        self.assertEqual(len(pmf), 7)

    def test_if1(self):
        def two_or_three():
            x = d(2)
            if x == 1:
                y = 2
            else:
                y = 3
            return y

        pmf = d9.run(two_or_three)

        for x in range(2, 4):
            self.assertAlmostEqual(pmf[(x,)], 0.5, places=5)

        self.assertEqual(len(pmf), 2)

    def test_if2(self):
        def two34or5():
            x = d(2)
            if x == 1:
                y = d(2)
                if y == 1:
                    z = 2
                else:
                    z = 3
            else:
                y = d(2)
                if y == 1:
                    z = 4
                else:
                    z = 5
            return z

        pmf = d9.run(two34or5)

        for x in range(2, 6):
            self.assertAlmostEqual(pmf[(x,)], 0.25, places=5)

        self.assertEqual(len(pmf), 4)

    def test_if3(self):
        def f():
            hp2 = 1

            if hp2 > 0:
                hp2 = hp2

            return hp2

        pmf = d9.run(f)

    def test_call1(self):
        def f(x):
            return x + d(2)

        def g():
            return f(0)

        pmf = d9.run(g)

        self.assertAlmostEqual(pmf[(1,)], 0.5, places=5)
        self.assertAlmostEqual(pmf[(2,)], 0.5, places=5)

    def test_call2(self):
        def f(x):
            return x + d(2)

        def g():
            return f(f(0))

        pmf = d9.run(g)

        self.assertAlmostEqual(pmf[(2,)], 0.25, places=5)
        self.assertAlmostEqual(pmf[(3,)], 0.5, places=5)
        self.assertAlmostEqual(pmf[(4,)], 0.25, places=5)

    def test_call3(self):
        def f(x, y):
            return x + y

        def g():
            return f(d(2), d(2))

        pmf = d9.run(g)

        self.assertAlmostEqual(pmf[(2,)], 0.25, places=5)
        self.assertAlmostEqual(pmf[(3,)], 0.5, places=5)
        self.assertAlmostEqual(pmf[(4,)], 0.25, places=5)

    def test_call4(self):
        def f(x, y):
            return x + y

        def g():
            return f(0)

        # make a proper error XXX
        #with self.assertRaisesRegex(KeyError, r".*y.*"):
        with self.assertRaisesRegex(d9.InterpreterError, r".*Call.*"):
            pmf = d9.run(g)

    def test_call5(self):
        def f(x):
            return x + d(6)

        def g():
            x = 0
            for i in range(3):
                x = f(x)
            return x

        pmf = d9.run(g)
        self.assertAlmostEqual(pmf[(3,)], 1 / 216, places=5)
        self.assertAlmostEqual(pmf[(10,)], 0.125, places=5)
        self.assertAlmostEqual(pmf[(18,)], 1 / 216, places=5)

    def test_if_for1(self):
        def f():
            x = 0
            if d(2) == 1:
                for i in range(3):
                    x += d(2)
            else:
                for i in range(2):
                    x += d(3)
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(2,)], 1 / 18, places=5)
        self.assertAlmostEqual(pmf[(3,)], 25 / 144, places=5)
        self.assertAlmostEqual(pmf[(4,)], 51 / 144, places=5)
        self.assertAlmostEqual(pmf[(5,)], 43 / 144, places=5)
        self.assertAlmostEqual(pmf[(6,)], 17 / 144, places=5)

    def test_for_if1(self):
        def f():
            x = 0
            for i in range(3):
                if d(2) == 1:
                    x += d(2)
                else:
                    x += d(3)
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(3,)], (5 / 12)**3, places=5)

    def test_for_if2(self):
        def f():
            x = 0
            y = 0
            for i in range(3):
                if d(2) == 1:
                    x += d(2)
                else:
                    y += d(2)
            return x + y

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(3,)], 1 / 8, places=5)
        self.assertAlmostEqual(pmf[(6,)], 1 / 8, places=5)

    def test_if3(self):
        def f():
            x = d(100)
            y = d(100)
            if x + y >= 198:
                z = 1
            else:
                z = 0
            return z

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 6 / 10000, places=5)

    def test_if4(self):
        def f():
            x = 0

            if d(2) == 1:
                x = -x
                x = x - 1

            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(-1,)], 1 / 2, places=5)
        self.assertAlmostEqual(pmf[(0,)], 1 / 2, places=5)

    def test_if5(self):
        def f():
            x = [1]
            if False:
                x = zeros(1)
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((1,),)], 1, places=5)

    def test_if6(self):
        def f():
            x = [1]
            if True:
                x = zeros(1)
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[((0,),)], 1, places=5)

    def test_if7(self):
        def f():
            x = 4
            if d(2) == 1:
                x = d(2)
            return x

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 0.25)
        self.assertAlmostEqual(pmf[(2,)], 0.25)
        self.assertAlmostEqual(pmf[(4,)], 0.5)

    def test_if8(self):
        def f():
            if d(2) == 1:
                x = 1
            return d(2)

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(1,)], 0.5)

    if 0:
        def test_for3(self):
            def f():
                x = constant([2, 3, 4])
                y = 0
                for i in x:
                    y += i
                return y

            pmf = d9.run(f)
            self.assertAlmostEqual(pmf[(9,)], 1, places=5)

    def test_for4(self):
        def f():
            xs = []
            for i in range(3):
                xs = concat([xs, reshape(d(2), [1])], -1)
            y = 0
            for x in xs:
                y = y + x
            return y

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(3,)], 1 / 8, places=5)

    def test_frame1(self):
        def g(n):
            result = 0
            if n == 0:
                result = 0

            return result

        def f():
            s = g(0)
            s = g(0)
            return s

        pmf = d9.run(f)
        self.assertAlmostEqual(pmf[(0,)], 1)

    def test_recurse1(self):
        def recurse4():
            return d(6)

        def recurse3():
            return recurse4() + recurse4()

        def recurse2():
            return recurse3() + recurse3()

        def recurse1():
            return recurse2() + recurse2()

        def f():
            return recurse1() + recurse1()

        def g():
            return sum(*(16 @ d(6)))

        pmf1 = d9.run(f)
        pmf2 = d9.run(g)

        for i in range(16, 16 * 6 + 1):
            self.assertAlmostEqual(pmf1[(i,)], pmf2[(i,)])

    def test_recurse2(self):
        def g(n):
            if n == 0:
                x = d(6)
            else:
                x = g(n - 1) + g(n - 1)
            return x

        def f():
            return g(4)

        def h():
            return sum(*(16 @ d(6)))

        pmf1 = d9.run(f)
        pmf2 = d9.run(h)

        for i in range(16, 16 * 6 + 1):
            self.assertAlmostEqual(pmf1[(i,)], pmf2[(i,)])

    def test_gen_if1(self):
        def g():
            for i in range(2):
                yield 0

        def f():
            y = 0
            for x in g():
                if x:
                    y = 0
            return y

        pmf = d9.run(f, squeeze=True)
        #self.assertAlmostEqual(pmf[0], 1)
        self.assertEqual(pmf, [0])

    def test_gen_if2(self):
        def g():
            for j in range(3):
                count = 0
                yield 0

        def f():
            x = 0
            for num in g():
                if True:
                    x = 0

            return x

        pmf = d9.run(f, squeeze=True)
        self.assertEqual(pmf, [0])
        #self.assertAlmostEqual(pmf[0], 1)

    def test_gen_del(self):
        def g():
            x = 1
            del x
            yield x

        def f():
            t = 0
            for i in g():
                t += i
            return t

        with self.assertRaisesRegex(d9.InterpreterError, r".*Call.*"):
            pmf = d9.run(f)

    def test_aug1(self):
        def f():
            x = d(2)
            x += 1
            return x

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[2], 1 / 2)
        self.assertAlmostEqual(pmf[3], 1 / 2)

    def test_aug2(self):
        def f():
            x = d(2)
            x -= 1
            return x

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[0], 1 / 2)
        self.assertAlmostEqual(pmf[1], 1 / 2)

    def test_error1(self):
        def f():
            del a.b

        # with self.assertRaisesRegex(ValueError, r".*not supported.*"):
        with self.assertRaisesRegex(ValueError, r".*move.*"):
            pmf = d9.run(f)

    def test_gen_for1(self):
        def g(x):
            for i in x:
                yield i

        def f():
            t = 0
            for j in g([1, 2, 3]):
                t += d(j)
            return t

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[3], 1 / 6)
        self.assertAlmostEqual(pmf[4], 1 / 3)
        self.assertAlmostEqual(pmf[5], 1 / 3)
        self.assertAlmostEqual(pmf[6], 1 / 6)

    def test_assert1(self):
        def f():
            d = d(6)
            assert d > 1
            return d

        pmf = d9.run(f, normalize=True, squeeze=True)
        for i in range(2, 7):
            self.assertAlmostEqual(pmf[i], 1 / 5)

    def test_assert2(self):
        def f():
            x = d(2)
            assert x > 2
            return x + d(2)

        # Don't care about result
        result = d9.run(f)
        self.assertEqual(len(result), 0)

    def test_args1(self):
        def f(n):
            return d(n)

        pmf = d9.run(f, 8, squeeze=True)
        for i in range(1, 9):
            self.assertAlmostEqual(pmf[i], 1 / 8)

    def test_args2(self):
        def f(n):
            return d(n)

        with self.assertRaisesRegex(TypeError, r".*Missing positional argument.*"):
            pmf = d9.run(f)

    def test_args3(self):
        def f():
            return d(6)

        with self.assertRaisesRegex(TypeError, r".*Extra argument.*"):
            pmf = d9.run(f, 17)

    def test_listcomp1(self):
        def f():
            return reduce_sum([10 * x for x in 3 @ d(6)], -1)

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[30], 1 / 216)
        self.assertAlmostEqual(pmf[100], 1 / 8)
        self.assertAlmostEqual(pmf[180], 1 / 216)

    def test_stream1(self):
        def f():
            t = 0
            for i in {d(2), d(2)}:
                t += i
            return t

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[2], 1 / 4)
        self.assertAlmostEqual(pmf[4], 1 / 4)

    def test_stream2(self):
        def f():
            return sum(*(2 @ d(3)), *(2 @ d(6)))

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[4], 1 / (3 * 3 * 6 * 6))
        self.assertAlmostEqual(pmf[18], 1 / (3 * 3 * 6 * 6))

    def test_genexp_iter(self):
        def grouped():
            for j in range(2):
                yield sum(*(j for i in range(2)))

        def f():
            for i in grouped():
                x = 1
            return d(6)

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[1], 1 / 6)
