# test_cmds.py

import unittest
import math

import dice9 as d9

#from dice9.problib import *

class TestProbLangSubscripts(unittest.TestCase):
    def test_subscript1(self):
        def f():
            x = [0, 1]
            return x[0]

        pmf = d9.run(f, squeeze=True)

        self.assertAlmostEqual(pmf[0], 1)

    def test_subscript2(self):
        def f():
            x = [d(6), d(4)]
            return x[0]

        pmf = d9.run(f, squeeze=True)

        for i in range(1, 7):
            self.assertAlmostEqual(pmf[i], 1 / 6)

    def test_subscript3(self):
        def f():
            x = [d(4), d(6)]
            return x[-1]

        pmf = d9.run(f, squeeze=True)

        for i in range(1, 7):
            self.assertAlmostEqual(pmf[i], 1 / 6)

    def test_subscript4(self):
        def f():
            x = [d(4), d(6), d(6)]
            return x[0:2]

        pmf = d9.run(f, squeeze=True)

        self.assertAlmostEqual(pmf[(1, 1)], 1 / 24)

    def test_subscript5(self):
        def f():
            x = [d(4), d(6), d(8)]
            i = d(3) - 1
            return x[i]

        pmf = d9.run(f, squeeze=True)

        self.assertAlmostEqual(pmf[1], (1 / 4 + 1 / 6 + 1 / 8) / 3)
        self.assertAlmostEqual(pmf[8], 1 / 24)

    def test_subscript6(self):
        def f():
            x = [d(4), d(6)]
            x[0] = 1
            return x[1]

        pmf = d9.run(f, squeeze=True)

        self.assertAlmostEqual(pmf[1], 1 / 6)

    def test_subscript7(self):
        def f():
            x = [d(4), d(6)]
            x[0] = d(6)
            return x

        pmf = d9.run(f, squeeze=True)

        self.assertAlmostEqual(pmf[(1, 1)], 1 / 36)

    def test_subscript8(self):
        def f():
            x = [d(4), d(6)]
            x[1] = x[0]
            return x

        pmf = d9.run(f, squeeze=True)

        self.assertAlmostEqual(pmf[(2, 2)], 1 / 4)

    def test_subscript9(self):
        def f():
            x = [d(4), d(6)]
            x[0:1] = 0
            return d(2)

        with self.assertRaisesRegex(d9.InterpreterError, r".*Assign.*"):
            pmf = d9.run(f)

    def test_subscript10(self):
        def f():
            x = [d(4), d(6)]
            x[0] = x[0] + 1
            return d(2)

        pmf = d9.run(f)

    def test_subscript11(self):
        def f():
            x = zeros(3)
            for i in range(3):
                x[i] = d(4 + 2 * i)
            return reduce_sum(x, -1)

        pmf = d9.run(f, squeeze=True)

        self.assertAlmostEqual(pmf[3], 1 / (4 * 6 * 8))
        self.assertAlmostEqual(pmf[18], 1 / (4 * 6 * 8))

    def test_subscript11(self):
        def f():
            x = [[1, 2, 3], 
                 [4, 5, d(6)]]
            i = d(2) - 1
            return x[i]

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[(1, 2, 3)], 1 / 2)
        self.assertAlmostEqual(pmf[(4, 5, 6)], 1 / 12)

    def test_subscript12(self):
        def f():
            x = [0, 0]
            for i in range(10):
                x[d(2) - 1] += 1
            return x

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[(5, 5)], 0.24609375)

    def test_subscript13(self):
        def f():
            x = [1, 1]
            for i in range(10):
                x[d(2) - 1] *= 2
            return x

        pmf = d9.run(f, squeeze=True)
        self.assertAlmostEqual(pmf[(32, 32)], 0.24609375)
