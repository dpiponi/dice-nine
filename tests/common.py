# common.py

import math
import unittest

import dice9 as d9

# from dice9.problib import *

# standard Python functions
class TestProbLang(unittest.TestCase):
    def coin_flip(self):
        def f1():
            x = d(2)
            return x

        pdf = d9.run(coin_flip)

        for x in range(1, 3):
            self.assertAlmostEqual(pdf[(x,)], 0.5, places=5)

        self.assertEqual(len(pdf), 2)

    def test2(self):
        def two_coins():
            x = d(2)
            y = d(2)
            return x, y

        pdf = d9.run(two_coins)

        for x in range(1, 3):
            for y in range(1, 3):
                self.assertAlmostEqual(pdf[(x, y)], 0.25, places=5)

        self.assertEqual(len(pdf), 4)

    def test3(self):
        def sum_two_coins():
            x = d(2)
            y = d(2)
            z = x + y
            return z

        pdf = d9.run(sum_two_coins)

        self.assertAlmostEqual(pdf[(2,)], 0.25, places=5)
        self.assertAlmostEqual(pdf[(3,)], 0.5, places=5)
        self.assertAlmostEqual(pdf[(4,)], 0.25, places=5)

        self.assertEqual(len(pdf), 3)

    def test4(self):
        def two_or_three():
            x = d(2)
            if x == 1:
                y = 2
            else:
                y = 3
            return y

        pdf = d9.run(two_or_three)

        for x in range(2, 4):
            self.assertAlmostEqual(pdf[(x,)], 0.5, places=5)

        self.assertEqual(len(pdf), 2)

    def test_sum_ten_coins(self):
        def sum_ten_coins():
            x = 0
            for i in range(10):
                x = x + d(2)
            return x

        pdf = d9.run(sum_ten_coins)

        for x in range(10, 21):
            self.assertAlmostEqual(pdf[(x,)], math.comb(10, x - 10) * 2**-10, places=5)

        self.assertEqual(len(pdf), 11)

    def test_fight(self):
        def fight():
            hp1 = 4
            hp2 = 4
            for i in range(8):
                if (hp1 > 0 and hp2 > 0) and d(20) > 10:
                    hp2 = max(hp2 - d(4), 0)
                if (hp1 > 0 and hp2 > 0) and d(20) > 10:
                    hp1 = max(hp1 - d(4), 0)
            return hp1, hp2

        pdf = d9.run(fight)

        matrix = [
            [
                0,
                0.10091449320316304,
                0.08755356073379526,
                0.07569718360900866,
                0.3046693801879884,
            ],
            [
                0.10894726216793038,
                0.0006201267242431642,
                0.0003647804260253907,
                0.00019454956054687503,
                9.727478027343751e-05,
            ],
            [
                0.09132260084152223,
                0.0003647804260253906,
                0.00021457672119140625,
                0.00011444091796875001,
                5.722045898437501e-05,
            ],
            [
                0.07593226432800292,
                0.000194549560546875,
                0.00011444091796875008,
                6.103515625000007e-05,
                3.051757812500002e-05,
            ],
            [
                0.15233469009399414,
                9.727478027343749e-05,
                5.722045898437501e-05,
                3.051757812500001e-05,
                1.5258789062499998e-05,
            ],
        ]

        for i in range(5):
            for j in range(5):
                if matrix[i][j] > 0:
                    self.assertAlmostEqual(pdf[(j, i)], matrix[i][j], places=5)

    def test9(self):
        def f9():
            dice = []
            dice = concat([[d(6)], dice], -1)
            for i in range(12):
                dice = concat([[d(6)], dice], -1)
                dice = sort(dice, -1)
            median = dice[6]
            return median

        def binomial_cdf(n: int, p: float, k: int) -> float:
            return sum(
                math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1)
            )

        def median_cdf(N: int, m: int) -> float:
            k = (N - 1) // 2
            cdf_m = 1 - binomial_cdf(N, m / 6, k)
            cdf_m_1 = 1 - binomial_cdf(N, (m - 1) / 6, k)
            return cdf_m - cdf_m_1

        median_dist = d9.run(f9)
        for i in range(1, 7):
            self.assertAlmostEqual(median_dist[(i,)], median_cdf(13, i))

#    def test11(self):
#        def binary_det():
#            x = list(16 @ d(2))
#            m = reshape(x, [4, 4])
#            return det(m) % 2
#
#        singular_dist = d9.run(binary_det)
#        self.assertAlmostEqual(singular_dist[(True,)], 315 / 1024)

    def test_roll_three(self):
        def roll_three():
            dice = list(3 @ d(6))
            total = 0
            for i in range(3):
                total = total + dice[i]
            return total

        total = d9.run(roll_three)

        expected = {
            3: 0.004629629629629629,
            4: 0.013888888888888888,
            5: 0.027777777777777776,
            6: 0.046296296296296294,
            7: 0.06944444444444445,
            8: 0.09722222222222218,
            9: 0.11574074074074067,
            10: 0.12499999999999992,
            11: 0.12499999999999992,
            12: 0.11574074074074067,
            13: 0.09722222222222218,
            14: 0.06944444444444445,
            15: 0.046296296296296294,
            16: 0.027777777777777776,
            17: 0.013888888888888888,
            18: 0.004629629629629629,
        }
        for i in range(4, 19):
            self.assertAlmostEqual(total[(i,)], expected[i])

    # exponentially tilted
    def test13(self):
        def conditional_sum_dice():
            init = d(6)
            total = init
            for i in range(49):
                total = total + d(6)
            return init, (total == 200)

        result = d9.run(conditional_sum_dice)

        p = [result[(i, True)] for i in range(1, 7)]
        total = sum(p)
        p = [x / total for x in p]

        expected = [
            0.1014269667663801,
            0.12273233569730187,
            0.14742337219877025,
            0.17578870491813472,
            0.20808829694631195,
            0.24454032347310117,
        ]

        for i in range(0, 6):
            self.assertAlmostEqual(p[i], expected[i])

    def test14(self):
        def missing_roll():
            tables = [0, 0, 0, 0, 0, 0]
            for i in range(4):
                t = d(6)
                tables = tables + one_hot(t - 1, 6)
            return argmin(tables, -1)

        result = d9.run(missing_roll)
        expected = [
            0.48225308641975423,
            0.2847222222222224,
            0.14969135802469133,
            0.06481481481481481,
            0.018518518518518517,
        ]
        for i in range(5):
            self.assertAlmostEqual(result[(i,)], expected[i])


if __name__ == "__main__":
    unittest.main()
