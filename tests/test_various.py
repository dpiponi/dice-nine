import math
import pytest
import dice9 as d9
from pytest import approx
from fractions import Fraction


class TestProbLang:

    def test1(self):

        @d9.dist
        def coin_flip(self):

            def f1():
                x = d(2)
                return x

            pdf = coin_flip()

            for x in range(1, 3):
                assert pdf[x] == approx(0.5, rel=1e-5)

            assert len(pdf) == 2

    def test2(self):

        @d9.dist
        def two_coins():
            x = d(2)
            y = d(2)
            return x, y

        pdf = two_coins()

        for x in range(1, 3):
            for y in range(1, 3):
                assert pdf[(x, y)] == approx(0.25, rel=1e-5)

        assert len(pdf) == 4

    def test3(self):

        @d9.dist
        def sum_two_coins():
            x = d(2)
            y = d(2)
            z = x + y
            return z

        pdf = sum_two_coins()

        assert pdf[2] == approx(0.25, rel=1e-5)
        assert pdf[3] == approx(0.5, rel=1e-5)
        assert pdf[4] == approx(0.25, rel=1e-5)

        assert len(pdf) == 3

    def test4(self):

        @d9.dist
        def two_or_three():
            x = d(2)
            if x == 1:
                y = 2
            else:
                y = 3
            return y

        pdf = two_or_three()

        for x in range(2, 4):
            assert pdf[x] == approx(0.5, rel=1e-5)

        assert len(pdf) == 2

    def test_sum_ten_coins(self):

        @d9.dist
        def sum_ten_coins():
            x = 0
            for i in range(10):
                x = x + d(2)
            return x

        pdf = sum_ten_coins()

        for x in range(10, 21):
            assert pdf[x] == approx(math.comb(10, x - 10) * 2**-10, rel=1e-5)

        assert len(pdf) == 11

    def test_fight(self):

        @d9.dist
        def fight():
            hp1 = 4
            hp2 = 4
            for i in range(8):
                if (hp1 > 0 and hp2 > 0) and d(20) > 10:
                    hp2 = max(hp2 - d(4), 0)
                if (hp1 > 0 and hp2 > 0) and d(20) > 10:
                    hp1 = max(hp1 - d(4), 0)
            return hp1, hp2

        pdf = fight()

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
                    assert pdf[(j, i)] == approx(matrix[i][j], rel=1e-5)

    def test9(self):

        @d9.dist
        def f9():
            dice = []
            dice = [d(6), *dice]
            for i in range(12):
                dice = [d(6), *dice]
                dice = sort(dice, -1)
            median = dice[6]
            return median

        def binomial_cdf(n: int, p: float, k: int) -> float:
            return sum(
                math.comb(n, i) * p**i * (1 - p)**(n - i)
                for i in range(0, k + 1))

        def median_cdf(N: int, m: int) -> float:
            k = (N - 1) // 2
            cdf_m = 1 - binomial_cdf(N, m / 6, k)
            cdf_m_1 = 1 - binomial_cdf(N, (m - 1) / 6, k)
            return cdf_m - cdf_m_1

        median_dist = f9()
        for i in range(1, 7):
            assert median_dist[i] == approx(median_cdf(13, i))

    def test_roll_three(self):

        @d9.dist
        def roll_three():
            dice = list(3 @ d(6))
            total = 0
            for i in range(3):
                total = total + dice[i]
            return total

        total = roll_three()

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
            assert total[i] == approx(expected[i])

    def test13(self):

        @d9.dist
        def conditional_sum_dice():
            init = d(6)
            total = init
            for i in range(49):
                total = total + d(6)
            return init, (total == 200)

        result = conditional_sum_dice()

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
            assert p[i] == approx(expected[i])

    def test14(self):

        @d9.dist
        def missing_roll():
            tables = [0, 0, 0, 0, 0, 0]
            for i in range(4):
                t = d(6)
                tables = tables + one_hot(t - 1, 6)
            return argmin(tables, -1)

        result = missing_roll()
        expected = [
            0.48225308641975423,
            0.2847222222222224,
            0.14969135802469133,
            0.06481481481481481,
            0.018518518518518517,
        ]
        for i in range(5):
            assert result[i] == approx(expected[i])

    def test15(self):

        def g(sides):
            for j in range(sides):
                yield sum(*(j for i in range(1)))

        @d9.dist
        def f(sides):
            successes = 0
            for num in g(sides):
                if 1 >= 2:
                    successes += num
            return successes

        pmf = f(6)
        assert pmf[0] == approx(1)

    def test16(self):

        def g(num_dice, sides):
            num_sixes = num_dice

            for j in range(1, sides + 1):
                count = 0
                loop_count = num_sixes
                for i in range(num_dice):
                    if i < loop_count:
                        if j <= sides:
                            r = d[j:sides]
                            if r == j:
                                count += 1
                                num_sixes -= 1
                yield count

        @d9.dist
        def f(num_dice, sides):
            return list(g(num_dice, sides))

        @d9.dist
        def f2(num_dice, sides):
            return lazy_bincount(num_dice @ d(sides), sides + 1)[1:]

        pmf = f(6, 4)
        pmf2 = f2(6, 4)
        for (k, v), (k2, v2) in zip(sorted(pmf.items()), sorted(pmf2.items())):
            assert k == k2
            assert v == approx(v2)

    def test_bingo(self):

        @d9.dist(semiring=d9.BigFraction(64))
        def bingo(cards, num, rounds):

            matches = reduce_sum(cards > 0)
            result = -1
            for i in lazy_perm(num, rounds):
                num = i + 1
                for j in range(len(cards)):
                    matches[j] -= num in cards[j]
                    if matches[j] == 0 and result < 0:
                        result = j

            return result

        cards = [[5, 6, 7, 8], [1, 2, 3, 0], [1, 2, 4, 0], [1, 3, 4, 0],
                 [2, 3, 4, 0]]

        pmf = bingo(cards, 8, 6)
        assert pmf[0] == Fraction(3, 14)
        for i in range(1, 5):
            assert pmf[i] == Fraction(11, 56)

        cards = [[1, 2, 4, 5], [1, 2, 3, 7], [1, 3, 5, 6], [1, 4, 6, 7],
                 [2, 3, 4, 6], [2, 5, 6, 7], [3, 4, 5, 7]]

        pmf = bingo(cards, 7, 5)
        for i in range(6):
            assert pmf[i] == Fraction(1, 7)

    def test_hit(self):
        s = d9.SemiringProduct(d9.BigFraction(64), d9.Real64())

        @d9.dist(semiring=s)
        def f():
            total = 0
            flag = False
            for i in range(20):
                total += d(6)
                flag = flag or total == 10
            return flag

        pmf = f()
        assert pmf[True] == (Fraction(17492167,
                                      60466176), approx(0.2892884610397718))
