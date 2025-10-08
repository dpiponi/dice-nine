import pytest
from pytest import approx
import dice9 as d9

def test_subscript1():
    @d9.dist
    def f():
        x = [0, 1]
        return x[0]

    pmf = f()
    assert pmf[0] == approx(1)

def test_subscript2():
    @d9.dist
    def f():
        x = [d(6), d(4)]
        return x[0]

    pmf = f()
    for i in range(1, 7):
        assert pmf[i] == approx(1 / 6)

def test_subscript3():
    @d9.dist
    def f():
        x = [d(4), d(6)]
        return x[-1]

    pmf = f()
    for i in range(1, 7):
        assert pmf[i] == approx(1 / 6)

def test_subscript4():
    @d9.dist
    def f():
        x = [d(4), d(6), d(6)]
        return x[0:2]

    pmf = f()
    assert pmf[(1, 1)] == approx(1 / 24)

def test_subscript5():
    @d9.dist
    def f():
        x = [d(4), d(6), d(8)]
        i = d(3) - 1
        return x[i]

    pmf = f()
    assert pmf[1] == approx((1 / 4 + 1 / 6 + 1 / 8) / 3)
    assert pmf[8] == approx(1 / 24)

def test_subscript6():
    @d9.dist
    def f():
        x = [d(4), d(6)]
        x[0] = 1
        return x[1]

    pmf = f()
    assert pmf[1] == approx(1 / 6)

def test_subscript7():
    @d9.dist
    def f():
        x = [d(4), d(6)]
        x[0] = d(6)
        return x

    pmf = f()
    assert pmf[(1, 1)] == approx(1 / 36)

def test_subscript8():
    @d9.dist
    def f():
        x = [d(4), d(6)]
        x[1] = x[0]
        return x

    pmf = f()
    assert pmf[(2, 2)] == approx(1 / 4)

def test_subscript9():
    @d9.dist
    def f():
        x = [d(4), d(6)]
        x[0:1] = 0
        return d(2)

    with pytest.raises(d9.InterpreterError, match=r".*Slice.*"):
        pmf = f()

def test_subscript10():
    @d9.dist
    def f():
        x = [d(4), d(6)]
        x[0] = x[0] + 1
        return d(2)

    pmf = f()

def test_subscript11():
    @d9.dist
    def f():
        x = zeros(3)
        for i in range(3):
            x[i] = d(4 + 2 * i)
        return reduce_sum(x, -1)

    pmf = f()
    assert pmf[3] == approx(1 / (4 * 6 * 8))
    assert pmf[18] == approx(1 / (4 * 6 * 8))

def test_subscript11():
    @d9.dist
    def f():
        x = [[1, 2, 3], 
             [4, 5, d(6)]]
        i = d(2) - 1
        return x[i]

    pmf = f()
    assert pmf[(1, 2, 3)] == approx(1 / 2)
    assert pmf[(4, 5, 6)] == approx(1 / 12)

def test_subscript12():
    @d9.dist
    def f():
        x = [0, 0]
        for i in range(10):
            x[d(2) - 1] += 1
        return x

    pmf = f()
    assert pmf[(5, 5)] == approx(0.24609375)

def test_subscript13():
    @d9.dist
    def f():
        x = [1, 1]
        for i in range(10):
            x[d(2) - 1] *= 2
        return x

    pmf = f()
    assert pmf[(32, 32)] == approx(0.24609375)

def test_subscript14():
    @d9.dist
    def main():
        a = zeros([3, 3])
        i = d(3) - 1
        a[i, i] += 1
        return a

    pmf = main()
    assert pmf[((1, 0, 0), (0, 0, 0), (0, 0, 0))] == approx(1 / 3)
    assert pmf[((0, 0, 0), (0, 1, 0), (0, 0, 0))] == approx(1 / 3)
    assert pmf[((0, 0, 0), (0, 0, 0), (0, 0, 1))] == approx(1 / 3)

def test_subscript15():
    @d9.dist
    def main():
        a = ones([3, 3])
        i = d(3) - 1
        a[i, i] *= 2
        return a

    pmf = main()
    assert pmf[((2, 1, 1), (1, 1, 1), (1, 1, 1))] == approx(1 / 3)
    assert pmf[((1, 1, 1), (1, 2, 1), (1, 1, 1))] == approx(1 / 3)
    assert pmf[((1, 1, 1), (1, 1, 1), (1, 1, 2))] == approx(1 / 3)

def test_subscript15():
    @d9.dist
    def main():
        a = ones([3, 3])
        i = d(3) - 1
        a[i, i] *= 2
        return a

    pmf = main()
    assert pmf[((2, 1, 1), (1, 1, 1), (1, 1, 1))] == approx(1 / 3)
    assert pmf[((1, 1, 1), (1, 2, 1), (1, 1, 1))] == approx(1 / 3)
    assert pmf[((1, 1, 1), (1, 1, 1), (1, 1, 2))] == approx(1 / 3)

def test_subscript16():
    @d9.dist
    def main():
        a = zeros([3, 3])
        for i in range(3):
            for j in range(3):
                a[i, j] = i + j
        return a[d(3) - 1, d(3) - 1]

    pmf = main()
    assert pmf[0] == approx(1 / 9)
    assert pmf[1] == approx(2 / 9)
    assert pmf[2] == approx(3 / 9)
    assert pmf[3] == approx(2 / 9)
    assert pmf[4] == approx(1 / 9)
