import pytest
import math
import dice9 as d9

class TestProbLangCmds:
    def test_return(self):
        @d9.dist
        def two_coins():
            x = d(2)
            y = d(2)
            return x, y

        pmf = two_coins()

        for x in range(1, 3):
            for y in range(1, 3):
                assert pmf[(x, y)] == pytest.approx(0.25, rel=1e-5)

        assert len(pmf) == 4

    def test_return3(self):
        def g():
            return d(6), d(6)

        @d9.dist
        def f():
            x, y = g()
            return x + y

        pmf = f()
        assert pmf[2] == pytest.approx(1 / 36)
        assert pmf[7] == pytest.approx(1 / 6)
        assert pmf[12] == pytest.approx(1 / 36)

    def test_for1(self):
        @d9.dist
        def sum_ten_coins():
            x = 0
            for i in range(10):
                x = x + d(2)
            return x

        pmf = sum_ten_coins()

        for x in range(10, 21):
            assert pmf[x] == pytest.approx(math.comb(10, x - 10) * 2**-10, rel=1e-5)

        assert len(pmf) == 11

    def test_for2a(self):
        @d9.dist
        def sum_ten_coins():
            long_name = 0
            for i in range(10):
                long_name = long_name + d(2)
            return long_name

        pmf = sum_ten_coins()

        for x in range(10, 21):
            assert pmf[x] == pytest.approx(math.comb(10, x - 10) * 2**-10, rel=1e-5)

        assert len(pmf) == 11

    def test_for2(self):
        @d9.dist
        def sum_six_coins():
            x = 0
            for i in range(2):
                for j in range(3):
                    x = x + d(2)
            return x

        pmf = sum_six_coins()

        for x in range(6, 13):
            assert pmf[x] == pytest.approx(math.comb(6, x - 6) * 2**-6, rel=1e-5)

        assert len(pmf) == 7

    def test_if1(self):
        @d9.dist
        def two_or_three():
            x = d(2)
            if x == 1:
                y = 2
            else:
                y = 3
            return y

        pmf = two_or_three()

        for x in range(2, 4):
            assert pmf[x] == pytest.approx(0.5, rel=1e-5)

        assert len(pmf) == 2

    def test_if2(self):
        @d9.dist
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

        pmf = two34or5()

        for x in range(2, 6):
            assert pmf[x] == pytest.approx(0.25, rel=1e-5)

        assert len(pmf) == 4

    def test_if3(self):
        @d9.dist
        def f():
            hp2 = 1

            if hp2 > 0:
                hp2 = hp2

            return hp2

        pmf = f()

    def test_call1(self):
        def f(x):
            return x + d(2)

        @d9.dist
        def g():
            return f(0)

        pmf = g()

        assert pmf[1] == pytest.approx(0.5, rel=1e-5)
        assert pmf[2] == pytest.approx(0.5, rel=1e-5)

    def test_call2(self):
        def f(x):
            return x + d(2)

        @d9.dist
        def g():
            return f(f(0))

        pmf = g()

        assert pmf[2] == pytest.approx(0.25, rel=1e-5)
        assert pmf[3] == pytest.approx(0.5, rel=1e-5)
        assert pmf[4] == pytest.approx(0.25, rel=1e-5)

    def test_call3(self):
        def f(x, y):
            return x + y

        @d9.dist
        def g():
            return f(d(2), d(2))

        pmf = g()

        assert pmf[2] == pytest.approx(0.25, rel=1e-5)
        assert pmf[3] == pytest.approx(0.5, rel=1e-5)
        assert pmf[4] == pytest.approx(0.25, rel=1e-5)

    def test_call4(self):
        def f(x, y):
            return x + y

        @d9.dist
        def g():
            return f(0)

        with pytest.raises(d9.InterpreterError, match=r".*bind.*"):
            pmf = g()

    def test_call5(self):
        def f(x):
            return x + d(6)

        @d9.dist
        def g():
            x = 0
            for i in range(3):
                x = f(x)
            return x

        pmf = g()
        assert pmf[3] == pytest.approx(1 / 216, rel=1e-5)
        assert pmf[10] == pytest.approx(0.125, rel=1e-5)
        assert pmf[18] == pytest.approx(1 / 216, rel=1e-5)

    def test_if_for1(self):
        @d9.dist
        def f():
            x = 0
            if d(2) == 1:
                for i in range(3):
                    x += d(2)
            else:
                for i in range(2):
                    x += d(3)
            return x

        pmf = f()
        assert pmf[2] == pytest.approx(1 / 18, rel=1e-5)
        assert pmf[3] == pytest.approx(25 / 144, rel=1e-5)
        assert pmf[4] == pytest.approx(51 / 144, rel=1e-5)
        assert pmf[5] == pytest.approx(43 / 144, rel=1e-5)
        assert pmf[6] == pytest.approx(17 / 144, rel=1e-5)

    def test_for_if1(self):
        @d9.dist
        def f():
            x = 0
            for i in range(3):
                if d(2) == 1:
                    x += d(2)
                else:
                    x += d(3)
            return x

        pmf = f()
        assert pmf[3] == pytest.approx((5 / 12)**3, rel=1e-5)

    def test_for_if2(self):
        @d9.dist
        def f():
            x = 0
            y = 0
            for i in range(3):
                if d(2) == 1:
                    x += d(2)
                else:
                    y += d(2)
            return x + y

        pmf = f()
        assert pmf[3] == pytest.approx(1 / 8, rel=1e-5)
        assert pmf[6] == pytest.approx(1 / 8, rel=1e-5)

    def test_if3(self):
        @d9.dist
        def f():
            x = d(100)
            y = d(100)
            if x + y >= 198:
                z = 1
            else:
                z = 0
            return z

        pmf = f()
        assert pmf[1] == pytest.approx(6 / 10000, rel=1e-5)

    def test_if4(self):
        @d9.dist
        def f():
            x = 0

            if d(2) == 1:
                x = -x
                x = x - 1

            return x

        pmf = f()
        assert pmf[-1] == pytest.approx(1 / 2, rel=1e-5)
        assert pmf[0] == pytest.approx(1 / 2, rel=1e-5)

    def test_if5(self):
        @d9.dist
        def f():
            x = [1]
            if False:
                x = zeros(1)
            return x

        pmf = f()
        assert pmf[(1,)] == pytest.approx(1, rel=1e-5)

    def test_if6(self):
        @d9.dist
        def f():
            x = [1]
            if True:
                x = zeros(1)
            return x

        pmf = f()
        assert pmf[(0,)] == pytest.approx(1, rel=1e-5)

    def test_if7(self):
        @d9.dist
        def f():
            x = 4
            if d(2) == 1:
                x = d(2)
            return x

        pmf = f()
        assert pmf[1] == pytest.approx(0.25)
        assert pmf[2] == pytest.approx(0.25)
        assert pmf[4] == pytest.approx(0.5)

    def test_if8(self):
        @d9.dist
        def f():
            if d(2) == 1:
                x = 1
            return d(2)

        pmf = f()
        assert pmf[1] == pytest.approx(0.5)

    def test_if9(self):
        @d9.dist
        def f():
            a = d(2)
            b = d(2)
            c = d(2)
            d = d(2)
            e = d(2)
            count = 0
            if e == 1:
                if d == 1:
                    count += d(2)
                else:
                    count += 10 * d(2)
            else:
                if c == 1:
                    count += 100 * d(2)
                else:
                    count += 1000 * d(2)
            return count

        @d9.dist
        def g():
            p = d(2) * 10000
            return p + f()

        pmf = g()
        print(pmf)
        assert pmf[10001] == pytest.approx(1 / 16)
        assert pmf[22000] == pytest.approx(1 / 16)

    def test_for4(self):
        @d9.dist
        def f():
            xs = []
            for i in range(3):
                xs = [*xs, d(2)]
            y = 0
            for x in xs:
                y = y + x
            return y

        pmf = f()
        assert pmf[3] == pytest.approx(1 / 8, rel=1e-5)

    def test_frame1(self):
        def g(n):
            result = 0
            if n == 0:
                result = 0

            return result

        @d9.dist
        def f():
            s = g(0)
            s = g(0)
            return s

        pmf = f()
        assert pmf[0] == pytest.approx(1)

    def test_recurse1(self):
        def recurse4():
            return d(6)

        def recurse3():
            return recurse4() + recurse4()

        def recurse2():
            return recurse3() + recurse3()

        def recurse1():
            return recurse2() + recurse2()

        @d9.dist
        def f():
            return recurse1() + recurse1()

        @d9.dist
        def g():
            return sum(*(16 @ d(6)))

        pmf1 = f()
        pmf2 = g()

        for i in range(16, 16 * 6 + 1):
            assert pmf1[i] == pytest.approx(pmf2[i])

    def test_recurse2(self):
        def g(n):
            if n == 0:
                x = d(6)
            else:
                x = g(n - 1) + g(n - 1)
            return x

        @d9.dist
        def f():
            return g(4)

        @d9.dist
        def h():
            return sum(*(16 @ d(6)))

        pmf1 = f()
        pmf2 = h()

        for i in range(16, 16 * 6 + 1):
            assert pmf1[i] == pytest.approx(pmf2[i])

    def test_gen_if1(self):
        def g():
            for i in range(2):
                yield 0

        @d9.dist
        def f():
            y = 0
            for x in g():
                if x:
                    y = 0
            return y

        pmf = f()
        assert pmf[0] == 1

    def test_gen_if2(self):
        def g():
            for j in range(3):
                count = 0
                yield 0

        @d9.dist
        def f():
            x = 0
            for num in g():
                if True:
                    x = 0

            return x

        pmf = f()
        assert pmf[0] == 1

    def test_gen_del(self):
        def g():
            x = 1
            del x
            yield x

        @d9.dist
        def f():
            t = 0
            for i in g():
                t += i
            return t

        with pytest.raises(d9.InterpreterError, match=r".*not found.*"):
            pmf = f()

    def test_aug1(self):
        @d9.dist
        def f():
            x = d(2)
            x += 1
            return x

        pmf = f()
        assert pmf[2] == pytest.approx(1 / 2)
        assert pmf[3] == pytest.approx(1 / 2)

    def test_aug2(self):
        @d9.dist
        def f():
            x = d(2)
            x -= 1
            return x

        pmf = f()
        assert pmf[0] == pytest.approx(1 / 2)
        assert pmf[1] == pytest.approx(1 / 2)

    def test_error1(self):
        @d9.dist
        def f():
            del a.b

        with pytest.raises(d9.InterpreterError, match=r".*Delet.*"):
            pmf = f()

    def test_gen_for1(self):
        def g(x):
            for i in x:
                yield i

        @d9.dist
        def f():
            t = 0
            for j in g([1, 2, 3]):
                t += d(j)
            return t

        pmf = f()
        assert pmf[3] == pytest.approx(1 / 6)
        assert pmf[4] == pytest.approx(1 / 3)
        assert pmf[5] == pytest.approx(1 / 3)
        assert pmf[6] == pytest.approx(1 / 6)

    def test_assert1(self):
        @d9.dist(normalize=True)
        def f():
            d = d(6)
            assert d > 1
            return d

        pmf = f()
        for i in range(2, 7):
            assert pmf[i] == pytest.approx(1 / 5)

    def test_assert2(self):
        @d9.dist
        def f():
            x = d(2)
            assert x > 2
            return x + d(2)

        result = f()
        assert len(result) == 0

    def test_args1(self):
        @d9.dist
        def f(n):
            return d(n)

        pmf = f(8, )
        for i in range(1, 9):
            assert pmf[i] == pytest.approx(1 / 8)

    def test_args2(self):
        @d9.dist
        def f(n):
            return d(n)

        with pytest.raises(TypeError, match=r".*missing.*"):
            pmf = f()

    def test_args3(self):
        @d9.dist
        def f():
            return d(6)

        with pytest.raises(TypeError, match=r".*positional.*"):
            pmf = f(17)

    def test_listcomp1(self):
        @d9.dist
        def f():
            return reduce_sum([10 * x for x in 3 @ d(6)], -1)

        pmf = f()
        assert pmf[30] == pytest.approx(1 / 216)
        assert pmf[100] == pytest.approx(1 / 8)
        assert pmf[180] == pytest.approx(1 / 216)

    def test_stream1(self):
        @d9.dist
        def f():
            t = 0
            for i in {d(2), d(2)}:
                t += i
            return t

        pmf = f()
        assert pmf[2] == pytest.approx(1 / 4)
        assert pmf[4] == pytest.approx(1 / 4)

    def test_stream2(self):
        @d9.dist
        def f():
            return sum(*(2 @ d(3)), *(2 @ d(6)))

        pmf = f()
        assert pmf[4] == pytest.approx(1 / (3 * 3 * 6 * 6))
        assert pmf[18] == pytest.approx(1 / (3 * 3 * 6 * 6))

    def test_genexp_iter(self):
        def grouped():
            for j in range(2):
                yield sum(*(j for i in range(2)))

        @d9.dist
        def f():
            for i in grouped():
                x = 1
            return d(6)

        pmf = f()
        assert pmf[1] == pytest.approx(1 / 6)

    def test_defarg1(self):
        def g(x=1+2):
            return x

        @d9.dist
        def f():
            return g(2)

        with pytest.raises(d9.InterpreterError, match=r".*literal.*"):
            pmf = f()

    def test_defarg2(self):
        def g(x, /, y):
            return d(6) + x + y

        @d9.dist
        def f():
            return g(2, y=3)

        pmf = f()
        assert pmf[11] == pytest.approx(1 / 6)

    def test_defarg3(self):
        def g(x, /, y):
            return d(6) + x + y

        @d9.dist
        def f():
            return g(x=2, y=3)

        with pytest.raises(d9.InterpreterError, match=r".*bind.*"):
            pmf = f()

    def test_kwonly1(self):
        def g(n=6):
            return d(n)

        @d9.dist
        def f():
            return g()

        pmf = f()
        assert pmf[1] == pytest.approx(1 / 6)

    def test_kwonly2(self):
        def g(n=6):
            return d(n)

        @d9.dist
        def f():
            return g(n=3)

        pmf = f()
        assert pmf[1] == pytest.approx(1 / 3)

    def test_bad_return(self):
        @d9.dist
        def f():
            if d(2) == 1:
                return d(2)
            return 1

        with pytest.raises(d9.InterpreterError, match=r".*return.*"):
            pmf = f()

    def test_yield_tuple1(self):
        def f():
            for i in range(2):
                yield d(6), d(6)

        @d9.dist
        def g():
            total = 0
            for x, y in f():
                total += x - y
            return total

        pmf = g()
        print(pmf)
        assert pmf[-10] == pytest.approx(1 / 6**4)
        assert pmf[10] == pytest.approx(1 / 6**4)
        
    def test_args4(self):
        @d9.dist
        def f(a, b=10, c=100):
            return a * d(2) + b * d(4) + c * d(6)

        pmf = f(1)
        assert pmf[642] == pytest.approx(1 / (2 * 4 * 6))

        pmf = f(1, 100, 10)
        assert pmf[462] == pytest.approx(1 / (2 * 4 * 6))

        pmf = f(1, c=10, b=100)
        assert pmf[462] == pytest.approx(1 / (2 * 4 * 6))
