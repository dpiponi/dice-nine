import pytest
import dice9 as d9

class TestProbLangErrors:
    def test_not_found(self):
        @d9.dist
        def g():
            return x

        with pytest.raises(d9.InterpreterError, match=r".*not found.*"):
            pmf = g()

    def test_return2(self):
        @d9.dist
        def g():
            return

        with pytest.raises(d9.InterpreterError, match=r".*must return.*"):
            pmf = g()

    def test_bounds1(self):
        @d9.dist
        def f():
            return [1, 2, 3, 4, 5, 6][d(6)]

        with pytest.raises(d9.InterpreterError, match=r".*Bounds.*"):
            pmf = f()

    def test_bounds2(self):
        @d9.dist
        def f():
            x = d(6)
            return [1, 2, 3, 4, 5, 6, 7][x, x]

        with pytest.raises(d9.InterpreterError, match=r".*subscript.*"):
            pmf = f()

    def test_bounds3(self):
        @d9.dist
        def f():
            x = d(6)
            return [1, 2, 3, 4, 5, 6, 7][1:x]

        with pytest.raises(d9.InterpreterError, match=r".*literal.*"):
            pmf = f()

    def test_bounds4(self):
        @d9.dist
        def f():
            x = d(6)
            return [1, 2, 3, 4, 5, 6, 7][x:1, 1:x]

        with pytest.raises(d9.InterpreterError, match=r".*literal.*"):
            pmf = f()

    def test_cond1(self):
        @d9.dist
        def f():
            if d(6) == 1:
                return 1
            else:
                return 2

        with pytest.raises(d9.InterpreterError, match=r".*conditional.*"):
            pmf = f()

    def test_yield1(self):
        @d9.dist
        def f():
            yield d(6)

        with pytest.raises(d9.InterpreterError, match=r".*top level.*"):
            pmf = f()

    def test_iterable1(self):
        @d9.dist
        def main():
            for i in 2:
                print(i)
            return 0

        with pytest.raises(d9.InterpreterError, match=r".*iterable.*"):
            pmf = main()
