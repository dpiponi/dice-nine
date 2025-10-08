import pytest
from pytest import approx
import dice9 as d9

def test_sum_two_coins():
    @d9.dist
    def sum_two_coins():
        x = d(2)
        y = d(2)
        z = x + y
        return z

    pmf = sum_two_coins()

    assert pmf[2] == approx(0.25)
    assert pmf[3] == approx(0.5)
    assert pmf[4] == approx(0.25)

    assert len(pmf) == 3

def test_flux():
    @d9.dist
    def flux():
        x = d(6)
        y = d(6)
        z = x - y
        return z

    pmf = flux()

    assert pmf[0] == approx(1 / 6)

    assert len(pmf) == 11

def test_product():
    @d9.dist
    def times_bits():
        x = 2 * d(2) - 3
        y = 2 * d(2) - 3
        z = x * y
        return z

    pmf = times_bits()

    assert pmf[-1] == approx(1 / 2)
    assert pmf[1] == approx(1 / 2)

    assert len(pmf) == 2

def test_compare():
    @d9.dist
    def comparisons():
        x = d(3)
        y = d(3)
        return x == y, x > y, x < y, x >= y, x <= y

    pmf = comparisons()

    assert pmf[(False, True, False, True, False)] == approx(1 / 3)
    assert pmf[(True, False, False, True, True)] == approx(1 / 3)
    assert pmf[(False, False, True, False, True)] == approx(1 / 3)

def test_broadcast_add():
    @d9.dist
    def add():
        x = [0, 1]
        y = [[10], [11]]
        return x + y

    pmf = add()
    assert pmf[((10, 11), (11, 12))] == approx(1.0)

def test_negation():
    @d9.dist
    def f():
        return -d(2)
        
    pmf = f()    
    assert pmf[-1] == approx(1 / 2)

def test_logical_op1():
    @d9.dist
    def f():
        return d(2) == 1 and d(2) == 1
        
    pmf = f()    
    assert pmf[1] == approx(1 / 4)

    @d9.dist
    def g():
        return d(2) == 1 or d(2) == 1
        
    pmf = g()    
    assert pmf[1] == approx(3 / 4)
    
    @d9.dist
    def h():
        return not (d(6) == 1)
        
    pmf = h()    
    assert pmf[1] == approx(5 / 6)

def test_chain1():
    @d9.dist
    def f():
        x = d(6)
        return 1 < x < 6

    pmf = f()
    assert pmf[True] == approx(2 / 3)

def test_chain2():
    @d9.dist
    def f():
        x = d(6)
        y = d(6)
        return 1 < x < y < 6

    pmf = f()
    assert pmf[True] == approx(1 / 6)

def test_chain3():
    @d9.dist
    def f():
        x = d(6)
        y = d(6)
        return 1 < x > y < 6

    pmf = f()
    assert pmf[True] == approx(5 / 12)
