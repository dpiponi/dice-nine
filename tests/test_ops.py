# test_ops.py

import unittest

import dice9 as d9


class TestProbLangOps(unittest.TestCase):
    def test_sum_two_coins(self):
        def sum_two_coins():
            x = d(2)
            y = d(2)
            z = x + y
            return z

        pmf = d9.run(sum_two_coins)

        self.assertAlmostEqual(pmf[(2,)], 0.25, places=5)
        self.assertAlmostEqual(pmf[(3,)], 0.5, places=5)
        self.assertAlmostEqual(pmf[(4,)], 0.25, places=5)

        self.assertEqual(len(pmf), 3)

    def test_flux(self):
        def flux():
            x = d(6)
            y = d(6)
            z = x - y
            return z

        pmf = d9.run(flux)

        self.assertAlmostEqual(pmf[(0,)], 1 / 6, places=5)

        self.assertEqual(len(pmf), 11)

    def test_product(self):
        def times_bits():
            x = 2 * d(2) - 3
            y = 2 * d(2) - 3
            z = x * y
            return z

        pmf = d9.run(times_bits)

        self.assertAlmostEqual(pmf[(-1,)], 1 / 2, places=5)
        self.assertAlmostEqual(pmf[(1,)], 1 / 2, places=5)

        self.assertEqual(len(pmf), 2)

    def test_compare(self):
        def comparisons():
            x = d(3)
            y = d(3)
            return x == y, x > y, x < y, x >= y, x <= y

        pmf = d9.run(comparisons)

        self.assertAlmostEqual(pmf[(False, True, False, True, False)], 1 / 3, places=5)
        self.assertAlmostEqual(pmf[(True, False, False, True, True)], 1 / 3, places=5)
        self.assertAlmostEqual(pmf[(False, False, True, False, True)], 1 / 3, places=5)

    def test_broadcast_add(self):
        def add():
            x = [0, 1]
            y = [[10], [11]]
            return x + y

        pmf = d9.run(add)
        self.assertAlmostEqual(pmf[(((10, 11), (11, 12)),)], 1.0, places=5)
        
    def test_negation(self):
        def f():
        	return -d(2)
        	
        pmf = d9.run(f)    
        self.assertAlmostEqual(pmf[(-1,)], 1 / 2)
        
    def test_logical_op1(self):
        def f():
        	return d(2) == 1 and d(2) == 1
        	
        pmf = d9.run(f)    
        self.assertAlmostEqual(pmf[(1,)], 1 / 4)


        def g():
        	return d(2) == 1 or d(2) == 1
        	
        pmf = d9.run(g)    
        self.assertAlmostEqual(pmf[(1,)], 3 / 4)
        
        def h():
        	return not (d(6) == 1)
        	
        pmf = d9.run(h)    
        self.assertAlmostEqual(pmf[(1,)], 5 / 6)
