from pytest import approx
import dice9 as d9

class TestProbLangAlg:
    def setup_method(self):
        self.z1 = d9.BigInteger()
        self.q1 = d9.BigFraction()

    def test_add(self):
        s = self.z1
        x = s.as_scalar(s.promote(1) + s.promote(2))
        assert x == 3

    def test_fact(self):
        s = self.z1
        one = s.promote(1)
        x = one
        p = x
        for i in range(40):
            p = s.mul(p, x)
            x = s.add(x, one)

        p = s.as_scalar(p)

        assert p == 815915283247897734345611269596115894272000000000

    def test_frac1(self):
        s = self.q1

        num = s.promote(1)
        den = s.promote(2)
        frac = s.divide(num, den)
        x = s.mul(frac, den)

        assert s.as_scalar(x) == 1

    def test_frac2(self):
        s = self.q1

        one = s.promote(1)

        x = one
        p = x
        for i in range(30):
            p = s.divide(p, x)
            x = s.add(x, one)

        x = one
        for i in range(30):
            p = s.mul(p, x)
            x = s.add(x, one)

        p = s.as_scalar(p)

        assert p == 1

    def test_prob1(self):
        s = self.q1

        @d9.dist(semiring=s)
        def main():
            return d(2)

        pmf = main()
        assert pmf[1] == approx(1 / 2)
        assert pmf[2] == approx(1 / 2)
