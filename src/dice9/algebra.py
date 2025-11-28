from abc import ABC, abstractmethod
from fractions import Fraction
from math import isqrt
import dice9.backends.numpy_impl as sx
import math
from math import gcd
import numpy as np
from typing import Sequence

prime_table = [
    32749, 32719, 32717, 32713, 32707, 32693, 32687, 32653, 32647, 32633, 32621,
    32611, 32609, 32603, 32587, 32579, 32573, 32569, 32563, 32561, 32537, 32533,
    32531, 32507, 32503, 32497, 32491, 32479, 32467, 32443, 32441, 32429, 32423,
    32413, 32411, 32401, 32381, 32377, 32371, 32369, 32363, 32359, 32353, 32341,
    32327, 32323, 32321, 32309, 32303, 32299, 32297, 32261, 32257, 32251, 32237,
    32233, 32213, 32203, 32191, 32189, 32183, 32173, 32159, 32143, 32141, 32119,
    32117, 32099, 32089, 32083, 32077, 32069, 32063, 32059, 32057, 32051, 32029,
    32027, 32009, 32003, 31991, 31981, 31973, 31963, 31957, 31907, 31891, 31883,
    31873, 31859, 31849, 31847, 31817, 31799, 31793, 31771, 31769, 31751, 31741,
    31729, 31727, 31723, 31721, 31699, 31687, 31667, 31663, 31657, 31649, 31643,
    31627, 31607, 31601, 31583, 31573, 31567, 31547, 31543, 31541, 31531, 31517,
    31513, 31511, 31489, 31481, 31477, 31469, 31397, 31393, 31391, 31387, 31379,
    31357, 31337, 31333, 31327, 31321, 31319, 31307, 31277, 31271, 31267, 31259,
    31253, 31249, 31247, 31237, 31231, 31223, 31219, 31193, 31189, 31183, 31181,
    31177, 31159, 31153, 31151, 31147, 31139, 31123, 31121, 31091, 31081, 31079,
    31069, 31063, 31051, 31039, 31033, 31019, 31013, 30983, 30977, 30971, 30949,
    30941, 30937, 30931, 30911, 30893, 30881, 30871, 30869, 30859, 30853, 30851,
    30841, 30839, 30829, 30817, 30809, 30803, 30781, 30773, 30763, 30757, 30727,
    30713, 30707, 30703, 30697, 30689, 30677, 30671, 30661, 30649, 30643, 30637,
    30631, 30593, 30577, 30559, 30557, 30553, 30539, 30529, 30517, 30509, 30497,
    30493, 30491, 30469, 30467, 30449, 30431, 30427, 30403, 30391, 30389, 30367,
    30347, 30341, 30323, 30319, 30313, 30307, 30293, 30271, 30269, 30259, 30253,
    30241, 30223, 30211, 30203, 30197, 30187, 30181, 30169, 30161, 30139, 30137,
    30133, 30119, 30113
]


class Semiring(ABC):
    """Operations on the probability axis."""

    dtype : type
    
    @abstractmethod
    def ones(self, shape: Sequence[int]):
        ...

    @abstractmethod
    def len(self, a):
        ...

    @abstractmethod
    def get(self, a, i):
        ...

    @abstractmethod
    def zeros(self, shape : Sequence[int]):
        ...

    @abstractmethod
    def add(self, a, b):
        ...

    @abstractmethod
    def mul(self, a, b):
        ...

    @abstractmethod
    def add_reduce(self, a, keepdims=False):
        ...

    @abstractmethod
    def segment_sum(self, values, segment_ids, num_segments):
        ...

    @abstractmethod
    def concat(self, bs, axis=0):
        ...

    @abstractmethod
    def kronecker(self, a, b):
        ...

    def as_scalar(self, x):
        return x

    def promote(self, x):
        return x

    def argsort(self, a):
        return np.arange(self.len(a))


# We're typically working with rings like ℤ/pℤ × ℤ/qℤ × ... × ℤ/rℤ
# which allow you to divide by any integer not divisible by any of
# p, q, ..., r. That's why it's "partial".
class PartialField(Semiring):

    @abstractmethod
    def reciprocal(self, x):
        ...

    def divide(self, a, b):
        return self.mul(a, self.reciprocal(b))

    @abstractmethod
    def const_ratio(self, num: int, den: int, shape=()):
        ...


class BaseField(PartialField):
    dtype : type

    def ones(self, shape):
        return sx.ones(shape, dtype=self.dtype)

    def zeros(self, shape):
        return sx.zeros(shape, dtype=self.dtype)

    def len(self, a):
        return len(a)

    def get(self, a, i):
        return a[i]

    def kronecker(self, a, b):
        m = self.len(a)
        n = self.len(b)
        return sx.reshape(a[:, None] * b[None, :], (m * n,))

    def concat(self, bs, axis=0):
        return np.concatenate(bs, axis)

    def promote(self, x):
        return x

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def reciprocal(self, x):
        return 1.0 / x

    def const_ratio(self, num, den, shape=()):
        n = sx.fill(shape, float(num))
        d = sx.fill(shape, float(den))
        return sx.cast(n, self.dtype) / sx.cast(d, self.dtype)

    def add_reduce(self, a, keepdims=False):
        return sx.reduce_sum(a, keepdims=keepdims)

    def segment_sum(self, values, segment_ids, num_segments):
        return sx.unsorted_segment_sum(values,
                                       segment_ids,
                                       num_segments=num_segments)
    def argsort(self, a):
        return np.argsort(a, axis=-1)

class Real64(BaseField):
    dtype = sx.float64

class Complex128(BaseField):
    dtype = sx.complex128


class LogReal64(PartialField):
    dtype = sx.float64

    def len(self, a):
        return len(a)

    def get(self, a, i):
        return a[i]

    def ones(self, shape):
        return sx.zeros(shape, dtype=self.dtype)

    def zeros(self, shape):
        return sx.fill(shape, -math.inf, dtype=self.dtype)

    def kronecker(self, a, b):
        m = self.len(a)
        n = self.len(b)
        return sx.reshape(a[:, None] + b[None, :], (m * n,))

    def concat(self, bs, axis=0):
        return np.concatenate(bs, axis)

    def add(self, a, b):
        return sx.logaddexp(a, b)

    def mul(self, a, b):
        return sx.cast(a, self.dtype) + sx.cast(b, self.dtype)

    def reciprocal(self, x):
        return -x

    def const_ratio(self, num, den, shape=()):
        val = math.log(num) - math.log(den)
        return sx.fill(shape, val, dtype=self.dtype)

    def add_reduce(self, a, keepdims=False):
        return sx.reduce_logsumexp(a, keepdims=keepdims)

    def segment_sum(self, values, segment_ids, num_segments):
        return sx.unsorted_segment_logsumexp(values, segment_ids, num_segments)

    # Tricky. I'm making promote take the log but
    # as_scalar won't. So we mustn't think of these as
    # inverses.
    def promote(self, x):
        return np.log(x)


class Int64(Semiring):
    dtype = sx.int64

    def ones(self, shape):
        return sx.ones(shape, dtype=self.dtype)

    def zeros(self, shape):
        return sx.zeros(shape, dtype=self.dtype)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def const_ratio(self, num, den, shape=()):
        del num, den
        return self.ones(shape) # Maybe should be num

    def add_reduce(self, a, keepdims=False):
        return sx.reduce_sum(a, keepdims=keepdims)

    def segment_sum(self, values, segment_ids, num_segments):
        return sx.unsorted_segment_sum(values,
                                       segment_ids,
                                       num_segments=num_segments)


class CRTBase:

    def __init__(self, bits=240):
        num_primes = (bits + 14) // 15
        primes = prime_table[:num_primes]
        self.primes = sx.constant(primes, dtype=sx.int64)
        self.num_primes = len(primes)

    def zeros(self, shape):
        return sx.zeros((*shape, self.num_primes), dtype=sx.int64)

    def ones(self, shape):
        return sx.ones((*shape, self.num_primes), dtype=sx.int64) % self.primes

    def promote(self, x):
        arr = np.asarray(x)
        arr64 = arr.astype(np.int64, copy=False)
        out = (arr64[..., None] % self.primes + self.primes) % self.primes
        return out
        # return out.copy() if copy else out

    def len(self, a):
        return len(a)

    def concat(self, bs, axis=0):
        return np.concatenate(bs, axis)

    def get(self, a, i):
        return a[i]

    def add(self, a, b):
        return (a + b) % self.primes

    def mul(self, a, b):
        return (a * b) % self.primes

    def add_reduce(self, a, keepdims=False):
        return np.sum(a, axis=0, keepdims=keepdims) % self.primes

    def segment_sum(self, values, segment_ids, num_segments):
        flat_vals = values.reshape(-1, self.num_primes)
        flat_seg = np.asarray(segment_ids).reshape(-1)
        out = np.zeros((num_segments, self.num_primes), dtype=np.int64)
        np.add.at(out, flat_seg, flat_vals)
        return out % self.primes

    def kronecker(self, a, b):
        m = self.len(a)
        n = self.len(b)
        return sx.reshape((a[:, None] * b[None, :]) % self.primes,
                          ((m * n), self.num_primes))

    def apply_crt(self, remainders):
        m, t = 1, 0
        for i in range(self.num_primes):
            prime = int(self.primes[i])
            rhs = (int(remainders[i]) - t % prime) % prime
            inv = pow(m % prime, prime - 2, prime)
            k = (rhs * inv) % prime
            t += m * k
            m *= prime
        return m, t


class BigInteger(CRTBase, Semiring):
    has_division = False
    dtype = sx.int64

    def const_ratio(self, num: int, den: int, shape=()):
        del den
        v = (int(num) % self.primes + self.primes) % self.primes
        return np.broadcast_to(v, (*shape, self.num_primes)).copy()

    def as_scalar(self, x, bound=None):
        m, t = self.apply_crt(x)

        if t > m // 2:
            t -= m

        if bound is not None and abs(t) > bound:
            raise ValueError("Unable to reconstruct integer from residues.")

        return t


class BigFraction(CRTBase, PartialField):
    has_division = True

    def __init__(self, bits=240):
        super().__init__(bits)
        # self._max_p = int(self.primes.max())
        self._inv_tables = sx.inv_tables(self.primes)

    def const_ratio(self, num: int, den: int, shape=()):
        num = int(num)
        den = int(den)
        if den == 0:
            raise ZeroDivisionError("const_ratio with den=0")
        num_mod = (num % self.primes + self.primes) % self.primes
        den_mod = (den % self.primes + self.primes) % self.primes
        inv_den = self._inv_tables[sx.range(self.num_primes), den_mod]
        val = (num_mod * inv_den) % self.primes
        return sx.broadcast_to(val, (*shape, self.num_primes)).copy()

    def reciprocal(self, x):
        x_mod = (x % self.primes).astype(np.int64, copy=False)
        if np.any(x_mod == 0):
            raise ZeroDivisionError(
                "reciprocal not defined for elements divisible by some p_i")
        num_primes = self.num_primes
        inv = self._inv_tables
        xk = np.moveaxis(x_mod, -1, 0)
        flat = xk.reshape(num_primes, -1)
        gathered = inv[sx.range(num_primes)[:, None], flat]
        outk = gathered.reshape((num_primes,) + x_mod.shape[:-1])
        out = np.moveaxis(outk, 0, -1)
        return out

    def divide(self, a, b):
        return (a * self.reciprocal(b)) % self.primes

    def as_scalar(self, x, num_max=None, den_max=None):
        m, t = self.apply_crt(x)

        if num_max is None or den_max is None:
            b = isqrt((m - 1) // 2)
            num_max = den_max = b

        # Wang algorithm, see
        # https://en.wikipedia.org/wiki/Rational_reconstruction
        t %= m
        s = np.array([[m, t], [0, 1]], dtype='object')
        p = np.array([[0, 1], [1, 0]], dtype='object')

        while s[0, 1]:
            p[1, 1] = -s[0, 0] // s[0, 1]
            s @= p
            num, den = -s[:, 0]

            if den == 0:
                continue
            if den < 0:
                num, den = -num, -den
            if (abs(num) <= num_max and 1 <= den <= den_max and
                (t * den - num) % m == 0 and gcd(abs(num), den) == 1):
                break
        else:
            raise ValueError("Unable to reconstruct rational from residues.")

        return Fraction(num, den)


def constant_one(semiring, shape=(1,)):
    return semiring.ones(shape)


class SemiringProduct(PartialField):

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def len(self, a):
        l1 = self.s1.len(a[0])
        l2 = self.s2.len(a[1])
        assert l1 == l2
        return l1

    #def reshape(self, a, shape):

    def concat(self, bs, axis=0):
        as1, as2 = zip(*bs)
        # print(bs, as1, as2)
        return (self.s1.concat(as1, axis), self.s2.concat(as2, axis))

    def kronecker(self, a, b):
        p1 = self.s1.kronecker(a[0], b[0])
        p2 = self.s2.kronecker(a[1], b[1])
        return (p1, p2)

    def promote(self, x):
        return (self.s1.promote(x), self.s2.promote(x))

    def zeros(self, shape):
        return (self.s1.zeros(shape), self.s2.zeros(shape))

    def ones(self, shape):
        return (self.s1.ones(shape), self.s2.ones(shape))

    def add(self, a, b):
        return (self.s1.add(a[0], b[0]), self.s2.add(a[1], b[1]))

    def mul(self, a, b):
        return (self.s1.mul(a[0], b[0]), self.s2.mul(a[1], b[1]))

    def divide(self, a, b):
        return (self.s1.divide(a[0], b[0]), self.s2.divide(a[1], b[1]))

    def reciprocal(self, x):
        return (self.s1.reciprocal(x[0]), self.s2.reciprocal(x[1]))

    def add_reduce(self, a, keepdims=False):
        return (self.s1.add_reduce(a[0], keepdims),
                self.s2.add_reduce(a[1], keepdims))

    def segment_sum(self, values, segment_ids, num_segments):
        return (self.s1.segment_sum(values[0], segment_ids, num_segments),
                self.s2.segment_sum(values[1], segment_ids, num_segments))

    def const_ratio(self, num: int, den: int, shape=()):
        return (self.s1.const_ratio(num, den, shape),
                self.s2.const_ratio(num, den, shape))

    def as_scalar(self, x):
        return (self.s1.as_scalar(x[0]), self.s2.as_scalar(x[1]))

    def get(self, a, i):
        return (self.s1.get(a[0], i), self.s2.get(a[1], i))

# class Dual(PartialField):
# 
#     def __init__(self, s):
#         self.s = s
# 
#     def len(self, a):
#         l1 = self.s.len(a[0])
#         l2 = self.s.len(a[1])
#         assert l1 == l2
#         return l1
# 
#     def concat(self, bs, axis=0):
#         as1, as2 = zip(*bs)
#         # print(bs, as1, as2)
#         return (self.s.concat(as1, axis), self.s.concat(as2, axis))
# 
#     def kronecker(self, a, b):
#         x = self.s.kronecker(a[0], b[0])
#         dx = self.s.kronecker(a[1], b[0]) + self.s2.kronecker(a[0], b[1])
#         return (x, dx)
# 
#     def promote(self, a):
#         return (self.s.promote(a), self.s.promote(0))
# 
#     def zeros(self, shape):
#         return (self.s.zeros(shape), self.s.zeros(shape))
# 
#     def ones(self, shape):
#         return (self.s.ones(shape), self.s.zeros(shape))
# 
#     def add(self, a, b):
#         return (self.s.add(a[0], b[0]), self.s.add(a[1], b[1]))
# 
#     def mul(self, a, b):
#         return (self.s.mul(a[0], b[0]), self.s.mul(a[1], b[1]))
# 
#     def divide(self, a, b):
#         return (self.s.divide(a[0], b[0]), self.s.divide(self.s.sub(self.s.mul(a[1], b[0]), self.s.mul(a[0], b[1])), s.mul(b[0], b[0])))
# 
#     def reciprocal(self, x):
#         return (self.s1.reciprocal(x[0]), self.s2.reciprocal(x[1]))
# 
#     def add_reduce(self, a, keepdims=False):
#         return (self.s1.add_reduce(a[0], keepdims),
#                 self.s2.add_reduce(a[1], keepdims))
# 
#     def segment_sum(self, values, segment_ids, num_segments):
#         return (self.s1.segment_sum(values[0], segment_ids, num_segments),
#                 self.s2.segment_sum(values[1], segment_ids, num_segments))
# 
#     def const_ratio(self, num: int, den: int, shape=()):
#         return (self.s1.const_ratio(num, den, shape),
#                 self.s2.const_ratio(num, den, shape))
# 
#     def as_scalar(self, a):
#         return (self.s1.as_scalar(a[0]), self.s2.as_scalar(a[1]))
# 
#     def get(self, a, i):
#         return (self.s1.get(a[0], i), self.s2.get(a[1], i))
