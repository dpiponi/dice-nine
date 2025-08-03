import math
import dice9 as pl

def binomial_cdf(n: int, p: float, k: int) -> float:
    """ d9(X ≤ k) for X ~ Binomial(n,p) """
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k+1))

def median_pmf_cdf(N: int, m: int) -> float:
    if N % 2 == 0:
        raise ValueError("N must be odd")
    if not (1 <= m <= 6):
        raise ValueError("m must be between 1 and 6 inclusive")
    k = (N - 1) // 2
    # d9(median ≤ m) = 1 - d9(≤ k successes with p=m/6)
    cdf_m   = 1 - binomial_cdf(N, m / 6,   k)
    cdf_m_1 = 1 - binomial_cdf(N, (m - 1 ) / 6, k)
    return cdf_m - cdf_m_1

def test():
    dice = []
    dice = concat([reshape(d(6), [1]), dice], -1)
    for i in range(12):
        dice = concat([reshape(d(6), [1]), dice], -1)
        dice = sort(dice, -1)
    median = dice[6]
    del dice

result = pl.run(test)
median_dist = result.distribution(["median"])

for i in range(1, 7):
    print(median_pmf_cdf(13, i))
print(median_dist)
