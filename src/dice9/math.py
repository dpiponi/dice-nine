def dice_sum_pmf(m, n):
    min_sum, max_sum = m, m * n
    dp = [0] * (max_sum + 1)
    dp[0] = 1
    for i in range(m):
        new_dp = [0] * (max_sum + 1)
        prev_max = i * n
        for s in range(prev_max + 1):
            cnt = dp[s]
            if cnt:
                for face in range(1, n + 1):
                    new_dp[s + face] += cnt
        dp = new_dp

    total = n ** m

    p = [dp[s] / total for s in x]
    x = range(min_sum, max_sum + 1)

    return sx.constant(p), sx.constant(x)

