# https://rpg.stackexchange.com/questions/212909/help-with-counting-successes-with-multiple-target-numbers

import time

import dice9 as d9

@d9.dist
def f():
    pchi = 20
    nchi = 20

    for rounds in range(41):
        p = lazy_sum(5 @ (d(6) >= 3)) >= lazy_sum(5 @ (d(6) >= 3))
        if nchi >= 0 and pchi >= 0:
            if p:
                pchi = max(move(pchi) - 1, -1)

        if nchi >= 0 and pchi >= 0:
            if not p:
                nchi = max(move(nchi) - 1, -1)

    return (nchi > -1, pchi > -1)




start = time.perf_counter()
result = f()
end = time.perf_counter()
print(result)

print(f"Time taken: {end - start:.6f} seconds")
