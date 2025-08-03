import dice9 as d9
d9.use('np', profile=False)

# Brachiosaurus vs Tyrannosaurus, rounds=14, p=0.362511

def f():
    hp1 = sum(36 @ d(8))
    hp2 = sum(18 @ d(8))

    for i in range(14):
        print("round", i)
        if hp1 > 0 and d(20) > 1:
            hp2 = sum((-x for x in 5 @ d(4)), hp2)
            hp2 = max(0, hp2)

        if hp2 > 0: 
            if d(20) > 1:
                hp1 -= d(6)
            if d(20) > 1:
                hp1 -= d(6)
            if d(20) > 1:
                hp1 = sum((-x for x in 5 @ d(8)), hp1)
            hp1 = max(hp1, 0)

    win1 = hp2 == 0
    win2 = hp1 == 0

    return win1, win2


import time

start = time.perf_counter()
result = d9.run(f)
end = time.perf_counter()
print(result)
print(f"Time taken: {end - start:.6f} seconds")
