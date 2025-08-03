import sys
import os

# Add ./src to the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import dice9 as ℙ
ℙ.use('np', profile=False)

def f(x, y):
    __listvars__()
    return x + y

def g():
    return f(d(2), d(2))

result = ℙ.run(g)
print(result)
