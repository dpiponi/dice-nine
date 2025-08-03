import dice9 as d9

d9.use("np", profile=False)

from dice9.problib import *


def f():
    return d(6)

print("*********************************************")
result = d9.run(f, debug=False, squeeze=True)
print(result)
