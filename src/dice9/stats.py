import operator
import itertools

def mean(pmf):
    return sum(itertools.starmap(operator.mul, pmf.items()))

