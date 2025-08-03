# https://rpg.stackexchange.com/questions/68860/anydice-functions-and-subsequent-dice-rolls
import dice9 as bbP
bbP.use('np')

def hitme():
    rolls = dseq(5, 6)
    hits = reduce_sum(reduce_sum(constant([5, 6])[:, None] == rolls, -1), -1)
    return hits

pmf = bbP.run(hitme)
print(pmf)
