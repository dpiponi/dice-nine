import math
import dice9 as pl
pl.use('tf', profile=False)

# +-----+-----+-----+-----+
# |Go to|     |7: $5|     |
# |Jail |  ?  |     |Jail |
# |     |     |   $3|     |
# +-----+-----+-----+-----+
# |10:  |           |     |
# | $4  |           |  ?  |
# | $2  |           |     |
# +-----+           +-----+
# |     |           |4:$4 |
# |  ?  |           |     |
# |     |           |  $2 |
# +-----+-----+-----+-----+
# |0    |1:$3 |     |Free |
# |     |     |  ?  |Park-|
# |Start|  $1 |     |ing  |
# +-----+-----+-----+-----+

# For 5 rounds:
# (0,) 0.06056358268794769
# (1,) 0.03110540883220396
# (2,) 0.050721523426584786
# (3,) 0.06314930309136794
# (4,) 0.0705139481100144
# (5,) 0.071807757706143
# (6,) 0.08701957345127953
# (7,) 0.09345156998393611
# (8,) 0.08896529245127055
# (9,) 0.08262907622312672
# (10,) 0.07490547173844758
# (11,) 0.07224343955949848
# (12,) 0.0551932080900072
# (13,) 0.035659327228664155
# (14,) 0.025680482558132985
# (15,) 0.019213175995287008
# (16,) 0.012814442118846737
# (17,) 0.004363416747240638

def game():
    player1 = 0
    player2 = 0
    money1 = 17
    money2 = 17

    price = [3, 4, 5, 4]
    rent = [1, 1, 2, 2]
    owners = [0, 0, 0, 0]

    for t in range(5):
        print("round", t)
        if money1 > 0 and money2 > 0:
            player1 = ((player1) + d(6)) % 12
            if player1 % 3 == 2:
                # Chance!
                money1 -= 1 if d(2) == 1 else 5

            if player1 % 3 == 1:
                prop = player1 // 3
                if owners[prop] == 0:
                    if price[prop] > 0 and money1 >= price[prop]:
                        # Buy property
                        owners += one_hot(prop, 4)
                        money1 -= price[prop]
                if owners[prop] == 2:
                    # Pay rent
                    money1 -= rent[prop]
                    money2 += rent[prop]
                # del prop

            money1 = max((money1), 0)

        if money1 > 0 and money2 > 0:
            player2 = ((player2) + d(6)) % 12
            if player2 % 3 == 2:
                # Chqnce!
                money2 -= 1 if d(2) == 1 else 5

            if player2 % 3 == 1:
                prop = player2 // 3
                if owners[prop] == 0:
                    if price[prop] > 0 and money2 >= price[prop]:
                        # Buy property
                        owners += 2 * one_hot(prop, 4)
                        money2 -= price[prop]
                if owners[prop] == 1:
                    # Pay rent
                    money2 -= rent[prop]
                    money1 += rent[prop]
                # del prop

            money2 = max((money2), 0)

        # __listvars__()

    return money1

result = pl.run(game)

for x, p in result.items():
    print(x, p)
