import dice9 as ℙ
ℙ.use('torch', profile=False)

# Brachiosaurus vs Tyrannosaurus, rounds=14, p=0.362511

def g4(m):
    for i in range(m):
        yield d(4)

def g8(m):
    for i in range(m):
        yield d(8)

def f8():
    #hp1 = 0
    #for i in range(36):
    #    hp1 = hp1 + d(8)
    hp1 = sum(g8(36))

    hp2 = sum(g8(18))

    for i in range(14):
        #__print__("round")
        if hp1 > 0 and d(20) > 1:
            hp2 = -sum(g4(5), -hp2)
            #hp2 = hp2 - d(4)
            #hp2 = hp2 - d(4)
            #hp2 = hp2 - d(4)
            #hp2 = hp2 - d(4)
            #hp2 = hp2 - d(4)
            hp2 = max(hp2, 0)

        if hp2 > 0: 
            if d(20) > 1:
                hp1 = hp1 - d(6)
            if d(20) > 1:
                hp1 = hp1 - d(6)
            if d(20) > 1:
                hp1 = -sum(g8(5), -hp1)
                #hp1 = sum((-x for x in g8(5)), hp1)
                #hp1 = hp1 - d(8)
                #hp1 = hp1 - d(8)
                #hp1 = hp1 - d(8)
                #hp1 = hp1 - d(8)
                #hp1 = hp1 - d(8)
            hp1 = max(hp1, 0)

    #del i
    win1 = hp2 == 0
    win2 = hp1 == 0

    return win1, win2


result = ℙ.run(f8, debug=True)
print(result)
