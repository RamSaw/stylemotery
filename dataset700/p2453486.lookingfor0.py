n = int(input())

d = {'X':2, 'O': 0, 'T':1, '.':-10}

def readGame():
    return [[d[c] for c in input().strip()] for i in range(4)]

def check(game):
    s = []
    for i in range(4): # horizontal
        s.append(sum(game[i]))
    for i in range(4): # vertical
        s.append(sum([li[i] for li in game]))
    d1 = d2 = 0 # diagonal
    for i in range(4):
        d1 += game[i][i]
        d2 += game[i][3-i]
    s.append(d1)
    s.append(d2)
    if max(s) >= 7:
        return "X won";
    gr0 = [n for n in s if n >= 0]
    if len(gr0) > 0 and min(gr0) <= 1:
        return "O won"
    if min(s) < 0:
        return "Game has not completed"
    return "Draw"

for i in range(n):
    game = readGame()
    if i + 1 < n:
        input()

    print("Case #%d: %s" % (i+1, check(game)))