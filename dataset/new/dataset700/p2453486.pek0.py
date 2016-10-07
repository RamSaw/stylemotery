import sys

stdin = sys.stdin
ixs = list(range(4))
for i in range(int(stdin.readline())):
    print("Case #%i: " % (i+1), end=' ')
    rows = [stdin.readline()[:4] for i in ixs]

    cols = [[row[i] for row in rows] for i in ixs]
    diags = [
        [rows[i][i] for i in ixs],
        [rows[i][3-i] for i in ixs]
    ]
    notdone = False
    for row in rows + cols + diags:
        x = None
        for t in row:
            if t == '.':
                notdone = True
                break
            elif t == 'T':
                continue
            elif x is not None and x != t:
                break
            else:
                x = t
        else:
            print(x, "won")
            break
    else:
        if notdone: print("Game has not completed")
        else:       print("Draw")

    stdin.readline()
