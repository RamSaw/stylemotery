def solve(pre):
    b = [input().strip() for _ in range(4)]
    input()
    for s in b + list(zip(*b)) + [''.join(b[i][i] for i in range(4)), ''.join(b[3-i][i] for i in range(4))]:
        for c in 'XO':
            if s.count('T') + s.count(c) == 4:
                print(pre, c, "won")
                return
    if ''.join(b).count('.'):
        print(pre, "Game has not completed")
    else:
        print(pre, "Draw")

n = int(input())
for i in range(n):
    solve("Case #%d:" % (i + 1))
