# -*- coding: utf-8 -*-

def solve(field, n, m):
    board = [[0 for a in range(m)] for i in range(n)]
    for a in range(0, n):
        hp, lp = 0, 0
        for b in range(0, m):
            hc = field[a][b]
            lc = field[a][m - 1 - b]
            if hp == hc:
                board[a][b] = 1
            elif hp < hc:
                board[a][b] = 1
                hp = hc
            if lp == lc:
                board[a][m - 1 - b] = 1
            elif lp < lc:
                board[a][m - 1 - b] = 1
                lp = lc
    for b in range(0, m):
        kp, jp = 0, 0
        for a in range(0, n):
            jc = field[a][b]
            kc = field[n - 1 - a][b]
            if jp == jc:
                board[a][b] = 1
            elif jp < jc:
                board[a][b] = 1
                jp = jc
            if kp == kc:
                board[n - 1 - a][b] = 1
            elif kp < kc:
                board[n - 1 - a][b] = 1
                kp = kc
    if sum([sum(board[a]) for a in range(0, n)]) == n * m:
        return 'YES'
    return 'NO'

if __name__ == '__main__':
    t = int(input())
    for i in range(1, t + 1):
        n, m = [int(e) for e in input().split(' ')]
        field = []
        for e in range(0, n):
            field.append([int(e) for e in input().split(' ')])
        print('Case #%d:' % i, solve(field, n, m))
