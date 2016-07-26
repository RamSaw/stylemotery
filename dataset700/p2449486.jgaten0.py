#!/usr/bin/env python
import sys

def solve(N, M, grid):
    possible = [[False for _ in range(M)] for _ in range(N)]
    for i in range(N):
        m = max(grid[i])
        for j in range(M):
            possible[i][j] = possible[i][j] or grid[i][j] == m

    for j in range(M):
        m = max(grid[_][j] for _ in range(N))
        for i in range(N):
            possible[i][j] = possible[i][j] or grid[i][j] == m

    if all(all(row) for row in possible):
        return "YES"
    else:
        return "NO"

if __name__ == '__main__':
    with open(sys.argv[1], 'rU') as fin, open(sys.argv[2], 'w') as fout:
        T = int(fin.readline())
        for case in range(1, T+1):
            print("Case #{0}:".format(case))

            N, M = list(map(int, fin.readline().split()))
            grid = [list(map(int, fin.readline().split())) for _ in range(N)]

            soln = solve(N, M, grid)
            print(soln)
            print("Case #{0}: {1}".format(case, soln), file=fout)
