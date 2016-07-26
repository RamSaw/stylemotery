import sys


def compute(N, M, a):
    rows = [0] * N
    cols = [0] * M
    for r in range(N):
        rows[r] = 0
        for c in range(M):
            if a[r][c] > rows[r]:
                rows[r] = a[r][c]
    for c in range(M):
        cols[c] = 0
        for r in range(N):
            if a[r][c] > cols[c]:
                cols[c] = a[r][c]
    for r in range(N):
        for c in range(M):
            if a[r][c] < rows[r] and a[r][c] < cols[c]:
                return "NO"
    return "YES"


def parse():
    N, M = list(map(int, sys.stdin.readline().strip().split()))
    a = []
    for i in range(N):
        a.append(list(map(int, sys.stdin.readline().strip().split())))
    return N, M, a,


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    T = int(sys.stdin.readline().strip())
    count = 1
    part = 0
    if len(sys.argv) == 3:
        part = int(sys.argv[1])
        count = int(sys.argv[2])
    for i in range(T):
        data = parse()
        if i * count >= part * T and i * count < (part + 1) * T:
            result = compute(*data)
            print("Case #%d: %s" % (i + 1, result))
