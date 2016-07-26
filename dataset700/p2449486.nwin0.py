def solve(pre):
    read_ints = lambda: list(map(int, input().split()))
    h, w = read_ints()
    to = [read_ints() for _ in range(h)]
    lawn = [[100] * w for _ in range(h)]
    for i, r in enumerate(to):
        cut = max(r)
        for j in range(w):
            lawn[i][j] = min(lawn[i][j], cut)
    for i, c in enumerate(zip(*to)):
        cut = max(c)
        for j in range(h):
            lawn[j][i] = min(lawn[j][i], cut)
    if lawn == to:
        print(pre, "YES")
    else:
        print(pre, "NO")

n = int(input())
for i in range(n):
    solve("Case #%d:" % (i + 1))
