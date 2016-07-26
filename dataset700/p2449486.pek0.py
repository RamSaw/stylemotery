import sys

stdin = sys.stdin
for c in range(int(stdin.readline())):
    n,m = list(map(int, stdin.readline().split()))
    rows = [list(map(int, stdin.readline().split())) for i in range(n)]
    cols = [[row[i] for row in rows] for i in range(m)]

    rowmaxs = [max(x) for x in rows]
    colmaxs = [max(x) for x in cols]

    verdict = "YES"
    for i in range(n):
        for k in range(m):
            if min(rowmaxs[i], colmaxs[k]) > rows[i][k]:
                verdict = "NO"
                break

        if verdict == "NO": break

    print("Case #%i: %s" % (c+1, verdict))
