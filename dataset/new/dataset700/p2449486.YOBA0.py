from itertools import chain


for i in range(int(eval(input()))):
    n, m = tuple(map(int, input().split()))
    l = dict(chain.from_iterable([list(zip(list(zip([j] * m, list(range(m)))), list(map(int, input().split())))) for j in range(n)]))
    ans = "YES"

    while len(l):
        ly, lx = min(l, key=l.get)
        lowest = l[(ly, lx)]

        if all([l.get((ly, j), lowest) == lowest for j in range(m)]):
            for j in range(m):
                l.pop((ly, j), None)

        elif all([l.get((j, lx), lowest) == lowest for j in range(n)]):
            for j in range(n):
                l.pop((j, lx), None)

        else:
            ans = "NO"
            break

    print(("Case #{}: {}".format(i + 1, ans)))
