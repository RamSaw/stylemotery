# -*- coding: utf-8 -*-

t = int(input())

for i in range(1, t+1):
    res = 0
    a, b = [int(e) for e in input().split(' ')]
    for n in range(a, b+1):
        ns = str(n)
        c = set()
        for k in range(1, len(ns)):
            m = int(ns[k:] + ns[:k])
            if a <= n < m <= b:
                c.add((n, m))
        res += len(c)
    print('Case #%d: %d' % (i, res))
