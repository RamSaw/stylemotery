# -*- coding: utf-8 -*-

t = int(input())

for i in range(1, t+1):
    res = 0
    line = [int(e) for e in input().split()]
    n, s, p = line[:3]
    t = line[3:]
    for e in t:
        if max(p*3-2, 0) <= e:
            res += 1
        elif s and p*3-4 <= e and p >= 2:
            s -= 1
            res += 1
    print('Case #%d: %d' % (i, res))
