# -*- coding: utf-8 -*-

def b(c, f, n):
    res = 0
    for i in range(1, n+1):
        res += (c / (2 + f * i - f))
    return res

def g(c, f, n):
    return b(c, f, n) + x / (2 + f * n)

t = int(input())

for i in range(1, t+1):
    c, f, x = [float(e) for e in input().split()]
    res = 10 ** 5
    n = 0
    while g(c, f, n) < res:
        res = g(c, f, n)
        n += 1
    print('Case #%d:' % i, res)
