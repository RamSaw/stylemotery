# -*- coding: utf-8 -*-

t = int(input())

for i in range(1, t+1):
    a = int(input())
    b = []
    for j in range(4):
        b.append([int(e) for e in input().split()])
    c = int(input())
    d = []
    for j in range(4):
        d.append([int(e) for e in input().split()])
    bb = set(b[a-1])
    dd = set(d[c-1])
    res = len(bb & dd)
    if res == 0:
        ans = 'Volunteer cheated!'
    elif res == 1:
        ans = list(bb & dd)[0]
    else:
        ans = 'Bad magician!'
    print('Case #%d:' % i, ans)
