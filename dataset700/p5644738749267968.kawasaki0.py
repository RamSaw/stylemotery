# -*- coding: utf-8 -*-

import bisect

T = int(input())
for test_case in range(1, T + 1):
    N = int(input())
    W1 = sorted(map(float, input().split()))
    W2 = sorted(map(float, input().split()))

    y = 0
    c1 = c2 = 0
    while c1 < N and c2 < N:
        if W2[c2] < W1[c1]:
            y += 1
            c1 += 1
            c2 += 1
        while c1 < N and c2 < N and W1[c1] < W2[c2]:
            c1 += 1

    z = N
    c1 = c2 = 0
    while c1 < N and c2 < N:
        if W1[c1] < W2[c2]:
            z -= 1
            c1 += 1
            c2 += 1
        while c1 < N and c2 < N and W2[c2] < W1[c1]:
            c2 += 1

    print('Case #{}: {} {}'.format(test_case, y, z))
