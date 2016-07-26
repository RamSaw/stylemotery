# -*- coding: utf-8 -*-

t = int(input())

for i in range(1, t+1):
    print('Case #%d:' % i)
    r, c, m = [int(e) for e in input().split()]
    if r == 1:
        print('*' * m + '.' * (c - m - 1) + 'c')
    elif c == 1:
        print('\n'.join('*' * m + '.' * (r - m - 1) + 'c'))
    elif r * c - m == 1:
        for j in range(r-1):
            print('*' * c)
        print('*' * (c-1) + 'c')
    elif r * c - m < 4:
        print('Impossible')
    else:
        for j in range(r):
            if j == r - 1:
                print('*' * m + '.' * (c - m - 1) + 'c')
            elif j == r - 2:
                if c-2 < m:
                    m -= (c-2)
                    print('*' * (c-2) + '.' * 2)
                else:
                    print('*' * m + '.' * (c-m))
                    m = 0
            else:
                if c < m:
                    m -= c
                    print('*' * c)
                else:
                    print('*' * m + '.' * (c - m))
                    m = 0
