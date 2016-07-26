# -*- coding: utf-8 -*-


def rotate(x, w):
    x = str(x)
    return int(x[-w:] + x[:-w])


T = int(input())
for case in range(1, T + 1):
    A, B = list(map(int, input().split()))

    l = len(str(A))
    assert l == len(str(B))

    s = []
    for n in range(A, B + 1):
        for w in range(1, len(str(n))):
            m = int(rotate(n, w))
            if n < m <= B:
                s.append((n, m))

    print('Case #%d: %d' % (case, len(set(s))))

