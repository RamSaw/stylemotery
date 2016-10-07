from itertools import product
def solve():
    h, w, m = list(map(int, input().split()))
    if h == 1:
        print('c' + '.' * (h * w - m - 1) + '*' * m)
    elif w == 1:
        for c in 'c' + '.' * (h * w - m - 1) + '*' * m:
            print(c)
    elif h * w - m == 1:
        print('c' + '*' * (w - 1))
        for _ in range(h-1):
            print('*' * w)
    else:
        m = h * w - m
        for i in range(h-1):
            for j in range(w-1):
                t = (i + 2) * 2 + (j + 2) * 2 - 4
                r = (i + 2) * (j + 2)
                if t <= m <= r:
                    a = [['*'] * w for _ in range(h)]
                    for k in range(i+2):
                        a[k][0] = '.'
                        a[k][1] = '.'
                    for k in range(j+2):
                        a[0][k] = '.'
                        a[1][k] = '.'
                    for y, x in product(list(range(2, i+2)), list(range(2, j+2))):
                        if y == 1 and x == 1:
                            continue
                        if t >= m:
                            break
                        a[y][x] = '.'
                        t += 1
                    a[0][0] = 'c'
                    for s in a:
                        print(''.join(s))
                    return
        print('Impossible')
for t in range(int(input())):
    print("Case #%d:" % (t + 1))
    solve()
