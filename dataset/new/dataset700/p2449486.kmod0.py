import sys

if __name__ == "__main__":
    f = sys.stdin
    if len(sys.argv) >= 2:
        fn = sys.argv[1]
        if fn != '-':
            f = open(fn)

    t = int(f.readline())
    for _t in range(t):
        n, m = list(map(int, f.readline().split()))
        b = []
        for i in range(n):
            b.append(list(map(int, f.readline().split())))
            assert len(b[-1]) == m
        # print b

        max_h = [0] * n
        max_v = [0] * m

        for i in range(n):
            for j in range(m):
                t = b[i][j]
                max_h[i] = max(max_h[i], t)
                max_v[j] = max(max_v[j], t)
        can = True
        for i in range(n):
            if not can:
                break
            for j in range(m):
                t = b[i][j]
                if max_h[i] > t and max_v[j] > t:
                    can = False
                    break

        print("Case #%d: %s" % (_t+1, "YES" if can else "NO"))
