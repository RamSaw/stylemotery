import sys

if __name__ == "__main__":
    f = sys.stdin
    if len(sys.argv) >= 2:
        fn = sys.argv[1]
        if fn != '-':
            f = open(fn)

    t = int(f.readline())
    for _t in range(t):
        a, b = list(map(int, f.readline().split()))

        total = 0
        for i in range(a, b):
            # print i
            s = set()
            cs = str(i)
            for j in range(1, len(cs)):
                k = int(cs[j:] + cs[:j])
                if i < k <= b:
                    s.add(k)
            # print s
            # print
            total += len(s)

        print("Case #%d: %d" % (_t + 1, total))
