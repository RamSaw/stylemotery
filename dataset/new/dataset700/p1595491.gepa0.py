import sys


if __name__ == "__main__":
    T = int(sys.stdin.readline().strip())
    for i in range(T):
        values = list(map(int, sys.stdin.readline().strip().split(' ')))
        _N, S, p = values[0:3]
        t = values[3:]
        min_normal = p + 2 * max(0, p - 1)
        min_surprising = p + 2 * max(0, p - 2)
        cnt_normal = len([x for x in t if x >= min_normal])
        cnt_surprising = len([x for x in t if x >= min_surprising]) - cnt_normal
        print("Case #%d: %s" % (i + 1, cnt_normal + min(cnt_surprising, S)))
