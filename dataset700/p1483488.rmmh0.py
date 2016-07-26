n_cases = eval(input())

for case in range(1, n_cases + 1):
    a, b = list(map(int, input().split()))

    out = 0
    e = 10 ** (len(str(a)) - 1)
    for n in range(a, b):
        s = str(n)
        m = n
        while True:
            m = (m / 10) + (m % 10 * e)
            if n < m <= b:
                out += 1
            if m == n:
                break

    print("Case #%d: %s" % (case, out))
