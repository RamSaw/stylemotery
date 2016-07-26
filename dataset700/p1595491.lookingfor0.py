T = int(input())

for z in range(1, T+1):
    a = list(map(int, input().split()))
    s, p = a[1:3]
    a = a[3:]
    A = 0 if p == 0 else 3*p - 2
    B = 0 if p == 0 else 1 if p == 1 else 3*p-4
    x = len([x for x in a if x >= A])
    y = len([x for x in a if x >= B]) - x
    res = x + min(s, y)
    print("Case #%d:" % z, res)