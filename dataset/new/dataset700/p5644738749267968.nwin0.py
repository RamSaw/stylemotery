from collections import deque
from bisect import *
def solve():
    n = int(input())
    a = list(map(float, input().split()))
    b = list(map(float, input().split()))
    a.sort()
    b.sort()
    da = deque(a)
    db = deque(b)
    k = 0
    while da:
        if da[0] < db[0]:
            da.popleft()
            db.pop()
        else:
            da.popleft()
            db.popleft()
            k += 1
    print(k, end=' ')
    k = 0
    for i, x in enumerate(a):
        j = bisect(b, x)
        k = max(k, j - i)
    print(k)
for t in range(int(input())):
    print("Case #%d:" % (t+1), end=' ')
    solve()
