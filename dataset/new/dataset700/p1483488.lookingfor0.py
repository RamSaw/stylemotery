T = int(input())

def isRecycle(x, y, d):
    k = 10**(d-1)
    for i in range(ndigits):
        y = k*(y%10) + y/10
        if x == y:
            return True
    return False

for z in range(1, T+1):
    res = 0
    A, B = list(map(int, input().split()))
    ndigits = len(str(A))
    for i in range(A, B):
        for j in range(i+1, B+1):
           if isRecycle(i, j, ndigits):
               res += 1
    print("Case #%d:" % z, res)