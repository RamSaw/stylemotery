from bisect import bisect_right

def generateTable(N):
    msq = list(range(1, 4)) + [11, 22] + [101, 111, 121, 202, 212]
    nums = revnums = [1, 2]

    digsums = [1, 4]

    for i in range(2, N+1):
        a, b, d = [], [], []
        p = 10**(i-1)
        for j in range(len(nums)):
            for k in range(3):
                n, rn, ds = 10*nums[j] + k, revnums[j] + k*p, digsums[j] + k*k
                if ds < 5:
                    a.append(n)
                    b.append(rn)
                    d.append(ds)
                    msq.append(10*p*n + rn) # even length
                    if i == N:
                        continue
                    for l in range(3): # odd length
                        if 2*ds + l*l < 10:
                            msq.append(100*p*n + 10*p*l + rn)
        nums, revnums, digsums = a, b, d

    msq.sort()
    return msq

def getNum(A, B):
    return bisect_right(tab2, B) - bisect_right(tab2, A-1)

N = 10
tab = generateTable(N)
tab2 = [n**2 for n in tab]

T = int(input())
for z in range(T):
    A, B = list(map(int, input().split()))
    print("Case #%d: %d" % (z+1, getNum(A, B)))
