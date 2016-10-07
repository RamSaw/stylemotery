T = int(input())

def readSq(n):
    res = []
    for i in range(n):
        res.append(set(map(int, input().split())))
    return res

def solve():
    a1 = int(input())
    s1 = readSq(4)
    a2 = int(input())
    s2 = readSq(4)
    ans = s1[a1-1] & s2[a2-1]
    if len(ans) == 0:
        return "Volunteer cheated!"
    if len(ans) > 1:
        return "Bad magician!"
    return str(list(ans)[0])

for z in range(T):
    print("Case #%d: %s" % (z+1, solve()))
