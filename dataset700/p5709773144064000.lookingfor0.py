T = int(input())

def solve(C, F, X):
    best = x/2
    buildTime, speed = 0, 2
    while True:
        buildTime += C/speed
        if buildTime > best:
            break
        speed += F
        best = min(best, buildTime + X/speed)
    return best

for z in range(T):
    c, f, x = list(map(float, input().split()))
    print("Case #%d: %.7f" % (z+1, solve(c, f, x)))