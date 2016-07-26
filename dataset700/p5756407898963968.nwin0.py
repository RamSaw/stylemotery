def solve():
    r1 = int(input())
    a1 = [list(map(int, input().split())) for i in range(4)]
    r2 = int(input())
    a2 = [list(map(int, input().split())) for i in range(4)]
    ans = -1
    for i in range(1, 17):
        if i in a1[r1-1] and i in a2[r2-1]:
            if ans != -1:
                return "Bad magician!"
            ans = i
    if ans == -1:
        return "Volunteer cheated!"
    return ans
for t in range(int(input())):
    print("Case #%d:" % (t + 1), solve())
