import sys
if len(sys.argv) == 1:
    sys.stdin = open("B.in")
else:
    sys.stdin = open(sys.argv[1])

def to_floats(s):
    return list(map(float, s.split()))

def get_floats():
    return to_floats(input())

n_cases = eval(input())

for case in range(1, n_cases + 1):
    farm_cost, farm_increase, goal = get_floats()

    best_time = float('inf')
    time = 0.0
    rate = 2.0
    while time < best_time:
        best_time = min(best_time, time + goal / rate)
        time += farm_cost / rate
        rate += farm_increase

    print("Case #%d: %.7f" % (case, best_time))
