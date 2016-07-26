__author__ = 'jrokicki'

import sys
RL = lambda: sys.stdin.readline().strip()
IA = lambda: list(map(int, RL().split(" ")))
LA = lambda: list(map(int, RL().split(" ")))
FA = lambda: list(map(float, RL().split(" ")))

T = int(sys.stdin.readline())

for CASE in range(T):
    C,F,X = FA()
    tick = 2.
    answer = X/tick

    game = 0
    while True:
        span = C / tick
        tick += F
        test = game + span + X/tick
        game = game + span

        if test < answer:
            answer = test
        else:
            if tick > X:
                break

    print("Case #%d: %s" % (CASE+1, answer))

