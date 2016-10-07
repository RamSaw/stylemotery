# -*- coding: utf-8 -*-

T = int(input())
for test_case in range(1, T + 1):
    N1 = int(input())
    for i in range(4):
        if i + 1 == N1:
            R1 = list(map(int, input().split(' ')))
        else:
            input()
    N2 = int(input())
    for i in range(4):
        if i + 1 == N2:
            R2 = list(map(int, input().split(' ')))
        else:
            input()
    assert 1 <= N1 <= 4
    assert 1 <= N2 <= 4
    assert len(R1) == len(R2) == 4

    num = set(R1) & set(R2)
    if len(num) == 1:
        answer = num.pop()
    elif 1 < len(num):
        answer = 'Bad magician!'
    else:
        answer = 'Volunteer cheated!'
    print('Case #{}: {}'.format(test_case, answer))
