from functools import reduce
# -*- coding: utf-8 -*-

T = int(input())
for test_case_id in range(1, T + 1):
    N, M = list(map(int, input().split()))
    A = []
    for i in range(N):
        A.append(list(map(int, input().split())))

    heights = reduce(lambda a, b: a | b, (set(row) for row in A))
    for y in range(N):
        for x in range(M):
            if (
                any(A[y][j] > A[y][x] for j in range(M)) and
                any(A[i][x] > A[y][x] for i in range(N))
            ):
                # Found a region surrounded by higher regions.
                print('Case #{}: NO'.format(test_case_id))
                break
        else:
            continue
        break
    else:
        print('Case #{}: YES'.format(test_case_id))
