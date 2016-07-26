# -*- coding: utf-8 -*-

T = int(input())
for test_case_id in range(1, T + 1):
    cells = []
    for i in range(4):
        cells.append(input())
    input()

    R = list(range(4))
    for c in 'XO':
        if (
            any(all(cells[i][j] in (c, 'T') for j in R) for i in R) or
            any(all(cells[i][j] in (c, 'T') for i in R) for j in R) or
            all(cells[i][i] in (c, 'T') for i in R) or
            all(cells[i][3 - i] in (c, 'T') for i in R)
        ):
            print('Case #{}: {} won'.format(test_case_id, c))
            break
    else:
        if '.' in ''.join(cells):
            print('Case #{}: Game has not completed'.format(test_case_id))
        else:
            print('Case #{}: Draw'.format(test_case_id))
