X_WON = 0
O_WON = 1
DRAW = 2
NEITHER = 3

messages = [
    "X won",
    "O won",
    "Draw",
    "Game has not completed"
]


def check_win(state):
    if state['X'] == 4 or state['X'] == 3 and state['T'] == 1:
        return X_WON
    elif state['O'] == 4 or state['O'] == 3 and state['T'] == 1:
        return O_WON
    else:
        return -1

def check_state(table):

    # Check rows
    for i in range(4):
        state = {'T': 0, 'X': 0, 'O': 0, '.': 0}
        for j in range(4):
            state[table[i][j]] += 1
        res = check_win(state)
        if res != -1:
            return res

    # Check cols
    for i in range(4):
        state = {'T': 0, 'X': 0, 'O': 0, '.': 0}
        for j in range(4):
            state[table[j][i]] += 1
        res = check_win(state)
        if res != -1:
            return res

    # Check diags
    # Normal
    state = {'T': 0, 'X': 0, 'O': 0, '.': 0}
    for i in range(4):
        state[table[i][i]] +=  1
    res = check_win(state)
    if res != -1:
        return res

    # Cross
    state = {'T': 0, 'X': 0, 'O': 0, '.': 0}
    for i in range(4):
        state[table[i][3 - i]] +=  1
    res = check_win(state)
    if res != -1:
        return res

    # Check not full
    for i in range(4):
        for j in range(4):
            if table[i][j] == '.':
                return NEITHER

    # It's a draw
    return DRAW

T = int(input().strip())
for i in range(T):
    table = [[] for _ in range(4)]
    for j in range(4):
        table[j] = list(input().strip())
    # print "table: %s" % table
    if i != T - 1:
        input()
    print("Case #%s: %s" % (i + 1, messages[check_state(table)]))