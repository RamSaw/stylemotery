YES = 0
NO = 1

messages = [
    "YES",
    "NO"
]

# def check_height(lawn, N, M, height, was, levels):
#     for x, y in levels[height]:
#         if x not in was['row'] and y not in was['col']:
#             # Try the row
#             good_row = True
#             for i in xrange(M):
#                 if lawn[x][i] > height:
#                     good_row = False
#                     break
#             if good_row:
#                 was['row'].add(x)
#                 continue
#             good_col = True
#             for i in xrange(N):
#                 if lawn[i][y] > height:
#                     good_col = False
#                     break
#             if good_col:
#                 was['col'].add(y)
#                 continue
#             # There is no good cut for this field
#             return False
#     return True

# def check_lawn(lawn, N, M):
#     heights_dict = {i: False for i in xrange(1,101)}
#     for i in xrange(N):
#         for j in xrange(M):
#             heights_dict[lawn[i][j]] = True

#     heights = [key for key in sorted(heights_dict.keys()) if heights_dict[key]]
#     if len(heights) == 1:
#         return YES
#     levels = {height: [] for height in heights}

#     for i in xrange(N):
#         for j in xrange(M):
#             levels[lawn[i][j]].append((i, j))

#     was = {
#         "row": set()
#         "col": set()
#     }

#     for i, height in enumerate(heights[:-1]):
#         if not check_height(lawn, N, M, height, was, levels):
#             return NO

#     return YES

def check_lawn(lawn, N, M):
    row_maxs = [0 for _ in range(N)]
    col_maxs = [0 for _ in range(M)]

    for i in range(N):
        cmax = -1
        for j in range(M):
            if lawn[i][j] > cmax:
                cmax = lawn[i][j]
        row_maxs[i] = cmax

    for j in range(M):
        cmax = -1
        for i in range(N):
            if lawn[i][j] > cmax:
                cmax = lawn[i][j]
        col_maxs[j] = cmax

    for i in range(N):
        for j in range(M):
            if row_maxs[i] > lawn[i][j] and col_maxs[j] > lawn[i][j]:
                return NO

    return YES


T = int(input().strip())
for i in range(T):
    N, M = list(map(int, input().strip().split(' ')))
    lawn = [[] for j in range(N)]
    for j in range(N):
        lawn[j] = list(map(int, input().strip().split(' ')))
    print("Case #%s: %s" % (i + 1, messages[check_lawn(lawn, N, M)]))
