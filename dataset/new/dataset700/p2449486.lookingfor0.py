T = int(input())

def readMatrix(nlines):
    return [list(map(int, input().split(' '))) for i in range(nlines)]

def transpose(A):
    return [[line[i] for line in A] for i in range(len(A[0]))]

for z in range(T):
    M, N = list(map(int, input().split(' ')))
    A = readMatrix(M)
    b = list(map(max, A))
    c = list(map(max, transpose(A)))
    fl = True
    for i in range(M):
        for j in range(N):
            if A[i][j] != min(b[i], c[j]):
                fl = False
    print("Case #%d: %s" % (z+1, "YES" if fl else "NO"))