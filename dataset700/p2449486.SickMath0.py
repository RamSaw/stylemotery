f = open('B-small-attempt0.in', 'r')
g = open('output', 'w')

T = int(f.readline()[:-1])

for case in range(T) :
    A = []
    N, M = list(map(int, f.readline()[:-1].split()))
    for i in range(N) : A.append(list(map(int, f.readline()[:-1].split())))
    for line in A : print(line)
    maxPerRow = list(map(max, A))
    maxPerColumn = list(map(max, list(zip(*A))))
    res = all(A[i][j] in (maxPerRow[i], maxPerColumn[j]) for i in range(N) for j in range(M))
    outString = 'Case #' + str(case+1) + ': ' + ('YES' if res else 'NO') + '\n'
    print(outString[:-1])
    g.write(outString)

f.close()
g.close()
