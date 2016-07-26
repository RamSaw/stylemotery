from sys import stdin

def read_ints():
    return list(map(int, stdin.readline().rstrip().split(' ')))

def print_lawn(lawn):
    for row in lawn:
        print(row)
        
def is_higher(a, i, j, N, M):
    s = a[i][j]
    v, h = False, False
    for ii in range(N):
        if a[ii][j] > s:
            v = True
            break
    for jj in range(M):
        if a[i][jj] > s:
            h = True
            break
    return v and h
    
def check(a, N, M):
    if N == 1 or M == 1:
        return True
    else:
        for i in range(N):
            for j in range(M):
                h = is_higher(a, i, j, N, M)
                if h:
                    return False
        return True
    
def main():
    T = int(stdin.readline())
    for Ti in range(T):
        N, M = read_ints()
        a = []
        for i in range(N):
            a.append(read_ints())
        answer = 'YES' if check(a, N, M) else 'NO'
        #print_lawn(a)
        print('Case #{}: {}'.format(Ti + 1, answer))
        
main()
