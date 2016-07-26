import copy
def is_mine_present(arr, m, n, x, y):
    for i in range(max(0, x-1), min(m, x+2)):
        for j in range(max(0, y-1), min(n, y+2)):
            if arr[i][j] == '*':
                return True
    return False

def is_solved(arr, m, n, x, y):
    for i in range(m):
        for j in range(n):
            if arr[i][j] == '.':
                return False
    global done
    done = x, y
    return True
    
def generate_combinations(arr, m, n, mines, x, y, cur):
    if cur == mines:
        #print arr, cur, mines
        return solve(arr, m, n)    
    if y == n:
        y = 0
        x += 1
        if x == m:
            return False
    
    arr[x][y] = '*'
    ans = generate_combinations(arr, m, n, mines, x, y+1, cur+1)
    if ans == False:
        arr[x][y] = '.'
        return generate_combinations(arr, m, n, mines, x, y+1, cur)
    return True
    
def solve(arr, m, n):
    for i in range(m):
        for j in range(n):
            if arr[i][j] != '*' and is_mine_present(arr, m, n, i, j) is False:
                temp = copy.deepcopy(arr)
                temp[i][j] = "#"
                find(temp, m, n, i, j)
                ans = is_solved(temp, m, n, i, j)
                if ans is True:
                    return True
    return False                                                

def find(arr, m, n, x, y):
    for i in range(max(0, x-1), min(m, x+2)):
        for j in range(max(0, y-1), min(n, y+2)):
            if arr[i][j] == "#":
                continue
            arr[i][j] = "#"
            if (i != x or j != y) and is_mine_present(arr, m, n, i, j) is False:
                find(arr, m, n, i, j)

#fi=open("C-small-attempt0.in",'r')#Input File
#fo=open("C-small-attempt0.out",'w')#Output File

fi=open("C-small-attempt5.in",'r')#Input File
fo=open("C-small-attempt5.out",'w')#Output File

#fi=open("C-large.in",'r')#Input File
#fo=open("C-large.out","w")#Output File

#fi=open("C.in",'r')#Input File
#fo=open("C.out","w")#Output File
    
T=int(fi.readline())
for case in range(1,T+1,1):
    ############################################
    m ,n, mines = list(map(int, fi.readline().split()))
    done = (0, 0)
    arr = [['.']*n for i in range(m)]
    if m*n-mines == 1:
        ans = True
        arr = [['*']*n for i in range(m)]
    elif m >= 3 and n >= 3 and m*n-mines < 4:
        ans = False
    else:    
        ans = generate_combinations(arr, m, n, mines, 0, 0, 0)
    if ans is True:
        arr[done[0]][done[1]] = "c"
        ans = ""
        for i in range(m):
            ans += ''.join(arr[i]) + "\n"
    else:
        ans = "Impossible\n"
    #print "Case #%s:\n%s"%(case, ans)
    ############################################
    fo.write("Case #%s:\n%s"%(case, ans))
